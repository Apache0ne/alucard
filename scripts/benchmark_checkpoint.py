#!/usr/bin/env python3
"""Fixed-prompt visual and quantitative benchmark for Alucard checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from alucard.model import UNet
from alucard.sample import model_kwargs_from_checkpoint, sample, tensor_to_rgba_image

PROMPTS = [
    "pixel art, gray, small, wizard, mage, spellcaster, front view",
    "pixel art, colorful, medium-sized, warrior, armored character, front view",
    "pixel art, green, small, slime, blob creature, front view",
    "pixel art, colorful, medium-sized, dragon, creature, side view",
    "pixel art, red, small, potion, bottle, item, front view",
    "pixel art, brown, medium-sized, treasure chest, item, front view",
    "pixel art, colorful, small, sword, weapon, item, side view",
    "pixel art, green, medium-sized, tree, environment object, front view",
]


def checkerboard(size: int, cell: int = 8) -> Image.Image:
    y, x = np.indices((size, size))
    mask = ((x // cell + y // cell) % 2).astype(bool)
    rgb = np.full((size, size, 4), 205, dtype=np.uint8)
    rgb[mask, :3] = 238
    rgb[..., 3] = 255
    return Image.fromarray(rgb, "RGBA")


def alpha_metrics(tensor: torch.Tensor) -> dict[str, float]:
    a = ((tensor[3].float() + 1.0) * 0.5).clamp(0, 1)
    dx = (a[:, 1:] - a[:, :-1]).abs().mean()
    dy = (a[1:, :] - a[:-1, :]).abs().mean()
    return {
        "partial_alpha_05_95": float(((a > 0.05) & (a < 0.95)).float().mean()),
        "strict_partial_alpha": float(((a > 0.0) & (a < 1.0)).float().mean()),
        "near_binary_alpha": float(((a <= 0.05) | (a >= 0.95)).float().mean()),
        "mean_alpha": float(a.mean()),
        "alpha_edge_strength": float((dx + dy) * 0.5),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark an Alucard training checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-text", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", action="append", default=None, help="Repeat to override defaults")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    prompts = args.prompt or PROMPTS

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet(**model_kwargs_from_checkpoint(ckpt)).to(device)
    model.load_state_dict(ckpt.get("ema_model", ckpt.get("model", ckpt)))
    model.eval()

    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    images = []
    alpha_stats = []
    per_prompt_seconds = []
    checker = checkerboard(128)

    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            tensor = sample(
                model,
                text_emb[i:i+1],
                num_steps=args.steps,
                cfg_text=args.cfg_text,
                cfg_ref=0.0,
                device=device,
                generator=gen,
            )[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_prompt_seconds.append(time.perf_counter() - t0)

        rgba = tensor_to_rgba_image(tensor)
        rgba.save(out / f"{i:02d}_native.png")
        alpha = rgba.getchannel("A")
        alpha.save(out / f"{i:02d}_alpha.png")
        composite = Image.alpha_composite(checker, rgba)
        composite.save(out / f"{i:02d}_checker.png")
        images.append((rgba, composite, alpha))
        alpha_stats.append(alpha_metrics(tensor))

    clip_scores = []
    with torch.no_grad():
        for i, (_, composite, _) in enumerate(images):
            image_input = preprocess(composite.convert("RGB")).unsqueeze(0).to(device)
            image_emb = clip_model.encode_image(image_input)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            clip_scores.append(float((image_emb * text_emb[i:i+1]).sum()))

    label_h = 32
    row_w = 128 * 2
    sheet = Image.new("RGBA", (row_w, (128 + label_h) * len(images)), (255, 255, 255, 255))
    for i, ((_, composite, alpha), prompt) in enumerate(zip(images, prompts)):
        row = Image.new("RGBA", (row_w, 128 + label_h), (255, 255, 255, 255))
        row.paste(composite, (0, 0))
        alpha_rgba = Image.merge("RGBA", (alpha, alpha, alpha, Image.new("L", alpha.size, 255)))
        row.paste(alpha_rgba, (128, 0))
        ImageDraw.Draw(row).text((4, 132), prompt[:68], fill=(0, 0, 0, 255))
        sheet.paste(row, (0, i * (128 + label_h)))
    sheet.save(out / "contact_sheet.png")

    peak_gb = 0.0
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    aggregate = {}
    for key in alpha_stats[0]:
        aggregate[key] = float(sum(x[key] for x in alpha_stats) / len(alpha_stats))

    report = {
        "checkpoint": str(args.checkpoint),
        "model_config": model_kwargs_from_checkpoint(ckpt),
        "steps": args.steps,
        "cfg_text": args.cfg_text,
        "seed": args.seed,
        "prompts": prompts,
        "seconds_per_image_mean": float(sum(per_prompt_seconds) / len(per_prompt_seconds)),
        "images_per_second": float(len(per_prompt_seconds) / sum(per_prompt_seconds)),
        "peak_gpu_memory_gb": peak_gb,
        "clip_cosine_mean": float(sum(clip_scores) / len(clip_scores)),
        "clip_cosine_per_prompt": clip_scores,
        "alpha_mean": aggregate,
        "alpha_per_prompt": alpha_stats,
    }
    (out / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved contact sheet: {out / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
