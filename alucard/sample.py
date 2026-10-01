"""Sampling / inference for flow-matching sprite generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@torch.no_grad()
def sample(
    model: nn.Module,
    text_emb: torch.Tensor,
    ref: torch.Tensor | None = None,
    num_steps: int = 20,
    cfg_text: float = 5.0,
    cfg_ref: float = 2.0,
    device: torch.device | str = "cuda",
    image_size: int = 128,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sprites by Euler integration from t=1 to t=0.

    Guidance uses the documented equation for every scale, including 0,
    fractional, and exactly-1 values:

        v = v_uncond + cfg_text * (v_text - v_uncond)
        v += cfg_ref * (v_both - v_text)   # when a reference exists
    """
    model.eval()
    device = torch.device(device)
    batch = text_emb.shape[0]
    text_emb = text_emb.to(device)
    if ref is not None:
        ref = ref.to(device)

    x = torch.randn(
        batch, 4, image_size, image_size, device=device, generator=generator
    )
    null_text = model.null_text_emb.unsqueeze(0).expand(batch, -1).to(text_emb.dtype)
    null_ref = torch.zeros(batch, 4, image_size, image_size, device=device, dtype=x.dtype)

    dt = -1.0 / num_steps
    for step in range(num_steps):
        t_val = 1.0 - step / num_steps
        t = torch.full((batch,), t_val, device=device)

        # Exact CFG equation. Avoid special branches that silently changed the
        # meaning of cfg_text<1 or cfg_ref==1 in the original sampler.
        v_uncond = model(x, t, null_text, null_ref)
        v_text = model(x, t, text_emb, null_ref)
        v_guided = v_uncond + cfg_text * (v_text - v_uncond)

        if ref is not None:
            v_both = model(x, t, text_emb, ref)
            v_guided = v_guided + cfg_ref * (v_both - v_text)

        x = x + dt * v_guided

    return x.clamp(-1, 1)


def tensor_to_rgba_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``(4,H,W)`` tensor in ``[-1,1]`` to RGBA PIL."""
    arr = ((tensor + 1) * 127.5).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGBA")


def model_kwargs_from_checkpoint(ckpt: dict) -> dict:
    """Return UNet constructor kwargs saved by v2, or old-checkpoint defaults."""
    cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    allowed = {
        "in_channels", "out_channels", "base_channels", "channel_mults",
        "num_res_blocks", "attn_resolutions", "text_dim", "dropout",
        "image_size", "timestep_scale",
    }
    out = {k: v for k, v in cfg.items() if k in allowed}
    if "channel_mults" in out:
        out["channel_mults"] = tuple(out["channel_mults"])
    if "attn_resolutions" in out:
        out["attn_resolutions"] = tuple(out["attn_resolutions"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate sprites with Alucard")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ref", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-text", type=float, default=2.5)
    parser.add_argument("--cfg-ref", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from alucard.model import UNet

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet(**model_kwargs_from_checkpoint(ckpt)).to(device)
    model.load_state_dict(ckpt.get("ema_model", ckpt.get("model", ckpt)))
    model.eval()

    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()
    tokens = tokenizer([args.prompt]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.expand(args.num_samples, -1)

    ref = None
    if args.ref:
        from alucard.dataset import load_rgba
        ref = load_rgba(Path(args.ref)).unsqueeze(0).expand(args.num_samples, -1, -1, -1).to(device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    sprites = sample(
        model, text_emb, ref,
        num_steps=args.num_steps,
        cfg_text=args.cfg_text,
        cfg_ref=args.cfg_ref,
        device=device,
        generator=generator,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.num_samples == 1:
        tensor_to_rgba_image(sprites[0]).save(output_path)
        print(f"Saved: {output_path}")
    else:
        for i in range(args.num_samples):
            p = output_path.parent / f"{output_path.stem}_{i:03d}{output_path.suffix}"
            tensor_to_rgba_image(sprites[i]).save(p)
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
