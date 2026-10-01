"""Training for the corrected Alucard v2 flow-matching pipeline.

Key differences from the original experimental run:
- semantic-changing augmentation is disabled by default;
- transparent RGB is canonicalized by the dataset loader;
- normalized t in [0,1] uses an explicit timestep embedding scale;
- alpha receives stronger flow supervision plus clean-image endpoint losses;
- gradient accumulation always flushes correctly at epoch boundaries;
- BF16 is preferred on supported CUDA hardware; FP16 scaler state is saved;
- best checkpoints use held-out validation loss, not training loss;
- fixed-prompt, fixed-seed RGBA/alpha/checkerboard previews are generated.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from alucard.dataset import SpriteDataset
from alucard.model import UNet
from alucard.sample import sample, tensor_to_rgba_image


class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_FlushHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

DEFAULT_EVAL_PROMPTS = [
    "pixel art, gray, small, wizard, mage, spellcaster, front view",
    "pixel art, colorful, medium-sized, warrior, armored character, front view",
    "pixel art, green, small, slime, blob creature, front view",
    "pixel art, colorful, medium-sized, dragon, creature, side view",
    "pixel art, red, small, potion, bottle, item, front view",
    "pixel art, brown, medium-sized, treasure chest, item, front view",
]


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1 - decay)


def alpha_edge_loss(pred_alpha: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    """L1 loss on horizontal/vertical alpha gradients."""
    pred_dx = pred_alpha[..., :, 1:] - pred_alpha[..., :, :-1]
    tgt_dx = target_alpha[..., :, 1:] - target_alpha[..., :, :-1]
    pred_dy = pred_alpha[..., 1:, :] - pred_alpha[..., :-1, :]
    tgt_dy = target_alpha[..., 1:, :] - target_alpha[..., :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy))


def flow_matching_losses(
    model: nn.Module,
    x_0: torch.Tensor,
    text_emb: torch.Tensor,
    ref: torch.Tensor,
    has_ref: torch.Tensor,
    text_drop_prob: float = 0.10,
    ref_drop_prob: float = 0.20,
    both_drop_prob: float = 0.05,
    null_text_emb: torch.Tensor | None = None,
    alpha_flow_weight: float = 2.0,
    alpha_recon_weight: float = 1.0,
    rgb_recon_weight: float = 0.5,
    alpha_edge_weight: float = 0.1,
    alpha_binary_weight: float = 0.05,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Compute rectified-flow loss plus alpha-aware endpoint supervision."""
    batch = x_0.shape[0]
    device = x_0.device

    t = torch.rand(batch, device=device, generator=generator)
    noise = torch.randn(x_0.shape, device=device, dtype=x_0.dtype, generator=generator)
    t_expand = t[:, None, None, None]
    x_t = (1 - t_expand) * x_0 + t_expand * noise
    target_v = noise - x_0

    drop_rand = torch.rand(batch, device=device, generator=generator)
    drop_both = drop_rand < both_drop_prob
    drop_text = (drop_rand >= both_drop_prob) & (drop_rand < both_drop_prob + text_drop_prob)
    drop_ref = (drop_rand >= both_drop_prob + text_drop_prob) & (
        drop_rand < both_drop_prob + text_drop_prob + ref_drop_prob
    )

    text_masked = text_emb.clone()
    null_mask = drop_both | drop_text
    _null_emb = null_text_emb if null_text_emb is not None else model.null_text_emb
    if null_mask.any():
        text_masked[null_mask] = _null_emb.to(device=device, dtype=text_emb.dtype)

    ref_masked = ref.clone()
    ref_null_mask = drop_both | drop_ref | ~has_ref
    if ref_null_mask.any():
        ref_masked[ref_null_mask] = 0.0

    pred_v = model(x_t, t, text_masked, ref_masked)

    channel_weights = torch.tensor(
        [1.0, 1.0, 1.0, alpha_flow_weight], device=device, dtype=pred_v.dtype
    )[None, :, None, None]
    flow = ((pred_v - target_v).square() * channel_weights).mean()

    x0_pred = x_t - t_expand * pred_v
    target_alpha = x_0[:, 3:4]
    pred_alpha = x0_pred[:, 3:4]
    alpha_recon = F.smooth_l1_loss(pred_alpha, target_alpha)

    alpha01 = ((target_alpha + 1.0) * 0.5).clamp(0, 1).detach()
    rgb_sq = (x0_pred[:, :3] - x_0[:, :3]).square()
    rgb_denom = alpha01.sum().clamp_min(1.0) * 3.0
    rgb_recon = (rgb_sq * alpha01).sum() / rgb_denom

    edge = alpha_edge_loss(pred_alpha, target_alpha)

    target01 = ((target_alpha + 1.0) * 0.5).clamp(0, 1)
    pred01 = ((pred_alpha + 1.0) * 0.5).clamp(0, 1)
    binary_mask = ((target01 <= 0.02) | (target01 >= 0.98)).float()
    binary_penalty = 4.0 * pred01 * (1.0 - pred01)
    binary = (binary_penalty * binary_mask).sum() / binary_mask.sum().clamp_min(1.0)

    total = (
        flow
        + alpha_recon_weight * alpha_recon
        + rgb_recon_weight * rgb_recon
        + alpha_edge_weight * edge
        + alpha_binary_weight * binary
    )
    return {
        "total": total,
        "flow": flow.detach(),
        "alpha_recon": alpha_recon.detach(),
        "rgb_recon": rgb_recon.detach(),
        "alpha_edge": edge.detach(),
        "alpha_binary": binary.detach(),
    }


def flow_matching_loss(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible scalar loss wrapper."""
    return flow_matching_losses(*args, **kwargs)["total"]


def _resolve_precision(device: torch.device, precision: str) -> tuple[torch.dtype | None, bool]:
    if device.type != "cuda" or precision == "fp32":
        return None, False
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            logger.warning("BF16 requested but unsupported; falling back to FP16")
            return torch.float16, True
        return torch.bfloat16, False
    if precision == "fp16":
        return torch.float16, True
    raise ValueError(f"Unknown precision: {precision}")


def _encode_eval_prompts(prompts: list[str]) -> torch.Tensor:
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.eval()
    tokens = tokenizer(prompts)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    del clip_model
    return emb.float().cpu()


def _checkerboard(size: int, cell: int = 8) -> Image.Image:
    arr = torch.empty(size, size, 4, dtype=torch.uint8)
    yy = torch.arange(size)[:, None]
    xx = torch.arange(size)[None, :]
    mask = ((xx // cell + yy // cell) % 2).bool()
    arr[..., :3] = 200
    arr[mask, :3] = 235
    arr[..., 3] = 255
    return Image.fromarray(arr.numpy(), "RGBA")


def _save_fixed_preview(samples: torch.Tensor, prompts: list[str], output_dir: Path, epoch: int) -> dict[str, float]:
    epoch_dir = output_dir / "samples" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    size = samples.shape[-1]
    checker = _checkerboard(size)
    label_h = 28
    rows = []
    partial_fracs = []
    binary_fracs = []

    for i, (tensor, prompt) in enumerate(zip(samples, prompts)):
        rgba = tensor_to_rgba_image(tensor)
        rgba.save(epoch_dir / f"{i:02d}_native.png")
        alpha = rgba.getchannel("A")
        alpha.save(epoch_dir / f"{i:02d}_alpha.png")
        composite = Image.alpha_composite(checker, rgba)
        composite.save(epoch_dir / f"{i:02d}_checker.png")

        alpha_t = ((tensor[3] + 1.0) * 0.5).clamp(0, 1)
        partial_fracs.append(float(((alpha_t > 0.05) & (alpha_t < 0.95)).float().mean()))
        binary_fracs.append(float(((alpha_t <= 0.05) | (alpha_t >= 0.95)).float().mean()))

        row = Image.new("RGBA", (size * 2, size + label_h), (255, 255, 255, 255))
        row.paste(composite, (0, 0))
        row.paste(Image.merge("RGBA", (alpha, alpha, alpha, Image.new("L", alpha.size, 255))), (size, 0))
        ImageDraw.Draw(row).text((4, size + 5), prompt[:70], fill=(0, 0, 0, 255))
        rows.append(row)

    sheet = Image.new("RGBA", (size * 2, (size + label_h) * len(rows)), (255, 255, 255, 255))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * (size + label_h)))
    sheet.save(output_dir / "samples" / f"epoch_{epoch:04d}_contact.png")

    return {
        "preview_partial_alpha": sum(partial_fracs) / len(partial_fracs),
        "preview_binary_alpha": sum(binary_fracs) / len(binary_fracs),
    }


def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    max_batches: int,
    loss_kwargs: dict,
    seed: int = 12345,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            x_0 = batch["image"].to(device, non_blocking=True)
            text_emb = batch["text_emb"].to(device, dtype=torch.float32, non_blocking=True)
            ref = batch["ref"].to(device, non_blocking=True)
            has_ref = batch["has_ref"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                losses = flow_matching_losses(
                    model, x_0, text_emb, ref, has_ref,
                    text_drop_prob=0.0, ref_drop_prob=0.0, both_drop_prob=0.0,
                    generator=generator, **loss_kwargs,
                )
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n += 1
    model.train()
    return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}


def train(
    data_dir: str,
    output_dir: str = "checkpoints_v2",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    ema_decay: float = 0.9999,
    grad_accum: int = 1,
    save_every: int = 1,
    sample_every: int = 1,
    num_workers: int = 4,
    resume: str | None = None,
    wandb_project: str | None = None,
    precision: str = "auto",
    val_fraction: float = 0.02,
    val_max_batches: int = 32,
    seed: int = 42,
    flip_prob: float = 0.0,
    palette_swap_prob: float = 0.0,
    timestep_scale: float = 1000.0,
    alpha_flow_weight: float = 2.0,
    alpha_recon_weight: float = 1.0,
    rgb_recon_weight: float = 0.5,
    alpha_edge_weight: float = 0.1,
    alpha_binary_weight: float = 0.05,
    cfg_text_preview: float = 2.5,
    preview_steps: int = 30,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_path = Path(output_dir)
    (output_path / "samples").mkdir(parents=True, exist_ok=True)

    train_base = SpriteDataset(
        data_dir,
        augment=(flip_prob > 0 or palette_swap_prob > 0),
        flip_prob=flip_prob,
        palette_swap_prob=palette_swap_prob,
        sanitize_transparent_rgb=True,
    )
    val_base = SpriteDataset(data_dir, augment=False, sanitize_transparent_rgb=True)

    n_total = len(train_base)
    if n_total < 2:
        raise ValueError("Need at least two samples for train/validation split")
    n_val = max(1, int(round(n_total * val_fraction)))
    n_val = min(n_val, n_total - 1)
    split_gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=split_gen).tolist()
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]
    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    logger.info(
        "Dataset: %d total | %d train | %d val | %d train batches/epoch",
        n_total, len(train_ds), len(val_ds), len(train_loader),
    )

    model_config = {
        "in_channels": 8,
        "out_channels": 4,
        "base_channels": 64,
        "channel_mults": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [32, 16],
        "text_dim": 512,
        "dropout": 0.0,
        "image_size": 128,
        "timestep_scale": timestep_scale,
    }
    model = UNet(**model_config).to(device)
    model.enable_gradient_checkpointing()
    ema_model = copy.deepcopy(model)
    ema_model.disable_gradient_checkpointing()
    ema_model.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.1fM)", total_params, total_params / 1e6)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info("Using %d GPUs with DataParallel", num_gpus)
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, epochs * updates_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    autocast_dtype, use_scaler = _resolve_precision(device, precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logger.info("Precision: %s", "fp32" if autocast_dtype is None else str(autocast_dtype).replace("torch.", ""))

    loss_kwargs = {
        "alpha_flow_weight": alpha_flow_weight,
        "alpha_recon_weight": alpha_recon_weight,
        "rgb_recon_weight": rgb_recon_weight,
        "alpha_edge_weight": alpha_edge_weight,
        "alpha_binary_weight": alpha_binary_weight,
    }

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if use_scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("best_loss", float("inf")))
        logger.info("Resumed epoch=%d step=%d best_val=%.6f", start_epoch, global_step, best_val_loss)

    eval_prompts = DEFAULT_EVAL_PROMPTS
    eval_embeddings = _encode_eval_prompts(eval_prompts)

    config = {
        "model": model_config,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "ema_decay": ema_decay,
            "grad_accum": grad_accum,
            "precision": precision,
            "val_fraction": val_fraction,
            "seed": seed,
            "flip_prob": flip_prob,
            "palette_swap_prob": palette_swap_prob,
            **loss_kwargs,
        },
        "eval_prompts": eval_prompts,
    }
    (output_path / "config.json").write_text(json.dumps(config, indent=2))

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=config)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = {"total": 0.0, "flow": 0.0, "alpha_recon": 0.0, "rgb_recon": 0.0,
                "alpha_edge": 0.0, "alpha_binary": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            x_0 = batch["image"].to(device, non_blocking=True)
            text_emb = batch["text_emb"].to(device, dtype=torch.float32, non_blocking=True)
            ref = batch["ref"].to(device, non_blocking=True)
            has_ref = batch["has_ref"].to(device, non_blocking=True)

            group_start = (batch_idx // grad_accum) * grad_accum
            group_end = min(group_start + grad_accum, len(train_loader))
            group_size = group_end - group_start

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                losses = flow_matching_losses(
                    model, x_0, text_emb, ref, has_ref,
                    null_text_emb=raw_model.null_text_emb,
                    **loss_kwargs,
                )
                backward_loss = losses["total"] / group_size

            if use_scaler:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            should_step = (batch_idx + 1) == group_end
            if should_step:
                optimizer_ran = True
                if use_scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if use_scaler:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_ran = scaler.get_scale() >= old_scale
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                if optimizer_ran:
                    scheduler.step()
                    update_ema(ema_model, raw_model, ema_decay)
                    global_step += 1

            for k in sums:
                sums[k] += float(losses[k])
            num_batches += 1

        train_metrics = {f"train_{k}": v / max(num_batches, 1) for k, v in sums.items()}
        val_metrics = _validate(ema_model, val_loader, device, autocast_dtype, val_max_batches, loss_kwargs)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %d/%d | train %.6f | val %.6f | alpha %.6f | LR %.2e | step %d",
            epoch + 1, epochs,
            train_metrics["train_total"], val_metrics["val_total"],
            val_metrics["val_alpha_recon"], current_lr, global_step,
        )

        preview_metrics = {}
        if (epoch + 1) % sample_every == 0 or epoch == 0:
            ema_model.eval()
            preview_gen = torch.Generator(device=device).manual_seed(424242)
            with torch.no_grad():
                samples = sample(
                    ema_model,
                    eval_embeddings.to(device),
                    num_steps=preview_steps,
                    cfg_text=cfg_text_preview,
                    cfg_ref=0.0,
                    device=device,
                    generator=preview_gen,
                )
            preview_metrics = _save_fixed_preview(samples, eval_prompts, output_path, epoch + 1)
            logger.info(
                "Preview alpha: partial=%.4f binary=%.4f",
                preview_metrics["preview_partial_alpha"],
                preview_metrics["preview_binary_alpha"],
            )
            model.train()

        ckpt_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": min(best_val_loss, val_metrics["val_total"]),
            "model_config": model_config,
            "loss_config": loss_kwargs,
            "model": raw_model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "metrics": {**train_metrics, **val_metrics, **preview_metrics},
        }

        if val_metrics["val_total"] < best_val_loss:
            best_val_loss = val_metrics["val_total"]
            ckpt_payload["best_val_loss"] = best_val_loss
            torch.save(ckpt_payload, output_path / "best.pt")
            logger.info("New best validation loss %.6f", best_val_loss)

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save(ckpt_payload, output_path / "latest.pt")

        metrics_record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "lr": current_lr,
            **train_metrics,
            **val_metrics,
            **preview_metrics,
        }
        with (output_path / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics_record) + "\n")

        if wandb_project:
            import wandb
            wandb.log(metrics_record, step=global_step)

    logger.info("Training complete. Best validation loss: %.6f", best_val_loss)


def main():
    parser = argparse.ArgumentParser(description="Train corrected Alucard v2")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--val-max-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flip-prob", type=float, default=0.0)
    parser.add_argument("--palette-swap-prob", type=float, default=0.0)
    parser.add_argument("--timestep-scale", type=float, default=1000.0)
    parser.add_argument("--alpha-flow-weight", type=float, default=2.0)
    parser.add_argument("--alpha-recon-weight", type=float, default=1.0)
    parser.add_argument("--rgb-recon-weight", type=float, default=0.5)
    parser.add_argument("--alpha-edge-weight", type=float, default=0.1)
    parser.add_argument("--alpha-binary-weight", type=float, default=0.05)
    parser.add_argument("--cfg-text-preview", type=float, default=2.5)
    parser.add_argument("--preview-steps", type=int, default=30)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
