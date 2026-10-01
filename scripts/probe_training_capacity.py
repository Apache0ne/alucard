#!/usr/bin/env python3
"""Probe real Alucard training throughput and VRAM capacity safely.

Each candidate batch size runs in a fresh subprocess. When that subprocess exits,
all CUDA allocations from the model, optimizer, activations, EMA model, and data
batch are released by the CUDA context. This avoids stale tensors or failed OOM
attempts contaminating later tests.

The worker executes the same core v2 training path: real SpriteDataset batches,
UNet with gradient checkpointing, BF16/FP16 autocast, AdamW, gradient clipping,
and EMA updates. The parent prints a compact table and recommends the fastest
batch that stays below the requested VRAM headroom.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _worker(args) -> int:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from alucard.dataset import SpriteDataset
    from alucard.model import UNet
    from alucard.train import flow_matching_losses, update_ema

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        precision = args.precision

    if precision == "bf16":
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif precision == "fp16":
        amp_dtype = torch.float16
        use_scaler = True
    elif precision == "fp32":
        amp_dtype = None
        use_scaler = False
    else:
        raise ValueError(precision)

    ds = SpriteDataset(
        args.data_dir,
        augment=False,
        flip_prob=0.0,
        palette_swap_prob=0.0,
        sanitize_transparent_rgb=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    iterator = iter(loader)

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
        "timestep_scale": 1000.0,
    }
    model = UNet(**model_config).to(device)
    model.enable_gradient_checkpointing()
    ema_model = copy.deepcopy(model)
    ema_model.disable_gradient_checkpointing()
    ema_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    loss_kwargs = dict(
        alpha_flow_weight=2.0,
        alpha_recon_weight=1.0,
        rgb_recon_weight=0.5,
        alpha_edge_weight=0.1,
        alpha_binary_weight=0.05,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    measured_times = []
    measured_losses = []
    total_steps = args.warmup_steps + args.measure_steps

    for step in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        x_0 = batch["image"].to(device, non_blocking=True)
        text_emb = batch["text_emb"].to(device, dtype=torch.float32, non_blocking=True)
        ref = batch["ref"].to(device, non_blocking=True)
        has_ref = batch["has_ref"].to(device, non_blocking=True)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            losses = flow_matching_losses(
                model,
                x_0,
                text_emb,
                ref,
                has_ref,
                null_text_emb=model.null_text_emb,
                **loss_kwargs,
            )
            loss = losses["total"]

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_ran = scaler.get_scale() >= old_scale
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_ran = True

        optimizer.zero_grad(set_to_none=True)
        if optimizer_ran:
            update_ema(ema_model, model, 0.9999)

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        if step >= args.warmup_steps:
            measured_times.append(dt)
            measured_losses.append(float(loss.detach()))

        # Do not keep batch tensors alive between steps.
        del batch, x_0, text_emb, ref, has_ref, losses, loss

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

    mean_step = sum(measured_times) / len(measured_times)
    images_per_sec = args.batch_size / mean_step

    result = {
        "ok": True,
        "batch_size": args.batch_size,
        "precision": precision,
        "mean_step_seconds": mean_step,
        "images_per_second": images_per_sec,
        "peak_allocated_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "total_vram_gb": total_vram,
        "vram_fraction_reserved": peak_reserved / total_vram,
        "loss_first": measured_losses[0],
        "loss_last": measured_losses[-1],
        "loss_mean": sum(measured_losses) / len(measured_losses),
    }
    print("PROBE_RESULT=" + json.dumps(result), flush=True)

    # Explicit cleanup is useful when worker mode is called directly; parent
    # mode additionally gets process-exit cleanup, which is the stronger guard.
    del iterator, loader, ds, optimizer, scaler, ema_model, model
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    return 0


def _parse_worker_output(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        if line.startswith("PROBE_RESULT="):
            return json.loads(line.split("=", 1)[1])
    return None


def _parent(args) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {name} | VRAM: {total:.2f} GiB")
    print("Each batch size is tested in a separate subprocess for hard CUDA cleanup.\n")

    candidates = [int(x) for x in args.batches.split(",") if x.strip()]
    results = []

    for bs in candidates:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--data-dir", args.data_dir,
            "--batch-size", str(bs),
            "--precision", args.precision,
            "--warmup-steps", str(args.warmup_steps),
            "--measure-steps", str(args.measure_steps),
            "--num-workers", str(args.num_workers),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
        ]
        print(f"Testing batch {bs:>4} ... ", end="", flush=True)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        result = _parse_worker_output(proc.stdout)

        if proc.returncode == 0 and result:
            results.append(result)
            print(
                f"{result['images_per_second']:.1f} img/s | "
                f"{result['mean_step_seconds']:.3f}s/step | "
                f"alloc {result['peak_allocated_gb']:.2f}G | "
                f"reserved {result['peak_reserved_gb']:.2f}G"
            )
        else:
            combined = (proc.stdout + "\n" + proc.stderr).lower()
            if "out of memory" in combined or "cuda error: out of memory" in combined:
                print("OOM")
            else:
                print("FAILED")
                print(proc.stdout[-2000:])
                print(proc.stderr[-4000:])
            if args.stop_on_oom:
                break

    if not results:
        raise RuntimeError("No batch size completed successfully")

    safe = [r for r in results if r["vram_fraction_reserved"] <= args.max_vram_fraction]
    pool = safe or results
    best = max(pool, key=lambda r: r["images_per_second"])
    max_fit = max(results, key=lambda r: r["batch_size"])

    print("\nResults")
    print("batch | img/s | sec/step | alloc GB | reserved GB | reserved % | mean loss")
    print("------|-------|----------|----------|-------------|------------|----------")
    for r in results:
        print(
            f"{r['batch_size']:>5} | "
            f"{r['images_per_second']:>5.1f} | "
            f"{r['mean_step_seconds']:>8.3f} | "
            f"{r['peak_allocated_gb']:>8.2f} | "
            f"{r['peak_reserved_gb']:>11.2f} | "
            f"{100*r['vram_fraction_reserved']:>9.1f}% | "
            f"{r['loss_mean']:.4f}"
        )

    recommendation = {
        "gpu": name,
        "total_vram_gb": total,
        "recommended_batch_size": best["batch_size"],
        "recommended_images_per_second": best["images_per_second"],
        "recommended_peak_reserved_gb": best["peak_reserved_gb"],
        "max_tested_batch_that_fit": max_fit["batch_size"],
        "max_vram_fraction": args.max_vram_fraction,
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recommendation, indent=2))

    print(
        f"\nRecommended batch: {best['batch_size']} "
        f"({best['images_per_second']:.1f} img/s, "
        f"{best['peak_reserved_gb']:.2f}/{total:.2f} GiB reserved)"
    )
    print(f"Largest tested batch that fit: {max_fit['batch_size']}")
    print(f"Saved: {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Probe Alucard v2 real training GPU capacity")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batches", default="16,32,48,64,80,96,112,128,160,192,224,256")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--measure-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-vram-fraction", type=float, default=0.90)
    p.add_argument("--stop-on-oom", action="store_true")
    p.add_argument("--output-json", default="capacity_probe.json")

    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, default=32, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
