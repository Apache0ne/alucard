#!/usr/bin/env python3
"""
Convert an Alucard training checkpoint to safetensors format for distribution.

Extracts the EMA model weights (or regular model if no EMA) and saves them
as a single .safetensors file along with a config JSON.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Convert Alucard checkpoint to safetensors")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output path (.safetensors)")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA weights (default: True)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use training weights instead of EMA")
    parser.add_argument("--half", action="store_true",
                        help="Convert to float16")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.suffix == ".safetensors":
        output_path = output_path.with_suffix(".safetensors")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.no_ema:
        state_dict = ckpt["model"]
        print("Using training weights")
    elif "ema_model" in ckpt:
        state_dict = ckpt["ema_model"]
        print("Using EMA weights")
    else:
        state_dict = ckpt["model"]
        print("No EMA weights found, using training weights")

    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}
        print("Converted to float16")

    save_file(state_dict, output_path)
    size_mb = output_path.stat().st_size / 1024**2
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    config_path = output_path.with_suffix(".json")
    model_config = ckpt.get("model_config", {
        "in_channels": 8,
        "out_channels": 4,
        "base_channels": 64,
        "channel_mults": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [32, 16],
        "text_dim": 512,
        "image_size": 128,
        "timestep_scale": 1.0,
    })
    config = {
        "architecture": "alucard-unet",
        "model": model_config,
        **model_config,
        "text_encoder": "openai/clip-vit-base-patch32",
        "parameters": sum(v.numel() for v in state_dict.values()),
        "dtype": "float16" if args.half else "float32",
        "source_checkpoint": str(args.checkpoint),
        "source_epoch": ckpt.get("epoch", "unknown"),
        "source_step": ckpt.get("global_step", "unknown"),
        "weights": "ema" if not args.no_ema and "ema_model" in ckpt else "training",
    }
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Config: {config_path}")

    print("\nModel summary:")
    print(f"  Parameters: {config['parameters']:,}")
    print(f"  Dtype: {config['dtype']}")
    print(f"  Epoch: {config['source_epoch']}")
    print(f"  Step: {config['source_step']}")


if __name__ == "__main__":
    main()
