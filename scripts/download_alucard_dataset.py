#!/usr/bin/env python3
"""Download evilsocket/alucard-sprites and build cached CLIP embeddings.

The Hugging Face dataset is kept in Arrow format via ``save_to_disk`` instead
of exploding 300k+ rows into individual PNG files. Training then memory-maps
that dataset and indexes a single consolidated embedding tensor.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare the published Alucard sprite dataset")
    parser.add_argument("--output-dir", type=str, default="data/alucard_sprites")
    parser.add_argument("--repo", type=str, default="evilsocket/alucard-sprites")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all rows")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_path = out / "hf_dataset"
    emb_path = out / "clip_embeddings.pt"
    meta_path = out / "metadata.json"

    previous_meta = {}
    if meta_path.exists():
        try:
            previous_meta = json.loads(meta_path.read_text())
        except Exception:
            previous_meta = {}

    requested_max = max(0, int(args.max_samples))
    cache_matches_request = (
        previous_meta.get("repo") == args.repo
        and previous_meta.get("split") == args.split
        and previous_meta.get("requested_max_samples") == requested_max
    )

    reuse_dataset = hf_path.exists() and not args.force and cache_matches_request
    if reuse_dataset:
        from datasets import load_from_disk
        ds = load_from_disk(str(hf_path))
        print(f"Using existing dataset: {hf_path} ({len(ds):,} rows)")
    else:
        if hf_path.exists() and not args.force:
            print("Cached dataset settings differ from this request; rebuilding it.")
        print(f"Downloading {args.repo} [{args.split}] ...")
        ds = load_dataset(args.repo, split=args.split)
        source_rows = len(ds)
        if requested_max > 0:
            ds = ds.select(range(min(requested_max, len(ds))))
        if "image" not in ds.column_names or "text" not in ds.column_names:
            raise ValueError(f"Expected image/text columns, found {ds.column_names}")
        if hf_path.exists():
            shutil.rmtree(hf_path)
        ds.save_to_disk(str(hf_path))
        print(f"Saved Arrow dataset: {hf_path} ({len(ds):,} rows from {source_rows:,} source rows)")
        # Any embedding tensor from another dataset selection is invalid.
        if emb_path.exists():
            emb_path.unlink()

    metadata = {
        "repo": args.repo,
        "split": args.split,
        "requested_max_samples": requested_max,
        "rows": len(ds),
        "columns": ds.column_names,
        "clip_model": "ViT-B-32",
        "clip_pretrained": "openai",
        "embedding_dtype": "float16",
    }

    if args.skip_embeddings:
        meta_path.write_text(json.dumps(metadata, indent=2))
        print("Skipped CLIP embeddings.")
        return

    if emb_path.exists() and not args.force:
        existing = torch.load(emb_path, map_location="cpu", weights_only=True)
        if len(existing) == len(ds):
            print(f"Using existing embeddings: {emb_path} {tuple(existing.shape)}")
            meta_path.write_text(json.dumps(metadata, indent=2))
            return
        print("Existing embedding count does not match dataset; rebuilding.")

    import open_clip

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Embedding device: {device}")

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    text_only = ds.select_columns(["text"])
    embeddings = torch.empty((len(ds), 512), dtype=torch.float16)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    batch_size = args.embedding_batch_size

    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        captions = text_only[start:end]["text"]
        captions = [c if isinstance(c, str) and c.strip() else "pixel art sprite" for c in captions]
        tokens = tokenizer(captions).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype if use_amp else torch.float32,
            enabled=use_amp,
        ):
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        embeddings[start:end] = emb.float().cpu().half()
        if end % 10000 == 0 or end == len(ds):
            print(f"Encoded {end:,}/{len(ds):,}")

    torch.save(embeddings, emb_path)
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved embeddings: {emb_path} {tuple(embeddings.shape)} {embeddings.dtype}")
    print(f"Dataset ready: {out}")


if __name__ == "__main__":
    main()
