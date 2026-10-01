"""Dataset utilities for Alucard sprite training.

Supports the original file layout and a Hugging Face ``datasets.save_to_disk``
layout produced by ``scripts/download_alucard_dataset.py``.

The v2 training path deliberately disables semantic-changing augmentation by
default. A cached CLIP embedding cannot be updated when an image is recolored
or mirrored, so those transforms can contradict captions such as "green" or
"facing left".
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _pil_to_rgba_tensor(
    img: Image.Image,
    size: int = 128,
    sanitize_transparent_rgb: bool = True,
) -> torch.Tensor:
    """Convert a PIL image to a normalized RGBA tensor.

    Fully transparent pixels are canonicalized to RGB=0 before normalization.
    This removes invisible RGB garbage from the learning target without
    modifying visible or partially transparent pixels.
    """
    img = img.convert("RGBA")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)

    arr_u8 = np.array(img, dtype=np.uint8, copy=True)
    if sanitize_transparent_rgb:
        transparent = arr_u8[..., 3] == 0
        arr_u8[transparent, :3] = 0

    arr = arr_u8.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_rgba(
    path: Path,
    size: int = 128,
    sanitize_transparent_rgb: bool = True,
) -> torch.Tensor:
    """Load an RGBA file as ``(4,H,W)`` in ``[-1,1]``."""
    with Image.open(path) as img:
        return _pil_to_rgba_tensor(img, size, sanitize_transparent_rgb)


def palette_swap(
    img: torch.Tensor,
    strength: float = 0.3,
    shift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one RGB shift and return both the image and the sampled shift.

    This remains available for experiments, but the v2 trainer defaults its
    probability to zero because cached text embeddings cannot be rewritten to
    match color-changing augmentation.
    """
    if shift is None:
        shift = (torch.rand(3, 1, 1, dtype=img.dtype) - 0.5) * 2 * strength
    rgb = (img[:3] + shift).clamp(-1, 1)
    return torch.cat([rgb, img[3:4]], dim=0), shift


class SpriteDataset(Dataset):
    """Sprite dataset with optional cached CLIP embeddings and frame pairs."""

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 128,
        augment: bool = False,
        flip_prob: float = 0.0,
        palette_swap_prob: float = 0.0,
        palette_swap_strength: float = 0.3,
        sanitize_transparent_rgb: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        self.flip_prob = flip_prob
        self.palette_swap_prob = palette_swap_prob
        self.palette_swap_strength = palette_swap_strength
        self.sanitize_transparent_rgb = sanitize_transparent_rgb
        self.hf_dataset = None
        self.mode = "files"

        consolidated_path = self.data_dir / "clip_embeddings.pt"
        self.clip_embeddings = None
        if consolidated_path.exists():
            load_kwargs = {"map_location": "cpu", "weights_only": True}
            try:
                self.clip_embeddings = torch.load(consolidated_path, mmap=True, **load_kwargs)
            except TypeError:
                self.clip_embeddings = torch.load(consolidated_path, **load_kwargs)

        hf_path = self.data_dir / "hf_dataset"
        if hf_path.exists():
            if self.clip_embeddings is None:
                raise ValueError(
                    f"{hf_path} exists but {consolidated_path.name} is missing. "
                    "Run scripts/download_alucard_dataset.py without --skip-embeddings."
                )
            try:
                from datasets import load_from_disk
            except ImportError as exc:
                raise ImportError("Hugging Face dataset mode requires `pip install datasets`.") from exc
            self.hf_dataset = load_from_disk(str(hf_path))
            if len(self.hf_dataset) != len(self.clip_embeddings):
                raise ValueError(
                    f"Dataset/embedding length mismatch: {len(self.hf_dataset)} vs "
                    f"{len(self.clip_embeddings)}"
                )
            self.mode = "hf"
            self.samples = list(range(len(self.hf_dataset)))
            return

        if self.clip_embeddings is not None:
            self.samples = []
            for i in range(len(self.clip_embeddings)):
                img_path = self.data_dir / f"sprite_{i:06d}.png"
                if img_path.exists():
                    self.samples.append(img_path)
                else:
                    break
        else:
            self.samples = []
            for img_path in sorted(self.data_dir.glob("*.png")):
                if img_path.stem.endswith(".prev"):
                    continue
                clip_path = img_path.with_suffix(".clip.pt")
                if clip_path.exists():
                    self.samples.append(img_path)

        if not self.samples:
            raise ValueError(
                f"No valid samples found in {data_dir}. Expected either "
                "hf_dataset/ + clip_embeddings.pt, or PNG files with cached CLIP embeddings."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_file_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        img_path = self.samples[idx]
        prev_path = img_path.parent / f"{img_path.stem}.prev.png"
        image = load_rgba(img_path, self.image_size, self.sanitize_transparent_rgb)

        if self.clip_embeddings is not None:
            text_emb = self.clip_embeddings[idx].float()
        else:
            clip_path = img_path.with_suffix(".clip.pt")
            text_emb = torch.load(clip_path, map_location="cpu", weights_only=True).float()

        has_ref = prev_path.exists()
        if has_ref:
            ref = load_rgba(prev_path, self.image_size, self.sanitize_transparent_rgb)
        else:
            ref = torch.zeros_like(image)
        return image, text_emb, ref, has_ref

    def _load_hf_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        item = self.hf_dataset[idx]
        image = _pil_to_rgba_tensor(
            item["image"], self.image_size, self.sanitize_transparent_rgb
        )
        text_emb = self.clip_embeddings[idx].float()
        ref = torch.zeros_like(image)
        return image, text_emb, ref, False

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "hf":
            image, text_emb, ref, has_ref = self._load_hf_sample(idx)
        else:
            image, text_emb, ref, has_ref = self._load_file_sample(idx)

        if self.augment:
            # Disabled by default because direction words in cached captions are
            # not rewritten. Enable only for a caption set known to be invariant.
            if self.flip_prob > 0 and random.random() < self.flip_prob:
                image = image.flip(-1)
                if has_ref:
                    ref = ref.flip(-1)

            # If enabled, use exactly the same color shift for current/reference
            # frames. This fixes the original pair-inconsistency bug.
            if self.palette_swap_prob > 0 and random.random() < self.palette_swap_prob:
                image, shift = palette_swap(image, self.palette_swap_strength)
                if has_ref:
                    ref, _ = palette_swap(ref, self.palette_swap_strength, shift=shift)

        return {
            "image": image,
            "text_emb": text_emb,
            "ref": ref,
            "has_ref": has_ref,
        }
