"""Alucard: compact text-to-sprite generation with flow matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from alucard.model import UNet
from alucard.sample import model_kwargs_from_checkpoint, sample, tensor_to_rgba_image


class Alucard:
    """High-level wrapper for sprite generation."""

    def __init__(self, model: UNet, clip_model: nn.Module, tokenizer, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.clip_model = clip_model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "Alucard":
        """Load a Hugging Face repo, local model directory, safetensors, or .pt checkpoint."""
        import open_clip

        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        path_obj = Path(path)
        model_kwargs: dict = {}

        if path_obj.is_file():
            state_dict, model_kwargs = cls._load_weights(path_obj, dev)
        elif path_obj.is_dir():
            safetensors_path = path_obj / "alucard_model.safetensors"
            pt_file = path_obj / "best.pt"
            if safetensors_path.exists():
                state_dict, model_kwargs = cls._load_weights(safetensors_path, dev)
            elif pt_file.exists():
                state_dict, model_kwargs = cls._load_weights(pt_file, dev)
            else:
                raise FileNotFoundError(f"No model weights found in {path}")
        else:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(repo_id=path, filename="alucard_model.safetensors")
            try:
                config_path = hf_hub_download(repo_id=path, filename="config.json")
            except Exception:
                config_path = None
            state_dict, model_kwargs = cls._load_weights(
                Path(local_path), dev, explicit_config=Path(config_path) if config_path else None
            )

        model = UNet(**model_kwargs)
        model.load_state_dict(state_dict)
        model.eval()

        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()
        return cls(model, clip_model, tokenizer, device=dev)

    @staticmethod
    def _config_to_model_kwargs(config: dict) -> dict:
        if "model" in config and isinstance(config["model"], dict):
            config = config["model"]
        allowed = {
            "in_channels", "out_channels", "base_channels", "channel_mults",
            "num_res_blocks", "attn_resolutions", "text_dim", "dropout",
            "image_size", "timestep_scale",
        }
        out = {k: v for k, v in config.items() if k in allowed}
        if "channel_mults" in out:
            out["channel_mults"] = tuple(out["channel_mults"])
        if "attn_resolutions" in out:
            out["attn_resolutions"] = tuple(out["attn_resolutions"])
        return out

    @classmethod
    def _load_weights(
        cls,
        path: Path,
        device: torch.device,
        explicit_config: Path | None = None,
    ) -> tuple[dict, dict]:
        """Load model weights and constructor metadata when available."""
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device=str(device))
            config_candidates = [explicit_config, path.with_suffix(".json"), path.parent / "config.json"]
            for candidate in config_candidates:
                if candidate and candidate.exists():
                    try:
                        cfg = json.loads(candidate.read_text())
                        return state, cls._config_to_model_kwargs(cfg)
                    except Exception:
                        pass
            return state, {}

        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("ema_model", ckpt.get("model", ckpt))
            return state, model_kwargs_from_checkpoint(ckpt)
        return ckpt, {}

    def encode_text(self, prompt: str | list[str]) -> torch.Tensor:
        if isinstance(prompt, str):
            prompt = [prompt]
        tokens = self.tokenizer(prompt).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @staticmethod
    def load_ref(image: str | Path | Image.Image, size: int = 128) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGBA")
        if image.size != (size, size):
            image = image.resize((size, size), Image.NEAREST)
        arr = np.array(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        ref: str | Path | Image.Image | None = None,
        num_samples: int = 1,
        num_steps: int = 30,
        cfg_text: float = 2.5,
        cfg_ref: float = 1.0,
        seed: int | None = None,
    ) -> Image.Image | list[Image.Image]:
        text_emb = self.encode_text(prompt).expand(num_samples, -1)
        ref_tensor = None
        if ref is not None:
            ref_tensor = self.load_ref(ref).unsqueeze(0).expand(num_samples, -1, -1, -1).to(self.device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        sprites = sample(
            self.model,
            text_emb,
            ref=ref_tensor,
            num_steps=num_steps,
            cfg_text=cfg_text,
            cfg_ref=cfg_ref,
            device=self.device,
            generator=generator,
        )
        images = [tensor_to_rgba_image(sprites[i]) for i in range(num_samples)]
        return images[0] if num_samples == 1 else images
