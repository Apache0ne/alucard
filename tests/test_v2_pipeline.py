import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from alucard.dataset import load_rgba
from alucard.model import timestep_embedding
from alucard.sample import sample


def test_transparent_rgb_is_canonicalized():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.png"
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[0, 0] = [255, 123, 77, 0]
        Image.fromarray(arr, "RGBA").save(path)
        t = load_rgba(path, size=2, sanitize_transparent_rgb=True)
        assert torch.allclose(t[:3, 0, 0], torch.tensor([-1.0, -1.0, -1.0]))
        assert t[3, 0, 0].item() == -1.0


def test_timestep_scale_uses_frequency_range():
    t = torch.tensor([0.0, 1.0])
    low = timestep_embedding(t, 64, scale=1.0)
    high = timestep_embedding(t, 64, scale=1000.0)
    assert (high[0] - high[1]).abs().mean() > (low[0] - low[1]).abs().mean() * 5


class ToyGuidanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.null_text_emb = nn.Parameter(torch.zeros(2), requires_grad=False)

    def forward(self, x, t, text_emb, ref=None):
        text_term = text_emb[:, :1, None, None]
        if ref is None:
            ref_term = 0.0
        else:
            ref_term = ref[:, :1].mean(dim=(2, 3), keepdim=True)
        return torch.ones_like(x) * (text_term + ref_term)


def test_cfg_fractional_and_unit_reference_are_respected():
    model = ToyGuidanceModel()
    text = torch.tensor([[0.2, 0.0]])
    ref = torch.ones(1, 4, 4, 4) * 0.3

    y0 = sample(model, text, num_steps=1, cfg_text=0.0, device="cpu", image_size=4,
                generator=torch.Generator().manual_seed(9))
    yhalf = sample(model, text, num_steps=1, cfg_text=0.5, device="cpu", image_size=4,
                   generator=torch.Generator().manual_seed(9))
    y1 = sample(model, text, num_steps=1, cfg_text=1.0, device="cpu", image_size=4,
                generator=torch.Generator().manual_seed(9))

    expected = (y0 + y1) * 0.5
    mask = (y0.abs() < 0.99) & (y1.abs() < 0.99) & (yhalf.abs() < 0.99)
    assert mask.any()
    assert torch.allclose(yhalf[mask], expected[mask], atol=1e-5)

    no_ref = sample(model, text, num_steps=1, cfg_text=1.0, device="cpu", image_size=4,
                    generator=torch.Generator().manual_seed(7))
    unit_ref = sample(model, text, ref=ref, num_steps=1, cfg_text=1.0, cfg_ref=1.0,
                      device="cpu", image_size=4, generator=torch.Generator().manual_seed(7))
    assert not torch.allclose(no_ref, unit_ref)
