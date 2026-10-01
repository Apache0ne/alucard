#!/usr/bin/env python3
"""Run alucard.train with live Colab-friendly progress bars.

This wrapper keeps the trainer math unchanged while adding two runtime controls:
- live tqdm progress for every DataLoader iterator, forced to stdout;
- ``--no-gradient-checkpointing`` to trade VRAM for speed on GPUs with headroom.

All other CLI arguments are passed through to ``python -m alucard.train``.
"""

from __future__ import annotations

import os
import sys

from tqdm import tqdm


def main() -> None:
    # Force immediate child-process output in Colab.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    print("[Alucard] progress wrapper started", flush=True)

    disable_checkpointing = "--no-gradient-checkpointing" in sys.argv
    if disable_checkpointing:
        # Remove our wrapper-only flag before alucard.train's argparse sees it.
        sys.argv.remove("--no-gradient-checkpointing")

    # Patch before importing alucard.train so all loaders created by the trainer
    # get visible iteration progress without duplicating the training code.
    from torch.utils.data import DataLoader

    original_iter = DataLoader.__iter__
    counter = {"n": 0}

    def progress_iter(self):
        iterator = original_iter(self)
        counter["n"] += 1
        return iter(
            tqdm(
                iterator,
                total=len(self),
                desc=f"batches #{counter['n']}",
                dynamic_ncols=True,
                leave=True,
                mininterval=0.25,
                file=sys.stdout,
                ascii=False,
            )
        )

    DataLoader.__iter__ = progress_iter

    if disable_checkpointing:
        # train.py calls enable_gradient_checkpointing() when it builds the UNet.
        # Override that one method for this process only. The subprocess exits
        # after training, so no modified state survives into benchmarking.
        from alucard.model import UNet

        def _keep_checkpointing_disabled(self):
            self.disable_gradient_checkpointing()

        UNet.enable_gradient_checkpointing = _keep_checkpointing_disabled
        print("[Alucard] Gradient checkpointing: OFF (fast mode)", flush=True)
    else:
        print("[Alucard] Gradient checkpointing: ON (memory-saving mode)", flush=True)

    from alucard.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
