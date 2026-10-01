#!/usr/bin/env python3
"""Run alucard.train with live DataLoader progress bars.

This wrapper intentionally leaves the trainer math unchanged. It wraps every
DataLoader iterator with tqdm so Colab shows batch progress immediately for
both training and validation. Use the same CLI arguments accepted by
``python -m alucard.train``.
"""

from __future__ import annotations

import sys

from tqdm.auto import tqdm


def main() -> None:
    # Patch before importing alucard.train so all loaders created by the trainer
    # get visible iteration progress without duplicating the training code.
    from torch.utils.data import DataLoader

    original_iter = DataLoader.__iter__
    counter = {"n": 0}

    def progress_iter(self):
        iterator = original_iter(self)
        counter["n"] += 1
        # The trainer iterates train then validation each epoch. A generic label
        # is deliberate because this wrapper does not alter trainer internals.
        return iter(
            tqdm(
                iterator,
                total=len(self),
                desc=f"batches #{counter['n']}",
                dynamic_ncols=True,
                leave=True,
                mininterval=0.5,
            )
        )

    DataLoader.__iter__ = progress_iter

    from alucard.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
