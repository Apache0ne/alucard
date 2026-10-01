# Alucard training pipeline v2

This branch is an experimental retraining path intended to diagnose and fix the failure mode of the published first-run checkpoint: fuzzy RGB, noisy backgrounds, weak prompt separation, and almost entirely fractional alpha.

## What changed

- **Transparent RGB canonicalization:** RGB is set to zero only where source alpha is exactly zero, so invisible color garbage is not learned as a target.
- **No semantic-changing augmentation by default:** horizontal flip and palette shifting default to 0 because cached captions/CLIP embeddings cannot be rewritten to keep direction/color words correct. If palette augmentation is explicitly enabled, reference/current animation frames now receive the same shift.
- **Better timestep encoding:** new checkpoints train with `timestep_scale=1000`; old checkpoints remain loadable with scale 1.
- **Alpha-aware objective:** the base rectified-flow velocity loss remains, with stronger alpha weighting plus endpoint alpha reconstruction, visible-RGB reconstruction, alpha-edge matching, and a small binary-confidence term only on nearly binary target pixels.
- **Correct gradient accumulation:** the last partial accumulation group of every epoch is normalized and stepped instead of leaking into the next epoch.
- **Correct mixed-precision bookkeeping:** BF16 is preferred; FP16 scaler state is checkpointed and scheduler/EMA/global-step advance only when the optimizer actually advances.
- **Real validation:** deterministic train/validation split and held-out loss. `best.pt` is selected by validation loss.
- **Fixed visual regression set:** same prompts, same seed, RGBA native files, checkerboard composites, and alpha masks every sampling epoch.
- **Correct CFG equation:** zero, fractional, unit, and amplified text/reference guidance all follow the documented equation.

## Published dataset

The current `evilsocket/alucard-sprites` dataset is 128x128 RGBA with `image` and `text` columns. The download helper keeps the Hugging Face dataset in Arrow format instead of writing hundreds of thousands of individual PNGs.

### Prepare a test subset

```bash
pip install -e .
python scripts/download_alucard_dataset.py \
  --output-dir data/alucard_sprites_20k \
  --max-samples 20000
```

Use `--max-samples 0` for the full dataset.

## Train

A conservative Colab/L4 test run:

```bash
alucard-train \
  --data-dir data/alucard_sprites_20k \
  --output-dir checkpoints_v2/test_20k \
  --epochs 10 \
  --batch-size 32 \
  --grad-accum 1 \
  --precision bf16 \
  --sample-every 1 \
  --save-every 1
```

The most important outputs are:

- `best.pt` — best held-out validation checkpoint
- `latest.pt` — resume checkpoint
- `metrics.jsonl` — train/validation/alpha metrics by epoch
- `samples/epoch_XXXX_contact.png` — fixed-seed checkerboard + alpha visual regression sheet
- `samples/epoch_XXXX/` — native RGBA, checkerboard composites, and alpha masks

## Benchmark a checkpoint

```bash
python scripts/benchmark_checkpoint.py \
  --checkpoint checkpoints_v2/test_20k/best.pt \
  --output-dir benchmark_v2/test_20k \
  --steps 30 \
  --cfg-text 2.5 \
  --seed 42
```

The benchmark writes `benchmark.json` and `contact_sheet.png`. It records:

- strict and 5–95% partial-alpha fractions
- near-binary-alpha fraction
- alpha edge strength
- CLIP image/text cosine similarity
- seconds/image and images/second
- peak GPU memory

Every prompt is generated from the same initial noise seed. That makes subject/prompt separation easy to inspect visually.

## Suggested experiment order

1. **20k rows / 10 epochs** to verify that alpha becomes crisp and prompt separation improves.
2. Compare `epoch_0001_contact.png` through the final epoch and inspect `benchmark.json`.
3. If the 20k run is clearly learning, run the full dataset with the same objective before changing architecture.
4. Only after that controlled run should token-level cross-attention or a larger model be evaluated; otherwise architecture changes are confounded with pipeline fixes.

## Colab

Open `notebooks/colab_train_v2.ipynb`. Its defaults clone this branch, prepare a 20k-row test dataset, train a test checkpoint, benchmark it, and display the contact sheet. Set `MAX_SAMPLES = 0` to prepare all rows.
