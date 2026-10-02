# Generator commands

Run from the repository root after installing the [dependencies](../README.md#installation).

```sh
# Evaluate and generate with all three checkpoints:
DEVICE=cuda bash run.sh generator

# Generate 50 attempts with one checkpoint on CPU:
python generator/sample.py --checkpoint generator/checkpoints/seed0.pt \
  --device cpu --per-template 1 --output outputs/example.csv

# Evaluate one checkpoint:
python generator/evaluate_checkpoint.py --checkpoint generator/checkpoints/seed0.pt \
  --device cuda --output outputs/development_nll.json

# Train or resume one seed from its packaged initialization:
python generator/train.py --seed 0 --device cuda --output outputs/dpo/seed0
python generator/train.py --seed 0 --device cuda --output outputs/dpo/seed0 --resume
```

Sampling options include `--mode selective|unconditional`, `--temperature`, `--seed`, `--per-template` and `--batch` (default 512). Default temperatures are 1.2 for conditional generation and 1.0 for unconditional generation. Use CUDA FP32 and batch 512 for the released sampling settings.

The output CSV retains all attempts. `sequence` contains the generated string; `stopped_normally`, `raw_length` and `filter_reason` record its status. `condition_train_index` identifies the fitting sequence supplying the requested profile, or is -1 for unconditional generation.

[Settings](configs/pareto_dpo.json) · [Input files](data/README.md).
