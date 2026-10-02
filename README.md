# PRISM

**Peptide Regression and Inverse design for Selective Metalloproteinase substrates**

PRISM predicts peptide activity across 18 MMPs and generates candidate substrates targeting MMP13. This repository provides the code, data and checkpoints for running PRISM.

![PRISM](assets/prism_overview.png)

## Installation

Use Python 3.12. Run all commands from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/predictor.txt -r requirements/generator.txt
```

## Run

### Check the data and checkpoints

```sh
bash run.sh verify
DEVICE=cpu bash run.sh smoke
```

The smoke check loads all three predictor and generator checkpoints and generates 50 attempts per generator.

### Predict and evaluate

![PRISM predictor](assets/predictor_model.png)

```sh
DEVICE=cuda FEATURE_PRECISION=bf16 bash run.sh predictor
# CPU:
# DEVICE=cpu FEATURE_PRECISION=fp32 bash run.sh predictor
```

This evaluates validation, benchmark test and OOD sequences with seeds 0, 1 and 2. ESM-2 650M is downloaded on first use; extracted features are cached in `outputs/features/`.

### Generate and score peptides

```sh
DEVICE=cuda bash run.sh generator
DEVICE=cuda bash run.sh score-generated
```

Generation uses 50 packaged MMP13 profiles, 400 attempts per profile and two sampling repeats for each checkpoint. Scoring uses the three PRISM predictors and exports predicted activity profiles and selected candidates.

```sh
# Unconditional generation:
MODE=unconditional TEMPERATURE=1.0 bash run.sh generator
# A single checkpoint and a custom output directory:
SEEDS="0" OUTPUT_DIR="$PWD/my_results" bash run.sh generator
```

### Train the generator

```sh
DEVICE=cuda bash run.sh train-generator
```

Runs 1,000 DPO updates from the packaged initialization weights. New checkpoints are saved in `outputs/generator/dpo/`. See [generator commands](generator/README.md) for resuming training and running individual checkpoints.

### Settings

| Variable | Default | Use |
| --- | --- | --- |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `FEATURE_PRECISION` | `bf16` | Use `fp32` for CPU feature extraction |
| `SEEDS` | `0 1 2` | Checkpoints to run |
| `MODE` | `selective` | `selective` or `unconditional` |
| `TEMPERATURE` | `1.2` / `1.0` | Conditional / unconditional sampling |
| `PER_TEMPLATE` | `400` | Attempts per profile |
| `SAMPLING_SEEDS` | `2026111201 2026111202` | Sampling repeats |
| `OUTPUT_DIR` | `outputs/` | Output directory |
| `FEATURE_DIR` | `$OUTPUT_DIR/features/` | Feature cache |
| `POOL_DIR` | `$OUTPUT_DIR/generator/pools/` | Generated pools to score |

`PYTORCH_PYTHON`, `GENERATOR_PYTHON` and `METRICS_PYTHON` can select separate Python executables. Run `bash run.sh help` for available commands.

## Files

| Location | Contents |
| --- | --- |
| `checkpoints/predictor/seed{0,1,2}.pt` | Predictor weights |
| `generator/checkpoints/seed{0,1,2}.pt` | Generator weights |
| `generator/initialization/seed{0,1,2}.pt` | Generator initialization for DPO |
| `data/benchmark/` | Predictor training, validation and test data |
| `data/ood/` | Labelled OOD evaluation data |
| `generator/data/` | Generator fitting data, preferences and generation profiles |
| `generator/configs/pareto_dpo.json` | Generator settings |
| `predictor/`, `generator/`, `evaluation/` | Model and evaluation code |
| `requirements/`, `tests/`, `assets/` | Dependencies, checkpoint checks and figures |

Data sizes and formats are listed in [data/README.md](data/README.md) and [generator/data/README.md](generator/data/README.md). Benchmark data originate from Kukreja et al. via the CleaveNet release; OOD labels are published CleaveNet experimental measurements.

### Outputs

| Location under `OUTPUT_DIR` | Contents |
| --- | --- |
| `predictor/predictions/` | NPZ files containing sequences, enzyme order and predictions |
| `predictor/metrics/` | Aggregate/per-target metrics, selected peptides and thresholds |
| `generator/nll/` | Generator development loss |
| `generator/pools/` | Raw generated sequences |
| `generator/quality/` | Sequence-quality summaries |
| `generator/prism_scores/` | Predicted activity profiles, scores and selected peptides |
| `generator/dpo/` | New generator training checkpoints |
