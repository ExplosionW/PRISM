# PRISM

**Peptide Regression and Inverse design for Selective Metalloproteinase substrates**

PRISM predicts peptide activity across 18 MMPs and generates candidate substrates targeting MMP13.

![PRISM](assets/prism_overview.png)

## Table of contents

- [Getting started](#getting-started)
  - [Installation](#installation)
  - [Project structure](#project-structure)
  - [Check the installation](#check-the-installation)
- [Training models](#training-models)
  - [PRISM predictor](#prism-predictor)
  - [PRISM generator full training pipeline](#prism-generator-full-training-pipeline)
- [Predicting peptide activity](#predicting-peptide-activity)
- [Generating peptides](#generating-peptides)
  - [Generate and score](#generate-and-score)
  - [Single-checkpoint generation and evaluation](#single-checkpoint-generation-and-evaluation)
- [Run settings](#run-settings)
- [Data](#data)
  - [Predictor data](#predictor-data)
  - [Generator data](#generator-data)
- [Outputs](#outputs)

## Getting started

### Installation

Use Python 3.12. Run all commands from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/predictor.txt -r requirements/generator.txt
```

### Project structure

```text
PRISM/
├── predictor/                 # Activity prediction and training
│   ├── checkpoints/           # Released predictor weights (seeds 0, 1, 2)
│   ├── initialization/        # Original task-LM weights and fixed enzyme inputs
│   └── configs/               # Fixed predictor training settings
├── generator/                 # Peptide generation and full training pipeline
│   ├── checkpoints/           # Released generator weights (seeds 0, 1, 2)
│   ├── initialization/        # Supervised weights for the DPO-only shortcut
│   ├── configs/               # Generation and fixed training settings
│   └── data/                  # Fitting data, preference pairs and requested profiles
├── data/                      # Predictor training, validation, test and OOD data
├── evaluation/                # Prediction and generation evaluation scripts
├── requirements/              # Python dependencies
├── tests/                     # Reference outputs and model/training checks
├── assets/                    # Images used in this README
├── README.md                  # Installation, commands and data formats
├── run.sh                     # Training, prediction, generation and evaluation entry point
└── .gitignore                 # Exclude caches, outputs and extra checkpoint files
```

### Check the installation

```sh
bash run.sh verify
DEVICE=cpu bash run.sh smoke
```

The smoke check loads all three predictor and generator checkpoints and generates 50 attempts per generator.

## Training models

### PRISM predictor

```sh
# All three seeds, original fixed recipe:
DEVICE=cuda FEATURE_PRECISION=bf16 bash run.sh train-predictor
# One seed, or resume interrupted training:
SEEDS="0" DEVICE=cuda bash run.sh train-predictor
SEEDS="0" RESUME=1 DEVICE=cuda bash run.sh train-predictor
```

This extracts training and validation features as needed and runs three stages:

1. Train the task backbone with frozen ESM features.
2. Add the original pretrained task LM as a residual branch; freeze inherited tensors for 10 epochs, then unfreeze them.
3. Freeze the selected backbone and train the shared-query adapter, including epoch zero in adapter checkpoint selection.

The fixed training loss is mean-activity MSE + 36 × centered-profile MSE + 2 × maximum-competitor-margin MSE + 0.5 × minibatch ListNet. Validation uses the first three terms only. Effective batch size is 128; each stage allows up to 300 epochs with patience 25. Training uses AdamW (learning rate 0.001, weight decay 0.001), cosine schedules, and adapter gradient clipping at 10. No ablation or loss-search options are exposed.

Task layers are initialized afresh; ESM is frozen and the task LM uses `predictor/initialization/lmw_256_8_iso.pt`. The packaged fixed inputs contain enzyme descriptors and the sequence-adapter masks/buffers, not trained task-layer parameters. Initialization hashes are checked before training. This command does not pretrain ESM or the task LM from random weights. It never reads benchmark test or OOD labels.

New models are written to `outputs/predictor/training/seed{0,1,2}/final.pt`, with per-stage best/resume checkpoints and training histories. Published checkpoints are not overwritten. To use a newly trained model, pass its `final.pt` to `predictor/predict.py`.

```sh
# Equivalent single-seed entry point; features must already exist:
python predictor/train.py --seed 0 --device cuda --features outputs/features \
  --output outputs/predictor/training/seed0
python predictor/train.py --seed 0 --device cuda --features outputs/features \
  --output outputs/predictor/training/seed0 --resume
```

Training follows the archived 7.09% recipe. Exact historical weights, early-stopping epochs and the reported score are not guaranteed by retraining. CUDA FP32 task training with historical BF16 ESM feature extraction is the reference setup; CPU/FP32 feature extraction is supported but changes the numerical setting. Recomputed features need not be byte-identical to the historical cache.

### PRISM generator full training pipeline

```sh
# Complete pipeline, all three seeds:
DEVICE=cuda bash run.sh train-generator
# One seed, or resume interrupted training:
SEEDS="0" DEVICE=cuda bash run.sh train-generator
SEEDS="0" RESUME=1 DEVICE=cuda bash run.sh train-generator
```

The fixed PRISM generator pipeline trains the base decoder from random initialization, builds two conditional experts, then performs supervised mixture training and Pareto-DPO:

| Stage | Training | Initialization |
| --- | --- | --- |
| Base decoder | 150 epochs; Transformer schedule with 4,000 warmup updates | Random initialization |
| Base refinement | 60 epochs; learning rate 5 × 10⁻⁵ | Selected base decoder |
| Profile expert | 300 epochs; profile-conditioned memory | Selected refined decoder |
| Competition expert | 300 epochs; profile and target-minus-competitor memory | The same selected refined decoder |
| Supervised mixture | 60 epochs; equal-weight experts with relative-position bias and channel modulation | Selected profile and competition experts; new modulation outputs initialized to zero |
| Pareto-DPO | 1,000 updates; learning rate 10⁻⁶ | Selected supervised mixture |

Supervised stages use Adam (epsilon 10⁻⁷), batch size 128, and a 50/50 mixture of conditional and unconditional examples. Each stage selects one checkpoint by minimum joint development NLL, including epoch zero; the fixed final epoch is also saved. The two experts retain the frozen unconditional decoder. Their 300-epoch schedules preserve optimizer state across the original 100 + 200 epoch boundary.

Supervised mixture training minimizes the mean of the two expert cross-entropies. Conditional fitting examples with measured MMP13 Z-score > 1 and higher than all other 17 enzymes receive weight 2; other examples receive weight 1. The loss is normalized by the expected mean weight. Pareto-DPO then updates the original fixed parameter subset using the packaged preference pairs, with no best-epoch selection. These preferences are frozen PRISM predictions, not experimental measurements; the command does not regenerate them.

```sh
# Equivalent single-seed entry point:
python generator/train_pipeline.py --seed 0 --device cuda \
  --output outputs/generator/training/seed0
```

The final model is saved as `outputs/generator/training/seed0/final.pt`. Per-stage best, last and resume checkpoints are retained. Published weights are not overwritten. To generate from a new model, pass its `final.pt` to `generator/sample.py`; the `run.sh generator` command uses the released checkpoints.

Supervised training and checkpoint selection use the packaged fitting and development splits; DPO uses the packaged training preference pairs. The architecture and schedule follow the released recipe; retraining does not guarantee identical historical weights or scores. Settings are in `generator/configs/training.json` and `generator/configs/pareto_dpo.json`.

#### DPO-only training

To start directly from the packaged supervised initialization instead of rebuilding the preceding stages:

```sh
DEVICE=cuda bash run.sh train-generator-dpo
SEEDS="0" RESUME=1 DEVICE=cuda bash run.sh train-generator-dpo
# Direct entry point:
python generator/train.py --seed 0 --device cuda --output outputs/generator/dpo/seed0
```

This writes `outputs/generator/dpo/seed{0,1,2}/last.pt`. The complete pipeline above passes its newly trained supervised checkpoint to this same DPO implementation.

## Predicting peptide activity

![PRISM predictor](assets/predictor_model.png)

```sh
DEVICE=cuda FEATURE_PRECISION=bf16 bash run.sh predictor
# CPU:
# DEVICE=cpu FEATURE_PRECISION=fp32 bash run.sh predictor
```

This evaluates validation, benchmark test and OOD sequences with seeds 0, 1 and 2. ESM-2 650M is downloaded on first use; extracted features are cached in `outputs/features/`.

The released predictor checkpoints are the original `prop7_listnet_batch / shared_query` models with three-seed mean benchmark Precision **7.0926%** under Count/Pareto-Max (Top100 per target, averaged over 18 targets). This is not MMP13-only precision. Checkpoint hashes are in `predictor/configs/prism.json`.

For custom ten-residue sequences, provide a CSV with a `sequence` column:

```sh
python predictor/extract_features.py --input peptides.csv --output features.npz --device cuda --precision bf16
python predictor/predict.py --checkpoint predictor/checkpoints/seed0.pt \
  --input features.npz --output predictions.npz --device cuda
```

The feature archive contains `sequences` and residue-level `esm33` features of shape N × 10 × 1280. Predictions contain `sequences`, `targets` and `predictions` of shape N × 18.

## Generating peptides

### Generate and score

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

### Single-checkpoint generation and evaluation

```sh
# Generate 50 attempts with one checkpoint on CPU:
python generator/sample.py --checkpoint generator/checkpoints/seed0.pt \
  --device cpu --per-template 1 --output outputs/example.csv
python generator/evaluate_checkpoint.py --checkpoint generator/checkpoints/seed0.pt \
  --device cuda --output outputs/development_nll.json
```

Sampling options include `--mode selective|unconditional`, `--temperature`, `--seed`, `--per-template` and `--batch` (default 512). The released settings use CUDA FP32, batch 512, canonical log-softmax two-draw sampling, conditional temperature 1.2 and unconditional temperature 1.0. There is no forced STOP, length correction or refill. Settings are in `generator/configs/pareto_dpo.json`.

Output CSVs retain every attempt. `sequence` contains the generated string; `stopped_normally`, `raw_length` and `filter_reason` record its status. `condition_train_index` identifies the fitting sequence supplying the requested profile, or is -1 for unconditional generation. Scoring exports activity profiles and active-bottleneck Top24/100 candidates; predicted qualification is not experimental cleavage validation.

## Run settings

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
| `RESUME` | `0` | Set `1` to resume predictor or generator training |
| `PREDICTOR_LM_INIT` | `predictor/initialization/lmw_256_8_iso.pt` | Original task-LM initialization; SHA checked |

`PYTORCH_PYTHON`, `GENERATOR_PYTHON` and `METRICS_PYTHON` can select separate Python executables. Run `bash run.sh help` for available commands.

## Data

### Predictor data

| Split | Peptides | Measured enzymes | Files |
| --- | ---: | ---: | --- |
| Training | 13,666 | 18 | `data/benchmark/train.csv`, `train.npz` |
| Validation | 1,200 | 18 | `data/benchmark/val.csv`, `val.npz` |
| Benchmark test | 2,901 | 18 | `data/benchmark/test.csv`, `test.npz` |
| OOD | 80 | 12 | `data/ood/ood.csv`, `ood.npz` |

The four splits have no exact sequence overlap; membership is in `data/split_ids.json`. Benchmark CSVs contain `sequence` followed by the 18 activity columns. NPZ arrays are `sequences` (N), `labels` (N × 18 cleavage Z-scores), `conditions` (N × 18 rounded to 0.1 for generation) and `tokens` (N × 10). Predictor training uses unrounded `labels`.

OOD NPZs contain `sequences`, `labels` (80 × 12 published efficiencies), `targets`, `substrate_ids` and `raw_fold_mean`. Benchmark Z-scores and OOD efficiencies have different units. The current OOD evaluation is retrospective: its qualification thresholds are computed from the external measurements under the fixed evaluation rule.

Enzyme orders are in `data/benchmark/target_order.json` and `data/ood/target_order.json`. Benchmark measurements originate from Kukreja et al., distributed with CleaveNet; OOD measurements come from published CleaveNet experiments. Source links and preprocessing metadata are in `data/source_manifest.json`.

### Generator data

| File under `generator/data/` | Contents |
| --- | --- |
| `fit.npz` | 10,954 fitting peptides with sequences, tokens, measured labels and rounded conditions |
| `development.npz` | 1,434 peptides for development loss evaluation |
| `preferences.csv` | 6,400 pairs: 5,101 training and 1,299 validation |
| `requests.json` | Requested 18-enzyme profiles, keyed by `fit_condition_index` |
| `templates.json` | 50 generation profiles and their fitting sequences, in sampling order |
| `known_sequences.txt` | 31,383 sequences excluded when counting novel outputs |
| `target_order.json` | Enzyme order; MMP13 has zero-based index 4 |

Profiles use the source Z-score scale, with conditions rounded to 0.1. Token order is `ACDEFGHIKLMNPQRSTVWY` (0–19); generator START = 20 and STOP = 21. Preference columns `delta_activity`, `delta_mean17` and `delta_max17` contain differences in PRISM predictions. The novelty exclusion file contains public dataset sequences and both members of the preference pairs.

## Outputs

| Location under `OUTPUT_DIR` | Contents |
| --- | --- |
| `predictor/predictions/` | NPZ files containing sequences, enzyme order and predictions |
| `predictor/metrics/` | Aggregate/per-target metrics, selected peptides and thresholds |
| `predictor/training/seed*/` | Newly trained predictor, per-stage checkpoints and histories |
| `generator/nll/` | Generator development loss |
| `generator/pools/` | Raw generated sequences |
| `generator/quality/` | Sequence-quality summaries |
| `generator/prism_scores/` | Predicted activity profiles, scores and selected peptides |
| `generator/training/seed*/` | Complete generator training: stage checkpoints, histories and final model |
| `generator/dpo/seed*/` | Checkpoints from the DPO-only shortcut |
