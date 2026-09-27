# PRISM

**Peptide Regression and Inverse design for Selective Metalloproteinase substrates**

PRISM connects peptide activity prediction with conditional sequence generation for selective MMP substrate discovery, with a focus on MMP13. The predictor estimates an 18-enzyme activity profile from a peptide sequence. The separately trained generator proposes peptides conditioned on a desired profile and target identity. Profile regression, ListNet ranking and Pareto-Max selection support competition-aware candidate identification.

![PRISM overview and predictor/generator architecture](assets/prism_overview.png)

[Editable manuscript overview (SVG)](assets/prism_overview.svg)

## Included

- Three released PRISM predictor checkpoints and three final generator checkpoints (seeds 0, 1 and 2).
- Labelled training, validation, benchmark test and OOD datasets, with sequence IDs and enzyme order.
- The packaged prediction, generation and generator-training implementations, called through `run.sh`.
- Evaluation scripts and small checkpoint-loading tests.

This release focuses on PRISM. The primary predictor reproduction route is inference from the released checkpoints. Baseline retraining and manuscript figure generation are outside this package.

## Data

| Dataset | Peptides | Measured enzymes | Location | Use |
| --- | ---: | ---: | --- | --- |
| Training | 13,666 | 18 | `data/benchmark/train.*` | Model fitting and generation templates |
| Validation | 1,200 | 18 | `data/benchmark/val.*` | Model and selection settings |
| Benchmark test | 2,901 | 18 | `data/benchmark/test.*` | Held-out evaluation |
| OOD | 80 | 12 | `data/ood/ood.*` | Generalization evaluation using published experimental labels |

All sequences are canonical 10-mers. These four sets have no exact sequence overlap. OOD peptides are separate from training, validation and benchmark test data; they are not used for model fitting. Predictions cover all 18 enzymes, including on OOD peptides.

The benchmark comes from the Kukreja et al. dataset distributed with CleaveNet; OOD labels come from CleaveNet's published experimental measurements. See [data documentation](data/README.md) for filtering, label units, array layouts and source links.

## Installation

Use separate Python environments for the PyTorch predictor and TensorFlow generator. The generator requirements record the versions used for the released training run. Python 3.12 and a Linux CUDA environment are recommended for reproducing training; CPU inference and small generation checks are also supported.

```sh
python3.12 -m venv .venv-predictor
.venv-predictor/bin/python -m pip install -r requirements/predictor.txt
python3.12 -m venv .venv-generator
.venv-generator/bin/python -m pip install -r requirements/generator.txt

export PYTORCH_PYTHON="$PWD/.venv-predictor/bin/python"
export TENSORFLOW_PYTHON="$PWD/.venv-generator/bin/python"
```

GPU runs require working CUDA support for the respective framework. For evaluating your saved predictions in a separate environment, install `requirements/metrics.txt` and set `METRICS_PYTHON` to that environment's Python.

## Run

Run these commands from the repository root. Results are written to `outputs/` by default.

### 1. Verify the release

```sh
bash run.sh verify
```

`verify` checks file hashes, sequence identities, label arrays and split separation.

### 2. Check all six model checkpoints

```sh
bash run.sh smoke
```

This compares all three predictors against reference outputs for four peptides and generates 50 sequences with each generator. It requires no encoder download.

### 3. Run predictor inference on validation, test and OOD data

```sh
DEVICE=cuda FEATURE_PRECISION=bf16 bash run.sh predictor
# CPU alternative:
# DEVICE=cpu FEATURE_PRECISION=fp32 bash run.sh predictor
```

This extracts frozen ESM-2 residue features, runs each PRISM checkpoint, selects candidates and calculates per-enzyme and aggregate results. ESM-2 650M is downloaded from Hugging Face on first use; the encoder revision is pinned in `predictor/extract_features.py`. Extracted features are reused from `outputs/features/`. The full feature cache is not included. Differences in device or numerical precision can slightly change predictions and rankings near ties.

### 4. Evaluate and sample the generator

```sh
bash run.sh generator
```

For each seed, this evaluates test-set conditional/unconditional NLL and samples 20,000 MMP13-conditioned attempts from 50 training-derived templates at temperature 1.2. It reports sequence validity, uniqueness, overlap and k-mer diversity. Invalid attempts and duplicates remain in the raw pools so their rates can be recomputed.

```sh
# Other supported generation settings:
MODE=efficient TEMPERATURE=1.0 bash run.sh generator
MODE=unconditional TEMPERATURE=1.0 bash run.sh generator
```

### 5. Score generated peptides with PRISM

```sh
bash run.sh score-generated
```

This scores unique novel canonical peptides with the three-seed PRISM predictor ensemble. It exports activity profiles, mean predicted MMP13 activity, target-minus-mean-competitor selectivity, target-dominance frequency and top-100 selectivity. The scorer is explicitly recorded as PRISM. See [evaluation details](docs/REPRODUCTION.md).

### Optional: train the generator

```sh
bash run.sh train-generator
```

This calls the existing G2 training pipeline: initial training, extended training and the final continuation with contrast weight zero. It requires a TensorFlow CUDA GPU. New training files are saved below `generator/checkpoints/` and `generator/optimization_v2/runs/`; released weights remain in `checkpoints/generator/`. Predictor training from scratch is not a supported entry point in this checkpoint release.

### Controls

```sh
SEEDS="0" OUTPUT_DIR="$PWD/my_results" bash run.sh generator
bash run.sh help
```

Defaults are seeds `0 1 2`, temperature `1.2`, mode `selective`, and 400 attempts per template. `PYTORCH_PYTHON`, `TENSORFLOW_PYTHON` and `METRICS_PYTHON` select the Python executables. `FEATURE_DIR` reuses precomputed features; `POOL_DIR` selects pools for scoring.

## Evaluation outputs

`run.sh predictor` writes metrics to `outputs/predictor/metrics/`:

- `by_target.csv`: precision and SR for each target and seed.
- `by_seed.csv` and `summary.csv`: per-seed and aggregate results.
- `selected_peptides.csv`: selected sequences and label-based outcomes.
- `regression_by_target.csv`: activity-prediction metrics.
- `thresholds.csv`: activity and selectivity thresholds.

The `dataset` column identifies `benchmark` or `ood`. The `protocol` column describes Pareto-Max selection: `pareto_max_fixed_pool_size` uses the validation-selected candidate-pool size for the benchmark, and `pareto_max_full_pool` uses all available OOD candidates. The selection budgets are 100 and 5 peptides per enzyme, respectively.

Generator NLL, sampled peptides, sequence-quality metrics and PRISM-scored activity profiles are written to `outputs/generator/`. These outputs are produced when you run the models and are excluded from version control. See [REPRODUCTION.md](docs/REPRODUCTION.md) for metric definitions.

## Layout

```text
assets/                 Manuscript PRISM overview (PNG and SVG)
checkpoints/            Released predictor and generator weights
data/                  Labelled splits, sequence IDs and source metadata
predictor/              Prediction, feature extraction and model definitions
generator/              Conditional decoder, sampling and training code
evaluation/             Selection, generation-quality and scoring evaluation
requirements/           Python dependencies
tests/                  Small predictor input and reference-output fixture
run.sh                  Unified entry point
SHA256SUMS              Release file hashes
```

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream code and data attribution. Generated outputs and local environments are excluded by `.gitignore`.
