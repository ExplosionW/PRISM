#!/usr/bin/env bash
# Thin entry point for the packaged PRISM inference, generation and evaluation code.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTORCH_PYTHON="${PYTORCH_PYTHON:-python}"
TENSORFLOW_PYTHON="${TENSORFLOW_PYTHON:-python}"
METRICS_PYTHON="${METRICS_PYTHON:-$PYTORCH_PYTHON}"
DEVICE="${DEVICE:-cuda}"
FEATURE_PRECISION="${FEATURE_PRECISION:-bf16}"
SEEDS="${SEEDS:-0 1 2}"
TEMPERATURE="${TEMPERATURE:-1.2}"
PER_TEMPLATE="${PER_TEMPLATE:-400}"
MODE="${MODE:-selective}"
OUT="${OUTPUT_DIR:-$ROOT/outputs}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
for seed in $SEEDS; do
  case "$seed" in 0|1|2) ;; *) echo 'SEEDS must contain only 0, 1 and 2.' >&2; exit 2;; esac
done
help() {
  cat <<'EOF'
Usage: bash run.sh COMMAND

  verify           Verify files, labelled datasets and disjoint sequence splits.
  smoke            Load all predictor weights and sample 50 peptides per generator.
  predictor        Extract ESM features; predict val/test/OOD; evaluate all three seeds.
  generator        Evaluate test NLL and generate MMP13-conditioned peptide pools.
  score-generated  Score generated pools using the three-seed PRISM predictor ensemble.
  all              Run verify, predictor, generator and score-generated.
  train-generator  Run the existing three-stage G2 training pipeline on a CUDA GPU.

Environment variables:
  PYTORCH_PYTHON, TENSORFLOW_PYTHON, METRICS_PYTHON   Python executables
  DEVICE=cuda, FEATURE_PRECISION=bf16              Predictor/ESM device and precision
  SEEDS="0 1 2", TEMPERATURE=1.2, MODE=selective     Experiment settings
  PER_TEMPLATE=400, OUTPUT_DIR=./outputs           Sampling count and output location
  FEATURE_DIR=./outputs/features                   Reuse extracted ESM features
  POOL_DIR=./outputs/generator/pools               Input pools for score-generated

CPU feature extraction: DEVICE=cpu FEATURE_PRECISION=fp32 bash run.sh predictor
EOF
}
if [[ "${1:-help}" == help || "${1:-}" == --help || "${1:-}" == -h ]]; then help; exit 0; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
FEATURE_DIR="${FEATURE_DIR:-$OUT/features}"
verify() { "$METRICS_PYTHON" evaluation/verify_data.py; }
smoke() {
  "$PYTORCH_PYTHON" evaluation/smoke_predictor.py
  mkdir -p "$OUT/smoke/pools"
  for seed in $SEEDS; do
    "$TENSORFLOW_PYTHON" generator/sample.py --checkpoint "checkpoints/generator/seed${seed}.weights.h5" --seed "$seed" --per-template 1 --output "$OUT/smoke/pools/PRISM_generator_seed${seed}_sample${seed}_T1.2_selective.csv"
  done
  "$METRICS_PYTHON" evaluation/generator_quality.py --pools "$OUT/smoke/pools" --output "$OUT/smoke/quality"
}
predictor() {
  mkdir -p "$FEATURE_DIR" "$OUT/predictor/predictions"
  for split in val test ood; do
    input="data/benchmark/$split.csv"
    [[ "$split" != ood ]] || input=data/ood/ood.csv
    if [[ ! -f "$FEATURE_DIR/$split.npz" ]]; then
      "$PYTORCH_PYTHON" predictor/extract_features.py --input "$input" --output "$FEATURE_DIR/$split.npz" --device "$DEVICE" --precision "$FEATURE_PRECISION"
    fi
    for seed in $SEEDS; do
      "$PYTORCH_PYTHON" predictor/predict.py --checkpoint "checkpoints/predictor/seed${seed}.pt" --input "$FEATURE_DIR/$split.npz" --output "$OUT/predictor/predictions/${split}_seed${seed}.npz" --device "$DEVICE"
    done
  done
  "$METRICS_PYTHON" evaluation/evaluate_selection.py --predictions "$OUT/predictor/predictions" --output "$OUT/predictor/metrics" --seeds $SEEDS
}
generator() {
  mkdir -p "$OUT/generator/nll" "$OUT/generator/pools"
  for seed in $SEEDS; do
    "$TENSORFLOW_PYTHON" generator/evaluate_checkpoint.py --checkpoint "checkpoints/generator/seed${seed}.weights.h5" --output "$OUT/generator/nll/nll_seed${seed}.json"
    "$TENSORFLOW_PYTHON" generator/sample.py --checkpoint "checkpoints/generator/seed${seed}.weights.h5" --seed "$seed" --temperature "$TEMPERATURE" --mode "$MODE" --per-template "$PER_TEMPLATE" --output "$OUT/generator/pools/PRISM_generator_seed${seed}_sample${seed}_T${TEMPERATURE}_${MODE}.csv"
  done
  "$METRICS_PYTHON" evaluation/summarize_nll.py --input "$OUT/generator/nll"
  "$METRICS_PYTHON" evaluation/generator_quality.py --pools "$OUT/generator/pools" --output "$OUT/generator/quality"
}
score_generated() {
  "$PYTORCH_PYTHON" evaluation/score_generated.py --pools "${POOL_DIR:-$OUT/generator/pools}" --output "$OUT/generator/prism_scores" --device "$DEVICE" --precision "$FEATURE_PRECISION"
}
train_generator() {
  for seed in $SEEDS; do
    "$TENSORFLOW_PYTHON" generator/src/train_generator.py --variant G2 --seed "$seed"
    "$TENSORFLOW_PYTHON" generator/optimization_v2/train.py --seed "$seed" --prob 0.5 --stage long
    "$TENSORFLOW_PYTHON" generator/optimization_v2/train.py --seed "$seed" --prob 0.5 --stage contrast --weight 0 --source "generator/optimization_v2/runs/long/p50/seed${seed}"
  done
}
case "$1" in
  verify) verify;; smoke) smoke;; predictor) predictor;; generator) generator;;
  score-generated) score_generated;; all) verify; predictor; generator; score_generated;;
  train-generator) train_generator;; *) help; exit 2;;
esac
