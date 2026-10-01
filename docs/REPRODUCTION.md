# Reproduction details

## Predictor

The released seed 0/1/2 checkpoints include the learned PRISM parameters and enzyme descriptors. Residue input is frozen ESM-2 650M layer 33, shape N × 10 × 1,280, from `facebook/esm2_t33_650M_UR50D` revision `08e4846e537177426273712802403f7ba8261b6c`. Published feature extraction used CUDA BF16 computation and FP16 storage. The task language-model state is embedded in the predictor checkpoints.

Prediction NPZ files have `sequences` (N), `targets` (18) and `predictions` (N × 18). Evaluation rejects mismatched sequence or enzyme orders. New predictions are saved under `outputs/predictor/predictions/`.

### Selection metrics

The evaluator calls the packaged Pareto selection implementation in `evaluation/selection.py`. A selected peptide is counted as precise when it exceeds both the target activity threshold and the positive target-versus-strongest-competitor margin threshold. SR is the fraction for which measured target activity does not exceed the strongest measured competitor. Results are computed per enzyme and macro-averaged.

For the benchmark, thresholds are the training median activity and the 90th percentile of positive training target-minus-maximum-competitor margins. If no positive margin exists, that margin threshold is infinite. The selection budget is 100 per enzyme. Validation considers activity-ranked candidate-pool fractions 0, 0.125, 0.167, 0.25, 0.375, 0.5, 0.75 and 1. It retains the largest feasible candidate count maintaining at least 80% of the mean measured activity of the validation top-100 predicted-activity reference. That count is transferred to test selection.

For OOD, the selection budget is five per measured enzyme and the candidate pool contains all 80 peptides. Following the supplied paper evaluation, activity and positive-margin thresholds are calculated within the measured OOD panel. These labels define the evaluation endpoint; model weights are unchanged. Predicted competition covers all 18 enzymes; measured competition covers the 12 available enzymes.

`evaluate_selection.py` writes per-seed, per-target and aggregate metrics, selected-peptide lists, thresholds, and per-enzyme correlations. Benchmark MAE uses the benchmark Z-score scale.

## Generator

The default generator is **PRISM G1 + Pareto-DPO**, with three released fixed-final checkpoints. Its standalone model, exact trainable parameter scope, DPO continuation, inputs and sampling kernel are documented in [generator/README.md](../generator/README.md). The earlier TensorFlow G2 sources and weights are retained but are not the default `run.sh` implementation.

Current NLL uses the 1,434-peptide generator development split and target MMP13, including STOP. Joint NLL averages conditional and unconditional token NLL. The split was repeatedly used for development and must not be described as independent confirmation.

Generation retains every attempt. Normal STOP plus ten canonical amino acids defines legality; uniqueness is measured among legal sequences. Novelty uses the fixed public-sequence/preference-pair union in `generator/data/known_sequences.txt` (31,383 IDs). This differs from the older 17,767-ID benchmark-only exclusion. Any external baseline must use the same exclusion set before comparing new-peptide yield. K-mer entropy is a descriptive distribution statistic, not a quantity assumed to improve whenever it increases.

### Scorer identity and target endpoint

`score-generated` averages the repository's three released PRISM predictors. It is a self-scoring utility; these outputs are not the historical official/native/ListNet cross-scores. The principal predicted joint event is MMP13 Z > 1 and MMP13 Z greater than the maximum of the other 17 scores. Count different legal novel hits over the full pool and divide by all attempts for yield per attempt. Target-minus-mean17 is a separate selectivity statistic.

Top-24/100 use `min(Z13 - 1, Z13 - max17)` in descending order, with sequence order breaking ties. Candidates are deduplicated before selection and never refilled. Selected rates are predicted outcomes, not measured precision. Replicates are averaged within training seed before means and sample SD across seeds; a single seed has no estimated training-seed SD.

## Output labels

The `dataset` field uses `benchmark` and `ood`. The `protocol` field uses `pareto_max_fixed_pool_size` for benchmark selection and `pareto_max_full_pool` for OOD selection. Fixed pool size means that the number of candidates selected on validation is carried over to benchmark evaluation. It is a selection setting, not a dataset name.

## Checkpoint-loading test

`bash run.sh smoke` loads the three predictor checkpoints, compares predictions for four peptides against a small numerical test fixture, and generates 50 attempts from each generator. The fixture is kept in `tests/` to check installation and checkpoint compatibility. Full predictions and generated pools are created by the main run commands.
