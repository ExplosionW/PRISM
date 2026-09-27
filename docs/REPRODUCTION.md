# Reproduction details

## Predictor

The released seed 0/1/2 checkpoints include the learned PRISM parameters and enzyme descriptors. Residue input is frozen ESM-2 650M layer 33, shape N × 10 × 1,280, from `facebook/esm2_t33_650M_UR50D` revision `08e4846e537177426273712802403f7ba8261b6c`. Published feature extraction used CUDA BF16 computation and FP16 storage. The task language-model state is embedded in the predictor checkpoints.

Prediction NPZ files have `sequences` (N), `targets` (18) and `predictions` (N × 18). Evaluation rejects mismatched sequence or enzyme orders. `reference_results/predictor/` contains validation, test and OOD predictions for each seed.

### Selection metrics

The evaluator calls the packaged Pareto selection implementation in `predictor/vendor/selection.py`. A selected peptide is counted as precise when it exceeds both the target activity threshold and the positive target-versus-strongest-competitor margin threshold. SR is the fraction for which measured target activity does not exceed the strongest measured competitor. Results are computed per enzyme and macro-averaged.

For the benchmark, thresholds are the training median activity and the 90th percentile of positive training target-minus-maximum-competitor margins. If no positive margin exists, that margin threshold is infinite. The selection budget is 100 per enzyme. Validation considers activity-ranked candidate-pool fractions 0, 0.125, 0.167, 0.25, 0.375, 0.5, 0.75 and 1. It retains the largest feasible candidate count maintaining at least 80% of the mean measured activity of the validation top-100 predicted-activity reference. That count is transferred to test selection.

For OOD, the selection budget is five per measured enzyme and the candidate pool contains all 80 peptides. Following the supplied paper evaluation, activity and positive-margin thresholds are calculated within the measured OOD panel. These labels define the evaluation endpoint; model weights are unchanged. Predicted competition covers all 18 enzymes; measured competition covers the 12 available enzymes.

`evaluate_selection.py` writes per-seed, per-target and aggregate metrics, selected-peptide lists, thresholds, and per-enzyme correlations. Benchmark MAE uses the benchmark Z-score scale. The published exact selection results are checked with `run.sh replay`.

## Generator

The generator is an independently trained autoregressive decoder with 2,628,758 parameters, conditioned on enzyme-response profiles and target identity. It is not a reversal of the predictor weights. The final G2 checkpoints are the three seed-specific checkpoints from the zero-contrast-weight continuation. The upstream directory name `contrast0` refers to that training implementation; the released final loss has contrast weight zero.

The training entry point calls the original packaged stages: 50 initial epochs, extended training up to 150 total epochs with validation stopping, then continuation for up to 50 epochs using Adam at 1e-4 and zero contrast weight. Validation stopping follows `generator/optimization_v2/protocol.json`. The conditional sampling probability during training is 0.5.

Conditional test NLL is averaged over 18 target identities. Unconditional NLL is evaluated separately. Joint NLL is their equally weighted mean. Each is teacher-forced token cross-entropy, including the termination token. `generator/evaluate_checkpoint.py` loads each final checkpoint before evaluation.

Default generation uses 50 training-derived MMP13-selective profiles and 400 attempts per profile at temperature 1.2. Selective templates rank measured MMP13 activity minus the mean activity of the other 17 enzymes; efficient templates rank MMP13 activity alone. Conditions use the rounded training profiles. The existing stateless sampler and repetition penalty 1.2 are preserved.

Raw pools retain all attempts. Validity requires normal termination and a canonical 10-mer. Uniqueness is calculated among valid sequences; known-sequence overlap uses the training, validation and benchmark-test sequence union. Novel unique sequences are used for predictor scoring. K-mer entropy uses novel valid attempts, retaining repeat occurrences, as in the supplied result analysis.

### Scorer identity

`run.sh score-generated` averages all three released PRISM predictors. Output metrics are predicted MMP13 activity, target-minus-mean-competitor selectivity, the fraction with MMP13 above all competitors, and top-100 selectivity. Its raw ensemble profiles are also saved.

The functional columns in `reference_results/generator/temperature_results.csv` were evaluated with CleaveNet's released five-member scoring ensemble; its quality/novelty/diversity columns are scorer-independent. `run.sh replay` verifies the latter columns for the three supplied selective pools at temperature 1.2. Reproducing the CleaveNet-scored functional columns additionally requires the upstream scoring ensemble; baseline scorers are not bundled here. PRISM-scored and CleaveNet-scored values should retain their respective scorer labels.

## Included checks

The release was checked locally as follows:

- Dataset labels, split sizes, target order and sequence separation were verified.
- Benchmark and OOD selection results were recomputed from all three saved prediction sets and matched the reference CSVs.
- Generation quality and k-mer entropy were recomputed from all three supplied temperature-1.2 pools and matched their reference values.
- All three predictor checkpoints reproduced the small reference fixture, with maximum absolute error below 2e-6.
- All three generator checkpoints loaded and generated a 50-attempt pool.
- Seed-0 full test NLL was evaluated from the generator checkpoint: conditional and unconditional values agreed with the reference within 4e-6 (TensorFlow 2.18.1, Keras 3.11.2, CPU).

Fresh full-dataset ESM inference and GPU retraining were not rerun during release packaging. The released weights, saved outputs and input datasets are included with file hashes.

Predictor checkpoint checks and metric replay used Python 3.13.5, PyTorch 2.8.0, Transformers 4.57.3, NumPy 2.1.3, pandas 2.2.3 and SciPy 1.15.3. Generator checks used Python 3.10 with the TensorFlow/Keras versions given above.
