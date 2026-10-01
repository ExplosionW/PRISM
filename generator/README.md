# PRISM generator

This directory contains the released **PRISM G1 + Pareto-DPO** generator: three training seeds, their G1 initialization checkpoints, portable PyTorch inference, and the fixed DPO continuation. The released files are the final checkpoints after 1,000 DPO updates; inference does not average the three training seeds.

## Contents

| Path | Contents |
| --- | --- |
| `checkpoints/seed{0,1,2}.pt` | Released Pareto-DPO checkpoints |
| `initialization/seed{0,1,2}.pt` | G1 starting checkpoints for reproducing DPO |
| `model.py` | Standalone model definition and strict checkpoint loader |
| `sampling.py`, `sample.py` | Released sampling kernel and CSV interface |
| `train.py` | Fixed DPO continuation from the packaged initialization |
| `evaluate_checkpoint.py` | Conditional, unconditional, joint and selective development NLL |
| `data/` | Preference pairs, requests, fit/development arrays, sampling templates and novelty exclusions |
| `configs/pareto_dpo.json` | Model, training and inference settings |

The older TensorFlow implementation remains under `src/` and `optimization_v2/`, with its earlier weights at `../checkpoints/generator/`. It is not the default generator in `run.sh`.

## Installation and generation

```sh
pip install -r requirements/generator.txt
DEVICE=cuda bash run.sh generator
# Small CPU example, 50 attempts:
python generator/sample.py --checkpoint generator/checkpoints/seed0.pt \
  --device cpu --per-template 1 --output outputs/example.csv
# Unconditional example:
python generator/sample.py --checkpoint generator/checkpoints/seed0.pt \
  --mode unconditional --device cuda --output outputs/unconditional.csv
```

The default conditional run makes 20,000 attempts: 400 for each of 50 fixed fit-derived profiles. Template eligibility is measured MMP13 Z > 1; ranking uses MMP13 minus the strongest of the other 17 enzymes, with sequence order breaking ties. Inputs are rounded to 0.1 in the fixed enzyme order. These are previously used fit conditions, not unseen-condition validation.

The released sampler first applies `log_softmax`, then draws at temperature 1.2 conditionally or 1.0 unconditionally. From the second position, an adjacent repeated draw triggers another draw at temperature 1.0 after dividing the selected canonical log-probability by 1.2. This is the evaluated G1 canonical two-draw procedure, **not** the ancestral sampler or CleaveNet's raw-logit procedure. It allows up to 15 tokens and does not mask START, force STOP/length, or refill invalid attempts. Changing batch size, device or numerical precision can change the sampled sequences; CUDA FP32 with batch 512 is the reproduction setting.

CSV files retain every attempt, including invalid, duplicate and known sequences. `stopped_normally` and `raw_length` identify normal-STOP canonical 10-mers. The `filter_reason` column is descriptive; rows are not removed.

## Model

The model has 1,949,048 stored parameters. Conditional probabilities are the equal-weight mixture of two autoregressive experts, initialized from profile and competition-profile representations. Each expert uses a three-layer, width-64 pre-normalized decoder, six attention heads with 64 dimensions per head, an 18-value condition prefix, enzyme-profile memory, relative attention biases and feature-wise linear modulation (FiLM). The competition representation is a deterministic transformation of the same 18-value profile, not additional measured information. The stored router is inactive. Unconditional generation uses the first expert's unconditional base; the checkpoint includes both base copies.

DPO updates exactly the parameter subset used in the released run: names containing `.base.` and names starting with `router.` are frozen. This also freezes the conditional decoder's `student.base` parameters, including its relative-attention parameters. The 617,574 trainable parameters are the profile/enzyme memory, memory cross-attention and gates, associated normalizations, student final normalization, and FiLM modules. All forwards during DPO use evaluation mode with gradients enabled. The unconditional distribution is unchanged from the packaged G1 initialization.

Relative attention, FiLM, mixtures and DPO are established techniques. This release combines them for the stated peptide-generation task; it does not claim their general invention.

## Reproduce the DPO continuation

```sh
DEVICE=cuda bash run.sh train-generator
# A single seed, with explicit output and recovery:
python generator/train.py --seed 0 --device cuda --output outputs/dpo/seed0
python generator/train.py --seed 0 --device cuda --output outputs/dpo/seed0 --resume
```

The packaged 6,400 same-request preference pairs contain 5,101 training pairs and 1,299 validation pairs, with no sequence shared between the pair splits. Chosen peptides satisfy the frozen PRISM predictor's MMP13 activity and maximum-competitor criteria and improve activity, target-minus-mean-competitor selectivity and target-minus-maximum-competitor margin relative to the rejected peptide. These are **predicted preferences**, not measured preference labels. Cross-scoring models were not used to label the pairs. Both members receive their original shared fit-template request, not their own predicted profile.

Training uses the standard reference-relative DPO objective with beta 0.1, summed sequence log-probability over ten residues plus STOP, Adam lr 1e-6 / eps 1e-7, gradient clipping at 1, batch 128 with replacement, and exactly 1,000 updates. The reference is the corresponding frozen G1 checkpoint. No early stopping or post-hoc best-checkpoint selection is applied. The entry point reproduces the DPO continuation from the supplied G1 initialization, not all preceding G1 development experiments. Numerical reproduction can depend on the PyTorch/CUDA environment.

## Evaluation and selection

```sh
python generator/evaluate_checkpoint.py --checkpoint generator/checkpoints/seed0.pt \
  --device cuda --output outputs/development_nll.json
bash run.sh score-generated
```

Development NLL is teacher-forced cross-entropy over ten residues and STOP. Joint NLL equally averages conditional and unconditional NLL. Selective NLL uses measured development MMP13 Z > 1 and activity above every other enzyme. This split was repeatedly used during model development; it is not an independent confirmation set.

The primary generated-pool endpoint is the **number of different, legal, novel peptides per fixed attempt budget with predicted MMP13 Z > 1 and MMP13 greater than the maximum of the other 17 predicted activities**. Novelty excludes the same 31,383 sequences for every compared model: all 18,583 public CleaveNet sequence IDs plus both members of all 6,400 preference pairs. No held-out measured labels are used for this exclusion. Training-pair sequences are therefore never counted as new target hits.

For selection, `active_bottleneck` ranks the entire legal, novel, deduplicated pool by `min(Z13 - 1, Z13 - max(other 17))`, with sequence order breaking ties. Top-24 and top-100 subsets are reported without refill. This differs from the paper's target-minus-mean-competitor selectivity, which is reported separately. Whole-pool yield does not depend on this selector.

`score-generated` uses the three PRISM predictor checkpoints already distributed in this repository. It is a PRISM self-score utility, not a substitute for external CleaveNet/ListNet cross-scoring or laboratory cleavage assays. Predicted joint satisfaction must not be called measured precision or demonstrated biological selectivity. Sampling repeats are averaged within each training seed before training-seed means and sample SD are calculated. Neither mean improvements nor correlated cross-scorers establish statistical significance or independent biological validation.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for CleaveNet-derived code and data attribution, and [DPO](https://arxiv.org/abs/2305.18290) for the preference objective.
