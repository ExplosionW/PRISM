# Datasets and split definitions

## Sources and filtering

The benchmark uses continuous cleavage Z-scores from the Kukreja et al. dataset distributed in the [CleaveNet data release](https://github.com/microsoft/cleavenet/tree/4dac67defc99ca35d967ddc76eca0fe8b74afdad/data). CleaveNet retained 2,901 of 3,717 test peptides after excluding peptides at Levenshtein distance below 3 from its original 14,866-peptide training pool. PRISM partitions that training pool into 13,666 training and 1,200 validation peptides, keeping connected components at Hamming distance at most 2 together. Released split membership is fixed in `split_ids.json`.

The OOD set contains 80 non-overlapping peptides with published CleaveNet experimental efficiency measurements for 12 MMPs. Source-library controls are excluded. OOD is used for evaluation and is distinct from the benchmark test set. `source_manifest.json` records the pinned upstream commit, original spreadsheet URLs and SHA-256 hashes.

## Benchmark files

`benchmark/{train,val,test}.csv` contains a `sequence` column followed by 18 labelled activity columns. Matching NPZ files contain:

| Array | Shape | Meaning |
| --- | --- | --- |
| `sequences` | N | Canonical amino-acid sequence strings |
| `labels` | N × 18 | Original continuous cleavage Z-scores |
| `conditions` | N × 18 | Generator conditions rounded to 0.1 |
| `tokens` | N × 10 | Generator amino-acid token IDs |

No extra normalization is applied to the benchmark labels. Only generator conditions are rounded, using NumPy ties-to-even rounding in float64. The alphabet is `ACDEFGHIKLMNPQRSTVWY`, with IDs 0–19; START and STOP have IDs 20 and 21. Predictor and generator exchange sequence strings rather than token IDs.

Enzyme order (`benchmark/target_order.json`):

```text
MMP1 MMP10 MMP11 MMP12 MMP13 MMP14 MMP15 MMP16 MMP17
MMP19 MMP2 MMP20 MMP24 MMP25 MMP3 MMP7 MMP8 MMP9
```

MMP13 is column 4 with zero-based indexing.

## OOD files

`ood/ood.csv` contains substrate identifiers, sequences and published efficiency labels. `ood/ood.npz` uses peptide-first axes; `labels` is 80 × 12, and `targets` gives the measured enzyme order. `raw_fold_mean` retains the 80 × 12 raw fold-change means, and `substrate_ids` records the 80 source identifiers. Raw fold changes are not substituted for efficiency labels. `ood/annotations.csv` preserves source grouping and sequence-distance annotations.

Measured enzymes:

```text
MMP1 MMP10 MMP12 MMP13 MMP14 MMP17 MMP2 MMP20 MMP3 MMP7 MMP8 MMP9
```

OOD efficiencies and benchmark Z-scores have different units. OOD candidate ranking uses the full 18-enzyme prediction; label-based evaluation uses the 12 measured enzymes. No labels are imputed for the other six enzymes.

## Verification

`bash run.sh verify` checks sequence identities, label parity between CSV and NPZ, canonical token encoding, target order and absence of exact sequence overlap between every pair of splits. Training and generation-template selection use only `benchmark/train.*`.
