# Predictor data

| Split | Peptides | Measured enzymes | Files |
| --- | ---: | ---: | --- |
| Training | 13,666 | 18 | `benchmark/train.csv`, `benchmark/train.npz` |
| Validation | 1,200 | 18 | `benchmark/val.csv`, `benchmark/val.npz` |
| Benchmark test | 2,901 | 18 | `benchmark/test.csv`, `benchmark/test.npz` |
| OOD | 80 | 12 | `ood/ood.csv`, `ood/ood.npz` |

The four splits have no exact sequence overlap. Membership is recorded in `split_ids.json`.

## File format

Benchmark CSVs contain `sequence` followed by 18 activity columns. NPZ files contain:

| Array | Shape | Contents |
| --- | --- | --- |
| `sequences` | N | Ten-residue peptide strings |
| `labels` | N × 18 | Cleavage Z-scores |
| `conditions` | N × 18 | Profiles rounded to 0.1 |
| `tokens` | N × 10 | Amino-acid token IDs |

Token order is `ACDEFGHIKLMNPQRSTVWY` (0–19); START = 20 and STOP = 21. Enzyme order is in `benchmark/target_order.json`; MMP13 has zero-based index 4.

OOD NPZ files contain `sequences`, `labels` (80 × 12 published efficiencies), `targets`, `substrate_ids` and `raw_fold_mean`. Use `labels` for evaluation; benchmark Z-scores and OOD efficiencies have different units. The measured enzyme order is in `ood/target_order.json`.

## Source

Benchmark measurements originate from Kukreja et al., distributed with CleaveNet; OOD labels come from published CleaveNet experiments. Source links and preprocessing metadata are in [source_manifest.json](source_manifest.json).
