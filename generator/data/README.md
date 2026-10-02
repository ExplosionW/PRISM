# Generator inputs

| File | Contents |
| --- | --- |
| `fit.npz` | 10,954 fitting peptides with sequences, tokens, measured labels and rounded conditions |
| `development.npz` | 1,434 peptides for development loss evaluation |
| `preferences.csv` | 6,400 pairs: 5,101 training and 1,299 validation |
| `requests.json` | Requested 18-enzyme profiles, keyed by `fit_condition_index` |
| `templates.json` | 50 generation profiles and their fitting sequences, in sampling order |
| `known_sequences.txt` | 31,383 sequences excluded when counting novel outputs |
| `target_order.json` | Enzyme order; MMP13 has zero-based index 4 |

Profiles use the source Z-score scale, with conditions rounded to 0.1. Token order is `ACDEFGHIKLMNPQRSTVWY` (0–19); START = 20 and STOP = 21.

Preference columns `delta_activity`, `delta_mean17` and `delta_max17` contain differences in PRISM predictions. The novelty exclusion file contains public dataset sequences and both members of the preference pairs. Measured-data sources are documented in [data/README.md](../../data/README.md).
