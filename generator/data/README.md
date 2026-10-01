# Generator data

These inputs belong to the released PRISM G1 + Pareto-DPO generator. They do not change the predictor's existing benchmark files.

- `fit.npz`: 10,954 measured peptides, with `tokens`, `labels`, rounded `conditions`, `sequences`, and indices into the earlier training split.
- `development.npz`: 1,434 measured peptides for development NLL. This split was reused during development and is not an independent test set.
- `preferences.csv`: 6,400 computational same-request preference pairs (5,101 train, 1,299 validation). `delta_activity`, `delta_mean17` and `delta_max17` are differences in frozen PRISM predictions, not measured labels. The two pair splits contain disjoint peptide sequences.
- `requests.json`: original rounded 18-enzyme fit request for each `fit_condition_index` in the preference table.
- `templates.json`: 50 fixed generation profiles and their source fit sequences. The order is part of the sampling procedure.
- `known_sequences.txt`: 31,383 sequence IDs excluded from generated-pool novelty, including the public 18,583 CleaveNet sequences and all 12,800 preference-pair peptides.
- `target_order.json`: output/condition order, with MMP13 at zero-based index 4.

Measured profiles use the source dataset's Z-score scale. The generator alphabet is `ACDEFGHIKLMNPQRSTVWY`, START = 20 and STOP = 21. The original dataset attribution is in [the repository data documentation](../../data/README.md). Public test/validation sequence IDs are used only for novelty exclusion; this directory adds no corresponding held-out measured labels.
