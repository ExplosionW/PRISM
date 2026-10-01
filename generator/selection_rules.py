"""Fixed MMP13 bottleneck selection from already scored novel peptides."""
import numpy as np


def select(sequences, profiles, k=100, target=4):
    sequences = np.asarray(sequences)
    profiles = np.asarray(profiles)
    if profiles.shape != (len(sequences), 18) or not np.isfinite(profiles).all():
        raise ValueError('Expected finite N x 18 profiles aligned to sequences.')
    if len(set(sequences)) != len(sequences):
        raise ValueError('Deduplicate the legal novel pool before selection.')
    activity = profiles[:, target]
    margin = activity - np.delete(profiles, target, 1).max(1)
    bottleneck = np.minimum(activity - 1, margin)
    return np.lexsort((sequences, -bottleneck))[:k]
