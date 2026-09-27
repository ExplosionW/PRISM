"""Pareto-Max selection and evaluation thresholds.

Arrays use enzyme-by-peptide axes. _pareto ranks activity against competitor risk;
the evaluator supplies maximum off-target activity as the risk.
"""
import numpy as np

FRACS = (0., .125, .167, .25, .375, .5, .75, 1.)

def margins(Y):
    P = len(Y)
    return np.stack([Y[t] - np.delete(Y, t, 0).max(0) for t in range(P)])

def thresholds(Ytr):
    """Compute activity and positive-margin thresholds from supplied labels (P, N)."""
    A = np.median(Ytr, 1)
    M = margins(Ytr)
    D = np.array([np.quantile(q[q > 0], .9) if (q > 0).any() else np.inf for q in M])
    return A, D

def _pareto(act, rsk, pool, k):
    sel, remain = [], np.asarray(pool)
    while len(sel) < k and len(remain):
        o = remain[np.lexsort((remain, rsk[remain], -act[remain]))]
        br, ba, front = np.inf, np.inf, []
        for j in o:
            if rsk[j] < br or (rsk[j] == br and act[j] == ba):
                front.append(j); br, ba = rsk[j], act[j]
        sel.extend(front)
        remain = np.setdiff1d(remain, front, assume_unique=True)
    return np.asarray(sel[:k], dtype=int)
