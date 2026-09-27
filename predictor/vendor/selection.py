#!/usr/bin/env python
"""终点选样评估（§104.7 第 2 项）——训练脚本内置，ESM 与 R8 共用同一份实现。

**为什么必须共用**：审查指出两边虽然用同样的损失名称，实际定义不同（§104.4-B）。
选样评估若再各写一份，同类分歧会重演。本模块被两个训练脚本 import，
保证「模型＋同一算法」里的「同一算法」在代码层面确实同一。

**协议**（与 `analysis/model_research_20260910/selection_protocol.json` 一致，
已由 `analysis/selection_recheck/recheck.py` 独立重写复现）：

  A_min  每酶**训练**活性中位数；Δ_min 每酶**训练**正边际 90 分位。
         两者都是计算阈值，**不是 W1 湿实验标定值**，报告时不得称为标定。
  usable (y ≥ A_min) 且 (margin ≥ Δ_min)，合取。
  risk   17 个脱靶预测中最高 floor(17/4)=4 个的均值（沿用历史约定，
         非严格分数权重 CVaR25%）。
  标定   在**验证集**上取满足「平均真实活性保持率 ≥ retain」的最大候选池，冻结。
  迁移   固定候选**比例** 与 固定候选**数量** 两种口径**同时**报告；
         不得依据测试成绩挑一种宣布成功。
  测试   真值只用于打分。活性保持失败原样报告，**绝不回调候选池**。

稳健性（recheck.py 实测）：λ_c=36 在 27 组 (K, retain, risk) 设定中 23 组增益为正、
17 组三 seed 全胜；失败集中在 K=50 且 risk 取全部脱靶均值处。故本模块默认
输出基准口径，同时提供 `sweep()` 用于报告协议敏感性。
"""
import numpy as np

FRACS = (0., .125, .167, .25, .375, .5, .75, 1.)


def margins(Y):
    P = len(Y)
    return np.stack([Y[t] - np.delete(Y, t, 0).max(0) for t in range(P)])


def thresholds(Ytr):
    """只用训练标签。Ytr 形状 (P, N_train)。"""
    A = np.median(Ytr, 1)
    M = margins(Ytr)
    D = np.array([np.quantile(q[q > 0], .9) if (q > 0).any() else np.inf for q in M])
    return A, D


def risk_of(pred, t, mode="top4"):
    off = np.delete(pred, t, 0)
    if mode == "max":
        return off.max(0)
    if mode == "mean":
        return off.mean(0)
    return np.sort(off, 0)[::-1][:max(1, off.shape[0] // 4)].mean(0)


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


def select(pred, t, k, frac, rule="pareto", rmode="top4"):
    """只接收预测，不接收任何真值。"""
    act = pred[t]
    order = np.argsort(-act, kind="stable")
    if rule == "activity":
        return order[:k]
    rsk = risk_of(pred, t, rmode)
    cnt = k if frac == 0 else max(k, int(np.ceil(frac * len(act))))
    pool = order[:cnt]
    if rule == "scalar":
        return pool[np.argsort(rsk[pool], kind="stable")[:k]]
    return _pareto(act, rsk, pool, k)


def evaluate(Pva, Yva, Pte, Yte, Ytr, k=100, retain=.8, rule="pareto", rmode="top4"):
    """返回逐 (靶点, 迁移口径) 的明细 dict 列表。标定只用验证集。"""
    A, D = thresholds(Ytr)
    hit = (Yte >= A[:, None]) & (margins(Yte) >= D[:, None])
    rows = []
    for t in range(len(Yte)):
        av = float(Yva[t, select(Pva, t, k, 0, "activity", rmode)].mean())
        at = float(Yte[t, select(Pte, t, k, 0, "activity", rmode)].mean())
        if not (av > 0 and at > 0):
            continue
        feas = [q for q in FRACS
                if Yva[t, select(Pva, t, k, q, rule, rmode)].mean() / av >= retain]
        frac = max(feas) if feas else 0.
        for transfer in ("count", "fraction"):
            ft = frac
            if transfer == "count" and frac != 0:
                j = max(k, int(np.ceil(frac * Yva.shape[1])))
                ft = np.nextafter(j / Pte.shape[1], 0.)
            ix = select(Pte, t, k, ft, rule, rmode)
            ret = float(Yte[t, ix].mean() / at)
            rows.append(dict(target=t, transfer=transfer, val_frac=frac, test_frac=float(ft),
                             usable=float(hit[t, ix].mean()), retention=ret,
                             pass_retain=bool(ret >= retain)))
    return rows, hit


def summarize(rows):
    """按迁移口径汇总：usable、真实活性保持率、逐靶点达标格数。"""
    out = {}
    for tf in ("count", "fraction"):
        g = [r for r in rows if r["transfer"] == tf]
        if not g:
            continue
        out[tf] = dict(usable=float(np.mean([r["usable"] for r in g])),
                       retention=float(np.mean([r["retention"] for r in g])),
                       n_pass=int(sum(r["pass_retain"] for r in g)), n=len(g))
    return out


def sweep(Pva, Yva, Pte, Yte, Ytr, ks=(50, 100, 200), retains=(.75, .8, .85),
          rmodes=("top4", "max", "mean")):
    """协议敏感性：不挑最好的一组报告，全部列出。"""
    res = []
    for k in ks:
        for r in retains:
            for m in rmodes:
                rows, _ = evaluate(Pva, Yva, Pte, Yte, Ytr, k, r, "pareto", m)
                s = summarize(rows)
                res.append(dict(k=k, retain=r, risk=m, **{f"{a}_{b}": c
                                                          for a, d in s.items()
                                                          for b, c in d.items()}))
    return res
