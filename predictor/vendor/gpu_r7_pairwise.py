#!/usr/bin/env python
"""R7：竞争酶对驱动的读取 + 隔离验证集（§95.6 的实现）。

**动机**：决策量 \\(M_{ts}=Y_{ts}-\\max_{o\\neq t}Y_{os}\\) **本身就是成对量**，
而 R1 的条件化是单酶的；§84.2 又测出模型只有 37.5% 的时候知道**哪个**酶是竞争者。
故把条件从单酶扩展到竞争酶对，在肽信息压缩**之前**决定「哪些位点组合能区分 p 与 q」。

**架构**（对每个无序酶对 {p,q}）：

    sym(e_p,e_q) = [e_p + e_q,  e_p ⊙ e_q]        对称，故只需 153 对而非 306
    γ_pq, β_pq   = MLP(sym)                       酶对条件的 FiLM
    e^{(pq)}_i   = (1+γ_pq) ⊙ e_i + β_pq          池化之前调制
    h_pq         = trunk_shared(pool(e^{(pq)}))
    D_pq         = ⟨W h_pq,  e_p − e_q⟩           **构造上反对称**，D_qp = −D_pq
    u_p          = (1/P) Σ_q D_pq                 最小二乘投影的闭式解
    Ŷ_p          = a_p + ĝ + u_p

**为什么这样构造**：审查（§95.6.3）指出，若 \\(D_{pq}\\) 可分解为 \\(b_p-b_q\\)，
投影后精确退回普通中心化 profile，则该方案无独立内容。本实现把
「读什么」（对称的酶对条件，决定 \\(h_{pq}\\)）与「往哪个方向读」（反对称的 \\(e_p-e_q\\)）分开：
\\(h_{pq}\\) 依赖**酶对**，故 \\(D_{pq}\\) 不可分解。

**关键消融**（`--no-pair-cond`）：令 \\(h_{pq}=h\\)（不做酶对条件），则
\\(D_{pq}=\\langle Wh, e_p\\rangle-\\langle Wh, e_q\\rangle = b_p-b_q\\)，
**可证地退回普通中心化**。故该臂与主臂之差，精确度量「压缩前的酶对共同条件读取」的贡献。

**隔离验证集**（`--iso-val`，针对 §96.2）：实测原 1,200 条验证集有 16.8%、
随机 5 折有 18.5% 距其训练集 ≤2，而过滤测试集为 **0%**（最小距离 3）。
故验证指标一直在**更容易的分布**上度量。本脚本按「距离 ≤2 连边」的连通分量
整体分配来构造验证集（实测 12,819 个分量、最大仅 539 条，可行），
使开发验证与测试集同为 ≥3 隔离。
"""
import argparse, itertools, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

B = os.environ.get("PROTEASE_DATA", "./data")
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_raw():
    Str = pd.read_csv(f"{B}/X_train.csv", header=None)[0].astype(str).str.strip().to_numpy()
    Ste = pd.read_csv(f"{B}/X_test.csv", header=None)[0].astype(str).str.strip().to_numpy()
    Stf = pd.read_csv(f"{B}/X_test_filtered.csv", header=None)[0].astype(str).str.strip().to_numpy()
    Yte = pd.read_csv(f"{B}/y_test.csv", header=None).to_numpy().T
    pos = {s: i for i, s in enumerate(Ste)}
    Yf = Yte[:, np.array([pos[s] for s in Stf])]
    seqs = pd.read_csv(f"{B}/seqs.csv", header=None)[0].astype(str).str.strip().to_numpy()
    Yall = np.load(f"{B}/Y.npy").astype(np.float32)
    si = {s: i for i, s in enumerate(seqs)}
    return Str, Yall[:, [si[s] for s in Str]].T, Stf, Yf


def encode(seqs, L=10):
    ix = {c: i for i, c in enumerate(AA)}
    return np.array([[ix.get(c, 20) for c in str(s).strip()[:L]] for s in seqs], np.int64)


def iso_split(Z, n_val, seed=0, thr=2):
    """按「距离 ≤thr 连边」的连通分量整体分配，切出与测试集同等隔离度的验证集。

    随机划分做不到这一点：实测原 1,200 条验证集 16.8%、随机 5 折 18.5%
    距其训练集 ≤2，而过滤测试集为 0%（§96.2）。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    N = len(Z); rows, cols = [], []
    for i in range(0, N, 500):
        d = (Z[i:i + 500, None, :] != Z[None, :, :]).sum(2)
        r, c = np.where(d <= thr)
        rows.append(r + i); cols.append(c)
    r = np.concatenate(rows); c = np.concatenate(cols)
    m = r != c; r, c = r[m], c[m]
    _, lab = connected_components(csr_matrix((np.ones(len(r)), (r, c)), shape=(N, N)),
                                  directed=False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(lab.max() + 1)
    sz = np.bincount(lab, minlength=lab.max() + 1)
    va_comp, tot = set(), 0
    for cmp_ in order:
        if tot >= n_val:
            break
        va_comp.add(cmp_); tot += sz[cmp_]
    is_va = np.array([l in va_comp for l in lab])
    return ~is_va, is_va


class R7Net(nn.Module):
    def __init__(s, P, L=10, d=96, m=32, hid=384, pdrop=0.15, de=32,
                 pair_cond=True, Emb=None, esm_dim=0):
        super().__init__()
        s.P, s.L, s.pair_cond = P, L, pair_cond
        s.pairs = list(itertools.combinations(range(L), 2))
        s.epairs = list(itertools.combinations(range(P), 2))          # 153 个无序酶对
        s.E = nn.Parameter(torch.randn(L, 21, d) * 0.05)
        s.esm = esm_dim > 0
        if s.esm:
            s.eproj = nn.Sequential(nn.Linear(esm_dim, d), nn.LayerNorm(d))
            s.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
        s.Wp = nn.Parameter(torch.randn(len(s.pairs), 2 * d, m) * (1.0 / (2 * d) ** .5))
        s.bp = nn.Parameter(torch.zeros(len(s.pairs), m))
        s.z = nn.Parameter(torch.zeros(len(s.pairs)))
        din = L * d + len(s.pairs) * m
        s.trunk = nn.Sequential(nn.Linear(din, hid), nn.ReLU(), nn.Dropout(pdrop),
                                nn.Linear(hid, hid // 2), nn.ReLU())
        s.a = nn.Parameter(torch.zeros(P))
        s.g = nn.Linear(hid // 2, 1)
        if Emb is None:
            s.ep = nn.Parameter(torch.randn(P, de) * 0.05); s.Eproj = None
        else:
            s.register_buffer("Ep_raw", torch.tensor(Emb))
            s.Eproj = nn.Sequential(nn.Linear(Emb.shape[1], de), nn.LayerNorm(de))
        # 对称的酶对条件 → FiLM；输入是 [e_p+e_q, e_p⊙e_q]，故对 p,q 交换不变
        s.film = nn.Sequential(nn.Linear(2 * de, 64), nn.ReLU(), nn.Linear(64, 2 * d))
        s.Wd = nn.Linear(hid // 2, de)                # 读出方向投影

    def ep_vec(s):
        return s.ep if s.Eproj is None else s.Eproj(s.Ep_raw)

    def body(s, e):
        i_, j_ = zip(*s.pairs)
        cat = torch.cat([e[:, list(i_)], e[:, list(j_)]], -1)
        u = torch.relu(torch.einsum("bpk,pkm->bpm", cat, s.Wp) + s.bp)
        gg = torch.sigmoid(s.z)[None, :, None]
        return s.trunk(torch.cat([e.flatten(1), (u * gg).flatten(1)], 1))

    def forward(s, Z, Zr=None, F=None):
        Bn = Z.shape[0]
        e = torch.stack([s.E[i][Z[:, i]] for i in range(s.L)], 1)
        if s.esm:
            e = s.fuse(torch.cat([e, s.eproj(F[Zr].float())], -1))
        h0 = s.body(e)
        gg = s.g(h0)
        ev = s.ep_vec()                                               # (P, de)
        pi = torch.tensor([p for p, _ in s.epairs], device=Z.device)
        qi = torch.tensor([q for _, q in s.epairs], device=Z.device)
        diff = ev[pi] - ev[qi]                                        # (K, de) 反对称方向
        if s.pair_cond:
            sym = torch.cat([ev[pi] + ev[qi], ev[pi] * ev[qi]], -1)   # (K, 2de) 对称
            gb = s.film(sym); gam, bet = gb.chunk(2, -1)              # 各 (K, d)
            em = e[:, None] * (1 + gam[None, :, None]) + bet[None, :, None]
            K = len(s.epairs)
            hp = s.body(em.reshape(Bn * K, s.L, -1)).reshape(Bn, K, -1)
        else:
            # 消融：h 不做酶对条件 ⇒ D_pq = ⟨Wh,e_p⟩−⟨Wh,e_q⟩ = b_p−b_q，可证退回普通中心化
            hp = h0[:, None, :].expand(-1, len(s.epairs), -1)
        D = (s.Wd(hp) * diff[None]).sum(-1)                           # (B, K) 反对称分数
        u = torch.zeros(Bn, s.P, device=Z.device, dtype=D.dtype)
        u.index_add_(1, pi, D); u.index_add_(1, qi, -D)               # u_p = Σ_q D_pq
        u = u / s.P
        return s.a[None, :] + gg + (u - u.mean(1, keepdim=True))


def hard_margin(Y, eye):
    Z = Y.unsqueeze(1).expand(-1, Y.shape[1], -1).masked_fill(eye, -1e9)
    return Y - Z.max(dim=2).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pair-cond", action="store_true",
                    help="消融：关闭酶对条件，D_pq 退化为 b_p−b_q（可证等价于普通中心化）")
    ap.add_argument("--iso-val", action="store_true",
                    help="按连通分量构造与测试集同等隔离（≥3）的验证集")
    ap.add_argument("--n-val", type=int, default=1200)
    ap.add_argument("--esm-feats", default=None)
    ap.add_argument("--esm-layer", type=int, default=33)
    ap.add_argument("--protease-emb", default=None)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--pair-dim", type=int, default=32)
    ap.add_argument("--hid", type=int, default=384)
    ap.add_argument("--de", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lam-g", type=float, default=1.0)
    ap.add_argument("--lam-c", type=float, default=18.0)
    ap.add_argument("--lam-m", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="r7.npz")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Str, Ypool, te_s, Yf = load_raw()
    P = Ypool.shape[1]; L = len(Str[0]); _, n = Yf.shape
    eye = torch.eye(P, dtype=torch.bool, device=dev)
    Zpool = encode(Str, L)
    BAR = "=" * 100
    print(BAR); print("R7：竞争酶对驱动的读取"); print(BAR)
    if a.iso_val:
        trm, vam = iso_split(Zpool, a.n_val)
        print(f"  **隔离验证集**（按连通分量）：训练 {trm.sum():,} / 验证 {vam.sum():,}")
    else:
        rng = np.random.default_rng(0); perm = rng.permutation(len(Str))
        vam = np.zeros(len(Str), bool); vam[perm[:a.n_val]] = True; trm = ~vam
        print(f"  随机验证集：训练 {trm.sum():,} / 验证 {vam.sum():,}")
    d = (Zpool[vam][:, None, :] != Zpool[trm][None, :, :]).sum(2).min(1)
    print(f"  验证集到训练集的最小距离 {d.min()}，≤2 的占 {(d <= 2).mean():.1%}"
          f"（测试集为 0.0%，最小 3）")
    print(f"  酶对条件 {'关闭（消融，可证退回普通中心化）' if a.no_pair_cond else '开启'}"
          f"；无序酶对 {P*(P-1)//2} 个\n")

    tr_s, va_s = Str[trm], Str[vam]
    Ytr, Yva = Ypool[trm], Ypool[vam]
    Emb = None
    if a.protease_emb:
        z = np.load(a.protease_emb, allow_pickle=True)
        genes = [str(g) for g in z["genes"]]; order = sorted(genes)
        E = z["ESM-650M"][[genes.index(g) for g in order]].astype(np.float32)
        Emb = (E - E.mean(0)) / (E.std(0) + 1e-6)
    T = lambda x: torch.tensor(x).to(dev)
    Ztr, Zva, Zte = T(encode(tr_s, L)), T(encode(va_s, L)), T(encode(te_s, L))
    F = None; esm_dim = 0; Rtr = Rva = Rte = None
    if a.esm_feats:
        zf = np.load(a.esm_feats, allow_pickle=True)
        row = {str(q): i for i, q in enumerate(zf["seqs"])}
        arr = zf[f"L{a.esm_layer}"]; esm_dim = arr.shape[2]; F = torch.tensor(arr).to(dev)
        Rtr, Rva, Rte = (T(np.array([row[q] for q in x], np.int64))
                         for x in (tr_s, va_s, te_s))
    ytr, yva = T(Ytr), T(Yva)
    gtr = ytr.mean(1, keepdim=True); Ctr = ytr - gtr
    gva = yva.mean(1, keepdim=True); Cva = yva - gva
    Mtr, Mva = hard_margin(ytr, eye), hard_margin(yva, eye)
    Yv = Yva.T
    Mval = np.stack([Yv[t] - np.delete(Yv, t, 0).max(0) for t in range(P)])
    Mfull = np.stack([Yf[t] - np.delete(Yf, t, 0).max(0) for t in range(P)])
    C_true = Yf - Yf.mean(0)

    def rs_of(Pm, Mref):
        Mp = np.stack([Pm[t] - np.delete(Pm, t, 0).max(0) for t in range(P)])
        return float(np.mean([spearmanr(Mp[q], Mref[q]).statistic for q in range(P)]))

    store = {"_Yf": Yf}; rows = []
    for seed in a.seeds:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        net = R7Net(P, L, a.dim, a.pair_dim, a.hid, a.dropout, a.de,
                    not a.no_pair_cond, Emb, esm_dim).to(dev)
        npar = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=a.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

        def loss(Yh, gt, Ct, Mt):
            g = Yh.mean(1, keepdim=True); C = Yh - g
            return (a.lam_g * ((g - gt) ** 2).mean() + a.lam_c * ((C - Ct) ** 2).mean()
                    + a.lam_m * ((hard_margin(Yh, eye) - Mt) ** 2).mean())

        def fwd(Z, R, i, j):
            return net(Z[i:j], None if R is None else R[i:j], F)

        best, bs, bad, ep = 1e18, None, 0, 0
        for ep in range(a.epochs):
            net.train()
            ii = torch.randperm(len(tr_s), device=dev)
            for i in range(0, len(ii), a.batch):
                b = ii[i:i + a.batch]
                opt.zero_grad(set_to_none=True)
                o = net(Ztr[b], None if Rtr is None else Rtr[b], F)
                loss(o, gtr[b], Ctr[b], Mtr[b]).backward()
                opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():
                # 每条验证肽等权，使早停损失不随推理batch大小变化。
                v = sum(loss(fwd(Zva, Rva, i, i + 128), gva[i:i+128], Cva[i:i+128],
                             Mva[i:i+128]).item() * len(Zva[i:i+128])
                        for i in range(0, len(Zva), 128)) / len(Zva)
            if v < best - 1e-7:
                best, bad = v, 0
                bs = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(bs); net.eval()
        with torch.no_grad():
            Pm = np.concatenate([fwd(Zte, Rte, i, i + 128).cpu().numpy()
                                 for i in range(0, n, 128)]).T
            Pv = np.concatenate([fwd(Zva, Rva, i, i + 128).cpu().numpy()
                                 for i in range(0, len(va_s), 128)]).T
        rt, rv = rs_of(Pm, Mfull), rs_of(Pv, Mval)
        Ch = Pm - Pm.mean(0)
        rc = float(np.mean([pearsonr(Ch[q], C_true[q]).statistic for q in range(P)]))
        store[f"R7|{seed}"] = Pm; store[f"VAL@R7|{seed}"] = Pv
        rows.append(dict(seed=seed, val_rho_sM=rv, rho_sM=rt, rho_C=rc))
        print(f"  seed {seed}  [验证] ρ_s(M) {rv:.4f}   [测试] ρ_s(M) {rt:.4f}  ρ(C) {rc:.4f}"
              f"   {npar/1e6:.2f}M   {time.time()-t0:6.1f}s ep{ep+1}", flush=True)

    d_ = os.path.dirname(a.out)
    if d_:
        os.makedirs(d_, exist_ok=True)
    np.savez_compressed(a.out, **store)
    pd.DataFrame(rows).to_csv(a.out.replace(".npz", ".csv"), index=False)
    df = pd.DataFrame(rows)
    print(f"\n  均值：验证 {df.val_rho_sM.mean():.4f}   测试 {df.rho_sM.mean():.4f}")
    print(f"  已写入 {a.out}")


if __name__ == "__main__":
    main()
