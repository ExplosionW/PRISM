#!/usr/bin/env python
"""R8：活性匹配难负例的选择性度量学习（§100 的实现）。

**动机**：此前所有损失（Prop-7 边际 MSE、最强脱靶分类、批内边际 KL）拟合的都是
**数值**；从未做过「在目标活性相当的肽之间，强制区分谁在竞争酶上也容易被切」。
后者才是选择性的定义。

**难负例的定义**（§100.3，决定实验是否成立）：

    x⁺ : 目标活性达标 ∧ 边际大        （合格候选）
    x⁻ : 目标活性达标 ∧ **边际小**     （竞争酶也切它 —— 难负例）
    x⁰ : 活性不达标                    （**不可用**：低活性负例太容易，学不到竞争差异）

**不能教模型「目标酶不切 x⁻」**——x⁻ 在活性任务中必须保持高分。
监督对象是 **(肽 x, 目标 t, 竞争集合 Q)**，「非选择性」不是肽本身的永久属性：
同一条 x⁻ 换一个 Q 就可能变成合格正例。

**度量分支**（复用 R1 已算出的逐酶 FiLM 表示 h_p，不另起编码器）：

    z_p(x) = normalize(W_z h_p(x)),  v_p = normalize(W_v e_p),  r_p(x) = κ⟨z_p, v_p⟩
    η_{t,Q}(x) = r_t(x) − max_{o∈Q\\{t}} r_o(x)                  相对优势
    Ŷ(x) = Ŷ⁽⁰⁾(x) + α·H_P r(x),   α 零初始化                    真正参与输出

归一化 + 受控尺度 κ 防止模型靠增大范数满足间隔。α 零初始化使**初始预测**与 R1 相同，
但不保证训练后不退步——这是实验安排，不是保证。

**配对损失**：
    L = E[ w · softplus( (m − [η(x⁺) − η(x⁻)]) / T ) ]
m 是嵌入空间的训练间隔，**与实验 Δ_min 不同单位**，不可混用。
阈值一律取自训练侧，不用测试标签。
"""
import argparse, importlib.util, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("r7", os.path.join(HERE, "gpu_r7_pairwise.py"))
r7 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(r7)
B = r7.B
AA = r7.AA
encode = r7.encode


def build_pairs(Ytr, q_act=0.5, n_bin=5, top_frac=0.25, max_per_bin=400, seed=0,
                mode="matched"):
    """构造活性匹配的 (目标 t, x⁺, x⁻) 三元组。

    **只用训练标签**。对每个目标 t：
      1. 用训练侧分位数定活性门槛（默认中位数），只保留活性达标者——
         低活性肽被排除，因为「靠低活性识别负例」学不到竞争差异（§100.3）。
      2. 按目标活性分箱，**箱内**活性相当，故配对是活性匹配的。
      3. 箱内按真实边际排序，取高边际 top 与低边际 bottom 配对。

    mode:
      matched — 活性分箱内配对（活性匹配）
      clean   — **干净随机对照**：负例从同一活性合格池随机抽，但**核验其真实边际更低**。
                旧的 `--plain-contrast` 从全体训练肽均匀抽，实测 8.67% 的负例真实边际
                **高于**正例（标签反了）、49.79% 低于活性门槛，故它同时改变了
                活性分布、边际难度与标签正确性，不是干净对照（§102.4）。
                本模式只解除「同箱」这一条，其余保持一致。

    另记（§102.3 实测）：本函数的「正例」是**相对高边际**，不是合格选择性肽——
    62.20% 的正例真实边际仍 < 0（有竞争酶切得更强），
    同时满足活性中位数与正边际 90 分位的仅 4.34%。故这是有效的**相对排序对**。
    """
    n, P = Ytr.shape
    rng = np.random.default_rng(seed)
    Y = Ytr.T                                                  # (P, n)
    M = np.stack([Y[t] - np.delete(Y, t, 0).max(0) for t in range(P)])
    out = []
    for t in range(P):
        a = Y[t]; thr = np.quantile(a, q_act)
        idx = np.where(a >= thr)[0]
        if len(idx) < 4 * n_bin:
            continue
        edges = np.quantile(a[idx], np.linspace(0, 1, n_bin + 1))
        for b in range(n_bin):
            lo, hi = edges[b], edges[b + 1]
            sel = idx[(a[idx] >= lo) & (a[idx] <= hi)]
            if len(sel) < 8:
                continue
            order = sel[np.argsort(-M[t][sel])]                # 边际从高到低
            k = max(1, int(top_frac * len(order)))
            pos, neg = order[:k], order[-k:]
            m = min(len(pos), len(neg), max_per_bin)
            pi = rng.choice(pos, m, replace=False)
            ni = rng.choice(neg, m, replace=False)
            for p_, n_ in zip(pi, ni):
                out.append((t, int(p_), int(n_)))
    out = np.array(out, np.int64)
    if mode == "clean" and len(out):
        # 保持正例与配对数不变，负例改从**同一活性合格池**抽，并核验边际更低
        for t in range(P):
            sel = out[:, 0] == t
            if not sel.any():
                continue
            a = Y[t]; pool = np.where(a >= np.quantile(a, q_act))[0]
            pos = out[sel, 1]
            cand = rng.choice(pool, (len(pos), 12))
            worse = M[t][cand] < M[t][pos][:, None]
            first = np.argmax(worse, 1)
            ok = worse.any(1)
            neg = cand[np.arange(len(pos)), first]
            neg[~ok] = out[sel, 2][~ok]                   # 抽不到更差的就保留原负例
            out[sel, 2] = neg
    return out


_sl = importlib.util.spec_from_file_location(
    "selection", os.path.join(HERE, "selection.py"))
selection = importlib.util.module_from_spec(_sl); _sl.loader.exec_module(selection)

_rm = importlib.util.spec_from_file_location(
    "runmeta", os.path.join(HERE, "runmeta.py"))
runmeta = importlib.util.module_from_spec(_rm); _rm.loader.exec_module(runmeta)


def load_lm(path, dev):
    """载入自建 DeCleave-LM（掩码残基预训练，权重全部本项目自训）。"""
    _s = importlib.util.spec_from_file_location(
        "mlm", os.path.join(HERE, "gpu_mlm_pretrain.py"))
    mlm = importlib.util.module_from_spec(_s); _s.loader.exec_module(mlm)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    lm = mlm.DeCleaveLM(**ck["cfg"]); lm.load_state_dict(ck["state"])
    return lm.to(dev)


class R8Net(nn.Module):
    """R1 的 FiLM 主干 + 度量分支。h_p 复用，不另起编码器。"""

    def __init__(s, P, L=10, d=96, m=32, hid=384, pdrop=0.15, de=32,
                 Emb=None, esm_dim=0, zdim=64, kappa=10.0, use_metric=True,
                 lm=None, lm_freeze=False, film_pos_rank=0, cross_attn=False,
                 lm_residual=False, ladder="none", n_layer=0,
                 geom_mode="none", geom_dim=28,
                 cdr_mode="off", cdr_dim=32, cdr_eps=0.1):
        super().__init__()
        import itertools
        s.P, s.L, s.use_metric, s.kappa, s.d_e = P, L, use_metric, kappa, d
        s.pairs = list(itertools.combinations(range(L), 2))
        s.E = nn.Parameter(torch.randn(L, 21, d) * 0.05)
        # ===== §107 酶几何分支 =====
        # 几何编码器：口袋残基图上的消息传递，边特征只用残基间距离的 RBF 展开。
        # 输入（距离、夹角）对整体旋转/平移不变，已实测差为 0.000e+00。
        s.gmode = geom_mode
        if geom_mode != "none":
            s.gnode = nn.Sequential(nn.Linear(geom_dim, d), nn.LayerNorm(d), nn.ReLU())
            s.gmsg = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + 16, d), nn.ReLU(),
                                                  nn.Linear(d, d)) for _ in range(2)])
            s.gupd = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
                                    for _ in range(2)])
            # 无实验结构的酶：可学习回退向量占位，**不用 AF 填充**（§106.4）
            s.gfall = nn.Parameter(torch.randn(d) * 0.02)
            if geom_mode == "interact":
                # 肽残基 × 酶亚位点交叉注意力。该项**依赖肽**，
                # 故不可被最终 Linear(h,18) 吸收——§89 的可吸收性只针对
                # 固定 e_p 的线性项。零初始化门控，开局等价于无此分支。
                s.gq = nn.Linear(d, d); s.gk = nn.Linear(d, d); s.gv = nn.Linear(d, d)
                s.gout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
                s.beta_g = nn.Parameter(torch.zeros(1))
            else:
                s.gemb = nn.Sequential(nn.Linear(d, de), nn.LayerNorm(de))
        # ===== §133.4 竞争差分残差（candidate-dependent competitive differential） =====
        # d_po(s) = h_p(s) - h_o(s)；z_p = Σ_{o≠p} w_po(s)·φ(d_po, e_p-e_o)；
        # ŷ_p = ŷ_base_p + η(r_p - mean_q r_q)，η 零初始化。
        #
        # 三个模式**参数量与额外前向开销完全相同**，只差喂进 φ/w 的是什么：
        #   diff    —— 本方案：差分 (h_p-h_o, e_p-e_o)
        #   plain   —— 同参数对照：竞争酶自身表示 (h_o, e_o)，无差分结构
        #   uniform —— 机制对照：用差分但 w 固定均匀，隔离「候选依赖的权重」
        # 不变性（可检查的构造，非统计独立性声明）：给全部 h_p 加同一向量时，
        # d 不变；竞争风险取 R_o-R_p 亦不变（rout 为线性）；故 diff/uniform 分支不变。
        # plain 不具此性质——这正是它作为对照的意义。
        s.cdr, s.cdr_eps = cdr_mode, cdr_eps
        if cdr_mode != "off":
            H_ = hid // 2
            s.cdr_phi = nn.Sequential(nn.Linear(H_ + de, cdr_dim), nn.ReLU())
            s.cdr_w = nn.Linear(H_ + de + 1, 1)      # +1 = 平移不变的竞争风险 R_o-R_p
            s.cdr_psi = nn.Linear(cdr_dim, 1)
            s.eta_cdr = nn.Parameter(torch.zeros(1))
            s.register_buffer("cdr_eye", torch.eye(P))
        s.ladder = ladder
        if ladder != "none":
            nL = n_layer
            # 每一级一个独立投影（冻结特征本身不训练，投影是我们自己的）
            s.lproj = nn.ModuleList([nn.Sequential(nn.Linear(esm_dim, d), nn.LayerNorm(d))
                                     for _ in range(nL)])
            if ladder == "linear":
                # 对照 b：多层**线性融合**，一次性压到 d，之后与单层完全相同
                s.lmix = nn.Linear(nL * d, d)
            else:
                # 对照 c（side）与本方案（cond）：逐级更新一个任务状态。
                # 二者结构完全相同，**唯一差别**是每一级是否做酶条件调制——
                # 这正是要单独隔离的创新点。
                s.lupd = nn.ModuleList([nn.GRUCell(d, d) for _ in range(nL)])
                if ladder == "cond":
                    s.lfilm = nn.ModuleList([nn.Sequential(
                        nn.Linear(de, 64), nn.ReLU(), nn.Linear(64, 2 * d))
                        for _ in range(nL)])
        s.esm = esm_dim > 0 and ladder == "none"
        if s.esm:
            s.eproj = nn.Sequential(nn.Linear(esm_dim, d), nn.LayerNorm(d))
            s.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
        # 自建 DeCleave-LM 作为残基表示来源（§100 的原意；R8 首轮误用了冻结 ESM）
        s.lm, s.lm_freeze, s.lm_residual = lm, lm_freeze, lm_residual
        if lm is not None:
            s.lmproj = nn.Sequential(nn.Linear(lm.d, d), nn.LayerNorm(d))
            s.lmfuse = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
            s.alpha_lm = nn.Parameter(torch.zeros(1))
            if lm_freeze:
                for q in lm.parameters():
                    q.requires_grad_(False)
        s.Wp = nn.Parameter(torch.randn(len(s.pairs), 2 * d, m) * (1.0 / (2 * d) ** .5))
        s.bp = nn.Parameter(torch.zeros(len(s.pairs), m))
        s.z = nn.Parameter(torch.zeros(len(s.pairs)))
        din = L * d + len(s.pairs) * m
        s.trunk = nn.Sequential(nn.Linear(din, hid), nn.ReLU(), nn.Dropout(pdrop),
                                nn.Linear(hid, hid // 2), nn.ReLU())
        s.a = nn.Parameter(torch.zeros(P))
        s.g = nn.Linear(hid // 2, 1)
        s.rout = nn.Linear(hid // 2, 1)
        if Emb is None:
            s.ep = nn.Parameter(torch.randn(P, de) * 0.05); s.Eproj = None
        else:
            s.register_buffer("Ep_raw", torch.tensor(Emb))
            s.Eproj = nn.Sequential(nn.Linear(Emb.shape[1], de), nn.LayerNorm(de))
        s.film = nn.Sequential(nn.Linear(de, 64), nn.ReLU(), nn.Linear(64, 2 * d))
        # 位点特异 FiLM（§103.13）：原 FiLM 的 γ、β 形状是 (P, d)，广播到全部 L 个位点，
        # 即**同一个酶对每个位点施加相同调制**。但 MMP 之间的差异集中在特定亚位点
        # （S1′、S3），不是均匀的。这里加一个低秩的位点特异修正：
        #   γ_{p,l} = γ_p + c^γ_{p,l} B_γ,   c ∈ R^{P×L×r}, B ∈ R^{r×d}
        # 参数量 de·L·2r + 2rd（r=8、L=10、d=96 时约 6.8K，可忽略）。
        # **零初始化**：训练开始时严格等价于原 FiLM，只能在其上做修正。
        # rank=0 时不创建任何参数，RNG 流与旧版逐位一致（保证与历史结果可比）。
        s.fp_r = film_pos_rank
        # 跨酶注意力（§103.13）：主干对 18 个酶各自独立前向，酶之间的竞争只经由
        # 损失（hard_margin）与中心化进入。但 ρ_s(M) 度量的边际 M_ts = Y_ts − max_{o≠t} Y_os
        # 本身就是竞争量。这里让 P 个酶表示在读出前互相看一眼。
        # 注意这**不可**被最终 Linear(h,18) 吸收：它是逐样本、跨酶的非线性交互，
        # 而 §89 指出可吸收的是 Uq(h)·Uk(e_p)ᵀ 这类固定 e_p 的线性项。
        # 零初始化门控 beta_x：开局严格等价于无此模块。
        s.xattn = cross_attn
        if cross_attn:
            s.xln = nn.LayerNorm(hid // 2)
            s.xmha = nn.MultiheadAttention(hid // 2, 4, batch_first=True)
            s.beta_x = nn.Parameter(torch.zeros(1))
        if s.fp_r > 0:
            s.filmP = nn.Linear(de, L * 2 * s.fp_r)
            nn.init.zeros_(s.filmP.weight); nn.init.zeros_(s.filmP.bias)
            s.basisG = nn.Parameter(torch.randn(s.fp_r, d) * 0.02)
            s.basisB = nn.Parameter(torch.randn(s.fp_r, d) * 0.02)
        if use_metric:
            s.Wz = nn.Linear(hid // 2, zdim)
            s.Wv = nn.Linear(de, zdim)
            s.alpha = nn.Parameter(torch.zeros(1))   # 零初始化：初始预测等于 R1

    def geom_sites(s):
        """(P,S,d) 亚位点表示。RBF 边、掩码求和；无结构的酶整体换成回退向量。"""
        GF, GD, GM, GH = s.geo
        h = s.gnode(GF)
        c = torch.linspace(2., 20., 16, device=GF.device)
        rbf = torch.exp(-((GD[..., None] - c) ** 2) / 4.)          # (P,S,S,16)
        m = GM[:, None, :, None]
        for msg, upd in zip(s.gmsg, s.gupd):
            n_ = h.shape[1]
            pair = torch.cat([h[:, :, None].expand(-1, -1, n_, -1),
                              h[:, None].expand(-1, n_, -1, -1), rbf], -1)
            agg = (msg(pair) * m).sum(2) / m.sum(2).clamp_min(1.)
            h = h + upd(torch.cat([h, agg], -1))
        return torch.where(GH[:, None, None] > .5, h, s.gfall[None, None])

    def geom_ep(s):
        S_ = s.geom_sites(); m = s.geo[2][..., None]
        return s.gemb((S_ * m).sum(1) / m.sum(1).clamp_min(1.))

    def ep_vec(s):
        base = s.ep if s.Eproj is None else s.Eproj(s.Ep_raw)
        if s.gmode == "embed":
            # 只在**有实验结构**的酶上用几何来源的 e_p；缺口酶保留各自可学的
            # e_p，而不是让 7 个酶共用同一个回退向量——后者会让它们彼此
            # 不可区分，把整个模型拖垮，使「几何是否可迁移」无法单独归因。
            gp = s.geom_ep()
            h = (s.geo[3] > .5).float()[:, None]
            return h * gp + (1 - h) * base
        return base

    def body(s, e):
        i_, j_ = zip(*s.pairs)
        cat = torch.cat([e[:, list(i_)], e[:, list(j_)]], -1)
        u = torch.relu(torch.einsum("bpk,pkm->bpm", cat, s.Wp) + s.bp)
        gg = torch.sigmoid(s.z)[None, :, None]
        return s.trunk(torch.cat([e.flatten(1), (u * gg).flatten(1)], 1))

    def forward(s, Z, Zr=None, F=None, ret_r=False):
        Bn = Z.shape[0]
        e = torch.stack([s.E[i][Z[:, i]] for i in range(s.L)], 1)
        if s.esm:
            e = s.fuse(torch.cat([e, s.eproj(F[Zr].float())], -1))
        if s.lm is not None:
            # 冻结时只挡 LM 自身的梯度，投影层始终可训练（§90.1 修过的同类 bug）
            if s.lm_freeze:
                was = s.lm.training; s.lm.eval()
                with torch.no_grad():
                    zl = s.lm.encode(Z)
                if was:
                    s.lm.train()
            else:
                zl = s.lm.encode(Z)
            if s.lm_residual:
                # §104.6 方向二：**保留**已有表示，只加一条零初始化的残差修正。
                # 现用的 concat+lmfuse+LayerNorm 会把已有表示重新映射，
                # 组合变差可能来自表示被覆盖而非信息重叠——两者此前没被分开。
                e = e + s.alpha_lm * s.lmproj(zl)
            else:
                e = s.lmfuse(torch.cat([e, s.lmproj(zl)], -1))
        if s.ladder == "linear":
            xs = [s.lproj[k](F[Zr][:, k].float()) for k in range(len(s.lproj))]
            e = e + s.lmix(torch.cat(xs, -1))
        h0 = s.body(e); gg = s.g(h0)
        ev = s.ep_vec()
        if s.ladder in ("side", "cond"):
            # 逐级递进读取：状态 (B,P,L,d) 沿冻结 PLM 的层向上更新。
            # side —— 不做逐级酶条件（通用侧网络，LST 式）；
            # cond —— 每一级用 e_p 调制该级读入，酶特异性沿层累积。
            st = e[:, None].expand(-1, s.P, -1, -1).reshape(-1, s.d_e)
            for k in range(len(s.lproj)):
                x = s.lproj[k](F[Zr][:, k].float())              # (B,L,d)
                if s.ladder == "cond":
                    gl, bl = s.lfilm[k](s.ep_vec()).chunk(2, -1)  # (P,d)
                    x = x[:, None] * (1 + gl[None, :, None]) + bl[None, :, None]
                else:
                    x = x[:, None].expand(-1, s.P, -1, -1)
                st = s.lupd[k](x.reshape(-1, s.d_e), st)
            hp = s.body(st.reshape(Bn * s.P, s.L, -1)).reshape(Bn, s.P, -1)
            R = s.rout(hp).squeeze(-1)
            r = None
            if s.use_metric:
                z = nn.functional.normalize(s.Wz(hp), dim=-1)
                v = nn.functional.normalize(s.Wv(ev), dim=-1)
                r = s.kappa * (z * v[None]).sum(-1)
                R = R + s.alpha * (r - r.mean(1, keepdim=True))
            Y = s.a[None, :] + gg + (R - R.mean(1, keepdim=True))
            return (Y, r) if ret_r else Y
        gam, bet = s.film(ev).chunk(2, -1)
        if s.fp_r > 0:
            c = s.filmP(ev).view(s.P, s.L, 2, s.fp_r)
            gm = gam[:, None] + c[:, :, 0] @ s.basisG          # (P, L, d)
            bt = bet[:, None] + c[:, :, 1] @ s.basisB
            em = e[:, None] * (1 + gm[None]) + bt[None]
        else:
            em = e[:, None] * (1 + gam[None, :, None]) + bet[None, :, None]
        hp = s.body(em.reshape(Bn * s.P, s.L, -1)).reshape(Bn, s.P, -1)   # (B,P,hid/2)
        if s.xattn:
            q = s.xln(hp)
            hp = hp + s.beta_x * s.xmha(q, q, q, need_weights=False)[0]
        R = s.rout(hp).squeeze(-1)
        if s.cdr != "off":
            Bn_, P_ = R.shape
            ev_ = s.ep_vec()                                          # (P,de)
            if s.cdr == "plain":
                dd = hp[:, None].expand(-1, P_, -1, -1)               # [p,o] -> h_o
                de_ = ev_[None].expand(P_, -1, -1)                    # [p,o] -> e_o
            else:
                dd = hp[:, :, None] - hp[:, None]                     # [p,o] -> h_p - h_o
                de_ = ev_[:, None] - ev_[None]                        # [p,o] -> e_p - e_o
            x = torch.cat([dd, de_[None].expand(Bn_, -1, -1, -1)], -1)
            phi = s.cdr_phi(x)                                        # (B,P,P,m)
            off_ = (1 - s.cdr_eye)[None, :, :, None]                  # 屏蔽 o=p
            if s.cdr == "uniform":
                w = off_ / (P_ - 1)
            else:
                # 竞争风险 R_o - R_p：对 h_p 的共同平移不变
                risk = (R[:, None, :] - R[:, :, None]).unsqueeze(-1)
                lg = s.cdr_w(torch.cat([x, risk], -1)).squeeze(-1)    # (B,P,P)
                lg = lg.masked_fill(s.cdr_eye[None] > .5, -1e9)
                # 保留非零均匀份额：起点漏掉一个危险竞争酶后不会永远看不到它
                w = ((1 - s.cdr_eps) * lg.softmax(-1)
                     + s.cdr_eps / (P_ - 1)).unsqueeze(-1) * off_
            zc = (w * phi).sum(2)                                     # (B,P,m)
            rc = s.cdr_psi(zc).squeeze(-1)                            # (B,P)
            R = R + s.eta_cdr * (rc - rc.mean(1, keepdim=True))
        if s.gmode == "interact":
            S_ = s.geom_sites()                                           # (P,S,d)
            q = s.gq(e); k = s.gk(S_); v = s.gv(S_)
            att = torch.einsum("bld,psd->bpls", q, k) / (q.shape[-1] ** .5)
            att = att.masked_fill(s.geo[2][None, :, None] < .5, -1e9).softmax(-1)
            ctx = torch.einsum("bpls,psd->bpld", att, v)
            R = R + s.beta_g * s.gout(ctx).squeeze(-1).mean(-1)
        r = None
        if s.use_metric:
            z = nn.functional.normalize(s.Wz(hp), dim=-1)                 # (B,P,zdim)
            v = nn.functional.normalize(s.Wv(ev), dim=-1)                 # (P,zdim)
            r = s.kappa * (z * v[None]).sum(-1)                           # (B,P)
            R = R + s.alpha * (r - r.mean(1, keepdim=True))               # H_P r
        Y = s.a[None, :] + gg + (R - R.mean(1, keepdim=True))
        return (Y, r) if ret_r else Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-metric", action="store_true", help="关闭度量分支（= R1 基线）")
    ap.add_argument("--lm-init", default=None,
                    help="自建 DeCleave-LM checkpoint。**R8 首轮从未用过它**（当时传的是 "
                         "--esm-feats，即冻结 ESM 特征），故 §100「自建 LM 接受选择性监督」"
                         "并未被检验（§102.2）。")
    ap.add_argument("--lm-freeze", action="store_true", help="冻结自建 LM，只更新其上的投影")
    ap.add_argument("--pair-mode", default="matched", choices=["matched", "clean"],
                    help="clean = 干净随机对照：同一活性合格池 + 核验边际更低")
    # ---- §104.6 方向一：把训练目标对准最终选样 ----
    ap.add_argument("--dfl", default="none", choices=["none", "final", "list"],
                    help="决策对齐的训练目标。none=沿用作用于辅助 r 分支的度量损失；"
                         "final=同样的成对排序但 η 从**最终预测** a+g+R+αr 计算；"
                         "list=在最终 profile 上做活性条件的软列表（ListNet）代理。"
                         "§104.4 指出 r 与最终评分不是同一个量，这里把它们分开测")
    ap.add_argument("--lam-dfl", type=float, default=0.5)
    ap.add_argument("--dfl-temp", type=float, default=1.0, help="列表 softmax 温度（只在开发折选）")
    ap.add_argument("--dfl-beta", type=float, default=1.0, help="列表评分里风险项的权重")
    ap.add_argument("--dfl-soft", type=float, default=0.25,
                    help="软标签的平滑尺度；越大边界附近越不确定（保留不确定度，不做硬阈值）")
    ap.add_argument("--n-list", type=int, default=512, help="每步列表代理采样的肽数")
    ap.add_argument("--select-eval", action="store_true",
                    help="训练后跑终点选样评估（§104.7）：usable、真实活性保持率、"
                         "逐靶点失败格，两种池规模迁移口径同时报告")
    ap.add_argument("--select-sweep", action="store_true",
                    help="额外报告 27 组 (K, 保持率门槛, risk) 协议敏感性")
    ap.add_argument("--ckpt-dir", default=None,
                    help="保存 checkpoint、完整配置、代码 hash、逐 epoch 指标与数据 ID（§104.7）")
    # ---- §104.6 方向二：保留原能力的双通路融合 ----
    # ---- §106/§107：PDB 实验结构的酶几何分支 ----
    # ---- §133.4 竞争差分残差 ----
    # ---- §133.5 约束训练 ----
    ap.add_argument("--act-constraint", action="store_true",
                    help="逐目标活性违约乘子：c_p 取起点模型的逐目标训练 MSE，"
                         "违约则对偶上升，满足则归零——任一目标的活性亏损不能被"
                         "其他目标的改善抵销")
    ap.add_argument("--dual-lr", type=float, default=0.05)
    ap.add_argument("--sel-focus", action="store_true",
                    help="选择聚焦代理：训练标签定义的严格 usable vs 竞争失败成对间隔，"
                         "负例取预测活性最高者（最可能被算法排进前列）")
    ap.add_argument("--lam-sel", type=float, default=0.5)
    ap.add_argument("--n-sel", type=int, default=512)
    ap.add_argument("--cdr-mode", default="off",
                    choices=["off", "diff", "plain", "uniform"],
                    help="diff=候选依赖的竞争差分残差（本方案）；"
                         "plain=同参数同开销对照，喂竞争酶自身表示而非差分；"
                         "uniform=用差分但权重固定均匀，隔离候选依赖权重的作用")
    ap.add_argument("--cdr-dim", type=int, default=32, help="φ 的输出宽度 m")
    ap.add_argument("--cdr-eps", type=float, default=0.1,
                    help="w 中保留的非零均匀份额")
    ap.add_argument("--lopo", type=int, default=-1,
                    help="留一酶：该酶的列从 g/C/边际三项损失中全部剔除，"
                         "训练完全不见其标签；测试时须从描述子推断它。"
                         "这是酶侧结构**唯一**具有函数空间价值的场景——"
                         "完整 panel 下可学 e_p 已能拟合任意逐酶调制（§105 的讨论）。"
                         "注：该酶的偏置 a_t 未训练，但 ρ_s(M) 对常数平移不变，故不受影响；"
                         "usable 依赖绝对活性，LOPO 下不作主指标")
    ap.add_argument("--geom", default=None,
                    help="mmp_geom.npz：实验 PDB 提取的口袋不变几何")
    ap.add_argument("--geom-mode", default="none",
                    choices=["none", "embed", "interact"],
                    help="embed=几何汇成每酶一个向量当 e_p（函数上等价于固定描述子）；"
                         "interact=肽残基×酶亚位点交叉注意力（依赖肽，不可被吸收）")
    ap.add_argument("--geom-ident-only", action="store_true",
                    help="只保留残基身份 one-hot，几何列置零——"
                         "§106.5 第 2 步的「只给同一组口袋残基身份」对照")
    ap.add_argument("--geom-shuffle", action="store_true",
                    help="打乱酶↔结构对应关系的内容对照")
    # ---- §104.6 方向三：冻结 PLM 多层特征的酶条件递进读取 ----
    ap.add_argument("--ladder", default="none",
                    choices=["none", "linear", "side", "cond"],
                    help="none=只读单层（现状）；linear=多层线性融合（对照 b）；"
                         "side=多层通用侧网络，不做逐级酶条件（对照 c）；"
                         "cond=多层**酶条件**递进读取（本方案）。"
                         "side 与 cond 结构完全相同，唯一差别是逐级是否调制")
    ap.add_argument("--esm-layers", type=int, nargs="+", default=[0, 20, 33],
                    help="参与递进读取的冻结层")
    ap.add_argument("--lm-residual", action="store_true",
                    help="自建 LM 以**零初始化残差**接入（e + α·lmproj(z)），"
                         "而非 concat+重映射。α 起始为 0，开局严格等价于无 LM")
    ap.add_argument("--init-from", default=None,
                    help="从同一 base checkpoint 初始化（路径中的 SEED 会被替换为 seed 号）")
    ap.add_argument("--freeze-base", action="store_true",
                    help="只训练新增模块，base checkpoint 已有张量全部冻结")
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="先冻结 base 训练新增模块 N 轮，之后解冻并重建优化器")
    ap.add_argument("--grad-probe", type=int, default=0,
                    help="每 N 个 epoch 在共享层上记录各损失项的梯度方向与大小"
                         "（§104.6：先测梯度冲突，不能先宣布冲突是原因）")
    ap.add_argument("--cross-attn", action="store_true",
                    help="读出前让 18 个酶表示互相注意（零初始化门控）。"
                         "边际是竞争量，但主干此前对各酶独立前向")
    ap.add_argument("--film-pos-rank", type=int, default=0,
                    help="位点特异 FiLM 的秩 r（0 = 关闭，行为与旧版逐位一致）。"
                         "让每个酶对不同位点施加不同调制，对应亚位点特异性")
    ap.add_argument("--neg-fresh", action="store_true",
                    help="每步重采负例（仍在活性合格池内、且核验边际更低）。"
                         "分离出「负例多样性」这一机制：旧 --plain-contrast 同时"
                         "带来了多样性与标签噪声，无法区分（§103.9）")
    ap.add_argument("--neg-noise", type=float, default=0.0,
                    help="负例污染比例 ρ：该比例的负例改为全库均匀抽（不核验），"
                         "即引入标签噪声。ρ=0 全干净，ρ=1 等价旧 --plain-contrast")
    ap.add_argument("--plain-contrast", action="store_true",
                    help="对照：不做活性匹配，负例从全体随机取（§100.8 要求的普通对比臂）")
    ap.add_argument("--lam-c", type=float, default=18.0)
    ap.add_argument("--lam-g", type=float, default=1.0)
    ap.add_argument("--lam-m", type=float, default=2.0)
    ap.add_argument("--lam-metric", type=float, default=0.5)
    ap.add_argument("--margin-m", type=float, default=0.5, help="嵌入空间训练间隔（非 Δ_min）")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--zdim", type=int, default=64)
    ap.add_argument("--esm-feats", default=None)
    ap.add_argument("--esm-layer", type=int, default=33)
    ap.add_argument("--protease-emb", default=None)
    ap.add_argument("--pemb-key", default="ESM-650M")
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--pair-dim", type=int, default=32)
    ap.add_argument("--hid", type=int, default=384)
    ap.add_argument("--de", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-pair", type=int, default=256, help="每步采样的配对数")
    ap.add_argument("--out", default="r8.npz")
    a = ap.parse_args()
    if a.lopo >= 0:
        raise RuntimeError("LOPO disabled after the 2026-09-10 audit: pair construction, "
                           "competitor margins and checkpoint provenance are not label-isolated. "
                           "Use the full-panel protocol; do not interpret legacy LOPO as generalization.")
    dev = torch.device(a.device)
    Str, Ypool, te_s, Yf = r7.load_raw()
    P = Ypool.shape[1]; L = len(Str[0]); _, n = Yf.shape
    eye = torch.eye(P, dtype=torch.bool, device=dev)
    trm, vam = r7.iso_split(encode(Str, L), 1200)
    tr_s, va_s = Str[trm], Str[vam]; Ytr, Yva = Ypool[trm], Ypool[vam]
    BAR = "=" * 100
    print(BAR); print("R8：活性匹配难负例的选择性度量学习"); print(BAR)
    print(f"  隔离验证集：训练 {len(tr_s):,} / 验证 {len(va_s):,}")

    trip = (build_pairs(Ytr, mode=a.pair_mode) if not a.no_metric
            else np.zeros((0, 3), np.int64))
    NEG = None
    if len(trip) and (a.neg_fresh or a.neg_noise > 0):
        # 每目标的活性合格池（与 build_pairs 同门槛），padded 成 (P, maxlen)
        Yt0 = Ytr.T
        M0 = np.stack([Yt0[t] - np.delete(Yt0, t, 0).max(0) for t in range(P)])
        pools = [np.where(Yt0[t] >= np.quantile(Yt0[t], 0.5))[0] for t in range(P)]
        mx = max(len(q) for q in pools)
        pad = np.zeros((P, mx), np.int64)
        for t, q in enumerate(pools):
            pad[t, :len(q)] = q; pad[t, len(q):] = q[0]
        NEG = (torch.tensor(pad, device=dev),
               torch.tensor([len(q) for q in pools], device=dev),
               torch.tensor(M0, dtype=torch.float32, device=dev))
        print(f"  负例重采：活性合格池每酶 {np.mean([len(q) for q in pools]):.0f} 条"
              f"（最小 {min(len(q) for q in pools)}）"
              f"   ρ={a.neg_noise:g}   fresh={a.neg_fresh}")
    if len(trip):
        Yt_ = Ytr.T
        Mt_ = np.stack([Yt_[t] - np.delete(Yt_, t, 0).max(0) for t in range(P)])
        # 曾用 trip[:2000]，但配对按目标顺序生成，前 2000 条只覆盖目标 0、1，
        # 报出的活性差 0.237 不代表全体（全体 0.216，§103.1）。改为全量。
        t_, p_, q_ = trip[:, 0], trip[:, 1], trip[:, 2]
        dm = float(np.mean(Mt_[t_, p_] - Mt_[t_, q_]))
        da = float(np.mean(np.abs(Yt_[t_, p_] - Yt_[t_, q_])))
        print(f"  活性匹配配对：{len(trip):,} 组   配对内目标活性差均值 {da:.3f}"
              f"（越小越匹配）   真实边际差均值 {dm:.3f}（越大越是难负例）")
    print(f"  度量分支 {'关闭（=R1 基线）' if a.no_metric else '开启'}"
          f"{'（λ=0，只留结构不加对比损失）' if not a.no_metric and a.lam_metric == 0 else ''}"
          f"；采样 {a.pair_mode}"
          f"{'；负例随机（旧对照，含 8.6% 标签反转）' if a.plain_contrast else ''}"
          f"；主干 {'自建 DeCleave-LM' if a.lm_init else ('冻结 ESM 特征' if a.esm_feats else '纯自建嵌入')}\n")

    GF = None
    if a.geom:
        zg = np.load(a.geom, allow_pickle=True)
        ggenes = [str(x) for x in zg["genes"]]
        assert ggenes == sorted(ggenes), "几何文件的酶顺序必须是字典序（与 Y.npy 一致）"
        GF = zg["feat"].astype(np.float32).copy()
        GD = zg["dist"].astype(np.float32).copy()
        GM = zg["mask"].astype(np.float32).copy()
        GH = zg["has_struct"].astype(np.float32).copy()
        if a.geom_ident_only:
            GF[:, :, 21:] = 0.; GD[:] = 0.       # 只留身份 one-hot
        if a.geom_shuffle:
            pr = np.random.default_rng(12345).permutation(len(GF))
            GF, GD, GM, GH = GF[pr], GD[pr], GM[pr], GH[pr]
        print(f"  酶几何：{a.geom_mode}"
              f"{'（仅残基身份）' if a.geom_ident_only else ''}"
              f"{'（打乱对应）' if a.geom_shuffle else ''}"
              f"   有实验结构 {int(GH.sum())}/{len(GH)}"
              f"   缺口以可学习回退向量占位：{[g for g,h in zip(ggenes,GH) if h<.5]}")
    Emb, genes, order = None, None, None
    if a.protease_emb:
        z = np.load(a.protease_emb, allow_pickle=True)
        genes = [str(g) for g in z["genes"]]; order = sorted(genes)
        E = z[a.pemb_key][[genes.index(g) for g in order]].astype(np.float32)
        Emb = (E - E.mean(0)) / (E.std(0) + 1e-6)
    T = lambda x: torch.tensor(x).to(dev)
    KEEP = None
    if a.lopo >= 0:
        KEEP = torch.tensor([q for q in range(P) if q != a.lopo], device=dev)
        print(f"  留一酶：索引 {a.lopo}（字典序第 {a.lopo+1} 个）"
              f"，其标签完全不参与训练；g/C/边际均在剩余 {P-1} 酶上计算")
    Ztr, Zva, Zte = T(encode(tr_s, L)), T(encode(va_s, L)), T(encode(te_s, L))
    F = None; esm_dim = 0; Rtr = Rva = Rte = None
    if a.esm_feats:
        zf = np.load(a.esm_feats, allow_pickle=True)
        row = {str(q): i for i, q in enumerate(zf["seqs"])}
        if a.ladder == "none":
            arr = zf[f"L{a.esm_layer}"]; esm_dim = arr.shape[2]
            F = torch.tensor(arr).to(dev)
        else:
            # §104.6 方向三：读**多层**冻结特征。R7/R8 此前只读第 33 层，
            # 早期层只做过线性探针，从未进入任务状态。
            F = torch.stack([torch.tensor(zf[f"L{ly}"]) for ly in a.esm_layers], 1).to(dev)
            esm_dim = F.shape[-1]                       # (N, n_layer, L, 1280)
            print(f"  多层冻结特征：层 {a.esm_layers}，"
                  f"{F.shape[1]}×{esm_dim} 维，{F.numel()*2/2**30:.2f} GB（不参与训练）")
        Rtr, Rva, Rte = (T(np.array([row[q] for q in x], np.int64))
                         for x in (tr_s, va_s, te_s))
    ytr, yva = T(Ytr), T(Yva)
    if KEEP is not None:
        gtr = ytr[:, KEEP].mean(1, keepdim=True); Ctr = ytr - gtr
        gva = yva[:, KEEP].mean(1, keepdim=True); Cva = yva - gva
    else:
        gtr = ytr.mean(1, keepdim=True); Ctr = ytr - gtr
        gva = yva.mean(1, keepdim=True); Cva = yva - gva
    Mtr, Mva = r7.hard_margin(ytr, eye), r7.hard_margin(yva, eye)
    Yv = Yva.T
    Mval = np.stack([Yv[t] - np.delete(Yv, t, 0).max(0) for t in range(P)])
    Mfull = np.stack([Yf[t] - np.delete(Yf, t, 0).max(0) for t in range(P)])
    C_true = Yf - Yf.mean(0)
    TRP = T(trip) if len(trip) else None

    def rs_of(Pm, Mref):
        Mp = np.stack([Pm[t] - np.delete(Pm, t, 0).max(0) for t in range(P)])
        return float(np.mean([spearmanr(Mp[q], Mref[q]).statistic for q in range(P)]))

    store = {"_Yf": Yf}; rows = []
    sel_rows, sel_detail, sel_sweep = [], [], []
    for seed in a.seeds:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        net = R8Net(P, L, a.dim, a.pair_dim, a.hid, a.dropout, a.de,
                    Emb, esm_dim, a.zdim, a.kappa, not a.no_metric,
                    load_lm(a.lm_init, dev) if a.lm_init else None, a.lm_freeze,
                    a.film_pos_rank, a.cross_attn, a.lm_residual,
                    a.ladder, len(a.esm_layers),
                    a.geom_mode, GF.shape[-1] if GF is not None else 28,
                    a.cdr_mode, a.cdr_dim, a.cdr_eps).to(dev)
        _ok = {}
        if a.init_from:
            # §104.6 方向二：加载**同一** base checkpoint，新增模块零初始化接入，
            # 使「加模块」不再等于「重训一个不同的模型」。
            _ck = torch.load(a.init_from.replace("SEED", str(seed)),
                             map_location="cpu", weights_only=False)["state"]
            _sd = net.state_dict()
            _ok = {k: v for k, v in _ck.items()
                   if k in _sd and _sd[k].shape == v.shape}
            net.load_state_dict({**_sd, **_ok})
            print(f"    从 {os.path.basename(a.init_from.replace('SEED', str(seed)))} "
                  f"载入 {len(_ok)}/{len(_sd)} 个张量"
                  f"（未匹配 {sorted(set(_sd) - set(_ok))[:4]}…）", flush=True)
        if a.freeze_base or a.freeze_epochs:
            # 只训练新增模块：base checkpoint 里已有的张量全部冻结。
            # --freeze-epochs N 则在第 N 轮后解冻（§104.6「先只训练新增模块，
            # 再按验证结果开放自建 LM」）。
            for k, q in net.named_parameters():
                q.requires_grad_(k not in _ok)
        if a.geom_mode != "none":
            net.geo = (T(GF), T(GD), T(GM), T(GH))
        npar = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad],
                                a.lr, weight_decay=a.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        gg_ = torch.Generator(device=dev); gg_.manual_seed(seed)
        USABLE_TR = CFAIL_TR = None
        if a.sel_focus:
            # 三类只用**训练**标签，阈值与 selection.py 同源（同一份 thresholds()）：
            #   活性合格   y ≥ A_min
            #   严格 usable  活性合格 且 真实边际 ≥ Δ_min
            #   竞争失败   活性合格 且 真实边际 ≤ 0
            _A, _D = selection.thresholds(Ytr.T)
            _M = selection.margins(Ytr.T)
            _qual = Ytr.T >= _A[:, None]
            USABLE_TR = torch.tensor(_qual & (_M >= _D[:, None]), device=dev)
            CFAIL_TR = torch.tensor(_qual & (_M <= 0), device=dev)
            print("    选择聚焦标签（仅训练集）：活性合格 %d，严格 usable %d，竞争失败 %d"
                  % (int(_qual.sum()), int(USABLE_TR.sum()), int(CFAIL_TR.sum())), flush=True)
        if a.dfl == "list":
            # 软 usable 权重，只用训练标签；阈值与 selection.py 同源
            _A, _D = selection.thresholds(Ytr.T)
            _M = selection.margins(Ytr.T)
            _q = (1 / (1 + np.exp(-(Ytr.T - _A[:, None]) / a.dfl_soft))) * \
                 (1 / (1 + np.exp(-(_M - _D[:, None]) / a.dfl_soft)))
            QSOFT = torch.tensor(_q, dtype=torch.float32, device=dev)   # (P, n_train)
        if NEG is not None:
            POOL, PLEN, MG = NEG

        def base_loss(Y, gt, Ct, Mt):
            if a.lopo >= 0:
                # 留一酶：把该列从 g、C、边际三项里全部剔除。
                # g 在剩余 17 酶上重算，故留出酶的标签一次也没进过损失。
                k_ = KEEP
                Yk = Y[:, k_]
                g = Yk.mean(1, keepdim=True); C = Yk - g
                gt_ = gt if gt.shape[1] == 1 else gt[:, k_]
                return (a.lam_g * ((g - gt_) ** 2).mean()
                        + a.lam_c * ((C - Ct[:, k_]) ** 2).mean()
                        + a.lam_m * ((r7.hard_margin(Yk, eye[k_][:, k_]) - Mt[:, k_]) ** 2).mean())
            g = Y.mean(1, keepdim=True); C = Y - g
            return (a.lam_g * ((g - gt) ** 2).mean() + a.lam_c * ((C - Ct) ** 2).mean()
                    + a.lam_m * ((r7.hard_margin(Y, eye) - Mt) ** 2).mean())

        def act_per_target(Y, Yt_):
            """逐目标活性损失：该酶自己那一列的 MSE。

            §133.5：现损失对目标取均值，一个目标的改善可以抵销另一个目标的活性亏损；
            这里改为逐目标各自成项，再由各自的对偶乘子加权。
            """
            return ((Y - Yt_) ** 2).mean(0)                    # (P,)

        def sel_focus_loss():
            """选择聚焦代理（§133.5）：只用**训练**标签定义的三类，
            并聚焦「模型容易排进前列的」竞争失败样本，同时保留随机样本。

            与 §101 的区别：正例是**严格 usable**（活性达标且真实边际 ≥ Δ_min），
            不把「边际较高但仍为负」的样本称作合格正例。
            与 §105 的区别：这是对 usable/竞争失败两类的成对间隔，不是标量 ListNet。
            """
            j = torch.randint(0, len(tr_s), (a.n_sel,), device=dev, generator=gg_)
            Y = net(Ztr[j], None if Rtr is None else Rtr[j], F)          # (n_sel,P)
            t_ = int(torch.randint(0, P, (1,), device=dev, generator=gg_).item())
            if a.lopo >= 0 and t_ == a.lopo:
                return Y.sum() * 0.0
            u = USABLE_TR[t_][j]; f = CFAIL_TR[t_][j]
            if u.sum() < 1 or f.sum() < 1:
                return Y.sum() * 0.0
            ar_ = torch.arange(len(j), device=dev)
            tt_ = torch.full((len(j),), t_, dtype=torch.long, device=dev)
            eta = final_eta(Y, tt_, ar_)                                 # (n_sel,)
            iu = torch.nonzero(u, as_tuple=True)[0]
            iff = torch.nonzero(f, as_tuple=True)[0]
            # 聚焦：在竞争失败里取**模型预测活性最高**的那些——即最可能被算法排进前列的负例
            kf = min(len(iff), max(1, a.n_sel // 16))
            iff = iff[Y[iff, t_].topk(kf).indices]
            npair = min(a.n_pair, len(iu) * len(iff))
            gu = iu[torch.randint(0, len(iu), (npair,), device=dev, generator=gg_)]
            gf = iff[torch.randint(0, len(iff), (npair,), device=dev, generator=gg_)]
            return nn.functional.softplus(
                (a.margin_m - (eta[gu] - eta[gf])) / a.temp).mean()

        def final_eta(Y, tt, ar):
            """η = Ŷ_t − max_{o≠t} Ŷ_o，**在最终预测上**算，与算法读的是同一个量。"""
            msk = torch.zeros_like(Y, dtype=torch.bool); msk[ar, tt] = True
            return Y[ar, tt] - Y.masked_fill(msk, -1e9).max(1).values

        def dfl_final():
            """与 metric_loss 同形，但 η 来自最终预测而非辅助 r 分支。"""
            k = torch.randint(0, len(TRP), (a.n_pair,), device=dev, generator=gg_)
            tt, ip, iq = TRP[k, 0], TRP[k, 1], TRP[k, 2]
            ar = torch.arange(a.n_pair, device=dev)
            idx = torch.cat([ip, iq])
            Y = net(Ztr[idx], None if Rtr is None else Rtr[idx], F)
            ep_ = final_eta(Y[:a.n_pair], tt, ar)
            eq_ = final_eta(Y[a.n_pair:], tt, ar)
            return nn.functional.softplus((a.margin_m - (ep_ - eq_)) / a.temp).mean()

        def dfl_list():
            """活性条件的软列表代理。

            算法按「活性高、风险低」取 Pareto 前沿；此处用可微评分
            score = Ŷ_t − β·risk 近似其排序，risk 取 17 个脱靶预测最高 4 个的均值
            （与 selection.py 同一约定）。目标分布用**训练真值**给出的软 usable：
            q ∝ σ((y_t − A_t)/s)·σ((m_t − Δ_t)/s)，s 越大边界越不确定——
            按 §104.6「以训练真值确定方向和连续权重，对边界附近标签保留不确定度」。
            完整 profile 回归照旧保留，不牺牲活性。
            """
            j = torch.randint(0, len(tr_s), (a.n_list,), device=dev, generator=gg_)
            Y = net(Ztr[j], None if Rtr is None else Rtr[j], F)          # (n_list, P)
            t = int(torch.randint(0, P, (1,), device=dev, generator=gg_).item())
            act = Y[:, t]
            off = torch.cat([Y[:, :t], Y[:, t + 1:]], 1)
            rk = off.topk(max(1, off.shape[1] // 4), dim=1).values.mean(1)
            lp = torch.log_softmax((act - a.dfl_beta * rk) / a.dfl_temp, 0)
            q = QSOFT[t][j]
            q = q / q.sum().clamp_min(1e-8)
            return -(q * lp).sum()

        def metric_loss():
            """在活性匹配的 (t, x⁺, x⁻) 上，要求 η(x⁺) − η(x⁻) ≥ m。"""
            k = torch.randint(0, len(TRP), (a.n_pair,), device=dev, generator=gg_)
            tt, ip, iq = TRP[k, 0], TRP[k, 1], TRP[k, 2]
            ar0 = torch.arange(a.n_pair, device=dev)
            if a.plain_contrast:                      # 旧对照：全库均匀，等价 fresh+ρ=1
                iq = torch.randint(0, len(tr_s), (a.n_pair,), device=dev, generator=gg_)
            elif NEG is not None:
                if a.neg_fresh:
                    # 池内重采 K 个候选，取第一个真实边际确实更低的；否则退回固定负例
                    K = 12
                    j = (torch.rand(a.n_pair, K, device=dev, generator=gg_)
                         * PLEN[tt][:, None]).long()
                    cand = POOL[tt[:, None], j]                       # (n_pair, K)
                    worse = MG[tt[:, None], cand] < MG[tt, ip][:, None]
                    first = worse.float().argmax(1)
                    iq = torch.where(worse.any(1), cand[ar0, first], iq)
                if a.neg_noise > 0:
                    # ρ 比例改为全库均匀抽（不核验）——这一路才是标签噪声
                    hit = torch.rand(a.n_pair, device=dev, generator=gg_) < a.neg_noise
                    rnd = torch.randint(0, len(tr_s), (a.n_pair,), device=dev,
                                        generator=gg_)
                    iq = torch.where(hit, rnd, iq)
            idx = torch.cat([ip, iq])
            _, r = net(Ztr[idx], None if Rtr is None else Rtr[idx], F, ret_r=True)
            rp, rq = r[:a.n_pair], r[a.n_pair:]
            ar = torch.arange(a.n_pair, device=dev)
            # η = r_t − max_{o≠t} r_o（完整 panel）
            msk = torch.zeros_like(rp, dtype=torch.bool); msk[ar, tt] = True
            ep_ = rp[ar, tt] - rp.masked_fill(msk, -1e9).max(1).values
            eq_ = rq[ar, tt] - rq.masked_fill(msk, -1e9).max(1).values
            return nn.functional.softplus((a.margin_m - (ep_ - eq_)) / a.temp).mean()

        # 共享层 = 残基嵌入 E 与共享主干 trunk（g、C、边际三个头都经过它）
        # 共享层按**名字**取，不按 requires_grad——冻结阶段它们仍然是共享层，
        # 只是此刻不可训练；探针在冻结阶段自动跳过。
        shared = [q for k, q in net.named_parameters()
                  if k.split(".")[0] in ("E", "trunk", "Wp")]
        assert shared, "共享层为空：参数名前缀与预期不符"

        def grad_probe(b):
            """在**共享层**上分别取各损失项的梯度，报告两两余弦与范数。

            §104.6：「训练前后记录活性、C 回归、边际/对比损失对共享层的
            梯度方向及大小」，且「不能先宣布冲突就是原因」。故这里只测量。
            """
            Y = net(Ztr[b], None if Rtr is None else Rtr[b], F)
            g_ = Y.mean(1, keepdim=True); C_ = Y - g_
            parts = {
                "g": a.lam_g * ((g_ - gtr[b]) ** 2).mean(),
                "C": a.lam_c * ((C_ - Ctr[b]) ** 2).mean(),
                "M": a.lam_m * ((r7.hard_margin(Y, eye) - Mtr[b]) ** 2).mean()}
            if a.dfl == "final" and TRP is not None:
                parts["dfl"] = a.lam_dfl * dfl_final()
            elif a.dfl == "list":
                parts["dfl"] = a.lam_dfl * dfl_list()
            elif TRP is not None and a.lam_metric > 0:
                parts["aux"] = a.lam_metric * metric_loss()
            live = [q for q in shared if q.requires_grad]
            if not live:
                net.zero_grad(set_to_none=True)
                return {}                      # 冻结阶段无梯度可测，跳过
            gs = {}
            for k, v in parts.items():
                gg2 = torch.autograd.grad(v, live, retain_graph=True,
                                          allow_unused=True)
                gs[k] = torch.cat([(x if x is not None else torch.zeros_like(y)).reshape(-1)
                                   for x, y in zip(gg2, live)])
            ks = list(gs)
            out = {f"norm_{k}": float(gs[k].norm()) for k in ks}
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    c = torch.nn.functional.cosine_similarity(
                        gs[ks[i]][None], gs[ks[j]][None]).item()
                    out[f"cos_{ks[i]}_{ks[j]}"] = float(c)
            net.zero_grad(set_to_none=True)
            return out

        LAM = None
        if a.act_constraint:
            # 约束水平 c_p = **起点模型**在训练集上的逐目标活性 MSE。
            # 含义：不许任何一个目标的活性比起点更差；满足时该项自动归零，
            # 故不能靠别的目标的改善来抵销（§133.5）。
            net.eval()
            with torch.no_grad():
                acc = torch.zeros(P, device=dev); n_ = 0
                for i in range(0, len(tr_s), 512):
                    Yb = net(Ztr[i:i + 512], None if Rtr is None else Rtr[i:i + 512], F)
                    acc += ((Yb - ytr[i:i + 512]) ** 2).sum(0); n_ += len(Yb)
                CLEV = acc / n_
            LAM = torch.zeros(P, device=dev)
            print("    活性约束水平 c_p（起点逐目标 MSE）：均值 %.5f  最差 %s=%.5f"
                  % (float(CLEV.mean()), order[int(CLEV.argmax())] if order else
                     int(CLEV.argmax()), float(CLEV.max())), flush=True)
            net.train()
        best, bs, bad, ep, hist, probes = 1e18, None, 0, 0, [], []
        for ep in range(a.epochs):
            net.train()
            ii = torch.randperm(len(tr_s), device=dev)
            for i in range(0, len(ii), a.batch):
                b = ii[i:i + a.batch]
                opt.zero_grad(set_to_none=True)
                Y = net(Ztr[b], None if Rtr is None else Rtr[b], F)
                l = base_loss(Y, gtr[b], Ctr[b], Mtr[b])
                if LAM is not None:
                    vio = act_per_target(Y, ytr[b]) - CLEV               # (P,)
                    l = l + (LAM * vio).sum() / P
                if a.sel_focus:
                    l = l + a.lam_sel * sel_focus_loss()
                if TRP is not None and a.lam_metric > 0 and a.dfl == "none":
                    l = l + a.lam_metric * metric_loss()
                elif a.dfl == "final" and TRP is not None:
                    l = l + a.lam_dfl * dfl_final()
                elif a.dfl == "list":
                    l = l + a.lam_dfl * dfl_list()
                l.backward(); opt.step()
                if LAM is not None:
                    # 对偶上升（Chamon & Ribeiro 式约束学习）：违约则乘子上升，
                    # 满足则回落到 0。乘子不参与反向，单独更新。
                    with torch.no_grad():
                        LAM.add_(a.dual_lr * vio.detach()).clamp_(min=0.)
            sch.step()
            if a.freeze_epochs and ep + 1 == a.freeze_epochs:
                for q in net.parameters():
                    q.requires_grad_(True)
                opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=a.wd)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, max(1, a.epochs - a.freeze_epochs))
                shared[:] = [q for k, q in net.named_parameters()
                             if k.split(".")[0] in ("E", "trunk", "Wp")]
                print(f"    ep{ep+1}: 解冻 base，重建优化器", flush=True)
            net.eval()
            with torch.no_grad():
                # 每条验证肽等权：末批不足时不能与完整批等权
                v = sum(base_loss(net(Zva[i:i+512], None if Rva is None else Rva[i:i+512], F),
                                  gva[i:i+512], Cva[i:i+512], Mva[i:i+512]).item()
                        * len(Zva[i:i+512]) for i in range(0, len(Zva), 512)) / len(Zva)
            hist.append(dict(ep=ep, val_loss=v, lr=sch.get_last_lr()[0]))
            if a.grad_probe and ep % a.grad_probe == 0:
                _pr = grad_probe(ii[:a.batch])
                if _pr:
                    probes.append(dict(ep=ep, **_pr))
            if v < best - 1e-7:
                best, bad = v, 0
                bs = {k: t.detach().clone() for k, t in net.state_dict().items()}
                hist[-1]["best"] = True
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(bs); net.eval()
        with torch.no_grad():
            Pm = np.concatenate([net(Zte[i:i+512], None if Rte is None else Rte[i:i+512], F)
                                 .cpu().numpy() for i in range(0, n, 512)]).T
            Pv = np.concatenate([net(Zva[i:i+512], None if Rva is None else Rva[i:i+512], F)
                                 .cpu().numpy() for i in range(0, len(Zva), 512)]).T
        rt, rv = rs_of(Pm, Mfull), rs_of(Pv, Mval)
        if a.lopo >= 0:
            Mp_ = np.stack([Pm[t] - np.delete(Pm, t, 0).max(0) for t in range(P)])
            rt_h = float(spearmanr(Mp_[a.lopo], Mfull[a.lopo]).statistic)
            rows_extra = dict(lopo_rho=rt_h)
        else:
            rows_extra = {}
        Ch = Pm - Pm.mean(0)
        rc = float(np.mean([pearsonr(Ch[q], C_true[q]).statistic for q in range(P)]))
        al = float(net.alpha.item()) if not a.no_metric else 0.0
        store[f"R8|{seed}"] = Pm; store[f"VAL@R8|{seed}"] = Pv
        runmeta.save_run(a.ckpt_dir, os.path.basename(a.out).replace(".npz", ""), seed,
                         state=bs, cfg=dict(vars(a), n_param=npar, stopped_ep=ep + 1),
                         hist=hist,
                         ids=dict(train=list(map(str, tr_s)), val=list(map(str, va_s)),
                                  test=list(map(str, te_s)),
                                  targets=list(order) if order else []),
                         code=runmeta.code_hash(os.path.join(HERE, "gpu_r8_metric.py"),
                                                os.path.join(HERE, "gpu_r7_pairwise.py"),
                                                os.path.join(HERE, "gpu_mlm_pretrain.py")),
                         extra=dict(val_rho_sM=rv, rho_sM=rt, rho_C=rc, alpha=al,
                                    alpha_lm=float(net.alpha_lm.item())
                                    if getattr(net, "lm", None) is not None
                                    and a.lm_residual else None,
                                    grad_probes=probes))
        rows.append(dict(seed=seed, val_rho_sM=rv, rho_sM=rt, rho_C=rc, alpha=al,
                         **rows_extra))
        if a.select_eval:
            sr, _hit = selection.evaluate(Pv, Yva.T, Pm, Yf, Ytr.T)
            sm = selection.summarize(sr)
            sel_rows.append(dict(seed=seed, **{f"{k1}_{k2}": v2
                                               for k1, d1 in sm.items()
                                               for k2, v2 in d1.items()}))
            sel_detail.extend([dict(seed=seed, **r) for r in sr])
            if a.select_sweep:
                sel_sweep.extend([dict(seed=seed, **r)
                                  for r in selection.sweep(Pv, Yva.T, Pm, Yf, Ytr.T)])
        if probes:
            kk = [k for k in probes[-1] if k.startswith("cos_")]
            print("    梯度探针（共享层，末次）："
                  + "  ".join(f"{k[4:]} {probes[-1][k]:+.3f}" for k in kk)
                  + "   ‖g‖ " + " ".join(
                      f"{k[5:]}={probes[-1][k]:.3g}"
                      for k in probes[-1] if k.startswith("norm_")), flush=True)
        if rows_extra:
            print(f"    留出酶 ρ_s(M) = {rows_extra['lopo_rho']:.4f}", flush=True)
        print(f"  seed {seed}  [验证] ρ_s(M) {rv:.4f}   [测试] ρ_s(M) {rt:.4f}  ρ(C) {rc:.4f}"
              f"   α={al:+.4f}   {npar/1e6:.2f}M   {time.time()-t0:6.1f}s ep{ep+1}", flush=True)

    if a.select_eval and sel_rows:
        sd = pd.DataFrame(sel_rows)
        print("\n  终点选样（阈值来自训练分布，非 W1 标定；测试真值只打分不回调）")
        for tf in ("count", "fraction"):
            u = sd[f"{tf}_usable"]; r = sd[f"{tf}_retention"]; n = sd[f"{tf}_n_pass"]
            print(f"   迁移={tf:8s} usable " + "/".join(f"{x:.4f}" for x in u)
                  + f"  平均 {u.mean():.5f}   真实活性保持 {r.mean():.5f}"
                  + f"   达标 {int(n.sum())}/{int(sd[f'{tf}_n'].sum())}")
        pd.DataFrame(sel_detail).to_csv(a.out.replace(".npz", "_select.csv"), index=False)
        if sel_sweep:
            pd.DataFrame(sel_sweep).to_csv(a.out.replace(".npz", "_sweep.csv"), index=False)

    d_ = os.path.dirname(a.out)
    if d_:
        os.makedirs(d_, exist_ok=True)
    np.savez_compressed(a.out, **store)
    df = pd.DataFrame(rows); df.to_csv(a.out.replace(".npz", ".csv"), index=False)
    print(f"\n  均值：验证 {df.val_rho_sM.mean():.4f}   测试 {df.rho_sM.mean():.4f}"
          f"   α {df.alpha.mean():+.4f}")
    print(f"  已写入 {a.out}")


if __name__ == "__main__":
    main()
