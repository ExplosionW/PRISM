#!/usr/bin/env python
"""DeCleave-LM：我们**自己**的小型蛋白质语言模型（掩码残基预训练）。

用户要求：训练一个在我们任务上的蛋白质语言头，装进我们的架构；可作 encoder，
也可像 T5 那样只提供嵌入。**不是微调 ESM 或任何现成大模型**——权重从零开始。

**语料**：只用 `X_train.csv` 的训练+验证部分（14,866 条 10-mer）。
**刻意不含测试集序列**——即使不用标签，把测试输入放进预训练也是转导学习，
会让后续比较有争议。宁可少 2,901 条，也不留这个口子。

**为什么这件事有可能有用**：实测底物库的位点间互信息均值 0.0977 bit、
最大 0.4667（有限样本下限约 0.014），即位点并非独立——库里存在可学的序列结构。
MLM 能把这部分结构编进残基表示，而监督训练只能从 18 维标签里间接学。

**为什么预期保守**：10-mer 很短，14,866 条也不算多；且 MLM 能学到的
位置特异分布与二阶相关，恰好也是 DeCleave-Pair 的显式归纳偏置已经覆盖的东西。
"""
import argparse, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

B = os.environ.get("PROTEASE_DATA", "./data")
AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, MASK = 20, 21                                   # 20 类残基 + PAD + MASK


class DeCleaveLM(nn.Module):
    """小型 encoder-only transformer；`encode()` 给下游用，`forward()` 出 MLM logits。"""

    def __init__(s, L=10, d=128, layers=4, heads=4, ff=256, pdrop=0.1):
        super().__init__()
        s.L, s.d = L, d
        s.emb = nn.Embedding(22, d)
        s.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        s.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, dropout=pdrop,
                                       batch_first=True, norm_first=True),
            num_layers=layers)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 20)                    # 只预测 20 类真实残基

    def encode(s, Z):
        return s.norm(s.tr(s.emb(Z) + s.pos))        # (B, L, d)

    def forward(s, Z):
        return s.head(s.encode(Z))


def encode_seqs(seqs, L=10):
    ix = {c: i for i, c in enumerate(AA)}
    Z = np.full((len(seqs), L), PAD, np.int64)
    for i, q in enumerate(seqs):
        for j, c in enumerate(str(q).strip()[:L]):
            Z[i, j] = ix.get(c, PAD)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.05,
                    help="线性预热占总轮数的比例。深层 transformer 无预热常训练不足——"
                         "实测 d=256/8层 在 400 轮后 val CE 仍单调下降（2.0853），"
                         "而 d=192/6层 早已收敛（1.6812）；无预热的规模比较是不公平的。")
    ap.add_argument("--extra", default=None,
                    help="附加预训练语料 csv（列 oct = 8-mer，如 MEROPS）。"
                         "MEROPS 给的是 P4–P4'（8 位），本库是 10-mer（P5–P5'），"
                         "故对齐到第 2–9 位、两端补 PAD。**只加进训练集**，"
                         "验证/早停仍在本库的 1,200 条上，故目标始终按我们的分布度量。")
    ap.add_argument("--extra-frac", type=float, default=1.0,
                    help="附加语料的采样比例（1.0 = 全用）")
    ap.add_argument("--iso-val", action="store_true",
                    help="按连通分量隔离切验证折，且训练语料排除该折（与下游一致）")
    ap.add_argument("--out", default="decleave_lm.pt")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    Str = pd.read_csv(f"{B}/X_train.csv", header=None)[0].astype(str).str.strip().to_numpy()
    L = len(Str[0])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Str))
    # 与下游完全相同的划分：前 1200 为验证。预训练用 train 学、val 早停，
    # 两者都不含测试序列。
    if a.iso_val:
        # 关键：下游用隔离验证时，预训练语料**必须**同样排除那一折。
        # 否则验证肽虽无标签、其序列已进过 MLM 语料，属转导学习，四臂对比失去意义。
        import importlib.util
        _s = importlib.util.spec_from_file_location(
            "r7", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "gpu_r7_pairwise.py"))
        r7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r7)
        trm, vam = r7.iso_split(encode_seqs(Str, L), 1200)   # 与下游同序同种子
        va, tr = Str[vam], Str[trm]
    else:
        va, tr = Str[perm[:1200]], Str[perm[1200:]]
    Ztr_np = encode_seqs(tr, L)
    n_own = len(Ztr_np)
    if a.extra:
        ex = pd.read_csv(a.extra)["oct"].astype(str).str.strip().to_numpy()
        if a.extra_frac < 1.0:
            rr = np.random.default_rng(1)
            ex = ex[rr.choice(len(ex), int(a.extra_frac * len(ex)), replace=False)]
        # 8 位对齐到 10 位坐标系的第 2–9 位；两端（P5、P5'）保持 PAD
        E8 = encode_seqs(ex, 8)
        pad = np.full((len(E8), L), PAD, np.int64)
        pad[:, 1:9] = E8
        Ztr_np = np.concatenate([Ztr_np, pad])
    Ztr = torch.tensor(Ztr_np).to(dev)
    Zva = torch.tensor(encode_seqs(va, L)).to(dev)
    BAR = "=" * 88
    print(BAR); print("DeCleave-LM：自建小型蛋白质语言模型（掩码残基预训练）"); print(BAR)
    print(f"  语料 train {len(Ztr):,}（本库 {n_own:,}"
          + (f" + 附加 {len(Ztr)-n_own:,}）" if a.extra else "）")
          + f" / val {len(va):,}"
          + ("（隔离折，**已从语料中排除**；亦不含测试集序列）" if a.iso_val
             else "（**不含测试集序列**）"))
    print(f"  d={a.dim} layers={a.layers} heads={a.heads} ff={a.ff} mask={a.mask_rate:g}\n")

    net = DeCleaveLM(L, a.dim, a.layers, a.heads, a.ff).to(dev)
    print(f"  参数量 {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    wu = max(1, int(a.warmup * a.epochs))
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, wu),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.epochs - wu))],
        milestones=[wu])
    ce = nn.CrossEntropyLoss()
    g = torch.Generator(device=dev); g.manual_seed(0)

    def corrupt(Z, gen):
        """标准 BERT 掩码：15% 选中，其中 80% → MASK，10% → 随机残基，10% → 保持。"""
        # **必须排除 PAD 位**：附加语料（MEROPS，8 位）补 PAD 后，
        # 若掩码选中 PAD，目标 id=20 超出 20 类分类头范围 → CUDA device-side assert。
        # 原语料 10 位全为真实残基，故此前从未触发。
        m = (torch.rand(Z.shape, device=Z.device, generator=gen) < a.mask_rate) & (Z != PAD)
        X = Z.clone()
        r = torch.rand(Z.shape, device=Z.device, generator=gen)
        X[m & (r < 0.8)] = MASK
        rnd = torch.randint(0, 20, Z.shape, device=Z.device, generator=gen)
        pick = m & (r >= 0.8) & (r < 0.9)
        X[pick] = rnd[pick]
        return X, m

    best, bs, bad = 1e18, None, 0
    for ep in range(a.epochs):
        net.train()
        ii = torch.randperm(len(Ztr), device=dev, generator=g)
        for i in range(0, len(ii), a.batch):
            b = ii[i:i + a.batch]
            X, m = corrupt(Ztr[b], g)
            opt.zero_grad(set_to_none=True)
            lo = net(X)
            l = ce(lo[m], Ztr[b][m])
            l.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            gv = torch.Generator(device=dev); gv.manual_seed(123)   # 验证用固定掩码
            Xv, mv = corrupt(Zva, gv)
            v = ce(net(Xv)[mv], Zva[mv]).item()
            acc = (net(Xv)[mv].argmax(1) == Zva[mv]).float().mean().item()
        if v < best - 1e-5:
            best, bad = v, 0
            bs = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
        if ep % 20 == 0:
            print(f"  ep{ep:>4d}  val CE {v:.4f}  掩码位准确率 {acc:.3f}", flush=True)

    net.load_state_dict(bs)
    torch.save(dict(state=bs, cfg=dict(L=L, d=a.dim, layers=a.layers,
                                       heads=a.heads, ff=a.ff)), a.out)
    print(f"\n  最优 val CE {best:.4f}（均匀猜测 = {np.log(20):.4f}）")
    print(f"  ⇒ 相对均匀基线降低 {(1-best/np.log(20))*100:.1f}%，即库里确有可学的序列结构")
    print(f"  已写入 {a.out}")


if __name__ == "__main__":
    main()
