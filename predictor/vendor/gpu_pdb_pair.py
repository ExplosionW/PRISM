#!/usr/bin/env python
"""§109: experimental pocket-conditioned residue-pair adapter; full panel only.

All arms load the exact same seed-specific frozen task model. New pair-feature
modulation starts at zero. No experimental complex/contact labels are used.
The learned correction changes covered enzymes' raw activities only; uncovered
activities remain at baseline (their competitive margins can still change).
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr

import gpu_r8_metric as r8


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_json(path, data):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)); tmp.replace(path)


def save_npz(path, data):
    path = Path(path); tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **data); tmp.replace(path)


def load_base(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    c, s = ck['cfg'], ck['state']
    assert c.get('lopo', -1) == -1
    assert c.get('ladder', 'none') == 'none' and c.get('geom_mode', 'none') == 'none'
    assert not c.get('cross_attn') and not c.get('film_pos_rank', 0)
    emb = s['Ep_raw'].numpy() if 'Ep_raw' in s else None
    lm = r8.load_lm(c['lm_init'], dev) if c.get('lm_init') else None
    base = r8.R8Net(len(ck['ids']['targets']), L=s['E'].shape[0],
        d=c['dim'], m=c['pair_dim'], hid=c['hid'], pdrop=c['dropout'], de=c['de'],
        Emb=emb, esm_dim=s['eproj.0.weight'].shape[1] if 'eproj.0.weight' in s else 0,
        zdim=c['zdim'], kappa=c['kappa'], use_metric=not c['no_metric'],
        lm=lm, lm_freeze=c.get('lm_freeze', False), lm_residual=c.get('lm_residual', False)).to(dev)
    base.load_state_dict(s, strict=True)
    base.eval().requires_grad_(False)
    return base, ck


@torch.no_grad()
def residue_features(base, z, f):
    e = torch.stack([base.E[i][z[:, i]] for i in range(base.L)], 1)
    if base.esm: e = base.fuse(torch.cat([e, base.eproj(f.float())], -1))
    if base.lm is not None:
        zl = base.lm.encode(z)
        if base.lm_residual: e = e + base.alpha_lm * base.lmproj(zl)
        else: e = base.lmfuse(torch.cat([e, base.lmproj(zl)], -1))
    return e


def condition(base, e):
    ev = base.ep_vec(); gam, bet = base.film(ev).chunk(2, -1)
    return e[:, None] * (1 + gam[None, :, None]) + bet[None, :, None]


def raw_scores(base, hp):
    raw = base.rout(hp).squeeze(-1)
    if base.use_metric:
        z = nn.functional.normalize(base.Wz(hp), dim=-1)
        v = nn.functional.normalize(base.Wv(base.ep_vec()), dim=-1)
        raw = raw + base.alpha * base.kappa * (z * v[None]).sum(-1)
    return raw


@torch.no_grad()
def cache_split(base, seqs, fmap, feats, dev, batch=128):
    es, ys, rs = [], [], []
    maxerr = 0.
    for i in range(0, len(seqs), batch):
        ss = seqs[i:i + batch]
        z = torch.as_tensor(r8.encode(ss), device=dev)
        f = torch.as_tensor(feats[[fmap[str(s)] for s in ss]], device=dev)
        e = residue_features(base, z, f)
        hp = base.body(condition(base, e).flatten(0, 1)).reshape(len(e), base.P, -1)
        raw = raw_scores(base, hp)
        y = base.a[None] + base.g(base.body(e)) + raw - raw.mean(1, keepdim=True)
        if i == 0:
            ref = base(z, torch.arange(len(z), device=dev), f)
            maxerr = float((y - ref).abs().max())
            assert maxerr < 2e-5, ('cache reconstruction mismatch', maxerr)
        es.append(e); ys.append(y); rs.append(raw)
    return tuple(torch.cat(x) for x in (es, ys, rs)), maxerr


class PocketPair(nn.Module):
    def __init__(self, base, geometry, mode, width=32):
        super().__init__()
        self.base = base
        gf, gd, gm, gh = [x.clone() for x in geometry]
        if mode == 'sequence': gf.zero_(); gd.zero_()
        if mode == 'identity': gf[:, :, 21:] = 0; gd.zero_()
        # Identity controls retain exactly the same pocket membership/mask.
        # They are not claimed to remove the structural origin of membership.
        if mode == 'geometry':
            gf[:, :, 21:23] /= 20.; gf[:, :, 24:27] /= 20.
        for name, arr in zip(('gf', 'gd', 'gm', 'gh'), (gf, gd, gm, gh)):
            self.register_buffer(name, arr)
        self.node = nn.Sequential(nn.Linear(27, width), nn.LayerNorm(width), nn.GELU())
        self.msg = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * width, width),
                                               nn.LayerNorm(width), nn.GELU()) for _ in range(2)])
        self.q = nn.Sequential(nn.Linear(2 * base.d_e, width), nn.LayerNorm(width), nn.GELU())
        self.key = nn.Linear(width, width); self.value = nn.Linear(width, width)
        self.combine = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                     nn.Linear(width, base.Wp.shape[-1]))
        nn.init.zeros_(self.combine[-1].weight); nn.init.zeros_(self.combine[-1].bias)
        ii, jj = zip(*base.pairs)
        self.register_buffer('ii', torch.tensor(ii)); self.register_buffer('jj', torch.tensor(jj))
        self.width = width

    def train(self, mode=True):
        super().train(mode); self.base.eval(); return self

    def adapter_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith('base.')}

    def sites(self):
        h = self.node(self.gf) * self.gm[..., None]
        # Fixed distance kernel: clean experimental distances only, no learned poses.
        w = torch.exp(-self.gd.square() / 64.) * self.gm[:, None]
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        for msg, upd in zip(self.msg, self.upd):
            agg = w @ msg(h)
            h = (h + upd(torch.cat([h, agg], -1))) * self.gm[..., None]
        return h

    def forward(self, e, baseline_y, baseline_raw):
        b = self.base; em = condition(b, e)
        pair = torch.cat([em[:, :, self.ii], em[:, :, self.jj]], -1)
        u = torch.relu(torch.einsum('btpk,pkm->btpm', pair, b.Wp) + b.bp)
        q = self.q(pair); sites = self.sites()
        att = torch.einsum('btpd,tsd->btps', q, self.key(sites)) / self.width ** .5
        att = att.masked_fill(self.gm[None, :, None] < .5, -1e4).softmax(-1)
        ctx = torch.einsum('btps,tsd->btpd', att, self.value(sites))
        delta = .25 * torch.tanh(self.combine(torch.cat([q, ctx, q * ctx], -1)))
        u = u * (1 + delta * self.gh[None, :, None, None])
        u = u * torch.sigmoid(b.z)[None, None, :, None]
        h = b.trunk(torch.cat([em.flatten(2), u.flatten(2)], -1))
        raw = raw_scores(b, h)
        # A residual in activity space: missing structures receive no correction.
        # g and C are always recomputed from this complete final profile in loss.
        return baseline_y + (raw - baseline_raw) * self.gh[None]


def profile_loss(pred, truth):
    g, gt = pred.mean(1, keepdim=True), truth.mean(1, keepdim=True)
    eye = torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device)
    return ((g - gt).square().mean() + 36 * ((pred - g) - (truth - gt)).square().mean()
            + 2 * (r8.r7.hard_margin(pred, eye) - r8.r7.hard_margin(truth, eye)).square().mean())


def pair_loss(pred, trip, n):
    eye = torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device)
    margin = r8.r7.hard_margin(pred, eye)
    ar = torch.arange(n, device=pred.device); t = trip[:, 0]
    return nn.functional.softplus(.5 - (margin[:n][ar, t] - margin[n:][ar, t])).mean()


@torch.no_grad()
def predict(net, cache, batch=128):
    net.eval()
    return torch.cat([net(*(x[i:i + batch] for x in cache))
                      for i in range(0, len(cache[0]), batch)])


def run_seed(a, seed, out, features, fmap, raw):
    dev = torch.device(a.device); basepath = a.base.replace('SEED', str(seed))
    base, ck = load_base(basepath, dev); original = state_digest(base.state_dict())
    ids = ck['ids']; pool_s, pool_y, test_s, test_y = raw
    assert len(ids['targets']) == 18
    assert len(set(pool_s)) == len(pool_s)
    assert not (set(ids['train']) & set(ids['val']))
    assert set(ids['train']) | set(ids['val']) == set(pool_s)
    assert not ((set(ids['train']) | set(ids['val'])) & set(ids['test']))
    assert list(test_s) == ids['test']
    rows = {str(s): i for i, s in enumerate(pool_s)}
    tr_y = pool_y[[rows[s] for s in ids['train']]]
    va_y = pool_y[[rows[s] for s in ids['val']]]
    train_y = torch.as_tensor(tr_y, device=dev); valid_y = torch.as_tensor(va_y, device=dev)
    geo = np.load(a.geometry, allow_pickle=True)
    assert list(geo['genes']) == ids['targets'], 'protease order mismatch'
    geometry = [torch.as_tensor(geo[k], device=dev, dtype=torch.float32)
                for k in ('feat', 'dist', 'mask', 'has_struct')]
    assert geometry[0].shape == (18, 32, 27)
    assert torch.isfinite(geometry[0]).all() and torch.isfinite(geometry[1]).all()
    caches = {}; checks = {}
    print(f'CACHE seed={seed} base={basepath}', flush=True)
    for split in ('train', 'val', 'test'):
        caches[split], checks[split] = cache_split(base, ids[split], fmap, features, dev)
    trips = torch.as_tensor(r8.build_pairs(tr_y, mode='clean', seed=0), device=dev)
    assert len(trips) and int(trips[:, 1:].max()) < len(ids['train'])
    seedout = out / f'seed{seed}'; seedout.mkdir(exist_ok=True)
    baseline = {'_Yf': test_y, f'BASE|{seed}': caches['test'][1].cpu().numpy().T,
                f'VAL@BASE|{seed}': caches['val'][1].cpu().numpy().T}
    save_npz(seedout / 'baseline.npz', baseline)
    provenance = dict(seed=seed, base_file=basepath, base_sha256=digest(basepath),
        base_state_sha256=original, geometry_sha256=digest(a.geometry),
        cache_initial_errors=checks, ids=ids, cfg=vars(a))
    save_json(seedout / 'provenance.json', provenance)
    for mode in a.modes:
        torch.manual_seed(seed); np.random.seed(seed)
        net = PocketPair(base, geometry, mode, a.width).to(dev)
        pars = [v for k, v in net.named_parameters() if not k.startswith('base.')]
        opt = torch.optim.AdamW(pars, lr=a.lr, weight_decay=a.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        generator = torch.Generator(device=dev).manual_seed(seed)
        arm = f'{mode}_seed{seed}'; start = time.time(); bestep = 0; bad = 0
        with torch.no_grad():
            yp = predict(net, caches['val']); initerr = float((yp - caches['val'][1]).abs().max())
            assert initerr < 2e-5, ('adapter initialization mismatch', initerr)
            best = float(profile_loss(yp, valid_y))
        beststate = net.adapter_state(); hist = [dict(epoch=0, val_loss=best, initial_error=initerr)]
        print(f'PREFLIGHT {arm} initial_error={initerr:.3g} trainable={sum(p.numel() for p in pars)} '
              f'val0={best:.6f} base_frozen=True', flush=True)
        torch.save(dict(adapter=beststate, epoch=0, provenance=provenance), seedout / f'{mode}_best.pt')
        for epoch in range(1, a.epochs + 1):
            epstart = time.time(); net.train()
            indices = torch.randperm(len(train_y), generator=generator, device=dev)
            total = 0.; steps = 0
            for i in range(0, len(indices), a.batch):
                ix = indices[i:i + a.batch]
                t = trips[torch.randint(len(trips), (a.n_pair,), generator=generator, device=dev)]
                pair_ix = torch.cat([t[:, 1], t[:, 2]])
                opt.zero_grad(set_to_none=True)
                pred = net(*(x[ix] for x in caches['train']))
                loss = profile_loss(pred, train_y[ix])
                pp = net(*(x[pair_ix] for x in caches['train']))
                loss = loss + .5 * pair_loss(pp, t, a.n_pair)
                if not torch.isfinite(loss): raise RuntimeError(f'{arm}: non-finite loss')
                loss.backward()
                if epoch == 1 and i == 0:
                    grad = sum(float(p.grad.square().sum()) for p in pars if p.grad is not None) ** .5
                    assert np.isfinite(grad) and grad > 0
                    assert all(p.grad is None for p in base.parameters())
                    print(f'FIRST_UPDATE {arm} loss={float(loss):.6f} adapter_grad={grad:.6f}', flush=True)
                torch.nn.utils.clip_grad_norm_(pars, 10.)
                opt.step(); total += float(loss.detach()); steps += 1
                if a.preflight and steps == 2: break
                if steps % 50 == 0:
                    save_json(out / 'status.json', dict(status='training', seed=seed, mode=mode,
                        epoch=epoch, batch=steps, best_epoch=bestep, updated_unix=time.time()))
            sch.step()
            with torch.no_grad():
                val = float(profile_loss(predict(net, caches['val']), valid_y))
            elapsed = time.time() - epstart
            rec = dict(epoch=epoch, val_loss=val, train_loss=total / steps, seconds=elapsed,
                       lr=sch.get_last_lr()[0])
            if val < best - 1e-7:
                best, bestep, bad = val, epoch, 0; beststate = net.adapter_state()
                torch.save(dict(adapter=beststate, epoch=epoch, provenance=provenance), seedout / f'{mode}_best.pt')
            else: bad += 1
            hist.append(rec); save_json(seedout / f'{mode}_history.json', hist)
            save_json(out / 'status.json', dict(status='training', seed=seed, mode=mode, epoch=epoch,
                best_epoch=bestep, best_val=best, epoch_seconds=elapsed, updated_unix=time.time()))
            print(f'EPOCH {arm} {epoch} val={val:.6f} best={best:.6f}@{bestep} seconds={elapsed:.1f}', flush=True)
            if a.preflight or bad >= a.patience: break
        assert state_digest(base.state_dict()) == original, 'frozen backbone changed'
        net.load_state_dict({**net.state_dict(), **beststate}, strict=True)
        pv = predict(net, caches['val']).cpu().numpy().T
        pt = predict(net, caches['test']).cpu().numpy().T
        missing = ~geo['has_struct'].astype(bool)
        assert np.array_equal(pt[missing], baseline[f'BASE|{seed}'][missing])
        result = dict(mode=mode, seed=seed, best_epoch=bestep, stopped_epoch=epoch,
                      initial_error=initerr, best_val=best, seconds=time.time() - start,
                      base_unchanged=True, uncovered_activity_unchanged=True)
        if not a.preflight:
            rr, _ = r8.selection.evaluate(pv, va_y.T, pt, test_y, tr_y.T)
            for rec in rr: rec['gene'] = ids['targets'][rec['target']]
            pd.DataFrame(rr).to_csv(seedout / f'{mode}_selection.csv', index=False)
            result['selection'] = r8.selection.summarize(rr)
            mt, mp = r8.selection.margins(test_y), r8.selection.margins(pt)
            result['rho_sM'] = float(np.mean([spearmanr(x, y).statistic for x, y in zip(mt, mp)]))
            ct, cp = test_y - test_y.mean(0), pt - pt.mean(0)
            result['rho_C'] = float(np.mean([pearsonr(x, y).statistic for x, y in zip(ct, cp)]))
        save_npz(seedout / f'{mode}.npz', {'_Yf': test_y, f'{mode}|{seed}': pt, f'VAL@{mode}|{seed}': pv})
        save_json(seedout / f'{mode}_result.json', result)
        print('ARM_DONE', json.dumps(result), flush=True)
        del net, opt
    del caches, base
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='ck/t2_res_stage_seedSEED.pt')
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--features', default='data/esm_feats.npz')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--n-pair', type=int, default=256)
    ap.add_argument('--lr', type=float, default=.001); ap.add_argument('--wd', type=float, default=.001)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--modes', nargs='+', choices=['sequence', 'identity', 'geometry'],
                    default=['sequence', 'identity', 'geometry'])
    ap.add_argument('--preflight', action='store_true'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists(): raise RuntimeError('Output exists; choose a new run directory')
    files = [__file__, r8.__file__, r8.r7.__file__, r8.selection.__file__,
             str(Path(__file__).parent / 'gpu_mlm_pretrain.py'), a.geometry]
    save_json(out / 'manifest.json', dict(cfg=vars(a), code={p: digest(p) for p in files},
        started_unix=time.time(), retrospective=True, full_panel_targets=18,
        selection='unchanged selection.py; count and fraction both reported',
        contact_supervision=False, predicted_structures=False))
    z = np.load(a.features, allow_pickle=True); features = z['L33']; ss = z['seqs']
    assert len(ss) == len(set(map(str, ss)))
    fmap = {str(s): i for i, s in enumerate(ss)}
    raw = r8.r7.load_raw()
    try:
        for seed in a.seeds: run_seed(a, seed, out, features, fmap, raw)
    except BaseException as exc:
        save_json(out / 'status.json', dict(status='failed', error=repr(exc), updated_unix=time.time()))
        raise
    save_json(out / 'status.json', dict(status='preflight_passed' if a.preflight else 'completed',
                                      seeds=a.seeds, modes=a.modes, updated_unix=time.time()))
    print('ALL_DONE', flush=True)


if __name__ == '__main__': main()
