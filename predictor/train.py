"""Train the fixed three-stage PRISM predictor recipe; no ablation options.

Task layers are initialized afresh. ESM is frozen; the task LM starts from its
original pretrained initialization, never from a released final predictor.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import AblatedBase, AblatedAdapter, DeCleaveLM, encode
from losses import PRISMLoss, validation_loss

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name('configs') / 'prism.json'
STAGES = ('task_base', 'lm_stage', 'adapter')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def save(path, data):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(data, tmp)
    tmp.replace(path)


def state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def check_assets(cfg, lm_path):
    if not lm_path.is_file():
        raise FileNotFoundError(
            f'Missing original task-LM initialization: {lm_path}. '
            'Supply --lm-init (or PREDICTOR_LM_INIT in run.sh). '
            f'Expected SHA256: {cfg["task_lm_sha256"]}. '
            'Released final predictor weights cannot substitute for this file.')
    if sha(lm_path) != cfg['task_lm_sha256']:
        raise ValueError('Task-LM initialization SHA256 does not match the historical recipe.')
    inputs = ROOT / cfg['fixed_inputs']
    if sha(inputs) != cfg['fixed_inputs_sha256']:
        raise ValueError('Fixed predictor inputs changed.')
    return inputs


def historical_pair_count(labels):
    """Retain the historical unused pair-index RNG draws, without pair forwards.

    Pair ranking was replaced by minibatch ListNet in the original 7.09 recipe.
    The old loop still drew 256 indices per batch. Only the original pool size
    affects these draws; pair identities no longer enter the loss.
    """
    count = 0
    for a in labels.T:
        ix = np.where(a >= np.quantile(a, .5))[0]
        if len(ix) < 20:
            continue
        edges = np.quantile(a[ix], np.linspace(0, 1, 6))
        for lo, hi in zip(edges[:-1], edges[1:]):
            n = int(((a[ix] >= lo) & (a[ix] <= hi)).sum())
            if n >= 8:
                count += min(max(1, int(.25 * n)), 400)
    if not count:
        raise ValueError('Empty historical pair pool.')
    return count


class Context:
    def __init__(self, cfg, feature_dir, fixed_path, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.ids = json.loads((ROOT / 'data/split_ids.json').read_text())
        fixed = np.load(fixed_path, allow_pickle=False)
        if fixed['targets'].tolist() != self.ids['targets']:
            raise ValueError('Fixed-input enzyme order mismatch.')
        self.emb = fixed['enzyme_embeddings'].copy()
        self.geo = [torch.tensor(fixed[k], dtype=torch.float32, device=self.device)
                    for k in ('gf', 'gd', 'gm', 'gh')]
        self.z, self.f, self.y, self.feature_metadata = {}, {}, {}, {}
        for split in ('train', 'val'):
            # Held-out labels are never loaded by the trainer.
            d = np.load(ROOT / f'data/benchmark/{split}.npz', allow_pickle=False)
            f = np.load(feature_dir / f'{split}.npz', allow_pickle=False)
            seq = d['sequences'].tolist()
            if seq != self.ids[split] or f['sequences'].tolist() != seq:
                raise ValueError(f'{split}: data/feature sequence order mismatch.')
            if d['labels'].shape != (len(seq), 18) or f['esm33'].shape != (len(seq), 10, 1280):
                raise ValueError(f'{split}: invalid data shape.')
            if not np.isfinite(d['labels']).all() or not np.isfinite(f['esm33']).all():
                raise ValueError(f'{split}: nonfinite labels/features.')
            meta = {k: str(f[k].item()) for k in ('encoder', 'revision', 'feature_precision')}
            if meta['encoder'] != cfg['esm_encoder'] or meta['revision'] != cfg['esm_revision']:
                raise ValueError(f'{split}: ESM identity/revision mismatch.')
            if meta['feature_precision'] not in ('bf16', 'fp32'):
                raise ValueError('Unknown feature precision.')
            self.feature_metadata[split] = meta
            self.z[split] = torch.tensor(encode(seq), device=self.device)
            self.f[split] = torch.as_tensor(f['esm33'].copy(), device=self.device)
            self.y[split] = torch.tensor(d['labels'].astype(np.float32), device=self.device)
        if self.feature_metadata['train'] != self.feature_metadata['val']:
            raise ValueError('Training and validation feature settings differ.')
        self.pair_count = historical_pair_count(self.y['train'].cpu().numpy())
        self.loss = PRISMLoss(self.y['train'])

    def data(self, split, ix):
        return self.z[split][ix], self.f[split][ix]


def construct(ctx, seed, lm_payload=None, previous=None):
    reset_seed(seed)
    lm = None
    if lm_payload is not None:
        # Construction also consumes RNG in the historical order.
        lm = DeCleaveLM(**lm_payload['cfg'])
        lm.load_state_dict(lm_payload['state'], strict=True)
    c = ctx.cfg['model']
    net = AblatedBase(18, L=10, d=c['dim'], m=c['pair_dim'], hid=c['hid'],
                      pdrop=c['dropout'], de=c['de'], Emb=ctx.emb, esm_dim=1280,
                      zdim=c['zdim'], kappa=c['kappa'], use_metric=True,
                      lm=lm, lm_freeze=False, lm_residual=True, arm='full').to(ctx.device)
    matched = []
    if previous is not None:
        own = net.state_dict()
        matched = [k for k, v in previous.items() if k in own and v.shape == own[k].shape]
        net.load_state_dict({**own, **{k: previous[k] for k in matched}}, strict=True)
    return net, matched


@torch.no_grad()
def cache_base(net, ctx, split):
    net.eval()
    rows = [[], [], []]
    for i in range(0, len(ctx.z[split]), 128):
        z, f = ctx.data(split, slice(i, i + 128))
        e = net.residues(z, f)
        hp = net.body(net.condition(e).flatten(0, 1)).reshape(len(e), 18, -1)
        raw = net.raw(hp)
        y = net.a[None] + net.g(net.body(e)) + raw - raw.mean(1, keepdim=True)
        for dest, value in zip(rows, (e, y, raw)):
            dest.append(value)
    return tuple(torch.cat(r) for r in rows)


@torch.no_grad()
def evaluate(net, ctx, cached=None):
    net.eval()
    batch = 128 if cached is not None else 512
    pred = []
    for i in range(0, len(ctx.y['val']), batch):
        ix = slice(i, i + batch)
        inputs = tuple(v[ix] for v in cached['val']) if cached is not None else ctx.data('val', ix)
        pred.append(net(*inputs))
    value = validation_loss(torch.cat(pred), ctx.y['val'])
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite validation loss.')
    return float(value)


def fit_stage(net, ctx, stage, out, seed, matched=(), cached=None, resume=False, after_epoch=None):
    """Fixed stage loop; after_epoch is used only by interruption/resume tests."""
    out.mkdir(parents=True, exist_ok=True)
    best_path, resume_path = out / 'best.pt', out / 'resume.pt'
    if (out / 'done.json').exists():
        result = json.loads((out / 'done.json').read_text())
        if sha(best_path) != result['best_sha256']:
            raise ValueError('Completed-stage checkpoint checksum mismatch.')
        net.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False)['state'])
        return net.eval(), result
    # Adapter initialization consumes RNG immediately before this loop in the
    # archived recipe; do not rewind its dropout stream after construction.
    if stage != 'adapter':
        reset_seed(seed)
    total, patience = ctx.cfg['max_epochs_per_stage'], ctx.cfg['patience']
    sampler = torch.Generator(device=ctx.device).manual_seed(seed)
    frozen = stage == 'lm_stage'
    adapter = stage == 'adapter'
    def set_trainable(freeze):
        for name, p in net.named_parameters():
            p.requires_grad_(not name.startswith('base.') if adapter else not freeze or name not in matched)
    def optimizer(remaining):
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.001, weight_decay=.001)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, remaining)
    set_trainable(frozen)
    opt, sch = optimizer(total)
    best, best_epoch, bad, start, history = float('inf'), 0, 0, 1, []
    if resume_path.exists():
        if not resume:
            raise FileExistsError('Use --resume or a new output directory.')
        r = torch.load(resume_path, map_location='cpu', weights_only=False)
        if r['stage'] != stage or r['seed'] != seed:
            raise ValueError('Resume stage/seed mismatch.')
        net.load_state_dict(r['state'])
        frozen = r['frozen']
        set_trainable(frozen)
        opt, sch = optimizer(total - 10 if stage == 'lm_stage' and r['epoch'] >= 10 else total)
        opt.load_state_dict(r['optimizer']); sch.load_state_dict(r['scheduler'])
        sampler.set_state(r['sampler'])
        torch.set_rng_state(r['cpu_rng']); np.random.set_state(r['numpy_rng'])
        if ctx.device.type == 'cuda':
            torch.cuda.set_rng_state_all(r['cuda_rng'])
        best, best_epoch, bad, history = r['best'], r['best_epoch'], r['bad'], r['history']
        start = r['epoch'] + 1
        if bad >= patience:
            start = total + 1
    elif adapter:
        best = evaluate(net, ctx, cached)
        history = [dict(epoch=0, val_loss=best)]
        save(best_path, dict(state=state(net), epoch=0, stage=stage, seed=seed))
    for epoch in range(start, total + 1):
        net.train()
        order = torch.randperm(len(ctx.y['train']), device=ctx.device, generator=sampler)
        losses = []
        for j in range(0, len(order), 128):
            ix = order[j:j + 128]
            torch.randint(ctx.pair_count, (256,), device=ctx.device, generator=sampler)
            inputs = tuple(v[ix] for v in cached['train']) if adapter else ctx.data('train', ix)
            opt.zero_grad(set_to_none=True)
            loss = ctx.loss(net(*inputs), ctx.y['train'][ix])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            if adapter:
                norm = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 10.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite adapter gradient.')
            opt.step()
            losses.append(float(loss.detach()))
        sch.step()
        if frozen and epoch == 10:
            frozen = False
            set_trainable(False)
            opt, sch = optimizer(total - 10)
        val = evaluate(net, ctx, cached)
        if val < best - 1e-7:
            best, best_epoch, bad = val, epoch, 0
            save(best_path, dict(state=state(net), epoch=epoch, stage=stage, seed=seed))
        else:
            bad += 1
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val, lr=sch.get_last_lr()[0])
        history.append(row)
        save(resume_path, dict(state=state(net), optimizer=opt.state_dict(), scheduler=sch.state_dict(),
                              sampler=sampler.get_state(), cpu_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if ctx.device.type == 'cuda' else [],
                              numpy_rng=np.random.get_state(), epoch=epoch, frozen=frozen,
                              best=best, best_epoch=best_epoch, bad=bad, history=history, stage=stage, seed=seed))
        write_json(out / 'history.json', history)
        print(json.dumps(dict(stage=stage, seed=seed, **row)), flush=True)
        if after_epoch is not None:
            after_epoch(epoch)
        if bad >= patience:
            break
    # Also works when resuming an already stopped last epoch.
    write_json(out / 'history.json', history)
    net.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False)['state'])
    result = dict(stage=stage, best_epoch=best_epoch, best_val=best,
                  epochs=history[-1]['epoch'], best_sha256=sha(best_path))
    write_json(out / 'done.json', result)
    return net.eval(), result


def run(args, cfg, fixed_path):
    out = args.output.resolve()
    # Training may never write to packaged data, models, or their ancestors.
    for p in (ROOT / 'predictor', ROOT / 'generator', ROOT / 'data', ROOT / 'tests'):
        if out == p or p in out.parents or out in p.parents:
            raise ValueError('Choose a separate training output directory, e.g. outputs/predictor/training/seed0.')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Output directory is not empty; use --resume or a new directory.')
    paths = [CONFIG, Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('losses.py'),
             args.lm_init, fixed_path, ROOT / 'data/split_ids.json']
    paths += [ROOT / f'data/benchmark/{s}.npz' for s in ('train', 'val')]
    paths += [args.features / f'{s}.npz' for s in ('train', 'val')]
    manifest = dict(seed=args.seed, device=str(args.device), torch_version=str(torch.__version__),
                    cuda_version=torch.version.cuda, inputs={str(p.resolve()): sha(p) for p in paths})
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['signature'] = signature
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text())['signature'] != signature:
            raise ValueError('Resume inputs, code, device or environment changed; use a new directory.')
    elif args.resume:
        raise FileNotFoundError('No matching training run to resume.')
    else:
        write_json(out / 'config.json', manifest)
    if (out / 'done.json').exists():
        done = json.loads((out / 'done.json').read_text())
        if sha(out / 'final.pt') != done['sha256']:
            raise ValueError('Final checkpoint changed.')
        print('Training is already complete.', flush=True)
        return
    ctx = Context(cfg, args.features, fixed_path, args.device)
    lm = torch.load(args.lm_init, map_location='cpu', weights_only=False)
    base, _ = construct(ctx, args.seed)
    base, first = fit_stage(base, ctx, 'task_base', out / 'task_base', args.seed, resume=args.resume)
    previous = state(base)
    del base
    base, matched = construct(ctx, args.seed, lm, previous)
    del previous, lm
    base, second = fit_stage(base, ctx, 'lm_stage', out / 'lm_stage', args.seed, matched, resume=args.resume)
    base.eval().requires_grad_(False)
    base.zero_grad(set_to_none=True)
    reset_seed(args.seed)
    net = AblatedAdapter(base, ctx.geo, arm='shared_query').to(ctx.device)
    cached = {s: cache_base(base, ctx, s) for s in ('train', 'val')}
    net, third = fit_stage(net, ctx, 'adapter', out / 'adapter', args.seed, cached=cached, resume=args.resume)
    save(out / 'final.pt', dict(state=state(net), model_cfg=cfg['model'], ids=ctx.ids, seed=args.seed,
                               recipe=dict(model='new:adapter:shared_query'), algorithm='prop7_listnet_batch',
                               training_signature=signature))
    write_json(out / 'done.json', dict(seed=args.seed, stages=[first, second, third],
                                       sha256=sha(out / 'final.pt'), feature_settings=ctx.feature_metadata,
                                       validation_loss=third['best_val']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--features', type=Path, default=ROOT / 'outputs/features')
    p.add_argument('--lm-init', type=Path, default=ROOT / 'predictor/initialization/lmw_256_8_iso.pt')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-inputs', action='store_true', help='Check fixed initialization assets without training or extracting ESM features.')
    args = p.parse_args()
    cfg = json.loads(CONFIG.read_text())
    fixed_path = check_assets(cfg, args.lm_init)
    if args.check_inputs:
        print('Original task-LM and fixed predictor inputs verified.')
        return
    if args.output is None:
        p.error('--output is required for training.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable; select --device cpu explicitly.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args, cfg, fixed_path)


if __name__ == '__main__':
    main()
