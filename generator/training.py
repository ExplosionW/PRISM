"""Fixed supervised stages of PRISM generator training."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import Attention, NormGenerator, MixtureGenerator, TARGET, STOP
from evaluate_checkpoint import evaluate

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'configs/training.json'


class ProfileExpert(nn.Module):
    """Trainable conditional decoder with a frozen unconditional decoder."""
    def __init__(self, variant):
        super().__init__()
        if variant not in ('profile', 'competition'):
            raise ValueError('Unknown expert.')
        self.variant = variant
        self.base = NormGenerator().requires_grad_(False).eval()
        self.student = NormGenerator()
        self.enzyme = nn.Embedding(18, 64)
        self.profile = nn.Linear(3, 64)
        self.cross = nn.ModuleList([Attention() for _ in range(3)])
        self.qnorm = nn.ModuleList([nn.LayerNorm(64, eps=.001) for _ in range(3)])
        self.gates = nn.Parameter(torch.zeros(3))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, tokens, y, target, conditional):
        unconditional = self.base(tokens, y, target, torch.zeros_like(conditional))
        b = self.student.base
        x = torch.cat([b.cond(y)[:, None], b.emb(tokens)], 1)
        x = F.dropout(x * 8 + b.pos[None, :x.shape[1]], .25, self.training)
        yt = y.gather(1, target[:, None])
        features = ([y, yt - y, F.one_hot(target, 18).to(y.dtype)]
                    if self.variant == 'competition'
                    else [y, torch.zeros_like(y), torch.zeros_like(y)])
        memory = self.enzyme(torch.arange(18, device=y.device))[None].expand(len(y), -1, -1)
        memory = memory + self.profile(torch.stack(features, -1))
        for i, block in enumerate(b.blocks):
            u = block.ln(x)
            x = x + block.att(u, u, True)
            x = x + self.gates[i].tanh() * self.cross[i](self.qnorm[i](x), memory)
            x = x + block.ff(block.ln2(x))
        return torch.where(conditional[:, None, None], b.out(self.student.finalnorm(x)), unconditional)


class TrainingMixture(MixtureGenerator):
    def components(self, tokens, y, target, conditional):
        a, b = [expert(tokens, y, target, conditional) for expert in self.experts]
        start = torch.full((len(tokens), 1), 20, device=tokens.device, dtype=torch.long)
        previous = torch.cat([start, tokens], 1)
        features = torch.cat([F.one_hot(previous, 22).to(y.dtype),
                              y[:, None].expand(-1, previous.shape[1], -1)], -1)
        weights = F.log_softmax(self.router(features) * 0, -1)
        logp = torch.logsumexp(torch.stack([F.log_softmax(a, -1), F.log_softmax(b, -1)], -2)
                               + weights[..., None], -2)
        return torch.where(conditional[:, None, None], logp, a), a, b

    def forward(self, *args):
        return self.components(*args)[0]


def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, payload):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def model_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def read_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)['model']


def initialize(stage, seed, sources):
    reset_seed(seed)
    if stage in ('pretrain', 'refine'):
        net = NormGenerator()
        if stage == 'refine':
            net.load_state_dict(read_state(sources[0]), strict=True)
        return net
    if stage in ('profile', 'competition'):
        # Preserve the original source-loader construction/RNG order.
        source = NormGenerator()
        source.load_state_dict(read_state(sources[0]), strict=True)
        net = ProfileExpert(stage)
        net.base.load_state_dict(source.state_dict(), strict=True)
        net.student.load_state_dict(source.state_dict(), strict=True)
        return net
    if stage != 'supervised':
        raise ValueError('Unknown training stage.')
    sources_loaded = []
    for variant, path in zip(('profile', 'competition'), sources):
        source = ProfileExpert(variant)
        source.load_state_dict(read_state(path), strict=True)
        sources_loaded.append(source)
    net = TrainingMixture()
    for expert, source in zip(net.experts, sources_loaded):
        result = expert.load_state_dict(source.state_dict(), strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith('channel_mod.') or '.rel_' in k for k in result.missing_keys)
    return net


def optimizer(net, stage):
    if stage in ('pretrain', 'refine'):
        return torch.optim.Adam(net.parameters(), lr=5e-5, eps=1e-7)
    if stage in ('profile', 'competition'):
        return torch.optim.Adam([
            {'params': net.student.parameters(), 'lr': 5e-5},
            {'params': [p for n, p in net.named_parameters() if not n.startswith(('base.', 'student.'))], 'lr': 5e-4},
        ], eps=1e-7)
    groups = [[], [], []]
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        local = '.'.join(name.split('.')[2:]) if name.startswith('experts.') else name
        index = (2 if name.startswith('router.') or local.startswith('channel_mod.') or '.rel_' in local
                 else 0 if local.startswith('student.') else 1)
        groups[index].append(param)
    return torch.optim.Adam([{'params': g, 'lr': lr} for g, lr in zip(groups, (5e-6, 5e-5, 5e-4))], eps=1e-7)


def nll(logits, tokens):
    labels = torch.cat([tokens, torch.full((len(tokens), 1), STOP, device=tokens.device, dtype=torch.long)], 1)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').reshape(len(tokens), 11).mean(1)


def objective(net, stage, x, y, flags, positive, fraction):
    targets = torch.full((len(x),), TARGET, device=x.device, dtype=torch.long)
    if stage != 'supervised':
        return nll(net(x, y, targets, flags), x).mean()
    _, a, b = net.components(x, y, targets, flags)
    losses = .5 * (nll(a, x) + nll(b, x))
    weights = 1 + (flags & positive).to(losses.dtype)
    return (losses * weights).mean() / (1 + .5 * fraction)


def frozen_parameters(net):
    return {n: p.detach().cpu().clone() for n, p in net.named_parameters() if not p.requires_grad}


def fit_stage(net, stage, seed, fit, development, out, spec, resume=False, after_epoch=None):
    """Run a fixed stage; the callback supports interruption tests, not a CLI sweep."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    best_path, resume_path = out / 'best.pt', out / 'resume.pt'
    if (out / 'done.json').exists():
        done = json.loads((out / 'done.json').read_text())
        for name, digest in done['sha256'].items():
            if sha(out / name) != digest:
                raise ValueError('Completed training stage changed.')
        return best_path
    device = next(net.parameters()).device
    x = torch.tensor(fit['tokens'], device=device, dtype=torch.long)
    y = torch.tensor(fit['conditions'], device=device)
    labels = fit['labels']
    positive = torch.tensor((labels[:, TARGET] > 1) & (labels[:, TARGET] > np.delete(labels, TARGET, 1).max(1)), device=device)
    fraction = float(positive.float().mean())
    frozen = frozen_parameters(net)
    opt = optimizer(net, stage)
    net.eval()
    initial = evaluate(net, development)
    best, best_epoch, start, step, history = initial['joint_nll'], 0, 1, 0, []
    def payload(epoch):
        result = dict(model=model_state(net), stage=stage, seed=seed, epoch=epoch)
        if stage == 'supervised':
            result.update(mixture_mode='uniform', training_objective='expert', training_dropout=True, selective_weight=2)
        return result
    if resume_path.exists():
        if not resume:
            raise FileExistsError('Use --resume or a new output directory.')
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        if saved['stage'] != stage or saved['seed'] != seed or saved['spec'] != spec:
            raise ValueError('Resume stage, seed or schedule changed.')
        net.load_state_dict(saved['model'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, step, history = saved['epoch'] + 1, saved['step'], saved['history']
        best, best_epoch = saved['best'], saved['best_epoch']
    else:
        save(best_path, payload(0))
    for epoch in range(start, spec['epochs'] + 1):
        net.train()
        rng = np.random.default_rng(spec['batch_seed'] + seed * 1000 + epoch)
        order = rng.permutation(len(x))
        loss_sum = 0.
        for lo in range(0, len(x), 128):
            ix = order[lo:lo + 128]
            flags = torch.tensor(rng.random(len(ix)) < .5, device=device)
            step += 1
            if stage == 'pretrain':
                for group in opt.param_groups:
                    group['lr'] = 64**-.5 * min(step**-.5, step * 4000**-1.5)
            opt.zero_grad(set_to_none=True)
            loss = objective(net, stage, x[ix], y[ix], flags, positive[ix], fraction)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(ix)
        net.eval()
        validation = evaluate(net, development)
        if validation['joint_nll'] < best:
            best, best_epoch = validation['joint_nll'], epoch
            save(best_path, payload(epoch))
        current = net.state_dict()
        if any(not torch.equal(current[n].cpu(), value) for n, value in frozen.items()):
            raise RuntimeError('A frozen parameter changed.')
        row = dict(epoch=epoch, loss=loss_sum / len(x), **validation)
        history.append(row)
        save(resume_path, dict(**payload(epoch), optimizer=opt.state_dict(), rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                              step=step, history=history, best=best, best_epoch=best_epoch, spec=spec))
        write_json(out / 'history.json', history)
        print(json.dumps(dict(stage=stage, seed=seed, **row)), flush=True)
        if after_epoch is not None:
            after_epoch(epoch)
    save(out / 'last.pt', payload(spec['epochs']))
    write_json(out / 'done.json', dict(stage=stage, seed=seed, epochs=spec['epochs'], best_epoch=best_epoch,
                                       best_joint_nll=best, sha256={name: sha(out / name) for name in ('best.pt', 'last.pt', 'resume.pt')}))
    return best_path
