"""PRISM generator with profile and competition experts."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AA = 'ACDEFGHIKLMNPQRSTVWY'
START, STOP, TARGET = 20, 21, 4


class Attention(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, 6 * d)
        self.k = nn.Linear(d, 6 * d)
        self.v = nn.Linear(d, 6 * d)
        self.out = nn.Linear(6 * d, d)

    def forward(self, x, memory, causal=False):
        batch = len(x)
        q = self.q(x).reshape(batch, -1, 6, self.d).transpose(1, 2)
        k = self.k(memory).reshape(batch, -1, 6, self.d).transpose(1, 2)
        v = self.v(memory).reshape(batch, -1, 6, self.d).transpose(1, 2)
        scores = (q / math.sqrt(self.d)) @ k.transpose(-1, -2)
        if causal:
            mask = torch.ones(x.shape[1], memory.shape[1], device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        weights = F.dropout(scores.softmax(-1), .25, self.training)
        return self.out((weights @ v).transpose(1, 2).reshape(batch, -1, 6 * self.d))


class RelativeAttention(Attention):
    def __init__(self, d=64):
        super().__init__(d)
        self.rel_base = nn.Parameter(torch.zeros(6, 21))
        self.rel_net = nn.Sequential(nn.Linear(36, 16), nn.SiLU(), nn.Linear(16, 126))
        nn.init.zeros_(self.rel_net[-1].weight)
        nn.init.zeros_(self.rel_net[-1].bias)

    def forward(self, x, memory, causal=False, features=None, enabled=True):
        batch = len(x)
        q = self.q(x).reshape(batch, -1, 6, self.d).transpose(1, 2)
        k = self.k(memory).reshape(batch, -1, 6, self.d).transpose(1, 2)
        v = self.v(memory).reshape(batch, -1, 6, self.d).transpose(1, 2)
        scores = (q / math.sqrt(self.d)) @ k.transpose(-1, -2)
        table = self.rel_base[None] + self.rel_net(features).reshape(batch, 6, 21)
        delta = (torch.arange(x.shape[1], device=x.device)[:, None]
                 - torch.arange(memory.shape[1], device=x.device)[None]).clamp(-10, 10) + 10
        scores = scores + int(enabled) * table[:, :, delta]
        if causal:
            mask = torch.ones(x.shape[1], memory.shape[1], device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        weights = F.dropout(scores.softmax(-1), .25, self.training)
        return self.out((weights @ v).transpose(1, 2).reshape(batch, -1, 6 * self.d))


class Block(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.att = Attention(d)
        self.ln = nn.LayerNorm(d, eps=.001)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Dropout(.1))
        self.ln2 = nn.LayerNorm(d, eps=.001)


class Decoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.Embedding(22, d)
        self.cond = nn.Linear(18, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(3)])
        self.out = nn.Linear(d, 22)
        angles = np.arange(32)[:, None] / (10000 ** (np.arange(d // 2)[None, :] / (d // 2)))
        self.register_buffer('pos', torch.tensor(np.concatenate([np.sin(angles), np.cos(angles)], 1), dtype=torch.float32), persistent=False)


class NormGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = Decoder()
        self.finalnorm = nn.LayerNorm(64, eps=.001)

    def forward(self, tokens, y, target, conditional):
        b = self.base
        start = torch.full((len(tokens),), START, device=tokens.device, dtype=torch.long)
        prefix = torch.where(conditional[:, None], b.cond(y), b.emb(start))
        x = torch.cat([prefix[:, None], b.emb(tokens)], 1)
        x = F.dropout(x * 8 + b.pos[None, :x.shape[1]], .25, self.training)
        for block in b.blocks:
            u = block.ln(x)
            x = x + block.att(u, u, True)
            x = x + block.ff(block.ln2(x))
        return b.out(self.finalnorm(x))


class SourceJointGenerator(nn.Module):
    def __init__(self, source_variant):
        super().__init__()
        self.source_variant = self.variant = source_variant
        self.joint_mode = 'joint'
        self.base = NormGenerator()
        self.base.requires_grad_(False)
        self.base.eval()
        self.student = NormGenerator()
        self.enzyme = nn.Embedding(18, 64)
        self.profile = nn.Linear(3, 64)
        self.cross = nn.ModuleList([Attention() for _ in range(3)])
        self.qnorm = nn.ModuleList([nn.LayerNorm(64, eps=.001) for _ in range(3)])
        self.gates = nn.Parameter(torch.zeros(3))
        for block in self.student.base.blocks:
            block.att = RelativeAttention()
        self.channel_mod = nn.ModuleList([
            nn.Sequential(nn.Linear(36, 16), nn.SiLU(), nn.Linear(16, 128)) for _ in range(3)
        ])
        for module in self.channel_mod:
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

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
        if self.source_variant == 'competition':
            memory_features = [y, yt - y, F.one_hot(target, 18).float()]
        else:
            memory_features = [y, torch.zeros_like(y), torch.zeros_like(y)]
        memory = self.enzyme(torch.arange(18, device=y.device))[None].expand(len(y), -1, -1)
        memory = memory + self.profile(torch.stack(memory_features, -1))
        features = torch.cat([y, yt - y], 1)
        for i, block in enumerate(b.blocks):
            u = block.ln(x)
            x = x + block.att(u, u, True, features, True)
            x = x + self.gates[i].tanh() * self.cross[i](self.qnorm[i](x), memory)
            u = block.ln2(x)
            gamma, beta = self.channel_mod[i](features).chunk(2, -1)
            x = x + block.ff((1 + gamma[:, None]) * u + beta[:, None])
        return torch.where(conditional[:, None, None], b.out(self.student.finalnorm(x)), unconditional)


class MixtureGenerator(nn.Module):
    def __init__(self, mode='uniform'):
        super().__init__()
        if mode != 'uniform':
            raise ValueError('This release implements the fixed equal-weight PRISM generator.')
        self.mixture_mode = mode
        self.experts = nn.ModuleList([SourceJointGenerator('profile'), SourceJointGenerator('competition')])
        self.router = nn.Sequential(nn.Linear(40, 16), nn.SiLU(), nn.Linear(16, 2))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    @property
    def base(self):
        return self.experts[0].base

    def forward(self, tokens, y, target, conditional):
        a, b = [expert(tokens, y, target, conditional) for expert in self.experts]
        start = torch.full((len(tokens), 1), START, device=tokens.device, dtype=torch.long)
        previous = torch.cat([start, tokens], 1)
        features = torch.cat([F.one_hot(previous, 22).to(y.dtype),
                              y[:, None].expand(-1, previous.shape[1], -1)], -1)
        weights = F.log_softmax(self.router(features) * 0, -1)
        conditional_logp = torch.logsumexp(torch.stack([F.log_softmax(a, -1), F.log_softmax(b, -1)], -2)
                                          + weights[..., None], -2)
        return torch.where(conditional[:, None, None], conditional_logp, a)


def load(path, device='cpu'):
    """Load a trusted PRISM generator checkpoint."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=False)
    model = MixtureGenerator(checkpoint['mixture_mode'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint


def configure_dpo(model):
    """Freeze decoder backbones and router for DPO."""
    for name, param in model.named_parameters():
        param.requires_grad_('.base.' not in name and not name.startswith('router.'))
    return model.eval()
