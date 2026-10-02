"""Reproduce fixed-1000-update Pareto-DPO from the packaged generator initialization.

Preferences are computational PRISM predictions, not experimental measurements.
Only the original DPO parameter subset is updated. No best-epoch selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from model import AA, TARGET, STOP, load, configure_dpo

ROOT = Path(__file__).resolve().parent


def logps(model, x, y):
    targets = torch.full((len(x),), TARGET, device=x.device, dtype=torch.long)
    flags = torch.ones(len(x), device=x.device, dtype=torch.bool)
    labels = torch.cat([x, torch.full((len(x), 1), STOP, device=x.device, dtype=torch.long)], 1)
    return F.log_softmax(model(x, y, targets, flags), -1).gather(-1, labels[:, :, None]).squeeze(-1).sum(1)


def pair_forward(model, chosen, rejected, conditions):
    result = logps(model, torch.cat([chosen, rejected]), torch.cat([conditions, conditions]))
    return result[:len(chosen)], result[len(chosen):]


def dpo_loss(chosen, rejected, reference):
    return F.softplus(-.1 * ((chosen - rejected) - (reference[:, 0] - reference[:, 1]))).mean()


def load_pairs(device):
    rows = pd.read_csv(ROOT / 'data/preferences.csv')
    requests = json.loads((ROOT / 'data/requests.json').read_text())
    def encode(sequences):
        return torch.tensor([[AA.index(c) for c in s] for s in sequences], device=device, dtype=torch.long)
    y = torch.tensor([requests[str(i)] for i in rows.fit_condition_index], device=device, dtype=torch.float32)
    return rows, encode(rows.chosen), encode(rows.rejected), y


@torch.no_grad()
def reference_probs(model, chosen, rejected, conditions):
    rows = []
    for lo in range(0, len(chosen), 256):
        a, b = pair_forward(model, chosen[lo:lo + 256], rejected[lo:lo + 256], conditions[lo:lo + 256])
        rows.append(torch.stack([a, b], 1))
    return torch.cat(rows)


def save_checkpoint(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026111400 + args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026111400 + args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, original = load(ROOT / f'initialization/seed{args.seed}.pt', args.device)
    configure_dpo(model)
    rows, chosen, rejected, conditions = load_pairs(args.device)
    train_indices = np.flatnonzero(rows.split.to_numpy() == 'train')
    reference = reference_probs(model, chosen, rejected, conditions)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6, eps=1e-7)
    args.output.mkdir(parents=True, exist_ok=True)
    resume_path = args.output / 'resume.pt'
    start = 1
    if args.resume:
        ck = torch.load(resume_path, map_location='cpu', weights_only=False)
        if ck['seed'] != args.seed:
            raise ValueError('Resume seed differs from the requested seed.')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['rng'])
        if str(args.device).startswith('cuda'):
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start = ck['update_step'] + 1
    elif resume_path.exists() or (args.output / 'last.pt').exists():
        raise FileExistsError('Use --resume or a new output directory.')
    def snapshot(step):
        # Keep architecture metadata; source paths are not needed for loading.
        return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    mixture_mode=original['mixture_mode'], seed=args.seed,
                    update_step=step, epoch=step, training_objective='pareto_dpo', training_dropout=False)
    for step in range(start, 1001):
        rng = np.random.default_rng(2026111400 + args.seed * 10000 + step)
        ix = rng.choice(train_indices, 128, replace=True)
        optimizer.zero_grad(set_to_none=True)
        a, b = pair_forward(model, chosen[ix], rejected[ix], conditions[ix])
        loss = dpo_loss(a, b, reference[ix])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite DPO loss.')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise FloatingPointError('Non-finite gradient.')
        optimizer.step()
        if step % 100 == 0:
            payload = snapshot(step)
            payload.update(optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            save_checkpoint(resume_path, payload)
            print(json.dumps(dict(step=step, loss=float(loss.detach()))), flush=True)
    save_checkpoint(args.output / 'last.pt', snapshot(1000))


if __name__ == '__main__':
    main()
