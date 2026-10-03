"""Evaluate generator NLL on the development split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from model import load, TARGET, STOP

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate(model, data, batch=256):
    device = next(model.parameters()).device
    values = {}
    for mode in ('conditional', 'unconditional'):
        rows = []
        for lo in range(0, len(data['tokens']), batch):
            x = torch.as_tensor(data['tokens'][lo:lo + batch], device=device, dtype=torch.long)
            y = torch.as_tensor(data['conditions'][lo:lo + batch], device=device, dtype=torch.float32)
            targets = torch.full((len(x),), TARGET, device=device, dtype=torch.long)
            flags = torch.full((len(x),), mode == 'conditional', device=device, dtype=torch.bool)
            labels = torch.cat([x, torch.full((len(x), 1), STOP, device=device, dtype=torch.long)], 1)
            loss = F.cross_entropy(model(x, y, targets, flags).flatten(0, 1), labels.flatten(), reduction='none').reshape(len(x), -1).mean(1)
            rows.extend(loss.cpu().tolist())
        values[mode] = np.asarray(rows)
    labels = data['labels']
    selective = (labels[:, TARGET] > 1) & (labels[:, TARGET] > np.delete(labels, TARGET, 1).max(1))
    result = {f'{k}_nll': float(v.mean()) for k, v in values.items()}
    result['joint_nll'] = (result['conditional_nll'] + result['unconditional_nll']) / 2
    result['selective_nll'] = float(values['conditional'][selective].mean()) if selective.any() else None
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/development.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = load(args.checkpoint, args.device)
    result = evaluate(model, np.load(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
