"""Generate peptide sequences with PRISM."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from model import AA, START, TARGET, load
from sampling import sample

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026111201, help='Sampling seed, independent of checkpoint training seed.')
    p.add_argument('--mode', choices=['selective', 'unconditional'], default='selective')
    p.add_argument('--temperature', type=float, help='Default: 1.2 conditional, 1.0 unconditional.')
    p.add_argument('--per-template', type=int, default=400)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.per_template < 1:
        p.error('--per-template must be positive')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = load(args.checkpoint, args.device)
    templates = json.loads((ROOT / 'data/templates.json').read_text())
    fit = np.load(ROOT / 'data/fit.npz')
    lookup = {s: i for i, s in enumerate(fit['sequences'])}
    fit_indices = [lookup[s] for s in templates['sequences']]
    ix = np.repeat(fit_indices, args.per_template)
    conditions = np.repeat(np.asarray(templates['conditions'], np.float32), args.per_template, axis=0)
    conditional = args.mode != 'unconditional'
    if not conditional:
        conditions = np.zeros_like(conditions)
    temperature = args.temperature if args.temperature is not None else (1.2 if conditional else 1.)
    known = set((ROOT / 'data/known_sequences.txt').read_text().splitlines())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'sequence', 'condition_train_index', 'stopped_normally', 'raw_length', 'filter_reason'])
        for lo, tokens, lengths, stopped in sample(model, conditions, TARGET, conditional, temperature, args.seed, args.batch):
            for j, (row, length, stop) in enumerate(zip(tokens, lengths, stopped)):
                tok = row[:length]
                seq = ''.join(AA[i] if i < 20 else '$' if i == START else '*' for i in tok)
                reason = ('no_stop' if not stop else 'special_token' if np.any(tok >= 20)
                          else 'length' if length != 10 else 'known_sequence' if seq in known else '')
                writer.writerow([lo + j, seq, int(ix[lo + j]) if conditional else -1, bool(stop), int(length), reason])
    print(f'Saved {len(conditions)} raw attempts to {args.output}')


if __name__ == '__main__':
    main()
