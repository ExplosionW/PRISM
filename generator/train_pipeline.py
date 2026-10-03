"""Train the PRISM generator from random initialization through Pareto-DPO."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from training import ROOT, CONFIG, sha, write_json, initialize, fit_stage


def validate_data():
    fit = dict(np.load(ROOT / 'data/fit.npz', allow_pickle=False))
    dev = dict(np.load(ROOT / 'data/development.npz', allow_pickle=False))
    for data, size in ((fit, 10954), (dev, 1434)):
        if len(data['sequences']) != size or len(set(data['sequences'])) != size:
            raise ValueError('Invalid split size or duplicate sequences.')
        if data['tokens'].shape != (size, 10) or data['conditions'].shape != (size, 18) or data['labels'].shape != (size, 18):
            raise ValueError('Invalid data shapes.')
        if not np.isfinite(data['conditions']).all() or not np.isfinite(data['labels']).all():
            raise ValueError('Nonfinite labels or conditions.')
        if not np.isin(data['tokens'], np.arange(20)).all():
            raise ValueError('Noncanonical training residues.')
    if set(fit['sequences']) & set(dev['sequences']):
        raise ValueError('Fitting and development sequences overlap.')
    return fit, dev


def run(args):
    out = args.output.resolve()
    for protected in ('generator', 'predictor', 'data', 'evaluation', 'tests', 'requirements', 'assets'):
        path = ROOT.parent / protected
        if out == path or path in out.parents or out in path.parents:
            raise ValueError('Use a separate output directory, e.g. outputs/generator/training/seed0.')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Use --resume or a new output directory.')
    cfg = json.loads(CONFIG.read_text())
    fit, development = validate_data()
    paths = [CONFIG, Path(__file__), ROOT / 'model.py', ROOT / 'training.py', ROOT / 'train.py',
             ROOT / 'evaluate_checkpoint.py', ROOT / 'configs/pareto_dpo.json']
    paths += [ROOT / 'data' / name for name in ('fit.npz', 'development.npz', 'preferences.csv', 'requests.json')]
    manifest = dict(seed=args.seed, device=args.device, torch=str(torch.__version__), cuda=torch.version.cuda,
                    inputs={str(p.relative_to(ROOT)): sha(p) for p in paths})
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != manifest:
            raise ValueError('Resume code, data, device or environment changed.')
    elif args.resume:
        raise FileNotFoundError('No matching training run to resume.')
    else:
        write_json(out / 'config.json', manifest)
    if (out / 'done.json').exists():
        if sha(out / 'final.pt') != json.loads((out / 'done.json').read_text())['sha256']:
            raise ValueError('Final checkpoint changed.')
        print('PRISM generator training is already complete.')
        return
    best = {}
    for stage, spec in cfg['stages'].items():
        sources = ([] if stage == 'pretrain' else [best['pretrain']] if stage == 'refine'
                   else [best['refine']] if stage in ('profile', 'competition')
                   else [best['profile'], best['competition']])
        model = initialize(stage, args.seed, sources).to(args.device)
        best[stage] = fit_stage(model, stage, args.seed, fit, development, out / stage, spec, resume=args.resume)
        del model
    dpo_out = out / 'dpo'
    command = [sys.executable, str(ROOT / 'train.py'), '--seed', str(args.seed), '--device', args.device,
               '--initialization', str(best['supervised']), '--output', str(dpo_out)]
    if args.resume and (dpo_out / 'resume.pt').exists():
        command.append('--resume')
    subprocess.run(command, check=True)
    tmp = out / 'final.tmp.pt'
    shutil.copyfile(dpo_out / 'last.pt', tmp)
    tmp.replace(out / 'final.pt')
    write_json(out / 'done.json', dict(seed=args.seed, sha256=sha(out / 'final.pt'),
                                      stages={stage: sha(path) for stage, path in best.items()},
                                      dpo_updates=1000))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu explicitly.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args)


if __name__ == '__main__':
    main()
