"""Check generator loading, causality and candidate selection."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'generator'))
from model import load, configure_dpo
from selection_rules import select


def main():
    torch.set_num_threads(2)
    reference = np.load(ROOT / 'tests/generator_expected.npz')
    x = torch.tensor(reference['tokens'], dtype=torch.long)
    y = torch.tensor(reference['conditions'], dtype=torch.float32)
    targets = torch.full((len(x),), 4, dtype=torch.long)
    for seed in range(3):
        model, ck = load(ROOT / f'generator/checkpoints/seed{seed}.pt')
        assert ck['update_step'] == 1000
        assert sum(p.numel() for p in model.parameters()) == 1949048
        configure_dpo(model)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 617574
        source, _ = load(ROOT / f'generator/initialization/seed{seed}.pt')
        for name, value in model.state_dict().items():
            if '.base.' in name or name.startswith('router.'):
                assert torch.equal(value, source.state_dict()[name]), name
        with torch.inference_mode():
            for flag in (False, True):
                flags = torch.full((len(x),), flag, dtype=torch.bool)
                out = model(x, y, targets, flags)
                np.testing.assert_allclose(out.numpy(), reference[f'seed{seed}_{int(flag)}'], atol=3e-5, rtol=1e-5)
                altered = x.clone()
                altered[:, 5:] = (altered[:, 5:] + 1) % 20
                torch.testing.assert_close(out[:, :6], model(altered, y, targets, flags)[:, :6], atol=0, rtol=0)
                if not flag:
                    torch.testing.assert_close(out, source(x, y, targets, flags), atol=0, rtol=0)
        print(f'Generator seed {seed}: checkpoint, fixed scope, reference outputs and causality passed.')
    sequences = np.array(['AAAAAAAAAA', 'CCCCCCCCCC', 'DDDDDDDDDD'])
    profiles = np.zeros((3, 18), np.float32)
    profiles[:, 4] = [1.1, 1.5, 1.5]
    profiles[:, 0] = [-3, .9, .9]
    assert select(sequences, profiles, 3).tolist() == [1, 2, 0]
    print('Bottleneck selection and stable sequence ties passed.')


if __name__ == '__main__':
    main()
