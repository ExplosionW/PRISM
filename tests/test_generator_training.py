"""Small CPU checks for the fixed generator pipeline; no production training."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'generator'))
import model
import training as tr
import train as dpo
import train_pipeline as pipeline


def data(seed):
    rng = np.random.default_rng(seed)
    labels = rng.normal(size=(8, 18)).astype('float32')
    labels[:2, 4] = 5
    return dict(tokens=rng.integers(0, 20, (8, 10)), conditions=labels.copy(), labels=labels)


class GeneratorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a: self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.assert_nested_equal(x, y)
        else: self.assertEqual(a, b)

    def test_complete_supervised_chain_and_dpo_update(self):
        fit, dev = data(1), data(2)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            best = {}
            for stage in ('pretrain', 'refine', 'profile', 'competition', 'supervised'):
                sources = ([] if stage == 'pretrain' else [best['pretrain']] if stage == 'refine'
                           else [best['refine']] if stage in ('profile', 'competition')
                           else [best['profile'], best['competition']])
                net = tr.initialize(stage, 0, sources)
                before = tr.model_state(net)
                frozen = tr.frozen_parameters(net)
                best[stage] = tr.fit_stage(net, stage, 0, fit, dev, root / stage,
                                          dict(epochs=2, batch_seed=2026101800))
                self.assertFalse(net.training, 'Checkpoint selection must disable dropout.')
                self.assertTrue(any(not torch.equal(before[k], v) for k, v in net.state_dict().items()))
                for k, v in frozen.items(): self.assertTrue(torch.equal(net.state_dict()[k], v))
                done = json.loads((root / stage / 'done.json').read_text())
                self.assertIn(done['best_epoch'], (0, 1, 2))
                self.assertEqual(len(json.loads((root / stage / 'history.json').read_text())), 2)
            loaded, checkpoint = model.load(best['supervised'], 'cpu')
            self.assertEqual(checkpoint['mixture_mode'], 'uniform')
            training_net = tr.initialize('supervised', 0, [best['profile'], best['competition']])
            training_net.load_state_dict(checkpoint['model'])
            training_net.eval()
            x = torch.tensor(fit['tokens'])
            y = torch.tensor(fit['conditions'])
            target = torch.full((8,), 4)
            flags = torch.arange(8) % 2 == 0
            with torch.no_grad():
                self.assertTrue(torch.equal(loaded(x, y, target, flags), training_net(x, y, target, flags)))
            model.configure_dpo(loaded)
            before = tr.model_state(loaded)
            frozen = tr.frozen_parameters(loaded)
            reference = dpo.reference_probs(loaded, x, x.flip(0), y)
            opt = torch.optim.Adam([p for p in loaded.parameters() if p.requires_grad], lr=1e-6, eps=1e-7)
            a, b = dpo.pair_forward(loaded, x, x.flip(0), y)
            loss = dpo.dpo_loss(a, b, reference)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loaded.parameters(), 1.)
            opt.step()
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in loaded.state_dict().items()))
            for k, v in frozen.items(): self.assertTrue(torch.equal(loaded.state_dict()[k], v))

    def test_resume_matches_uninterrupted_parameters_optimizer_and_rng(self):
        fit, dev = data(3), data(4)
        spec = dict(epochs=3, batch_seed=2026101800)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            full = tr.initialize('pretrain', 1, [])
            tr.fit_stage(full, 'pretrain', 1, fit, dev, root / 'full', spec)
            interrupted = tr.initialize('pretrain', 1, [])
            def interrupt(epoch):
                if epoch == 1: raise InterruptedError()
            with self.assertRaises(InterruptedError):
                tr.fit_stage(interrupted, 'pretrain', 1, fit, dev, root / 'resume', spec, after_epoch=interrupt)
            resumed = tr.initialize('pretrain', 1, [])
            tr.fit_stage(resumed, 'pretrain', 1, fit, dev, root / 'resume', spec, resume=True)
            a = torch.load(root / 'full/resume.pt', weights_only=False)
            b = torch.load(root / 'resume/resume.pt', weights_only=False)
            self.assert_nested_equal(a, b)

    def test_pipeline_passes_new_supervised_checkpoint_to_dpo(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve() / 'run'
            def fit(net, stage, seed, fit, dev, path, spec, resume=False):
                path.mkdir(parents=True)
                checkpoint = path / 'best.pt'
                checkpoint.write_bytes(stage.encode())
                return checkpoint
            def subprocess(command, check):
                self.assertEqual(command[command.index('--initialization') + 1], str(out / 'supervised/best.pt'))
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir()
                (destination / 'last.pt').write_bytes(b'new-dpo-model')
            args = SimpleNamespace(output=out, seed=2, device='cpu', resume=False)
            with patch.object(pipeline, 'validate_data', return_value=({}, {})), \
                 patch.object(pipeline, 'initialize', return_value=torch.nn.Linear(1, 1)), \
                 patch.object(pipeline, 'fit_stage', side_effect=fit), \
                 patch.object(pipeline.subprocess, 'run', side_effect=subprocess):
                pipeline.run(args)
            self.assertEqual((out / 'final.pt').read_bytes(), b'new-dpo-model')
            args.resume = True
            with patch.object(pipeline, 'validate_data', return_value=({}, {})), \
                 patch.object(pipeline, 'initialize') as initialize:
                pipeline.run(args)
                initialize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
