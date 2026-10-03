"""CPU contract tests; synthetic labels only, no full benchmark training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'predictor'))
import train as training
from losses import PRISMLoss, validation_loss
from model import AblatedAdapter
from predict import load, predict


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Repeated indexed-gradient reductions can vary with CPU threading.
        # Test checkpoint/RNG restoration independently of that runtime effect.
        torch.set_num_threads(1)
        cls.cfg = json.loads(training.CONFIG.read_text())
        cls.fixed = training.check_assets(cls.cfg, ROOT / 'predictor/initialization/lmw_256_8_iso.pt')
        cls.lm = torch.load(ROOT / 'predictor/initialization/lmw_256_8_iso.pt', map_location='cpu', weights_only=False)

    def context(self, epochs=12):
        z = np.load(self.fixed)
        cfg = dict(self.cfg, max_epochs_per_stage=epochs)
        rng = np.random.default_rng(912)
        ctx = SimpleNamespace(cfg=cfg, device=torch.device('cpu'), emb=z['enzyme_embeddings'],
                              geo=[torch.tensor(z[k]) for k in ('gf', 'gd', 'gm', 'gh')])
        ctx.z = {s: torch.tensor(rng.integers(0, 20, (8, 10))) for s in ('train', 'val')}
        ctx.f = {s: torch.tensor(rng.normal(size=(8, 10, 1280)), dtype=torch.float32) for s in ('train', 'val')}
        ctx.y = {s: torch.tensor(rng.normal(size=(8, 18)), dtype=torch.float32) for s in ('train', 'val')}
        ctx.data = lambda s, ix: (ctx.z[s][ix], ctx.f[s][ix])
        ctx.loss, ctx.pair_count = PRISMLoss(ctx.y['train']), 512
        return ctx

    def test_loss_matches_archived_formula_and_gradient(self):
        torch.manual_seed(13)
        truth = torch.randn(128, 18)
        p = torch.randn(128, 18, requires_grad=True)
        q = p.detach().clone().requires_grad_(True)
        loss = PRISMLoss(truth)
        def margin(x):
            off = x[:, None, :].expand(-1, 18, -1).masked_fill(torch.eye(18, dtype=torch.bool)[None], -torch.inf)
            return x - off.max(-1).values
        g, gt = q.mean(1, keepdim=True), truth.mean(1, keepdim=True)
        core = (g - gt).square().mean() + 36 * ((q - g) - (truth - gt)).square().mean() + 2 * (margin(q) - margin(truth)).square().mean()
        ma = torch.quantile(truth, .5, dim=0)
        md = torch.stack([torch.quantile(v[v > 0], .9) if (v > 0).any() else torch.tensor(float('inf')) for v in margin(truth).T])
        weights = torch.sigmoid((truth - ma) / .25) * torch.sigmoid((margin(truth) - md) / .25)
        weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-8)
        off = q[:, None, :].expand(-1, 18, -1).masked_fill(torch.eye(18, dtype=torch.bool)[None], -torch.inf)
        expected = core + .5 * (-weights * torch.log_softmax(q - off.topk(4, dim=-1).values.mean(-1), dim=0)).sum(0).mean()
        actual = loss(p, truth)
        actual.backward(); expected.backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(p.grad, q.grad))
        self.assertTrue(torch.equal(validation_loss(q, truth), core))

    def test_freeze_unfreeze_and_exact_epoch_resume(self):
        ctx = self.context()
        base, _ = training.construct(ctx, 0)
        initial = training.state(base)
        net, matched = training.construct(ctx, 0, self.lm, initial)
        before = training.state(net)
        checks = []
        def check(epoch):
            if epoch == 10:
                self.assertTrue(all(torch.equal(net.state_dict()[k], before[k]) for k in matched))
                self.assertTrue(all(p.requires_grad for p in net.parameters()))
                checks.append(epoch)
            if epoch == 11:
                self.assertTrue(any(not torch.equal(net.state_dict()[k], before[k]) for k in matched))
                checks.append(epoch)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            training.fit_stage(net, ctx, 'lm_stage', root / 'full', 0, matched, after_epoch=check)
            other, matched = training.construct(ctx, 0, self.lm, initial)
            def stop(epoch):
                if epoch == 10:
                    raise InterruptedError('simulated interruption after optimizer reset')
            with self.assertRaises(InterruptedError):
                training.fit_stage(other, ctx, 'lm_stage', root / 'resume', 0, matched, after_epoch=stop)
            other, matched = training.construct(ctx, 0, self.lm, initial)
            training.fit_stage(other, ctx, 'lm_stage', root / 'resume', 0, matched, resume=True)
            full = torch.load(root / 'full/resume.pt', weights_only=False)
            resumed = torch.load(root / 'resume/resume.pt', weights_only=False)
            self.assertTrue(all(torch.equal(v, resumed['state'][k]) for k, v in full['state'].items()))
            self.assertEqual(full['history'], resumed['history'])
            self.assertTrue(torch.equal(full['cpu_rng'], resumed['cpu_rng']))
            self.assertTrue(torch.equal(full['sampler'], resumed['sampler']))
            for k, value in full['optimizer']['state'].items():
                for name, tensor in value.items():
                    self.assertTrue(torch.equal(tensor, resumed['optimizer']['state'][k][name]))
        self.assertEqual(checks, [10, 11])

    def test_adapter_freezes_base_and_exports_loadable_model(self):
        ctx = self.context(epochs=2)
        base, _ = training.construct(ctx, 1, self.lm)
        base.eval().requires_grad_(False)
        before = training.state(base)
        training.reset_seed(1)
        net = AblatedAdapter(base, ctx.geo, arm='shared_query')
        cached = {s: training.cache_base(base, ctx, s) for s in ('train', 'val')}
        net.eval()
        with torch.no_grad():
            torch.testing.assert_close(net(*cached['val']), cached['val'][1], atol=2e-5, rtol=0)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp)
            training.fit_stage(net, ctx, 'adapter', path / 'adapter', 1, cached=cached)
            self.assertTrue(all(torch.equal(v, net.base.state_dict()[k]) for k, v in before.items()))
            self.assertTrue(all(p.grad is None for p in net.base.parameters()))
            history = json.loads((path / 'adapter/history.json').read_text())
            self.assertEqual([r['epoch'] for r in history], [0, 1, 2])
            targets = json.loads((ROOT / 'data/benchmark/target_order.json').read_text())
            training.save(path / 'final.pt', dict(state=training.state(net), model_cfg=self.cfg['model'], ids={'targets': targets}, recipe={'model': 'new:adapter:shared_query'}))
            restored, order = load(path / 'final.pt')
            seqs = [''.join('ACDEFGHIKLMNPQRSTVWY'[j] for j in row) for row in ctx.z['val'].tolist()]
            actual = predict(restored, seqs, ctx.f['val'].numpy(), batch=8)
            net.eval()
            with torch.no_grad():
                expected = net(*cached['val']).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
            self.assertEqual(order, targets)

    def test_archive_weights_unchanged(self):
        for seed, expected in self.cfg['archived_checkpoints'].items():
            self.assertEqual(training.sha(ROOT / f'predictor/checkpoints/seed{seed}.pt'), expected)

    def test_adapter_resume_preserves_post_initialization_rng(self):
        ctx = self.context(epochs=3)
        def build():
            base, _ = training.construct(ctx, 2, self.lm)
            base.eval().requires_grad_(False)
            training.reset_seed(2)
            net = AblatedAdapter(base, ctx.geo, arm='shared_query')
            cached = {s: training.cache_base(base, ctx, s) for s in ('train', 'val')}
            return net, cached
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            net, cache = build()
            initial = training.state(net)
            training.fit_stage(net, ctx, 'adapter', root / 'full', 2, cached=cache)
            other, cache = build()
            def stop(epoch):
                if epoch == 1:
                    raise InterruptedError('simulated adapter interruption')
            with self.assertRaises(InterruptedError):
                training.fit_stage(other, ctx, 'adapter', root / 'resume', 2, cached=cache, after_epoch=stop)
            other, cache = build()
            training.fit_stage(other, ctx, 'adapter', root / 'resume', 2, cached=cache, resume=True)
            full = torch.load(root / 'full/resume.pt', weights_only=False)
            resumed = torch.load(root / 'resume/resume.pt', weights_only=False)
            self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in full['state'].items()))
            self.assertTrue(all(torch.equal(v, resumed['state'][k]) for k, v in full['state'].items()))
            self.assertEqual(full['history'], resumed['history'])
            self.assertTrue(torch.equal(full['cpu_rng'], resumed['cpu_rng']))
            self.assertTrue(torch.equal(full['sampler'], resumed['sampler']))


if __name__ == '__main__':
    unittest.main()
