"""Fixed training objective of the archived PRISM 7.09% predictor."""
import torch


def margins(y):
    eye = torch.eye(y.shape[1], device=y.device, dtype=torch.bool)
    off = y[:, None, :].expand(-1, y.shape[1], -1).masked_fill(eye[None], -torch.inf)
    return y - off.max(-1).values


def validation_loss(pred, truth):
    pred, truth = pred.float(), truth.float()
    g, gt = pred.mean(1, keepdim=True), truth.mean(1, keepdim=True)
    return ((g - gt).square().mean()
            + 36 * ((pred - g) - (truth - gt)).square().mean()
            + 2 * (margins(pred) - margins(truth)).square().mean())


class PRISMLoss:
    def __init__(self, train_labels):
        self.activity = torch.quantile(train_labels, .5, dim=0)
        self.margin = torch.stack([
            torch.quantile(v[v > 0], .9) if (v > 0).any()
            else train_labels.new_tensor(float('inf'))
            for v in margins(train_labels).T
        ])

    def __call__(self, pred, truth):
        pred, truth = pred.float(), truth.float()
        core = validation_loss(pred, truth)
        eye = torch.eye(18, device=pred.device, dtype=torch.bool)
        off = pred[:, None, :].expand(-1, 18, -1).masked_fill(eye[None], -torch.inf)
        score = pred - off.topk(4, dim=-1).values.mean(-1)
        q = torch.sigmoid((truth - self.activity) / .25) * torch.sigmoid(
            (margins(truth) - self.margin) / .25)
        q = q / q.sum(0, keepdim=True).clamp_min(1e-8)
        return core + .5 * (-q * torch.log_softmax(score, dim=0)).sum(0).mean()
