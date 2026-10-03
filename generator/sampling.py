"""Canonical two-draw peptide sampling."""
import torch


@torch.inference_mode()
def sample(net, y, target, conditional, temperature, seed, batch=512):
    if temperature <= 0 or batch < 1:
        raise ValueError('Temperature and batch size must be positive.')
    device = next(net.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    for lo in range(0, len(y), batch):
        yy = torch.as_tensor(y[lo:lo + batch], device=device, dtype=torch.float32)
        n = len(yy)
        xx = torch.empty((n, 0), device=device, dtype=torch.long)
        stopped = torch.zeros(n, device=device, dtype=torch.bool)
        length = torch.full((n,), 15, device=device, dtype=torch.long)
        target_ids = torch.full((n,), target, device=device, dtype=torch.long)
        flags = torch.full((n,), conditional, device=device, dtype=torch.bool)
        for k in range(15):
            logits = torch.log_softmax(net(xx, yy, target_ids, flags), -1)[:, -1]
            u = torch.rand((n, 1), device=device, generator=rng)
            draw = ((logits / temperature).softmax(-1).cumsum(-1) < u).sum(-1).clamp_max(21)
            if k:
                repeated = draw == xx[:, -1]
                penalized = logits.clone()
                penalized.scatter_(1, draw[:, None], logits.gather(1, draw[:, None]) / 1.2)
                u2 = torch.rand((n, 1), device=device, generator=rng)
                again = (penalized.softmax(-1).cumsum(-1) < u2).sum(-1).clamp_max(21)
                draw = torch.where(repeated, again, draw)
            first = (~stopped) & (draw == 21)
            length = torch.where(first, k, length)
            stopped |= draw == 21
            xx = torch.cat([xx, draw[:, None]], 1)
        yield lo, xx.cpu().numpy(), length.cpu().numpy(), stopped.cpu().numpy()
