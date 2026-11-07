"""S1 pilot common pieces: surface-code data (Stim), PyMatching baseline, small 3D-CNN decoder (PyTorch, GPU).

Noise model: standard uniform circuit-level depolarizing noise with strength p
(after_clifford_depolarization = after_reset_flip = before_measure_flip = before_round_data_depolarization = p).
Task: rotated surface code memory-Z, distance d, d rounds. Label = whether the logical observable flipped.
The decoder sees detection events laid out on a (time, x, y) grid from Stim detector coordinates.
"""
import numpy as np
import stim
import pymatching
import torch
import torch.nn as nn


def circuit(d, p, rounds=None):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds or d,
                                  after_clifford_depolarization=p, after_reset_flip_probability=p,
                                  before_measure_flip_probability=p, before_round_data_depolarization=p)


class Layout:
    """Maps the flat detector vector to a dense (T, X, Y) grid."""

    def __init__(self, circ):
        coords = circ.get_detector_coordinates()
        c = np.array([coords[i] for i in range(circ.num_detectors)])       # (x, y, t)
        xs, ys, ts = c[:, 0], c[:, 1], c[:, 2]
        ux, uy, ut = np.unique(xs), np.unique(ys), np.unique(ts)
        self.ix = np.searchsorted(ux, xs); self.iy = np.searchsorted(uy, ys); self.it = np.searchsorted(ut, ts)
        self.shape = (len(ut), len(ux), len(uy))

    def to_grid(self, det):                       # det: (B, D) uint8/bool -> (B, 1, T, X, Y) float32
        B = det.shape[0]
        g = np.zeros((B,) + self.shape, np.float32)
        g[:, self.it, self.ix, self.iy] = det
        return g[:, None]


def sample(circ, n, seed=None):
    s = circ.compile_detector_sampler(seed=seed)
    det, obs = s.sample(n, separate_observables=True)
    return det.astype(np.uint8), obs[:, 0].astype(np.uint8)


def matcher(circ):
    return pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))


def mwpm_ler(circ, n, seed=None, m=None):
    det, obs = sample(circ, n, seed)
    m = m or matcher(circ)
    pred = m.decode_batch(det)[:, 0]
    return float((pred != obs).mean())


class CNN3D(nn.Module):
    def __init__(self, ch=64, depth=4):
        super().__init__()
        layers, c_in = [], 1
        for _ in range(depth):
            layers += [nn.Conv3d(c_in, ch, 3, padding=1), nn.GELU()]
            c_in = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def train(model, X, y, w=None, epochs=10, bs=2048, lr=1e-3, device="cuda", seed=0, log=None):
    """X: (N,1,T,X,Y) float32 numpy, y: (N,) {0,1}, w: optional per-sample weights."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    with torch.no_grad():
        model(torch.as_tensor(X[:2], device=device))           # init lazy layers
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = len(y); steps = epochs * ((N + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xt = torch.as_tensor(X); yt = torch.as_tensor(y, dtype=torch.float32)
    wt = torch.as_tensor(w, dtype=torch.float32) if w is not None else None
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        perm = torch.randperm(N)
        model.train(); tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx].to(device, non_blocking=True); yb = yt[idx].to(device)
            l = lossf(model(xb), yb)
            if wt is not None:
                wb = wt[idx].to(device); l = (l * wb).sum() / wb.sum()
            else:
                l = l.mean()
            opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sched.step()
            tot += float(l.detach()) * len(idx)
        if log is not None:
            log.append(dict(epoch=ep, loss=tot / N))
    return model


@torch.no_grad()
def nn_ler(model, layout, circ, n, seed=None, bs=8192, device="cuda", chunk=200_000):
    model.eval(); wrong = 0; done = 0
    s = circ.compile_detector_sampler(seed=seed)
    while done < n:
        k = min(chunk, n - done)
        det, obs = s.sample(k, separate_observables=True)
        X = layout.to_grid(det.astype(np.uint8))
        for i in range(0, k, bs):
            pr = (model(torch.as_tensor(X[i:i + bs], device=device)) > 0).cpu().numpy()
            wrong += int((pr != obs[i:i + bs, 0]).sum())
        done += k
    return wrong / n
