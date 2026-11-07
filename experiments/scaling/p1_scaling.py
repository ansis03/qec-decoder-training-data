"""P1 pilot: data scaling of a neural decoder vs code distance and training noise level.
Question: N*(d) = training samples a fixed-architecture 3D-CNN needs to match MWPM at the target p,
and how the best training noise level p_train/p_target moves with d.
Reuses ../sampling/common.py (Stim circuit, Layout, CNN3D, MWPM). Grids are stored as uint8 to save RAM.
Output: results/p1_<stamp>.json (rewritten after every run, so partial results survive)."""
import os, sys, json, time, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sampling'))
from common import circuit, Layout, matcher, CNN3D

stamp = time.strftime('%Y%m%d_%H%M%S')
P_T = 0.005
DS = [3, 5, 7]
NS = {3: [10_000, 30_000, 100_000, 300_000, 1_000_000], 5: [30_000, 100_000, 300_000, 1_000_000, 3_000_000],
      7: [100_000, 300_000, 1_000_000, 3_000_000]}
BETAS = [1.0, 1.5, 2.0, 3.0]
NTEST = 1_000_000
dev = 'cuda'


def grids(circ, lay, n, seed):
    det, obs = circ.compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    g = np.zeros((n,) + lay.shape, np.uint8)
    g[:, lay.it, lay.ix, lay.iy] = det
    return g, obs[:, 0].astype(np.float32)


def train(model, G, y, Gva, yva, min_steps=6000, epochs=10, bs=2048, lr=1e-3, seed=0, evals=12):
    """Steps = max(epochs over the data, min_steps) so small N is not under-optimized; keep the checkpoint
    with the best validation accuracy (evaluated `evals` times), i.e. early stopping without a fixed budget bias."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(dev)
    with torch.no_grad(): model(torch.as_tensor(G[:2], device=dev).float()[:, None])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = len(y); steps = max(epochs * ((N + bs - 1) // bs), min_steps)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Gt = torch.as_tensor(G); yt = torch.as_tensor(y)
    lf = nn.BCEWithLogitsLoss()
    best, best_state = 1e9, None
    every = max(steps // evals, 1); step = 0
    while step < steps:
        perm = torch.randperm(N); model.train()
        for i in range(0, N, bs):
            if step >= steps: break
            idx = perm[i:i + bs]
            xb = Gt[idx].to(dev, non_blocking=True).float()[:, None]
            loss = lf(model(xb), yt[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1
            if step % every == 0 or step == steps:
                v = ler(model, Gva, yva); model.train()
                if v < best: best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, steps


@torch.no_grad()
def ler(model, Gte, yte, bs=8192):
    model.eval(); wrong = 0
    for i in range(0, len(yte), bs):
        pr = (model(torch.as_tensor(Gte[i:i + bs], device=dev).float()[:, None]) > 0).cpu().numpy()
        wrong += int((pr != yte[i:i + bs]).sum())
    return wrong / len(yte)


out = dict(p_target=P_T, ntest=NTEST, arch='CNN3D(ch=64,depth=4)+LazyLinear head, steps=max(10 epochs, 6000), OneCycle lr1e-3, best-val (target p, 1e5) checkpoint', runs={})
for d in DS:
    ct = circuit(d, P_T); lay = Layout(ct)
    Gte, yte = grids(ct, lay, NTEST, seed=777)
    Gva, yva = grids(ct, lay, 100_000, seed=555)   # validation at the TARGET p (early stopping)
    det_te = None
    m = matcher(ct)
    det, obs = ct.compile_detector_sampler(seed=777).sample(NTEST, separate_observables=True)
    mw = float((m.decode_batch(det)[:, 0] != obs[:, 0]).mean()); del det, obs
    out['runs'][f'd{d}'] = dict(mwpm=mw, shape=list(lay.shape), points=[])
    print(f'd={d} MWPM LER={mw:.3e} grid={lay.shape}', flush=True)
    for beta in BETAS:
        ctr = circuit(d, P_T * beta)
        assert Layout(ctr).shape == lay.shape
        Nmax = max(NS[d])
        Gtr, ytr = grids(ctr, lay, Nmax, seed=int(1000 * beta) + d)
        for N in NS[d]:
            t0 = time.time()
            model, steps = train(CNN3D(), Gtr[:N], ytr[:N], Gva, yva, seed=0)
            L = ler(model, Gte, yte)
            pt = dict(beta=beta, N=N, ler=L, ratio=L / mw, steps=steps, sec=round(time.time() - t0, 1))
            out['runs'][f'd{d}']['points'].append(pt)
            print(f'  d={d} beta={beta} N={N:>9} LER={L:.3e} ratio={L / mw:.3f} ({pt["sec"]}s)', flush=True)
            json.dump(out, open(f'results/p1_{stamp}.json', 'w'), indent=1)
            del model; torch.cuda.empty_cache()
        del Gtr, ytr
print('done', stamp)
