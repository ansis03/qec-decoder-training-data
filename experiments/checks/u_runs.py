"""Supplementary controls for the ICCDE paper, pre-registered before any result was seen.
The registered pass rules are evaluated in analysis/numbers.py (supp).
  a: d=5, p=0.3%, beta=1: hard vs label smoothing eps=0.1, N=1e6, 6000 steps
  b: d=3, p=0.3%, beta=3: hard, eps=0.1, exact posterior, N in {1e4,3e4,1e5,3e5}, 6000 steps
  c: d=5, p=0.5%, N=3e5, beta in {1,2}, steps in {1764,6000}: each run tests both the final and the best-validation state
Training code is teacher/t2_train.py (on the server an unchanged copy sat next to this file); c uses train2, the same loop with the final state kept too.
Usage: python experiments/checks/u_runs.py a:1,2 c:1 b:1,2     Output: experiments/checks/results/u_<part>_<stamp>_s<seeds>.json"""
import os, sys, json, time, numpy as np, stim, pymatching, torch, torch.nn as nn
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, '..', 'teacher'))
sys.path.insert(0, os.path.join(here, '..'))
import t2_train as T
from exact_posterior import exact_joint, enc
stamp = time.strftime('%Y%m%d_%H%M%S')
dev = 'cuda'
os.makedirs(os.path.join(here, 'results'), exist_ok=True)


def base(d, p, nval=100_000):
    ct = T.circ(d, p); lay = T.Layout(ct)
    det_te, obs_te = ct.compile_detector_sampler(seed=777).sample(1_000_000, separate_observables=True)
    Gte, yte = T.grids(lay, det_te), obs_te[:, 0]
    det_va, obs_va = ct.compile_detector_sampler(seed=555).sample(nval, separate_observables=True)
    Gva, yva = T.grids(lay, det_va), obs_va[:, 0]
    mw = float((pymatching.Matching.from_detector_error_model(ct.detector_error_model(decompose_errors=True)).decode_batch(det_te)[:, 0] != obs_te[:, 0]).mean())
    return ct, lay, Gte, yte, Gva, yva, mw


def dump(part, seeds, out):
    json.dump(out, open(os.path.join(here, f'results/u_{part}_{stamp}_s{"-".join(map(str, seeds))}.json'), 'w'), indent=1)


def part_a(seeds):
    P_T, N = 3e-3, 1_000_000
    ct, lay, Gte, yte, Gva, yva, mw = base(5, P_T)
    assert mw == 0.00324, mw                                  # same test set as T2-T9
    det, obs = T.circ(5, P_T).compile_detector_sampler(seed=100).sample(2_000_000, separate_observables=True)
    det, obs = det[:N], obs[:N]
    Gtr = T.grids(lay, det); hard = obs[:, 0].astype(np.float32)
    out = dict(part='a', d=5, p=P_T, beta=1.0, mwpm=mw, seeds=seeds, runs=[])
    print(f'a: MWPM {mw}', flush=True)
    for s in seeds:
        for name, y in (('hard', hard), ('hard_ls10', (hard * 0.9 + 0.05).astype(np.float32))):
            t1 = time.time(); model, steps = T.train(Gtr, y, Gva, yva, seed=s); L = T.ler(model, Gte, yte)
            rec = dict(seed=s, method=name, N=N, steps=steps, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t1))
            out['runs'].append(rec); print(json.dumps(rec), flush=True); dump('a', seeds, out)
            del model; torch.cuda.empty_cache()


def part_b(seeds):
    P_T, BETA, NS = 3e-3, 3.0, [10_000, 30_000, 100_000, 300_000]
    ct, lay, Gte, yte, Gva, yva, mw = base(3, P_T)
    P, D, _ = exact_joint(ct); Px0, Px1 = P[:2 ** D], P[2 ** D:]
    eta_tab = np.where(Px0 + Px1 > 0, Px1 / np.maximum(Px0 + Px1, 1e-300), 0.0)
    det, obs = T.circ(3, P_T * BETA).compile_detector_sampler(seed=300).sample(max(NS), separate_observables=True)
    Gtr = T.grids(lay, det); hard = obs[:, 0].astype(np.float32); exact = eta_tab[enc(det)].astype(np.float32)
    out = dict(part='b', d=3, p=P_T, beta=BETA, mwpm=mw, seeds=seeds, runs=[])
    print(f'b: MWPM {mw}, mean exact label {exact.mean():.4f}, hard {hard.mean():.4f}', flush=True)
    for s in seeds:
        for name, y in (('hard', hard), ('hard_ls10', (hard * 0.9 + 0.05).astype(np.float32)), ('exact', exact)):
            for N in NS:
                t1 = time.time(); model, steps = T.train(Gtr[:N], y[:N], Gva, yva, seed=s); L = T.ler(model, Gte, yte)
                rec = dict(seed=s, method=name, N=N, steps=steps, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t1))
                out['runs'].append(rec); print(json.dumps(rec), flush=True); dump('b', seeds, out)
                del model; torch.cuda.empty_cache()


def train2(Gtr, y, Gva, yva, steps, bs=2048, lr=1e-3, seed=0, evals=12):
    """T.train with a fixed step count, returning the final and the best-validation state."""
    torch.manual_seed(seed)
    model = T.CNN3D().to(dev)
    with torch.no_grad(): model(torch.as_tensor(Gtr[:2], device=dev).float()[:, None])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = len(y)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Gt = torch.as_tensor(Gtr).to(dev); yt = torch.as_tensor(y, dtype=torch.float32).to(dev)
    lf = nn.BCEWithLogitsLoss(); best, state, step = 1e9, None, 0; every = max(steps // evals, 1)
    while step < steps:
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, bs):
            if step >= steps: break
            idx = perm[i:i + bs]
            loss = lf(model(Gt[idx].float()[:, None]), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1
            if step % every == 0 or step == steps:
                v = T.ler(model, Gva, yva); model.train()
                if v < best: best, state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    final = {k: t.detach().clone() for k, t in model.state_dict().items()}
    return model, final, state


def part_c(seeds):
    P_T, N = 5e-3, 300_000
    ct, lay, Gte, yte, Gva, yva, mw = base(5, P_T)
    out = dict(part='c', d=5, p=P_T, N=N, mwpm=mw, seeds=seeds, runs=[])
    print(f'c: MWPM {mw}', flush=True)
    for s in seeds:
        for beta in (1.0, 2.0):
            det, obs = T.circ(5, P_T * beta).compile_detector_sampler(seed=1000 * s).sample(N, separate_observables=True)
            Gtr = T.grids(lay, det); y = obs[:, 0].astype(np.float32)
            for steps in (1764, 6000):
                t1 = time.time(); model, final, best = train2(Gtr, y, Gva, yva, steps, seed=s)
                model.load_state_dict(final); Lf = T.ler(model, Gte, yte)
                model.load_state_dict(best); Lb = T.ler(model, Gte, yte)
                rec = dict(seed=s, beta=beta, N=N, steps=steps, ler_final=Lf, ler_best=Lb, ratio_final=Lf / mw, ratio_best=Lb / mw, sec=round(time.time() - t1))
                out['runs'].append(rec); print(json.dumps(rec), flush=True); dump('c', seeds, out)
                del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    for job in sys.argv[1:]:
        part, s = job.split(':'); seeds = [int(x) for x in s.split(',')]
        dict(a=part_a, b=part_b, c=part_c)[part](seeds)
    print('done', flush=True)
