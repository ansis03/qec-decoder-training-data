"""Smoke test before the supplementary runs: imports, MWPM on the shared test set, exact table, one very short training of each kind."""
import os, sys, time
sys.argv = ['u_runs.py']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u_runs as U
import numpy as np

ct, lay, Gte, yte, Gva, yva, mw = U.base(5, 3e-3)
print('mwpm d5', mw, flush=True)
assert mw == 0.00324
t = time.time()
P, D, _ = U.exact_joint(U.T.circ(3, 3e-3))
print('exact joint D', D, 'sec', round(time.time() - t, 1), 'sum', float(P.sum()), flush=True)
det, obs = U.T.circ(5, 3e-3).compile_detector_sampler(seed=100).sample(4096, separate_observables=True)
G = U.T.grids(lay, det); y = obs[:, 0].astype(np.float32)
m, f, b = U.train2(G, y, Gva[:2000], yva[:2000], 40, seed=1, evals=4)
m.load_state_dict(f); a = U.T.ler(m, Gte[:20000], yte[:20000])
m.load_state_dict(b); c = U.T.ler(m, Gte[:20000], yte[:20000])
print('train2 ok', a, c, flush=True)
m, st = U.T.train(G, y * 0.9 + 0.05, Gva[:2000], yva[:2000], min_steps=30, seed=1)
print('T.train ok', st, flush=True)
print('smoke done', flush=True)
