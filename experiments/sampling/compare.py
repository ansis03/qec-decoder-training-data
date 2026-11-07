"""S1 step 2: training-data selection methods for a neural surface-code decoder, all baselines from the start.

Target: d=5, p_target. Train size N in N_LIST, same epochs per N, same test set (seed 99, 1e6 shots at p_target).
Methods
  uniform        : N samples at p_target
  highp_X        : N samples at p = X (Cascade-style "train at higher p")
  beta_B         : N samples at B * p_target (Peters-style scalar knob, unweighted)
  strat / strat_w: pool of POOL*N at p_target; group by defect count w; resample N with q(w) ∝ sqrt(P(w) * F(w))
                   where F(w) = MWPM failure rate in that group (failure spectrum); strat_w adds likelihood-ratio
                   weights P(w)/q(w) to the loss (unbiased w.r.t. p_target)
  fail_F         : pool of POOL*N at p_target; mix fraction F of MWPM-failure samples, rest uniform from the pool
Cost bookkeeping: N_train (main axis) and N_generated (pool size; Stim + MWPM are cheap but counted).
Usage: python compare.py [p_target] [seeds] [methods comma list]
"""
import sys, time, json, datetime
import numpy as np
from common import circuit, Layout, sample, matcher, CNN3D, train, nn_ler

D = 5
P = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
N_LIST = [100_000, 300_000, 1_000_000]
EPOCHS = {100_000: 20, 300_000: 12, 1_000_000: 8}
POOL = 5
ALL = ["uniform", "highp_0.007", "highp_0.01", "beta_1.25", "beta_1.5", "beta_2", "strat", "strat_w", "fail_0.2", "fail_0.4"]
METHODS = sys.argv[3].split(",") if len(sys.argv) > 3 else ALL
OUT = f"results/compare_d{D}_p{P}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"

c_t = circuit(D, P); lay = Layout(c_t); m_t = matcher(c_t)


def make_data(method, N, seed):
    """returns det (N,D), obs (N,), weights or None, n_generated"""
    rs = 1000 * seed
    if method == "uniform":
        det, obs = sample(c_t, N, seed=rs); return det, obs, None, N
    if method.startswith("highp_") or method.startswith("beta_"):
        pp = float(method.split("_")[1]) if method.startswith("highp_") else P * float(method.split("_")[1])
        det, obs = sample(circuit(D, pp), N, seed=rs); return det, obs, None, N
    det, obs = sample(c_t, POOL * N, seed=rs)
    fail = (m_t.decode_batch(det)[:, 0] != obs)
    rng = np.random.default_rng(rs + 7)
    if method.startswith("fail_"):
        F = float(method.split("_")[1])
        fi, ok = np.nonzero(fail)[0], np.nonzero(~fail)[0]
        nf = min(int(F * N), len(fi))
        idx = np.concatenate([rng.choice(fi, nf, replace=nf > len(fi)), rng.choice(ok, N - nf, replace=False)])
        return det[idx], obs[idx], None, POOL * N
    # stratified by defect count
    w = det.sum(1)
    ws, inv = np.unique(w, return_inverse=True)
    Pw = np.bincount(inv) / len(w)
    Fw = np.bincount(inv, weights=fail) / np.bincount(inv)
    q = np.sqrt(Pw * (Fw + 1e-4)); q /= q.sum()
    probs = q[inv] / np.bincount(inv)[inv]                 # per-sample selection prob so that group mass = q(w)
    probs /= probs.sum()
    idx = rng.choice(len(w), N, replace=True, p=probs)
    wt = (Pw / q)[inv[idx]] if method == "strat_w" else None
    return det[idx], obs[idx], wt, POOL * N


res = []
for N in N_LIST:
    for m in METHODS:
        for s in range(SEEDS):
            t0 = time.time()
            det, obs, wt, ngen = make_data(m, N, s)
            X = lay.to_grid(det)
            model = train(CNN3D(), X, obs, w=wt, epochs=EPOCHS[N], seed=s)
            ler = nn_ler(model, lay, c_t, 1_000_000, seed=99)
            res.append(dict(method=m, N=N, seed=s, ler=ler, n_generated=ngen, label_flip_rate=float(obs.mean()),
                            sec=time.time() - t0))
            print(f"N={N:8d} {m:12s} seed={s}: LER={ler:.3e}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(d=D, p=P, epochs=EPOCHS, pool=POOL, results=res), open(OUT, "w"), indent=1)
            del X
print("saved", OUT)
