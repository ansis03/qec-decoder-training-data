"""S1 step 1 sanity: PyMatching LER at a few p (d=5), and whether a small 3D-CNN trained on uniform samples
at the target p gets near MWPM as N grows. If it cannot get near MWPM, data-selection comparisons are moot.
Usage: python sanity.py [d] [p] """
import sys, time, json, datetime
import numpy as np, torch
from common import circuit, Layout, sample, matcher, mwpm_ler, CNN3D, train, nn_ler

d = int(sys.argv[1]) if len(sys.argv) > 1 else 5
p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
out = {"d": d, "p": p, "mwpm": {}, "nn": []}
for pp in [0.003, 0.005, 0.007]:
    c = circuit(d, pp); t = time.time()
    out["mwpm"][pp] = mwpm_ler(c, 1_000_000, seed=1)
    print(f"MWPM d={d} p={pp}: LER={out['mwpm'][pp]:.2e}  ({time.time()-t:.0f}s)", flush=True)
c = circuit(d, p); lay = Layout(c)
print(f"grid {lay.shape}, detectors {c.num_detectors}", flush=True)
for N in [100_000, 1_000_000, 4_000_000]:
    det, obs = sample(c, N, seed=10)
    print(f"N={N}: label balance P(obs flip)={obs.mean():.3f}", flush=True)
    X = lay.to_grid(det); t = time.time()
    model = train(CNN3D(), X, obs, epochs=8 if N >= 1_000_000 else 20)
    ler = nn_ler(model, lay, c, 1_000_000, seed=99)
    out["nn"].append(dict(N=N, ler=ler, sec=time.time() - t))
    print(f"NN uniform N={N}: LER={ler:.2e}  vs MWPM {out['mwpm'][p]:.2e}  ({time.time()-t:.0f}s)", flush=True)
    del X
json.dump(out, open(f"results/sanity_d{d}_p{p}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json", "w"), indent=1)
