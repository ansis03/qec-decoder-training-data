"""B3: Posterior relabeling for neural decoder training.
Sample syndromes at elevated noise (beta * p_target, as in Peters' knob) but train on soft labels = posterior of a
logical flip under the TARGET noise model: exact (Walsh-Hadamard joint, d=3) or approximate (matching complementary
gap with target-p weights: eta_hat = sigmoid(-(W_flip - W_noflip))). Turns the knob's joint shift into pure covariate shift.
Compared: uniform@target (hard), knob (hard), knob+exact posterior (d=3), knob+gap posterior (d=3,5), target+gap (d=5).
Kill: at d=5 with gap labels, < 2x sample efficiency vs best knob (hard).
Output: results/b3_<stamp>.json"""
import sys, os, json, time, math
import numpy as np, stim, pymatching, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sampling'))
from common import Layout, CNN3D
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from exact_posterior import exact_joint, enc

stamp = time.strftime('%Y%m%d_%H%M%S')
P_T = 3e-3
dev = 'cuda'


def circ(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p, before_round_data_depolarization=p)


def gap_matcher(c):
    """Matching on the target DEM with observable L0 turned into an extra detector (index D)."""
    D = c.num_detectors
    txt = str(c.detector_error_model(decompose_errors=True)).replace(' L0', f' D{D}')
    return pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(txt)), D


def gap_posterior(m, D, det):
    n = len(det)
    z0 = np.concatenate([det, np.zeros((n, 1), np.uint8)], 1); z1 = np.concatenate([det, np.ones((n, 1), np.uint8)], 1)
    _, w0 = m.decode_batch(z0, return_weights=True); _, w1 = m.decode_batch(z1, return_weights=True)
    return 1 / (1 + np.exp(np.clip(w1 - w0, -50, 50)))     # P(flip) ~ exp(-W1) / (exp(-W0) + exp(-W1))


def grids(lay, det):
    g = np.zeros((len(det),) + lay.shape, np.uint8); g[:, lay.it, lay.ix, lay.iy] = det; return g


def train(G, y, Gva, yva, min_steps=6000, epochs=10, bs=2048, lr=1e-3):
    torch.manual_seed(0)
    model = CNN3D().to(dev)
    with torch.no_grad(): model(torch.as_tensor(G[:2], device=dev).float()[:, None])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = len(y); steps = max(epochs * ((N + bs - 1) // bs), min_steps)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Gt = torch.as_tensor(G); yt = torch.as_tensor(y, dtype=torch.float32)
    lf = nn.BCEWithLogitsLoss(); best, state, step = 1e9, None, 0; every = max(steps // 10, 1)
    while step < steps:
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            if step >= steps: break
            idx = perm[i:i + bs]
            loss = lf(model(Gt[idx].to(dev).float()[:, None]), yt[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1
            if step % every == 0 or step == steps:
                v = ler(model, Gva, yva); model.train()
                if v < best: best, state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); return model


@torch.no_grad()
def ler(model, G, y, bs=8192):
    model.eval(); w = 0
    for i in range(0, len(y), bs):
        w += int(((model(torch.as_tensor(G[i:i + bs], device=dev).float()[:, None]) > 0).cpu().numpy() != y[i:i + bs]).sum())
    return w / len(y)


out = dict(p_target=P_T, runs=[])
for d in (3, 5):
    ct = circ(d, P_T); lay = Layout(ct)
    det_te, obs_te = ct.compile_detector_sampler(seed=777).sample(1_000_000, separate_observables=True)
    Gte, yte = grids(lay, det_te), obs_te[:, 0]
    det_va, obs_va = ct.compile_detector_sampler(seed=555).sample(100_000, separate_observables=True)
    Gva, yva = grids(lay, det_va), obs_va[:, 0]
    mw = float((pymatching.Matching.from_detector_error_model(ct.detector_error_model(decompose_errors=True)).decode_batch(det_te)[:, 0] != obs_te[:, 0]).mean())
    gm, Dg = gap_matcher(ct)
    eta_tab = None
    if d == 3:
        P, D, _ = exact_joint(ct); Px0, Px1 = P[:2 ** D], P[2 ** D:]
        eta_tab = np.where(Px0 + Px1 > 0, Px1 / np.maximum(Px0 + Px1, 1e-300), 0.0)
    print(f'd={d} MWPM LER={mw:.3e}', flush=True)
    methods = [('uniform', 1.0, 'hard'), ('knob2', 2.0, 'hard'), ('knob3', 3.0, 'hard'), ('target+gap', 1.0, 'gap'),
               ('knob2+gap', 2.0, 'gap'), ('knob3+gap', 3.0, 'gap')]
    if d == 3: methods += [('knob2+exact', 2.0, 'exact'), ('knob3+exact', 3.0, 'exact'), ('target+exact', 1.0, 'exact')]
    Ns = [30_000, 100_000, 300_000, 1_000_000] if d == 5 else [10_000, 30_000, 100_000, 300_000]
    pool = {}
    for name, beta, lab in methods:
        if beta not in pool:
            det, obs = circ(d, P_T * beta).compile_detector_sampler(seed=int(100 * beta)).sample(max(Ns), separate_observables=True)
            pool[beta] = (det, obs[:, 0].astype(np.float32))
        det, hard = pool[beta]
        if lab == 'hard': y = hard
        elif lab == 'gap': y = gap_posterior(gm, Dg, det).astype(np.float32)
        else: y = eta_tab[enc(det)].astype(np.float32)
        G = grids(lay, det)
        for N in Ns:
            t0 = time.time()
            model = train(G[:N], y[:N], Gva, yva)
            L = ler(model, Gte, yte)
            rec = dict(d=d, method=name, beta=beta, labels=lab, N=N, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t0))
            out['runs'].append(rec)
            print(f"  d={d} {name:13s} N={N:>8} LER={L:.3e} (x{L/mw:.2f} MWPM) [{rec['sec']}s]", flush=True)
            json.dump(out, open(os.path.join(here, f'results/b3_{stamp}.json'), 'w'), indent=1)
            del model; torch.cuda.empty_cache()
print('done', stamp)
