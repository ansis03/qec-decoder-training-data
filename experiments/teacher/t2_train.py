"""T2 (teacher study): posterior relabeling with a near-ML teacher at d=5 (same protocol as ../posterior/b3_relabel.py).
Surface code memory-Z d=5 r=5, target p=3e-3. Training syndromes at beta*p (beta=3, Peters' knob). Labels:
  hard     : sampled logical flip at beta*p (the knob, baseline)
  mwpm_gap : sigmoid(W0-W1) from MWPM with L0 as an extra detector, TARGET-p weights (the posterior study)
  tess_gap : same with Tesseract (near-ML, handles Y correlations) — the new teacher (T1: 3.4-12.7x closer to exact eta)
Model/training identical to the posterior study: CNN3D(ch=64, depth=4), AdamW 1e-3, OneCycle, bs 2048, steps=max(10 epochs, 6000),
best-validation checkpoint (validation = 1e5 target-p shots, hard labels). Test = 1e6 target-p shots.
Pre-registered pass (same as the posterior study): tess_gap at N=1e6 has LER <= hard at N=2e6 (>= 2x data efficiency).
Output: results/t2_<stamp>.json"""
import os, sys, json, time, numpy as np, stim, pymatching, torch, torch.nn as nn
from multiprocessing import Pool
here = os.path.dirname(os.path.abspath(__file__)); stamp = time.strftime('%Y%m%d_%H%M%S')
P_T, BETA, D_ = 3e-3, 3.0, 5
dev = 'cuda'
G = {}


def circ(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p, before_round_data_depolarization=p)


class Layout:
    def __init__(self, c):
        co = c.get_detector_coordinates(); a = np.array([co[i] for i in range(c.num_detectors)])
        ux, uy, ut = np.unique(a[:, 0]), np.unique(a[:, 1]), np.unique(a[:, 2])
        self.ix = np.searchsorted(ux, a[:, 0]); self.iy = np.searchsorted(uy, a[:, 1]); self.it = np.searchsorted(ut, a[:, 2])
        self.shape = (len(ut), len(ux), len(uy))


def grids(lay, det):
    g = np.zeros((len(det),) + lay.shape, np.uint8); g[:, lay.it, lay.ix, lay.iy] = det; return g


class CNN3D(nn.Module):
    def __init__(self, ch=64, depth=4):
        super().__init__(); layers, c_in = [], 1
        for _ in range(depth): layers += [nn.Conv3d(c_in, ch, 3, padding=1), nn.GELU()]; c_in = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x): return self.head(self.body(x)).squeeze(-1)


@torch.no_grad()
def ler(model, G_, y, bs=16384):
    model.eval(); w = 0
    for i in range(0, len(y), bs):
        w += int(((model(torch.as_tensor(G_[i:i + bs], device=dev).float()[:, None]) > 0).cpu().numpy() != y[i:i + bs]).sum())
    return w / len(y)


def train(Gtr, y, Gva, yva, min_steps=6000, epochs=10, bs=2048, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = CNN3D().to(dev)
    with torch.no_grad(): model(torch.as_tensor(Gtr[:2], device=dev).float()[:, None])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = len(y); steps = max(epochs * ((N + bs - 1) // bs), min_steps)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Gt = torch.as_tensor(Gtr).to(dev); yt = torch.as_tensor(y, dtype=torch.float32).to(dev)
    lf = nn.BCEWithLogitsLoss(); best, state, step = 1e9, None, 0; every = max(steps // 10, 1)
    while step < steps:
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, bs):
            if step >= steps: break
            idx = perm[i:i + bs]
            loss = lf(model(Gt[idx].float()[:, None]), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1
            if step % every == 0 or step == steps:
                v = ler(model, Gva, yva); model.train()
                if v < best: best, state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); return model, steps


def init_tess(beam):
    from tesseract_decoder import tesseract
    dem = circ(D_, P_T).detector_error_model(decompose_errors=False)
    dem = stim.DetectorErrorModel(str(dem.flattened()).replace(' L0', f' D{dem.num_detectors}'))
    G['dec'] = tesseract.TesseractConfig(dem=dem, det_beam=beam, beam_climbing=True, no_revisit_dets=True).compile_decoder()


def tess_gap(det):
    dec = G['dec']; c = []
    for k in (0, 1):
        c.append(dec.cost_from_errors(dec.decode_to_errors(np.concatenate([det.astype(bool), [bool(k)]]))))
    return 1 / (1 + np.exp(np.clip(c[1] - c[0], -50, 50)))


if __name__ == '__main__':
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    W = min(int(os.environ.get('WORKERS', '64')), 256)
    os.makedirs(os.path.join(here, 'results'), exist_ok=True)
    ct = circ(D_, P_T); lay = Layout(ct)
    det_te, obs_te = ct.compile_detector_sampler(seed=777).sample(1_000_000, separate_observables=True)
    Gte, yte = grids(lay, det_te), obs_te[:, 0]
    det_va, obs_va = ct.compile_detector_sampler(seed=555).sample(100_000, separate_observables=True)
    Gva, yva = grids(lay, det_va), obs_va[:, 0]
    mw = float((pymatching.Matching.from_detector_error_model(ct.detector_error_model(decompose_errors=True)).decode_batch(det_te)[:, 0] != obs_te[:, 0]).mean())
    out = dict(p_target=P_T, beta=BETA, d=D_, mwpm=mw, runs=[]); print(f'MWPM LER {mw:.3e}', flush=True)
    t0 = time.time()
    det, obs = circ(D_, P_T * BETA).compile_detector_sampler(seed=300).sample(NMAX, separate_observables=True)
    hard = obs[:, 0].astype(np.float32); Gtr = grids(lay, det)
    demd = ct.detector_error_model(decompose_errors=True); demd = stim.DetectorErrorModel(str(demd.flattened()).replace(' L0', f' D{demd.num_detectors}'))
    mm = pymatching.Matching.from_detector_error_model(demd)
    z = np.zeros((NMAX, 1), np.uint8)
    _, w0 = mm.decode_batch(np.concatenate([det, z], 1), return_weights=True)
    _, w1 = mm.decode_batch(np.concatenate([det, z + 1], 1), return_weights=True)
    mgap = (1 / (1 + np.exp(np.clip(w1 - w0, -50, 50)))).astype(np.float32); print(f'mwpm labels [{time.time()-t0:.0f}s]', flush=True)
    NT = min(NMAX, 1_000_000)
    with Pool(W, initializer=init_tess, initargs=(20,)) as pool:
        tgap = np.array(pool.map(tess_gap, list(det[:NT]), chunksize=256), dtype=np.float32)
    print(f'tesseract labels ({NT}) [{time.time()-t0:.0f}s]', flush=True)
    np.savez_compressed(os.path.join(here, f'results/t2_labels_{stamp}.npz'), tgap=tgap, mgap=mgap[:NT], hard=hard[:NT])
    plan = [('hard', hard, [300_000, 1_000_000, 2_000_000]), ('mwpm_gap', mgap, [300_000, 1_000_000]),
            ('tess_gap', tgap, [100_000, 300_000, 1_000_000])]
    for name, y, Ns in plan:
        for N in Ns:
            if N > len(y): continue
            t1 = time.time(); model, steps = train(Gtr[:N], y[:N], Gva, yva); L = ler(model, Gte, yte)
            rec = dict(method=name, N=N, steps=steps, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t1))
            out['runs'].append(rec); print(json.dumps(rec), flush=True)
            json.dump(out, open(os.path.join(here, f'results/t2_{stamp}.json'), 'w'), indent=1)
            del model; torch.cuda.empty_cache()
    print('done', flush=True)
