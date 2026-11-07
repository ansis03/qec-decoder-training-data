"""T7: d=7 posterior relabeling (protocol of T2/T3). Training syndromes regenerated with sampler seed 700 and the same
NMAX as T6 (checked against saved hard labels). Methods per seed: hard@{1e6,2e6,4e6}, tess_gap@{5e5,1e6}, mwpm_gap@1e6.
Usage: python t7_train7.py <labels7.npz> <seed,seed> [methods]   Output: results/t7_<stamp>_s<seeds>.json"""
import os, sys, json, time, numpy as np, pymatching, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_train as T

if __name__ == '__main__':
    lab = np.load(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(',')]
    only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    here = os.path.dirname(os.path.abspath(__file__)); stamp = time.strftime('%Y%m%d_%H%M%S')
    D_, P_T, BETA, NMAX = 7, 3e-3, 3.0, int(lab['nmax'])
    ct = T.circ(D_, P_T); lay = T.Layout(ct)
    det_te, obs_te = ct.compile_detector_sampler(seed=777).sample(1_000_000, separate_observables=True)
    Gte, yte = T.grids(lay, det_te), obs_te[:, 0]
    det_va, obs_va = ct.compile_detector_sampler(seed=555).sample(200_000, separate_observables=True)
    Gva, yva = T.grids(lay, det_va), obs_va[:, 0]
    mw = float((pymatching.Matching.from_detector_error_model(ct.detector_error_model(decompose_errors=True)).decode_batch(det_te)[:, 0] != obs_te[:, 0]).mean())
    det, obs = T.circ(D_, P_T * BETA).compile_detector_sampler(seed=700).sample(NMAX, separate_observables=True)
    hard = obs[:, 0].astype(np.float32); nl = len(lab['tgap'])
    assert np.array_equal(hard[:nl], lab['hard']), 'data mismatch'
    Gtr = T.grids(lay, det); del det
    out = dict(d=D_, mwpm=mw, seeds=seeds, labels=nl, runs=[]); print(f'd=7 MWPM LER {mw:.3e}, labels {nl}', flush=True)
    def temp(q, T):
        z = np.log(np.clip(q, 1e-12, 1 - 1e-12) / np.clip(1 - q, 1e-12, 1)); return (1 / (1 + np.exp(-z / T))).astype(np.float32)
    ext = [(f'tess_T{T}', temp(lab['tgap'], T), 1_000_000) for T in (2, 4, 8)] + [(f'mwpm_T{T}', temp(lab['mgap'], T), 1_000_000) for T in (2,)]
    ext += [(f'hard_ls{int(e*100):02d}', (hard * (1 - e) + 0.5 * e).astype(np.float32), 1_000_000) for e in (0.1, 0.3)]
    plan = ext + [('hard', hard, 1_000_000), ('hard', hard, 1_500_000), ('hard', hard, 2_000_000), ('hard', hard, 3_000_000),
            ('hard', hard, 4_000_000),
            ('tess_gap', lab['tgap'], 500_000), ('tess_gap', lab['tgap'], 1_000_000), ('mwpm_gap', lab['mgap'], 1_000_000)]
    for s in seeds:
        for name, y, N in plan:
            if only and f'{name}@{N}' not in only: continue
            if N > len(y): continue
            t1 = time.time(); model, steps = T.train(Gtr[:N], y[:N], Gva, yva, seed=s); L = T.ler(model, Gte, yte)
            rec = dict(seed=s, method=name, N=N, steps=steps, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t1))
            out['runs'].append(rec); print(json.dumps(rec), flush=True)
            json.dump(out, open(os.path.join(here, f'results/t7_{stamp}_s{"-".join(map(str, seeds))}.json'), 'w'), indent=1)
            del model; torch.cuda.empty_cache()
    print('done', flush=True)
