"""T9: d=5 controls for the soft-label effect (same data/labels as T2/T3, N=1e6, 6000 steps).
  hard_ls{05,10,20} : label smoothing of the hard labels toward 0.5 (no teacher)
  tess_T2           : Tesseract gap labels softened with temperature 2
Compared against T3 (hard 1.35, mwpm_gap 1.23, tess_gap 0.985, hard@2e6 1.00 x MWPM; 9 seeds).
Usage: python t9_d5_controls.py <labels.npz> <seed,seed>   Output: results/t9_<stamp>_s<seeds>.json"""
import os, sys, json, time, numpy as np, pymatching, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_train as T

if __name__ == '__main__':
    lab = np.load(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(',')]
    here = os.path.dirname(os.path.abspath(__file__)); stamp = time.strftime('%Y%m%d_%H%M%S')
    ct = T.circ(T.D_, T.P_T); lay = T.Layout(ct)
    det_te, obs_te = ct.compile_detector_sampler(seed=777).sample(1_000_000, separate_observables=True)
    Gte, yte = T.grids(lay, det_te), obs_te[:, 0]
    det_va, obs_va = ct.compile_detector_sampler(seed=555).sample(100_000, separate_observables=True)
    Gva, yva = T.grids(lay, det_va), obs_va[:, 0]
    mw = float((pymatching.Matching.from_detector_error_model(ct.detector_error_model(decompose_errors=True)).decode_batch(det_te)[:, 0] != obs_te[:, 0]).mean())
    det, obs = T.circ(T.D_, T.P_T * T.BETA).compile_detector_sampler(seed=300).sample(2_000_000, separate_observables=True)
    det, obs = det[:1_000_000], obs[:1_000_000]
    Gtr = T.grids(lay, det); hard = obs[:, 0].astype(np.float32); assert np.array_equal(hard, lab['hard'])
    z = np.log(np.clip(lab['tgap'], 1e-12, 1 - 1e-12) / np.clip(1 - lab['tgap'], 1e-12, 1))
    eps = [float(x) for x in os.environ.get('LS_EPS', '0.05,0.1,0.2').split(',') if x]
    methods = [(f'hard_ls{int(e*100):02d}', (hard * (1 - e) + 0.5 * e).astype(np.float32)) for e in eps]
    methods += [('tess_T2', (1 / (1 + np.exp(-z / 2))).astype(np.float32))]
    if os.environ.get('WITH_TESS1'): methods += [('tess_gap', lab['tgap'].astype(np.float32))]
    out = dict(mwpm=mw, seeds=seeds, runs=[])
    for s in seeds:
        for name, y in methods:
            t1 = time.time(); model, steps = T.train(Gtr, y, Gva, yva, seed=s); L = T.ler(model, Gte, yte)
            rec = dict(seed=s, method=name, N=1_000_000, steps=steps, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t1))
            out['runs'].append(rec); print(json.dumps(rec), flush=True)
            json.dump(out, open(os.path.join(here, f'results/t9_{stamp}_s{"-".join(map(str, seeds))}.json'), 'w'), indent=1)
            del model; torch.cuda.empty_cache()
    print('done', flush=True)
