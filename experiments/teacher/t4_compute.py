"""T4: compute-matched control for T3. In T2/T3, hard@2e6 got 9770 steps vs 6000 for the 1e6 runs (10 epochs vs
min_steps). Here every method at N=1e6 gets 20 epochs (9780 steps), so data and compute are separated.
Usage: python t4_compute.py <labels.npz> <seed,seed>.  Output: results/t4_<stamp>_s<seeds>.json"""
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
    det, obs = T.circ(T.D_, T.P_T * T.BETA).compile_detector_sampler(seed=300).sample(2_000_000, separate_observables=True); det, obs = det[:1_000_000], obs[:1_000_000]
    Gtr = T.grids(lay, det); hard = obs[:, 0].astype(np.float32); assert np.array_equal(hard, lab['hard'])
    out = dict(mwpm=mw, seeds=seeds, epochs=20, runs=[])
    for s in seeds:
        for name, y in (('hard', hard), ('tess_gap', lab['tgap']), ('mwpm_gap', lab['mgap'])):
            t1 = time.time(); model, steps = T.train(Gtr, y, Gva, yva, seed=s, epochs=20); L = T.ler(model, Gte, yte)
            rec = dict(seed=s, method=name, N=1_000_000, steps=steps, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t1))
            out['runs'].append(rec); print(json.dumps(rec), flush=True)
            json.dump(out, open(os.path.join(here, f'results/t4_{stamp}_s{"-".join(map(str, seeds))}.json'), 'w'), indent=1)
            del model; torch.cuda.empty_cache()
    print('done', flush=True)
