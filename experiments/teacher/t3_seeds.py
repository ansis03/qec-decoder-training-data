"""T3: seed replication of T2 (d=5, beta=3). Same data (sampler seed 300), same saved labels; training seed varies.
Methods: hard@1e6, hard@2e6, tess_gap@1e6, mwpm_gap@1e6.  Usage: python t3_seeds.py <labels.npz> <seed,seed,...>
Output: results/t3_<stamp>_s<seeds>.json"""
import os, sys, json, time, numpy as np, pymatching, stim, torch
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
    Gtr = T.grids(lay, det); hard = obs[:, 0].astype(np.float32)
    assert np.array_equal(hard[:1_000_000], lab['hard']), 'data mismatch with saved labels'
    out = dict(mwpm=mw, seeds=seeds, runs=[])
    plan = [('hard', hard, 1_000_000), ('hard', hard, 2_000_000), ('tess_gap', lab['tgap'], 1_000_000), ('mwpm_gap', lab['mgap'], 1_000_000)]
    for s in seeds:
        for name, y, N in plan:
            t1 = time.time(); model, steps = T.train(Gtr[:N], y[:N], Gva, yva, seed=s); L = T.ler(model, Gte, yte)
            rec = dict(seed=s, method=name, N=N, steps=steps, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t1))
            out['runs'].append(rec); print(json.dumps(rec), flush=True)
            json.dump(out, open(os.path.join(here, f'results/t3_{stamp}_s{"-".join(map(str, seeds))}.json'), 'w'), indent=1)
            del model; torch.cuda.empty_cache()
    print('done', flush=True)
