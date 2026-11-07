"""T8: label diagnostics. For d=5 (beam 20) and d=7 (beam 10) saved labels: agreement of argmax(label) with the sampled
hard label, fraction of near-certain labels, mean binary entropy, correlation between teachers, and how often the two
teachers disagree on the argmax. Also: at d=7, re-label a 20k subsample with beam 20 and beam 40 to measure beam effect.
Output: printed + results/t8_<stamp>.json"""
import os, sys, json, time, numpy as np, stim
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t6_labels7 as L7
here = os.path.dirname(os.path.abspath(__file__)); stamp = time.strftime('%Y%m%d_%H%M%S')


def stats(name, t, m, h):
    H = lambda q: -(np.clip(q, 1e-12, 1) * np.log2(np.clip(q, 1e-12, 1)) + np.clip(1 - q, 1e-12, 1) * np.log2(np.clip(1 - q, 1e-12, 1)))
    r = dict(set=name, n=len(t),
             tess_agree_hard=float(((t > .5) == (h > .5)).mean()), mwpm_agree_hard=float(((m > .5) == (h > .5)).mean()),
             teachers_disagree=float(((t > .5) != (m > .5)).mean()), corr=float(np.corrcoef(t, m)[0, 1]),
             tess_entropy=float(H(t).mean()), mwpm_entropy=float(H(m).mean()),
             tess_extreme=float(((t < 1e-3) | (t > 1 - 1e-3)).mean()), mwpm_extreme=float(((m < 1e-3) | (m > 1 - 1e-3)).mean()),
             tess_mean=float(t.mean()), mwpm_mean=float(m.mean()), hard_mean=float(h.mean()))
    print(json.dumps(r), flush=True); return r


if __name__ == '__main__':
    out = []
    a = np.load(sys.argv[1]); out.append(stats('d5_beam20', a['tgap'], a['mgap'], a['hard']))
    b = np.load(sys.argv[2]); out.append(stats('d7_beam10', b['tgap'], b['mgap'], b['hard']))
    det, _ = L7.circ(7, L7.P_T * L7.BETA).compile_detector_sampler(seed=700).sample(int(b['nmax']), separate_observables=True)
    sub = det[:20000]; W = min(int(os.environ.get('WORKERS', '64')), 240)
    for beam in (20, 40):
        t0 = time.time()
        with Pool(W, initializer=L7.init_tess, initargs=(beam,)) as pool:
            tb = np.array(pool.map(L7.tess_gap, list(sub), chunksize=16), dtype=np.float32)
        t10 = b['tgap'][:20000]
        r = dict(set=f'd7_beam{beam}_vs_beam10', n=20000, sec=round(time.time() - t0),
                 argmax_change=float(((tb > .5) != (t10 > .5)).mean()), mae=float(np.abs(tb - t10).mean()),
                 agree_hard=float(((tb > .5) == (b['hard'][:20000] > .5)).mean()),
                 agree_hard_beam10=float(((t10 > .5) == (b['hard'][:20000] > .5)).mean()),
                 agree_mwpm=float(((tb > .5) == (b['mgap'][:20000] > .5)).mean()))
        print(json.dumps(r), flush=True); out.append(r)
        np.save(os.path.join(here, f'results/t8_d7_beam{beam}_20k.npy'), tb)
    json.dump(out, open(os.path.join(here, f'results/t8_{stamp}.json'), 'w'), indent=1)
    print('done', flush=True)
