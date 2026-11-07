"""T1 (teacher study): teacher quality for posterior relabeling. Surface code d=3 r=3 (D=24 detectors, exact joint
distribution by Walsh-Hadamard, from ../exact_posterior.py), target p=3e-3, syndromes sampled at beta*p (beta=1,3).
Teachers (all give eta_hat = P(logical flip | syndrome) under the TARGET model):
  mwpm_gap : sigmoid(W0 - W1), W_k = min-weight matching with L0 turned into an extra detector set to k (the posterior study)
  tess_gap : same with Tesseract (handles hyperedges / Y correlations; near-ML error)
Score vs exact eta: MAE, KL, argmax disagreement, correlation.  Output: results/t1_<stamp>.json"""
import os, sys, json, time, numpy as np, stim, pymatching
from multiprocessing import Pool
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from exact_posterior import exact_joint, enc
P_T = 3e-3
G = {}


def circ(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p, before_round_data_depolarization=p)


def obs_as_detector(dem):
    """L0 -> extra detector index D (last)."""
    D = dem.num_detectors
    return stim.DetectorErrorModel(str(dem).replace(' L0', f' D{D}')), D


def init_tess(d, beam):
    from tesseract_decoder import tesseract
    dem, D = obs_as_detector(circ(d, P_T).detector_error_model(decompose_errors=False))
    G['dec'] = tesseract.TesseractConfig(dem=dem, det_beam=beam, beam_climbing=True, no_revisit_dets=True).compile_decoder()
    G['D'] = D


def tess_costs(det):
    dec = G['dec']; out = []
    for k in (0, 1):
        s = np.concatenate([det.astype(bool), [bool(k)]])
        errs = dec.decode_to_errors(s); out.append(dec.cost_from_errors(errs))
    return out


def score(eh, eta):
    e = np.clip(eh, 1e-9, 1 - 1e-9); t = np.clip(eta, 1e-12, 1 - 1e-12)
    kl = t * np.log(t / e) + (1 - t) * np.log((1 - t) / (1 - e))
    return dict(mae=float(np.abs(eh - eta).mean()), kl=float(kl.mean()), argmax_dis=float(((eh > .5) != (eta > .5)).mean()),
                corr=float(np.corrcoef(eh, eta)[0, 1]))


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000; W = min(int(os.environ.get('WORKERS', '64')), 128); beam = 20
    stamp = time.strftime('%Y%m%d_%H%M%S'); os.makedirs(os.path.join(here, 'results'), exist_ok=True)
    ct = circ(3, P_T); P, D, _ = exact_joint(ct); Px0, Px1 = P[:2 ** D], P[2 ** D:]
    eta_tab = np.where(Px0 + Px1 > 0, Px1 / np.maximum(Px0 + Px1, 1e-300), 0.0)
    demd, Dg = obs_as_detector(ct.detector_error_model(decompose_errors=True))
    mm = pymatching.Matching.from_detector_error_model(demd)
    out = dict(n=n, beam=beam, runs=[])
    pool = Pool(W, initializer=init_tess, initargs=(3, beam))
    for beta in (1.0, 3.0):
        det, _ = circ(3, P_T * beta).compile_detector_sampler(seed=int(100 * beta)).sample(n, separate_observables=True)
        eta = eta_tab[enc(det)]
        z0 = np.concatenate([det, np.zeros((n, 1), np.uint8)], 1); z1 = np.concatenate([det, np.ones((n, 1), np.uint8)], 1)
        _, w0 = mm.decode_batch(z0, return_weights=True); _, w1 = mm.decode_batch(z1, return_weights=True)
        r = score(1 / (1 + np.exp(np.clip(w1 - w0, -50, 50))), eta); r.update(teacher='mwpm_gap', beta=beta); out['runs'].append(r); print(r, flush=True)
        t0 = time.time()
        C = np.array(pool.map(tess_costs, list(det), chunksize=64))
        r = score(1 / (1 + np.exp(np.clip(C[:, 1] - C[:, 0], -50, 50))), eta)
        r.update(teacher='tess_gap', beta=beta, sec=round(time.time() - t0)); out['runs'].append(r); print(r, flush=True)
        json.dump(out, open(os.path.join(here, f'results/t1_{stamp}.json'), 'w'), indent=1)
