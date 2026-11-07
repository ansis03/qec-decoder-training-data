"""B4a: teacher quality check at d=3 (exact posterior known). Cheap teachers for posterior relabeling:
  gap       : sigmoid(-(W1-W0)) from plain MWPM (B3)
  corrgap   : same with pymatching enable_correlations=True
  cal(X)    : logistic recalibration of teacher X: eta = sigmoid(a*(W0-W1) + b*|x| + c), fit on TARGET-noise samples with
              hard labels (free in simulation; eta_T(x) is a function of x, so the fit transfers to knob samples)
Scored on beta=3 knob samples against exact target posterior: mean |eta_hat - eta|, KL, argmax disagreement.
Output: results/b4a_<stamp>.json"""
import os, sys, json, time
import numpy as np, stim, pymatching, torch
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from exact_posterior import exact_joint, enc
stamp = time.strftime('%Y%m%d_%H%M%S')
P_T = 3e-3


def circ(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p, before_round_data_depolarization=p)


def gap_matcher(c, corr=False):
    D = c.num_detectors
    txt = str(c.detector_error_model(decompose_errors=True)).replace(' L0', f' D{D}')
    return pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(txt), enable_correlations=corr), D


def gap_delta(m, det, corr):
    n = len(det)
    z0 = np.concatenate([det, np.zeros((n, 1), np.uint8)], 1); z1 = np.concatenate([det, np.ones((n, 1), np.uint8)], 1)
    _, w0 = m.decode_batch(z0, return_weights=True, enable_correlations=corr)
    _, w1 = m.decode_batch(z1, return_weights=True, enable_correlations=corr)
    return w0 - w1      # logit of P(flip)


def feats(delta, det):
    return np.stack([delta, det.sum(1).astype(np.float64), np.ones(len(det))], 1)


def fit_logistic(X, y, iters=200):
    Xt = torch.tensor(X, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    w = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w, yt); l.backward(); return l
    opt.step(cl); return w.detach().numpy()


def sig(z): return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def score(eh, eta):
    e = np.clip(eh, 1e-9, 1 - 1e-9); t = np.clip(eta, 1e-12, 1 - 1e-12)
    kl = t * np.log(t / e) + (1 - t) * np.log((1 - t) / (1 - e))
    return dict(mae=float(np.abs(eh - eta).mean()), kl=float(kl.mean()), argmax_dis=float(((eh > .5) != (eta > .5)).mean()),
                corr=float(np.corrcoef(eh, eta)[0, 1]))


if __name__ == '__main__':
    ct = circ(3, P_T)
    P, D, _ = exact_joint(ct); Px0, Px1 = P[:2 ** D], P[2 ** D:]
    eta_tab = np.where(Px0 + Px1 > 0, Px1 / np.maximum(Px0 + Px1, 1e-300), 0.0)
    gm, _ = gap_matcher(ct, True)
    dc, oc = ct.compile_detector_sampler(seed=9).sample(1_000_000, separate_observables=True)     # calibration set (target)
    out = dict(runs=[])
    for beta in (1.0, 3.0):
        det, _ = circ(3, P_T * beta).compile_detector_sampler(seed=int(100 * beta)).sample(300_000, separate_observables=True)
        eta = eta_tab[enc(det)]
        for corr in (False, True):
            name = 'corrgap' if corr else 'gap'
            dl = gap_delta(gm, det, corr)
            r = score(sig(dl), eta); r.update(teacher=name, beta=beta); out['runs'].append(r); print(r, flush=True)
            w = fit_logistic(feats(gap_delta(gm, dc, corr), dc), oc[:, 0].astype(np.float64))
            r = score(sig(feats(dl, det) @ w), eta); r.update(teacher='cal_' + name, beta=beta, w=w.tolist())
            out['runs'].append(r); print(r, flush=True)
    os.makedirs(os.path.join(here, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(here, f'results/b4a_{stamp}.json'), 'w'), indent=1)
