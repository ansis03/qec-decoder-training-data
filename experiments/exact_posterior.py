"""Exact posterior of a logical flip for small circuits (d = 3: 24 detectors).
The joint distribution P(x, y) over all 2^(D+1) outcomes comes from the Walsh-Hadamard (XOR-convolution) identity
P_hat(k) = prod_i (1 - 2 p_i [k . v_i odd]), where v_i is the (detectors, observable) pattern of error mechanism i.
Then eta(x) = P(x, 1) / P(x). Used by posterior/, teacher/t1_teacher.py and checks/.
These functions are unchanged from a larger pilot script, of which the experiments used only this part."""
import numpy as np


def fwht(a):
    h = 1; n = a.shape[0]
    while h < n:
        a = a.reshape(-1, 2, h)
        x, y = a[:, 0, :].copy(), a[:, 1, :].copy()
        a[:, 0, :] = x + y; a[:, 1, :] = x - y
        a = a.reshape(n); h *= 2
    return a


def exact_joint(circ):
    dem = circ.detector_error_model(decompose_errors=False)
    D = circ.num_detectors; B = D + 1
    mech = {}
    for inst in dem.flattened():
        if inst.type != 'error': continue
        q = inst.args_copy()[0]; v = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id(): v ^= 1 << t.val
            elif t.is_logical_observable_id() and t.val == 0: v ^= 1 << D
        if v == 0: continue
        o = mech.get(v, 0.0); mech[v] = o * (1 - q) + q * (1 - o)
    k = np.arange(2 ** B, dtype=np.uint64)
    F = np.ones(2 ** B)
    for v, q in mech.items():
        odd = (np.bitwise_count(k & np.uint64(v)) & 1).astype(bool)
        F[odd] *= (1 - 2 * q)
    P = fwht(F) / 2 ** B
    P = np.clip(P, 0, None)
    return P.reshape(2, 2 ** D) if False else P, D, len(mech)


def enc(det):
    return (det.astype(np.uint64) * (np.uint64(1) << np.arange(det.shape[1], dtype=np.uint64))).sum(1)
