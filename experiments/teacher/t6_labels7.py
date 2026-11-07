"""T6: labels for d=7 (surface code memory-Z d=7 r=7, target p=3e-3, beta=3). Training syndromes: sampler seed 700,
NMAX shots (sampled at once, so slices are reproducible only with the same NMAX). Labels:
  hard (all NMAX), mwpm_gap (first NL), tess_gap (first NL, Tesseract beam BEAM, DEM flattened before L0->detector).
Output: results/t6_labels7_<stamp>.npz (+ progress lines)."""
import os, sys, time, numpy as np, stim, pymatching
from multiprocessing import Pool
here = os.path.dirname(os.path.abspath(__file__)); stamp = time.strftime('%Y%m%d_%H%M%S')
P_T, BETA, D_ = 3e-3, 3.0, 7
G = {}


def circ(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p, before_round_data_depolarization=p)


def obs_det(dem):
    return stim.DetectorErrorModel(str(dem.flattened()).replace(' L0', f' D{dem.num_detectors}'))


def init_tess(beam):
    from tesseract_decoder import tesseract
    G['dec'] = tesseract.TesseractConfig(dem=obs_det(circ(D_, P_T).detector_error_model(decompose_errors=False)), det_beam=beam,
                                         beam_climbing=True, no_revisit_dets=True).compile_decoder()


def tess_gap(det):
    dec = G['dec']; c = [dec.cost_from_errors(dec.decode_to_errors(np.concatenate([det.astype(bool), [bool(k)]]))) for k in (0, 1)]
    return 1 / (1 + np.exp(np.clip(c[1] - c[0], -50, 50)))


if __name__ == '__main__':
    NMAX, NL, BEAM = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    W = min(int(os.environ.get('WORKERS', '64')), 240)
    os.makedirs(os.path.join(here, 'results'), exist_ok=True); t0 = time.time()
    det, obs = circ(D_, P_T * BETA).compile_detector_sampler(seed=700).sample(NMAX, separate_observables=True)
    hard = obs[:, 0].astype(np.float32)
    mm = pymatching.Matching.from_detector_error_model(obs_det(circ(D_, P_T).detector_error_model(decompose_errors=True)))
    z = np.zeros((NL, 1), np.uint8)
    _, w0 = mm.decode_batch(np.concatenate([det[:NL], z], 1), return_weights=True)
    _, w1 = mm.decode_batch(np.concatenate([det[:NL], z + 1], 1), return_weights=True)
    mgap = (1 / (1 + np.exp(np.clip(w1 - w0, -50, 50)))).astype(np.float32); print(f'mwpm labels [{time.time()-t0:.0f}s]', flush=True)
    tg = np.zeros(NL, np.float32); CH = 50_000
    with Pool(W, initializer=init_tess, initargs=(BEAM,)) as pool:
        for a in range(0, NL, CH):
            tg[a:a + CH] = pool.map(tess_gap, list(det[a:a + CH]), chunksize=64)
            print(f'tesseract {a + CH}/{NL} [{time.time()-t0:.0f}s]', flush=True)
            np.savez_compressed(os.path.join(here, f'results/t6_labels7_{stamp}.npz'), tgap=tg[:a + CH], mgap=mgap[:a + CH],
                                hard=hard[:a + CH], nmax=NMAX, beam=BEAM)
    print('done', flush=True)
