"""Tesseract gap-label speed at d=7 (beta=3 syndromes), single process, beams 20 and 10."""
import time, numpy as np, stim
from tesseract_decoder import tesseract
P_T = 3e-3
c = stim.Circuit.generated('surface_code:rotated_memory_z', distance=7, rounds=7, after_clifford_depolarization=P_T,
                           before_measure_flip_probability=P_T, after_reset_flip_probability=P_T, before_round_data_depolarization=P_T)
dem = c.detector_error_model(decompose_errors=False); dem = stim.DetectorErrorModel(str(dem.flattened()).replace(' L0', f' D{dem.num_detectors}'))
cb = stim.Circuit.generated('surface_code:rotated_memory_z', distance=7, rounds=7, after_clifford_depolarization=3 * P_T,
                            before_measure_flip_probability=3 * P_T, after_reset_flip_probability=3 * P_T, before_round_data_depolarization=3 * P_T)
det, _ = cb.compile_detector_sampler(seed=1).sample(20, separate_observables=True)
for beam in (20, 10, 5):
    dec = tesseract.TesseractConfig(dem=dem, det_beam=beam, beam_climbing=True, no_revisit_dets=True).compile_decoder()
    t = time.time()
    for s in det:
        for k in (0, 1): dec.cost_from_errors(dec.decode_to_errors(np.concatenate([s.astype(bool), [bool(k)]])))
    print(f'beam {beam}: {(time.time()-t)/len(det):.2f} s per shot (two decodes)', flush=True)
