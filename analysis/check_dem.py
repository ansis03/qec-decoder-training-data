"""Checks the L0-as-detector trick on the DEMs used for the gap labels.
Reports, for d = 3, 5, 7 at p = 3e-3: whether the DEM has repeat blocks (replacing L0 by text inside a repeat block breaks the DEM, so the teacher scripts flatten it first), and how many
decomposed components would become hyperedges (three detectors) once L0 is rewritten as an extra detector.
Pure DEM inspection: no sampling, no training."""
import stim


def circ(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p, before_round_data_depolarization=p)


for d in (3, 5, 7):
    for p in (3e-3, 5e-3):
        c = circ(d, p)
        dem = c.detector_error_model(decompose_errors=True)
        has_repeat = 'repeat' in str(dem)
        flat = dem.flattened()
        comps = tripled = with_l0 = 0
        for inst in flat:
            if inst.type != 'error':
                continue
            cur_d = cur_l = 0
            for t in inst.targets_copy() + [stim.target_separator()]:
                if t.is_separator():
                    comps += 1
                    if cur_l:
                        with_l0 += 1
                        if cur_d == 2:
                            tripled += 1
                    cur_d = cur_l = 0
                elif t.is_relative_detector_id():
                    cur_d += 1
                elif t.is_logical_observable_id():
                    cur_l += 1
        print(f'd={d} p={p}: repeat_block={has_repeat} detectors={dem.num_detectors} components={comps} with_L0={with_l0} '
              f'two_detector_with_L0={tripled}')
