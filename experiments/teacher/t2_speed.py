import time, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_train as T
T.init_tess(20)
det, _ = T.circ(5, T.P_T * 3).compile_detector_sampler(seed=1).sample(40, separate_observables=True)
t = time.time()
for s in det: T.tess_gap(s)
print(f'{(time.time()-t)/40*1000:.1f} ms per shot (two decodes)')
