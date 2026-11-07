"""B4b: last round for posterior relabeling at d=5 (p_target=3e-3), beta=3 samples.
Pre-registered pass (2026-10-10): best new method at N=1e6 has LER <= min(knob2 hard, knob3 hard) at N=2e6 (>= 2x data
efficiency). Otherwise the line is closed.
  ref   : knob2/knob3 hard @2e6, knob3+gap @2e6 (+ knob3+gap @3e5,1e6 retrained as teacher = reproducibility check vs B3)
  mix_a : y = (1-a)*gap + a*hard(beta=3)                         a in {0.3, 0.6}
  sd    : self-distillation, y = teacher net prob (teacher = knob3+gap trained on the same N samples)
  sdmix : y = 0.5*gap + 0.5*teacher
Output: results/b4b_<stamp>.json"""
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'b3_relabel.py'), encoding='utf-8').read().split('out = dict(p_target')[0])
here = os.path.dirname(os.path.abspath(__file__))
stamp = time.strftime('%Y%m%d_%H%M%S')


@torch.no_grad()
def prob(model, G, bs=8192):
    model.eval(); o = []
    for i in range(0, len(G), bs):
        o.append(torch.sigmoid(model(torch.as_tensor(G[i:i + bs], device=dev).float()[:, None])).cpu().numpy().reshape(-1))
    return np.concatenate(o).astype(np.float32)


d = 5
ct = circ(d, P_T); lay = Layout(ct)
det_te, obs_te = ct.compile_detector_sampler(seed=777).sample(1_000_000, separate_observables=True)
Gte, yte = grids(lay, det_te), obs_te[:, 0]
det_va, obs_va = ct.compile_detector_sampler(seed=555).sample(100_000, separate_observables=True)
Gva, yva = grids(lay, det_va), obs_va[:, 0]
mw = float((pymatching.Matching.from_detector_error_model(ct.detector_error_model(decompose_errors=True)).decode_batch(det_te)[:, 0] != obs_te[:, 0]).mean())
gm, Dg = gap_matcher(ct)
print(f'd={d} MWPM LER={mw:.3e}', flush=True)
out = dict(p_target=P_T, d=d, mwpm=mw, runs=[])


def run(name, G, y, N):
    t0 = time.time(); model = train(G[:N], y[:N], Gva, yva); L = ler(model, Gte, yte)
    rec = dict(method=name, N=N, ler=L, ratio_mwpm=L / mw, sec=round(time.time() - t0)); out['runs'].append(rec)
    print(f"  {name:12s} N={N:>8} LER={L:.3e} (x{L/mw:.2f} MWPM) [{rec['sec']}s]", flush=True)
    json.dump(out, open(os.path.join(here, f'results/b4b_{stamp}.json'), 'w'), indent=1)
    return model


NMAX = 2_000_000
pool = {}
for beta in (2.0, 3.0):
    det, obs = circ(d, P_T * beta).compile_detector_sampler(seed=int(100 * beta)).sample(NMAX, separate_observables=True)
    pool[beta] = (grids(lay, det), obs[:, 0].astype(np.float32), det)
# references at 2e6 (denominator of the pass criterion)
for beta in (2.0, 3.0):
    run(f'knob{int(beta)}', pool[beta][0], pool[beta][1], NMAX)
G3, hard3, det3 = pool[3.0]; del pool
gap3 = gap_posterior(gm, Dg, det3).astype(np.float32)
run('knob3+gap', G3, gap3, NMAX)
for N in (300_000, 1_000_000):
    teacher = run('knob3+gap', G3, gap3, N)
    tp = prob(teacher, G3[:N]); del teacher; torch.cuda.empty_cache()
    run('sd', G3, tp, N)
    run('sdmix', G3, 0.5 * gap3[:N] + 0.5 * tp, N)
    for a in ((0.3, 0.6) if N == 1_000_000 else (0.3,)):
        run(f'mix{a}', G3, (1 - a) * gap3 + a * hard3, N)
print('done', stamp)
