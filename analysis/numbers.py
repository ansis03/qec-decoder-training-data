"""Recomputes every number in the manuscript from the copied result files, then checks the manuscript against them.

Sources are the JSON/NPZ files under data/, as the experiments wrote them.
Writes data/numbers.json (figs.py reads it, so the figures and the text share one source).
Usage: python analysis/numbers.py            recompute, print, write data/numbers.json, run the assertions in CLAIMS
       python analysis/numbers.py --no-tex   same, but do not require each claim to appear in paper/main.tex
The manuscript (paper/) is not in this repository, so here each claim is checked against its printed value only.
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _here]      # this file shadows the stdlib `numbers` module
import glob, json, math
import numpy as np
from scipy import stats

ROOT = os.path.dirname(_here)
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA, 'numbers.json')
BOOT = 20000


def load(*parts):
    return json.load(open(os.path.join(DATA, *parts), encoding='utf-8'))


def interp_N(Ns, ys, target):
    """Smallest N where y falls to target, by log-log interpolation between neighbouring grid points. None if it never does."""
    for i in range(len(Ns) - 1):
        a, b = ys[i], ys[i + 1]
        if a >= target >= b and a > b:
            return Ns[i] * (Ns[i + 1] / Ns[i]) ** (math.log(a / target) / math.log(a / b))
    return None


def boot_ci(x, f=np.mean):
    r = np.random.default_rng(0)
    x = np.asarray(x, float)
    return np.percentile(f(x[r.integers(0, len(x), (BOOT, len(x)))], axis=1), [2.5, 97.5]).tolist()


def summary(a):
    x = np.array(sorted(a.values()), float)
    s = dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), min=float(x.min()), max=float(x.max()))
    if len(x) > 1:
        s['sd'] = float(x.std(ddof=1)); s['ci_mean'] = boot_ci(x)
    return s


def paired(a, b):
    """Paired comparison over the training seeds common to both: geometric-mean ratio a/b with a bootstrap CI."""
    k = sorted(set(a) & set(b))
    x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); lr = np.log(x / y)
    out = dict(n=len(k), ratio=float(np.exp(lr.mean())), ci=np.exp(boot_ci(lr)).tolist(), wins=int((x < y).sum()), mean_diff=float((x - y).mean()))
    if len(k) >= 6:
        out['wilcoxon_p'] = float(stats.wilcoxon(x, y).pvalue)
    return out


# ---------------------------------------------------------------------------------------------- S1: sample selection
def s1():
    j = load('s1', 'results', 'compare_d5_p0.005_20261007_1551.json')
    san = load('s1', 'results', 'sanity_d5_p0.005_20261007_1550.json')
    t = {}
    for r in j['results']:
        t.setdefault(r['method'], {}).setdefault(r['N'], {})[r['seed']] = r['ler']
    Ns = [100_000, 300_000, 1_000_000]
    assert t['highp_0.01'] == t['beta_2'] and t['fail_0.2'] == t['fail_0.4']        # same data and seed: one condition each
    base = np.mean(list(t['uniform'][1_000_000].values()))
    out = dict(base=float(base), mwpm=san['mwpm']['0.005'], ns=Ns, methods={})
    for m in ('uniform', 'beta_1.25', 'highp_0.007', 'beta_1.5', 'beta_2', 'strat', 'strat_w', 'fail_0.2'):
        means = [float(np.mean(list(t[m][N].values()))) for N in Ns]
        Nstar = interp_N(Ns, means, base) if m != 'uniform' else 1e6
        per = []
        for sd in (0, 1):
            n = interp_N(Ns, [t[m][N][sd] for N in Ns], t['uniform'][1_000_000][sd]) if m != 'uniform' else 1e6
            per.append(1e6 / n if n else None)
        out['methods'][m] = dict(mean=means, seeds={str(N): [t[m][N][0], t[m][N][1]] for N in Ns}, Nstar=Nstar, eff=(1e6 / Nstar if Nstar else None), eff_seed=per)
    # share of the failure set: the pool holds 5N shots, of which about MWPM-LER are failures, and all of them are kept
    F = out['mwpm']; POOL = 5
    kept = POOL * F                                       # fraction of N
    w = 1.0 / ((1 - kept) / (POOL * (1 - F)))             # keep rate of failures (1) over keep rate of successes
    out['fail_mix'] = dict(frac=kept, boost=w, flip_threshold=1 / (1 + w))
    out['sanity_nn'] = san['nn']
    sp = {N: [abs(t[m][N][0] - t[m][N][1]) / np.mean(list(t[m][N].values())) for m in ('uniform', 'highp_0.007', 'beta_1.25', 'beta_1.5', 'beta_2', 'strat', 'strat_w', 'fail_0.2')] for N in Ns}
    out['two_seed_rel_diff_median'] = {str(N): float(np.median(v)) for N, v in sp.items()}
    return out


# ---------------------------------------------------------------------------------------------- P1: scale
def p1():
    j = load('p1', 'results', 'p1_20261008_144832.json')
    out = dict(ntest=j['ntest'], d={})
    B = [1.0, 1.5, 2.0, 3.0]
    for d in (3, 5, 7):
        r = j['runs'][f'd{d}']
        Ns = sorted({x['N'] for x in r['points']})
        tab = {(x['beta'], x['N']): x for x in r['points']}
        rows = [[tab[(b, N)]['ratio'] for b in B] for N in Ns]
        env = [min(x) for x in rows]
        out['d'][str(d)] = dict(mwpm=r['mwpm'], shape=r['shape'], ns=Ns, betas=B, ratio=rows, steps=[tab[(1.0, N)]['steps'] for N in Ns],
                                best_beta=[B[int(np.argmin(x))] for x in rows], envelope=env, ler={str(b): [tab[(b, N)]['ler'] for N in Ns] for b in B},
                                Nstar_env=interp_N(Ns, env, 1.0), hi_lo=[x[3] / x[0] for x in rows], mid_lo=[x[2] / x[0] for x in rows])
    r5, r7 = out['d']['5'], out['d']['7']
    t5 = {b: [row[i] for row in r5['ratio']] for i, b in enumerate(B)}
    t7 = {b: [row[i] for row in r7['ratio']] for i, b in enumerate(B)}
    base = t5[1.0][r5['ns'].index(1_000_000)]
    out['eff_d5'] = {str(b): 1e6 / interp_N(r5['ns'], t5[b], base) for b in (1.5, 2.0, 3.0)}
    # d = 7 extrapolation, three ways. Each uses the best envelope point at N = 3e6 (beta = 1.5, ratio 2.12)
    y1, y3 = t7[1.5][r7['ns'].index(1_000_000)], t7[1.5][r7['ns'].index(3_000_000)]
    slope = math.log(y3 / y1) / math.log(3)
    powerlaw = 3e6 * (1 / y3) ** (1 / slope)
    slope5 = math.log(t5[1.0][r5['ns'].index(3_000_000)] / t5[1.0][r5['ns'].index(300_000)]) / math.log(10)     # d = 5, beta = 1, N = 3e5 to 3e6
    rule = 3e6 * y3 ** (-1 / slope5)
    N5eq = interp_N(r5['ns'], r5['envelope'], y3)                           # shift the d = 5 envelope along N until it passes through the d = 7 point
    shift = 3e6 / N5eq
    curve = shift * 3e6
    out['extrap_d7'] = dict(y1e6=y1, y3e6=y3, slope=slope, powerlaw=powerlaw, slope5=slope5, rule=rule, N5eq=N5eq, shift=shift, curve=curve, low=min(powerlaw, rule, curve), high=max(powerlaw, rule, curve))
    return out


# ---------------------------------------------------------------------------------------------- ex12: posterior labels
def ex12():
    b = load('ex12', 'results', 'b3_20261010_134242.json')
    T = {(r['d'], r['method'], r['N']): r['ratio_mwpm'] for r in b['runs']}
    N3 = [10_000, 30_000, 100_000, 300_000]; N5 = [30_000, 100_000, 300_000, 1_000_000]
    out = dict(mwpm={str(r['d']): r['ler'] / r['ratio_mwpm'] for r in b['runs']})
    out['d3'] = {m: [T[(3, m, n)] for n in N3] for m in ('uniform', 'knob2', 'knob3', 'knob3+gap', 'knob3+exact', 'knob2+exact', 'target+exact')}
    out['d5'] = {m: [T[(5, m, n)] for n in N5] for m in ('uniform', 'knob2', 'knob3', 'knob3+gap')}
    out['ns3'] = N3; out['ns5'] = N5
    h = out['d3']['knob3']
    out['exact_eff'] = {str(n): ((interp_N(N3, h, y) / n) if interp_N(N3, h, y) else None) for n, y in zip(N3, out['d3']['knob3+exact'])}
    out['eff_d5'] = {m: 1e6 / interp_N(N5, out['d5'][m], out['d5']['uniform'][3]) for m in ('knob2', 'knob3')}
    out['eff_d3'] = {m: 3e5 / interp_N(N3, out['d3'][m], out['d3']['uniform'][3]) for m in ('knob2', 'knob3')}
    b4 = load('ex12', 'results', 'b4b_20261010_223510.json')
    out['b4b'] = [dict(method=r['method'], N=r['N'], ratio=r['ratio_mwpm']) for r in b4['runs']]
    a4 = load('ex12', 'results', 'b4a_20261010_223204.json')
    out['b4a'] = {f"{r['teacher']}/{r['beta']:g}": r['mae'] for r in a4['runs']}
    new = [r['ratio_mwpm'] for r in b4['runs'] if r['N'] == 1_000_000 and r['method'] != 'knob3+gap']
    out['b4b_best_new'] = min(new)
    out['b4b_bar'] = min(r['ratio_mwpm'] for r in b4['runs'] if r['N'] == 2_000_000 and r['method'] in ('knob2', 'knob3'))
    return out


# ---------------------------------------------------------------------------------------------- ex17: teachers and label smoothing
def ex17():
    rec = {}
    dup = 0
    for f in sorted(glob.glob(os.path.join(DATA, 'ex17', 'results', 't[2-9]_*.json'))):
        n = os.path.basename(f)
        if n.startswith('t8'):
            continue
        dd = 7 if n.startswith('t7') else 5
        for r in json.load(open(f, encoding='utf-8'))['runs']:
            k = (dd, r['method'], r['N'], r['steps'], r.get('seed', 0))
            if k in rec:
                assert rec[k] == r['ratio_mwpm'], k            # repeated runs reproduce exactly
                dup += 1
            rec[k] = r['ratio_mwpm']

    def get(d, m, N, st):
        return {k[4]: v for k, v in rec.items() if k[:4] == (d, m, N, st)}
    d5 = dict(hard=get(5, 'hard', 1_000_000, 6000), hard2e6=get(5, 'hard', 2_000_000, 9770), hard_9780=get(5, 'hard', 1_000_000, 9780),
              mwpm=get(5, 'mwpm_gap', 1_000_000, 6000), mwpm_9780=get(5, 'mwpm_gap', 1_000_000, 9780), tess=get(5, 'tess_gap', 1_000_000, 6000),
              tess_9780=get(5, 'tess_gap', 1_000_000, 9780), tess_T2=get(5, 'tess_T2', 1_000_000, 6000), ls05=get(5, 'hard_ls05', 1_000_000, 6000),
              ls10=get(5, 'hard_ls10', 1_000_000, 6000), ls20=get(5, 'hard_ls20', 1_000_000, 6000))
    d7 = dict(hard=get(7, 'hard', 1_000_000, 6000), hard15=get(7, 'hard', 1_500_000, 7330), hard20=get(7, 'hard', 2_000_000, 9770), hard30=get(7, 'hard', 3_000_000, 14650),
              hard40=get(7, 'hard', 4_000_000, 19540), mwpm=get(7, 'mwpm_gap', 1_000_000, 6000), mwpm_T2=get(7, 'mwpm_T2', 1_000_000, 6000), tess05=get(7, 'tess_gap', 500_000, 6000),
              tess=get(7, 'tess_gap', 1_000_000, 6000), tess_T2=get(7, 'tess_T2', 1_000_000, 6000), tess_T4=get(7, 'tess_T4', 1_000_000, 6000), tess_T8=get(7, 'tess_T8', 1_000_000, 6000),
              ls10=get(7, 'hard_ls10', 1_000_000, 6000), ls30=get(7, 'hard_ls30', 1_000_000, 6000))
    out = dict(runs=len(rec), duplicates=dup, d5={k: summary(v) for k, v in d5.items()}, d7={k: summary(v) for k, v in d7.items()},
               d5_seeds={k: {str(s): x for s, x in v.items()} for k, v in d5.items()}, d7_seeds={k: {str(s): x for s, x in v.items()} for k, v in d7.items()})
    H, L, TG, M, H2 = d5['hard'], d5['ls10'], d5['tess'], d5['mwpm'], d5['hard2e6']
    out['d5_paired'] = dict(ls10_vs_tess=paired(L, TG), ls10_vs_hard=paired(L, H), tess_vs_hard=paired(TG, H), mwpm_vs_hard=paired(M, H), tess_vs_2e6=paired(TG, H2), ls10_vs_2e6=paired(L, H2),
                            hard9780_vs_hard=paired(d5['hard_9780'], H), hard9780_vs_2e6=paired(d5['hard_9780'], H2), tess9780_vs_tess=paired(d5['tess_9780'], TG), tess_vs_hard_9780=paired(d5['tess_9780'], d5['hard_9780']))
    out['d7_paired'] = dict(ls10_vs_hard=paired(d7['ls10'], d7['hard']), mwpm_vs_hard=paired(d7['mwpm'], d7['hard']), tess_vs_hard=paired(d7['tess'], d7['hard']), T4_vs_hard=paired(d7['tess_T4'], d7['hard']), ls30_vs_hard=paired(d7['ls30'], d7['hard']))
    # d = 7 learning curve of hard labels on medians, and the data factor implied by each soft label
    Ns = [1_000_000, 1_500_000, 2_000_000, 3_000_000, 4_000_000]
    med = [out['d7'][k]['median'] for k in ('hard', 'hard15', 'hard20', 'hard30', 'hard40')]
    out['d7_curve'] = dict(ns=Ns, median=med, slope_last=math.log(med[4] / med[3]) / math.log(4 / 3))
    out['d7_curve']['powerlaw_Nstar'] = 4e6 * med[4] ** (-1 / out['d7_curve']['slope_last'])
    out['d7_eff'] = {k: interp_N(Ns, med, out['d7'][k]['median']) / 1e6 for k in ('mwpm', 'ls10')}
    # a run is "failed" when its ratio exceeds twice the median of hard labels at the same N (same reference for every label)
    ref = out['d7']['hard']['median'] * 2
    out['d7_failed'] = {k: dict(failed=sum(v > ref for v in d7[k].values()), n=len(d7[k])) for k in ('hard', 'mwpm', 'mwpm_T2', 'ls10', 'ls30', 'tess', 'tess_T2', 'tess_T4', 'tess_T8')}
    out['d7_failed_ref'] = ref
    hard_runs = [(k, d7[k]) for k in ('hard', 'hard15', 'hard20', 'hard30', 'hard40')]
    soft = [k for k in ('mwpm', 'mwpm_T2', 'tess', 'tess_T2', 'tess_T4', 'tess_T8', 'ls10', 'ls30')]
    nf, nn = sum(out['d7_failed'][k]['failed'] for k in soft), sum(out['d7_failed'][k]['n'] for k in soft)
    out['d7_soft_failed'] = dict(failed=nf, n=nn, fisher_vs_hard_1e6=float(stats.fisher_exact([[nf, nn - nf], [0, 8]])[1]), fisher_vs_hard_all=float(stats.fisher_exact([[nf, nn - nf], [1, 23]])[1]))
    out['d7_hard_failed'] = dict(failed=int(sum(sum(v > 2 * np.median(list(c.values())) for v in c.values()) for _, c in hard_runs)), n=sum(len(c) for _, c in hard_runs), worst=max(d7['hard15'].values()))
    ref5 = 2 * out['d5']['hard']['median']
    out['d5_failed'] = {k: dict(failed=sum(v > ref5 for v in d5[k].values()), n=len(d5[k])) for k in ('hard', 'mwpm', 'tess', 'tess_T2', 'ls05', 'ls10', 'ls20')}
    out['d5_failed_ref'] = ref5
    out['d5_failed']['hard2e6'] = dict(failed=sum(v > 2 * out['d5']['hard2e6']['median'] for v in d5['hard2e6'].values()), n=len(d5['hard2e6']))    # its own N
    t1 = load('ex17', 'results', 't1_20261013_180457.json')
    out['t1'] = t1['runs']; out['t1_n'] = t1['n']; out['t1_beam'] = t1['beam']
    out['t8'] = load('ex17', 'results', 't8_20261014_021112.json')
    t2 = load('ex17', 'results', 't2_20261013_182227.json')
    out['t2'] = [dict(method=r['method'], N=r['N'], steps=r['steps'], ratio=r['ratio_mwpm']) for r in t2['runs']]
    out['mwpm'] = {'5': t2['mwpm'], '7': 0.001341}
    assert all(json.load(open(f, encoding='utf-8'))['mwpm'] == 0.001341 for f in glob.glob(os.path.join(DATA, 'ex17', 'results', 't7_*.json')))
    return out


def overlap():
    """On stored beta = 3 labels: how often the near-ML teacher also fails on the shots MWPM fails on."""
    out = {}
    for name, f in (('5', 't2_labels_20261013_182227.npz'), ('7', 't6_labels7_20261013_222513.npz')):
        a = np.load(os.path.join(DATA, 'ex17', 'results', f))
        t, m, h = a['tgap'], a['mgap'], a['hard']
        mw = (m > .5) != (h > .5); tw = (t > .5) != (h > .5)
        # calibration of the target-weight teachers against the labels actually drawn at beta*p
        bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(t, bins) - 1, 0, 9)
        ece = sum(abs(h[idx == i].mean() - t[idx == i].mean()) * (idx == i).mean() for i in range(10) if (idx == i).any())
        idm = np.clip(np.digitize(m, bins) - 1, 0, 9)
        ecem = sum(abs(h[idm == i].mean() - m[idm == i].mean()) * (idm == i).mean() for i in range(10) if (idm == i).any())
        out[name] = dict(n=len(h), mwpm_wrong=float(mw.mean()), tess_wrong=float(tw.mean()), both=float((mw & tw).mean()), tess_given_mwpm=float(tw[mw].mean()), mwpm_given_tess=float(mw[tw].mean()), ece_tess=float(ece), ece_mwpm=float(ecem))
    return out


def supp():
    """Pre-registered controls A, B, C (experiments/checks/u_runs.py). The verdicts are the registered rules, evaluated here."""
    files = sorted(glob.glob(os.path.join(DATA, 'supp', 'u_*.json')))
    if not files:
        return None
    A, B, C = {}, {}, []
    mw = {}
    for f in files:
        j = json.load(open(f, encoding='utf-8'))
        mw[j['part']] = j['mwpm']
        for r in j['runs']:
            if j['part'] == 'a':
                A.setdefault(r['method'], {})[r['seed']] = r['ratio_mwpm']
            elif j['part'] == 'b':
                B.setdefault((r['method'], r['N']), {})[r['seed']] = r['ratio_mwpm']
            else:
                C.append(r)
    out = dict(mwpm=mw)
    pa = paired(A['hard_ls10'], A['hard'])
    lo, hi = pa['ci']; REF = (0.691, 0.819)
    verdict = 'generic' if (hi < 1 and lo <= REF[1] and hi >= REF[0]) else ('compatible' if (lo <= 1 <= hi or lo > REF[1]) else 'undetermined')
    ref_a = 2 * float(np.median(list(A['hard'].values())))
    failed = [s for s, x in A['hard_ls10'].items() if x > ref_a]
    kept = {s: x for s, x in A['hard_ls10'].items() if s not in failed}
    out['A'] = dict(hard=summary(A['hard']), ls10=summary(A['hard_ls10']), paired=pa, verdict=verdict, seeds={k: {str(s): x for s, x in v.items()} for k, v in A.items()},
                    failed_seeds=failed, posthoc_ratio=paired(kept, A['hard'])['ratio'], hard_failed=sum(x > ref_a for x in A['hard'].values()))
    NS = [10_000, 30_000, 100_000, 300_000]
    med = {m: [float(np.median(list(B[(m, n)].values()))) for n in NS] for m in ('hard', 'hard_ls10', 'exact')}
    fac = {str(n): (lambda x: x / n if x else None)(interp_N(NS, med['hard'], y)) for n, y in zip(NS, med['exact'])}
    facls = {str(n): (lambda x: x / n if x else None)(interp_N(NS, med['hard'], y)) for n, y in zip(NS, med['hard_ls10'])}
    pe = {str(n): paired(B[('hard_ls10', n)], B[('exact', n)]) for n in NS}
    ph = {str(n): paired(B[('exact', n)], B[('hard', n)]) for n in NS}
    pl = {str(n): paired(B[('hard_ls10', n)], B[('hard', n)]) for n in NS}
    f1, f3 = fac['10000'], fac['30000']
    b1 = 'claim' if (f1 and f3 and f1 >= 3 and f3 >= 3) else ('retract' if (f1 is None or f3 is None or f1 < 2 or f3 < 2) else 'small')
    worse = [n for n in NS if pe[str(n)]['ci'][0] > 1]
    b2 = 'not_reproduced' if len(worse) >= 2 else ('reproduced' if all(pe[str(n)]['ci'][0] <= 1 <= pe[str(n)]['ci'][1] for n in NS) else 'partly')
    out['B'] = dict(ns=NS, median=med, summary={f'{m}@{n}': summary(B[(m, n)]) for m in ('hard', 'hard_ls10', 'exact') for n in NS}, factor_exact=fac, factor_ls=facls, ls_vs_exact=pe, exact_vs_hard=ph, ls_vs_hard=pl, b1=b1, b2=b2, b2_ns=worse)
    cell = {}
    for r in C:
        for sel in ('final', 'best'):
            cell.setdefault((r['beta'], r['steps'], sel), {})[r['seed']] = r['ler_' + sel]
    medc = {f'{b:g}/{st}/{sel}': float(np.median(list(v.values()))) for (b, st, sel), v in cell.items()}
    RC = medc['2/6000/best'] / medc['2/1764/final']
    same = sum(cell[(2.0, 6000, 'best')][s] > cell[(2.0, 1764, 'final')][s] for s in cell[(2.0, 1764, 'final')])
    n = len(cell[(2.0, 1764, 'final')])
    c = 'explains' if (RC >= 1.3 and same >= 3) else ('not_explained' if 0.87 <= RC <= 1.15 else 'partly')
    out['C'] = dict(median=medc, RC=RC, same_direction=int(same), n=n, verdict=c, beta_gain_short=medc['2/1764/final'] / medc['1/1764/final'], beta_gain_long=medc['2/6000/best'] / medc['1/6000/best'],
                    seeds={f'{b:g}/{st}/{sel}': {str(s): x for s, x in v.items()} for (b, st, sel), v in cell.items()})
    return out


def params():
    """Trainable parameters of CNN3D(64, 4) for each grid, by hand (conv 3x3x3 + bias; dense 256 head)."""
    out = {}
    conv = (27 * 1 + 1) * 64 + 3 * (27 * 64 + 1) * 64
    for d in (3, 5, 7):
        flat = 64 * (d + 1) ** 3
        out[str(d)] = conv + (flat + 1) * 256 + (256 + 1)
    return out


V = dict(s1=s1(), p1=p1(), ex12=ex12(), ex17=ex17(), overlap=overlap(), params=params(), supp=supp())
# parity scale at p = 0.3%, beta = 3: d = 3 from ex12 (one seed), d = 5 from ex17 (nine seeds), d = 7 extrapolated from ex17 medians
V['scale03'] = dict(Nstar3=interp_N(V['ex12']['ns3'], V['ex12']['d3']['knob3'], 1.0), Nstar5=interp_N([1_000_000, 2_000_000], [V['ex17']['d5']['hard']['mean'], V['ex17']['d5']['hard2e6']['mean']], 1.0),
                   ratio5=V['ex17']['d5']['hard2e6']['mean'], Nstar7_low=V['ex17']['d7_curve']['powerlaw_Nstar'])
V['scale03']['grow35'] = V['scale03']['Nstar5'] / V['scale03']['Nstar3']
V['scale03']['grow57'] = V['scale03']['Nstar7_low'] / V['scale03']['Nstar5']
V['scale05'] = dict(grow35=3e6 / V['p1']['d']['3']['Nstar_env'], grow57_low=V['p1']['extrap_d7']['low'] / 3e6, grow57_high=V['p1']['extrap_d7']['high'] / 3e6)
json.dump(V, open(OUT, 'w', encoding='utf-8'), indent=1)


def fmt_sci(x, digits=2):
    """1.68e-2 -> '1.68e-2' (no zero padding), for matching the text of tables."""
    m, e = f'{x:.{digits}e}'.split('e')
    return f'{m}e{int(e)}'


# ---------------------------------------------------------------------------------------------- manuscript check
# Each claim is (name, computed value, value as printed in the manuscript, literal string that must appear in main.tex).
# Filled in as the manuscript is written; numbers go in main.tex by hand, and this list is what ties them to the data.
CLAIMS = []
exec(open(os.path.join(_here, 'claims.py'), encoding='utf-8').read()) if os.path.exists(os.path.join(_here, 'claims.py')) else None

def manuscript():
    """The prose of the paper: abstract and sections, comments removed. Tables are generated, so they are not listed here."""
    out = []
    for f in ['abstract.tex'] + sorted(os.path.basename(x) for x in glob.glob(os.path.join(ROOT, 'paper', 'sec_*.tex'))):
        t = open(os.path.join(ROOT, 'paper', f), encoding='utf-8').read()
        out.append('\n'.join(l for l in t.splitlines() if not l.lstrip().startswith('%')))
    return '\n'.join(out)


def unregistered(tex, claims):
    """Numbers in the prose that no claim's literal covers, for the human check in the review step."""
    import re
    t = re.sub(r'\\(cite|label|ref|input|includegraphics|Description)(\[[^\]]*\])?\{[^}]*\}', ' ', tex)
    t = re.sub(r'\\begin\{tabular\}\{[^\n]*', ' ', t)
    for _, _, _, lit in claims:
        t = t.replace(lit, ' ')
    return sorted(set(re.findall(r'(?<![A-Za-z{\\])\d+(?:\.\d+)?(?:e\d+)?(?:\\,\d+)?', t)))


if __name__ == '__main__':
    tex = manuscript() if os.path.exists(os.path.join(ROOT, 'paper', 'abstract.tex')) and '--no-tex' not in sys.argv else None
    bad = 0
    for name, val, printed, lit in CLAIMS:
        dec = len(printed.split('.')[1]) if '.' in printed else 0
        ok = abs(float(printed) - val) <= 0.5 * 10 ** (-dec) + 1e-9
        found = tex is None or lit in tex
        if not (ok and found):
            bad += 1
            print(f'FAIL {name}: computed {val}, printed {printed}, in tex {found}')
    print(f'{len(CLAIMS)} claims checked, {bad} failed. Wrote {OUT}')
    if tex and '--scan' in sys.argv:
        print('numbers in prose not covered by a claim:', unregistered(tex, CLAIMS))
    assert bad == 0
