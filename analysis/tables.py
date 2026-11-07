"""Writes the numeric tables of the manuscript as LaTeX fragments in out/tables/, from data/numbers.json only.
Numbers in tables are therefore never typed by hand. Prose numbers are checked by numbers.py through claims.py."""
import os, sys, json
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _here]      # analysis/numbers.py shadows the stdlib `numbers` module

ROOT = os.path.dirname(_here)
V = json.load(open(os.path.join(ROOT, 'data', 'numbers.json'), encoding='utf-8'))
OUT = os.path.join(ROOT, 'out', 'tables')
os.makedirs(OUT, exist_ok=True)


def write(name, text):
    open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8').write(text + '\\bottomrule\n')     # \bottomrule right after \input breaks tabular


def e2(x):
    return f'{100 * x:.2f}'


def t_d3():
    su = V['supp']
    if su is None:
        return
    B = su['B']
    ns = [str(n) for n in B['ns']]

    def med(k, label):
        return label + ' & ' + ' & '.join(f'{x:.2f}' for x in B['median'][k]) + ' \\\\'

    def rat(k, label):
        return label + ' & ' + ' & '.join(f"{B[k][n]['ratio']:.2f} [{B[k][n]['ci'][0]:.2f}, {B[k][n]['ci'][1]:.2f}]" for n in ns) + ' \\\\'
    rows = [med('hard', 'Hard targets'), med('hard_ls10', 'Label smoothing'), med('exact', 'Exact posterior'), '\\addlinespace', rat('exact_vs_hard', 'Exact / hard'), rat('ls_vs_hard', 'Smoothing / hard'), rat('ls_vs_exact', 'Smoothing / exact')]
    write('d3', '\n'.join(rows) + '\n')


def t_labels():
    e = V['ex17']
    fail5, fail7 = e['d5_failed'], e['d7_failed']

    def cells(s, f):
        if s is None:
            return ['', '', '', '']
        return [str(s['n']), f"{s['mean']:.2f} $\\pm$ {s['sd']:.2f}", f"{s['median']:.2f}", f"{s['min']:.2f}--{s['max']:.2f}", f"{f['failed']}/{f['n']}" if f else '--'][0:5]

    rows = [('Hard', 'hard', 'hard'), ('MWPM gap', 'mwpm', 'mwpm'), ('MWPM gap, $T=2$', None, 'mwpm_T2'), ('Tesseract gap', 'tess', 'tess'), ('Tesseract gap, $T=2$', 'tess_T2', 'tess_T2'), ('Tesseract gap, $T=4$', None, 'tess_T4'), ('Tesseract gap, $T=8$', None, 'tess_T8'),
            ('Smoothing, $\\varepsilon=0.05$', 'ls05', None), ('Smoothing, $\\varepsilon=0.1$', 'ls10', 'ls10'), ('Smoothing, $\\varepsilon=0.2$', 'ls20', None), ('Smoothing, $\\varepsilon=0.3$', None, 'ls30'), ('Hard, $N=2\\times10^{6}$', 'hard2e6', None)]
    out = []
    for lbl, k5, k7 in rows:
        c5 = cells(e['d5'][k5], fail5.get(k5)) if k5 else [''] * 5
        c7 = cells(e['d7'][k7], fail7.get(k7)) if k7 else [''] * 5
        out.append(lbl + ' & ' + ' & '.join(c5 + c7) + ' \\\\')
    write('labels', '\n'.join(out) + '\n')


t_d3(); t_labels()
print('tables written to', OUT)
