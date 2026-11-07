"""Draws the manuscript figures from data/numbers.json (written by numbers.py), so figures and text share one source.
Output: out/figs/*.pdf, set by pdflatex through the pgf backend so figure text uses the paper's Libertine and Biolinum.
All figures are full text width (5.95 in) with one message each. Grey is the reference, blue a higher training error rate,
orange the item the reader should look at. Series also differ by marker, line pattern, hatch or direct label, as the ACM template asks,
so they survive grayscale printing."""
import os, sys, json
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _here]      # analysis/numbers.py shadows the stdlib `numbers` module
import numpy as np
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = json.load(open(os.path.join(ROOT, 'data', 'numbers.json'), encoding='utf-8'))
OUT = os.path.join(ROOT, 'out', 'figs')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3dc'
GRAY, BLUE, ORANGE, TEAL, PALE = '#85857f', '#1f6fb2', '#c8551b', '#0b9483', '#86b6ef'
GREEN = '#199e70'      # third series next to BLUE and ORANGE; TEAL sits too close to BLUE (normal-vision Delta E 15.0), this one passes all pairs
RAMP = {1.5: '#86b6ef', 2.0: '#3987e5', 3.0: '#0d366b'}
MARK = {1.5: 's', 2.0: '^', 3.0: 'D'}
WIDTH = 5.95
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False, 'font.family': 'sans-serif',
    'pgf.preamble': r'\usepackage[T1]{fontenc}\usepackage{libertine}\usepackage[libertine]{newtxmath}\renewcommand{\familydefault}{\sfdefault}',
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.edgecolor': '#b9b8ae', 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.solid_capstyle': 'round', 'hatch.linewidth': 0.6})


def clean(ax, grid='both'):
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis=grid, color=GRID, lw=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def title(ax, text):
    ax.set_title(text, loc='left', fontweight='bold', pad=4)


def logx(ax, ticks, labels):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.set_xticklabels(labels); ax.xaxis.set_minor_locator(NullLocator())


def logy(ax, ticks, labels):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.set_yticklabels(labels); ax.yaxis.set_minor_locator(NullLocator())


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), backend='pgf')
    plt.close(fig)


def bars(ax, xs, heights, los, his, colors, width=0.68, hatch=None):
    """Vertical bars from zero with error bars, as in Yan et al. Fig. 3(b) and AlphaQubit Fig. 2(B)."""
    ax.bar(xs, heights, width=width, color=colors, edgecolor='white', linewidth=0.5, hatch=hatch, zorder=2)
    for x, h, lo, hi in zip(xs, heights, los, his):
        ax.errorbar([x], [h], yerr=[[h - lo], [hi - h]], color=INK, capsize=1.8, elinewidth=0.7, capthick=0.7, lw=0, zorder=3)


def fig_sampling():
    """Sampling study: test LER by sampling method at a small and a large training set."""
    s = V['s1']['methods']; mw = 100 * V['s1']['mwpm']
    rows = [('uniform', r'$\beta=1$', GRAY), ('beta_1.25', r'$\beta=1.25$', BLUE), ('highp_0.007', r'$\beta=1.4$', BLUE), ('beta_1.5', r'$\beta=1.5$', BLUE), ('beta_2', r'$\beta=2$', BLUE),
            ('strat', 'Stratified', GRAY), ('strat_w', 'Strat., weighted', GRAY), ('fail_0.2', 'Failure-enriched', ORANGE)]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 1.6), constrained_layout=True)
    idx = {'100000': 0, '1000000': 2}
    xs = np.arange(len(rows))
    for ax, n, label in ((axs[0], '100000', r'(a) $N=10^{5}$'), (axs[1], '1000000', r'(b) $N=10^{6}$')):
        h = [100 * s[k]['mean'][idx[n]] for k, _, _ in rows]
        lo = [100 * min(s[k]['seeds'][n]) for k, _, _ in rows]
        hi = [100 * max(s[k]['seeds'][n]) for k, _, _ in rows]
        bars(ax, xs, h, lo, hi, [c for _, _, c in rows])
        ax.axhline(mw, color=INK, lw=0.9, ls='--', zorder=4)
        ax.set_xticks(xs); ax.set_xticklabels([r[1] for r in rows], rotation=32, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', length=0, labelcolor=INK)
        ax.set_ylim(0, 10); ax.set_xlim(-0.6, len(rows) - 0.4)      # one scale for both panels, so (a) and (b) compare by bar height
        title(ax, label); clean(ax, 'y'); ax.grid(False, axis='x')
    axs[0].set_ylabel('Test LER (\\%)')
    axs[0].legend(handles=[plt.Line2D([], [], color=INK, lw=0.9, ls='--', label=f'MWPM ({mw:.2f}\\%)')], frameon=False, loc='upper left', handlelength=2, borderaxespad=0.1)
    save(fig, 'fig_sampling')


def fig_parity():
    """R against N on shared axes: the crossing of R = 1 moves right with d and is absent at d = 7."""
    p, e12, e17 = V['p1']['d'], V['ex12'], V['ex17']
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 1.85), sharex=True, sharey=True, constrained_layout=True)
    ticks, labels = [1e4, 1e5, 1e6], ['$10^{4}$', '$10^{5}$', '$10^{6}$']
    for ax, d, name in zip(axs, ('3', '5', '7'), ('(a)', '(b)', '(c)')):
        ax.axhline(1, color=INK, lw=0.9, ls='--', zorder=2)
        ax.plot(p[d]['ns'], p[d]['envelope'], color=BLUE, marker='o', ms=4, lw=1.4, mec='white', mew=0.4, zorder=4, label=r'$p=0.5\%$, best $\beta$, 1 seed')
        if d != '7':
            ax.plot(e12['ns' + d], e12['d' + d]['knob3'], color=GREEN, marker='s', ms=5, lw=1.1, ls=(0, (4, 2)), mfc='none', mew=0.8, zorder=6, label=r'$p=0.3\%$, $\beta=3$, 1 seed')
        title(ax, f'{name} $d={d}$')
        ax.set_xlabel('Training set size $N$')
        logx(ax, ticks, labels); logy(ax, [1, 3, 10, 30], ['1', '3', '10', '30'])
        clean(ax)
    for ax, xs, ss in ((axs[1], [1e6, 2e6], [e17['d5']['hard'], e17['d5']['hard2e6']]), (axs[2], e17['d7_curve']['ns'], [e17['d7'][k] for k in ('hard', 'hard15', 'hard20', 'hard30', 'hard40')])):
        med = [s['median'] for s in ss]
        ax.errorbar(xs, med, yerr=[[s['median'] - s['min'] for s in ss], [s['max'] - s['median'] for s in ss]], color=ORANGE, marker='^', ms=4.6, lw=1.4, mec='white', mew=0.4, capsize=1.8, elinewidth=0.8, zorder=5, label=r'$p=0.3\%$, $\beta=3$, 4--9 seeds')
    axs[0].set_ylabel('LER / MWPM LER'); axs[0].set_xlim(7e3, 6e6); axs[0].set_ylim(0.5, 32)
    # N*: a full-height band from the earliest to the latest parity crossing among the series; every panel states N* in the same place, right of its title
    s03 = V['scale03']
    for ax, lo, hi in ((axs[0], p['3']['Nstar_env'], s03['Nstar3']), (axs[1], s03['Nstar5'], p['5']['ns'][-1])):
        ax.axvspan(lo, hi, color='#cfcec6', lw=0, zorder=1)
    for ax, text in zip(axs, (r'parity at $N^{*}\approx10^{5}$', r'parity at $N^{*}\approx2$--$3\times10^{6}$', r'no parity by $N=4\times10^{6}$')):
        ax.set_title(text, loc='right', fontsize=8, pad=4)
    axs[0].legend(handles=[plt.Line2D([], [], color=INK, lw=0.9, ls='--', label='MWPM')], frameon=False, loc='upper left', handlelength=2, borderaxespad=0.1)
    h, l = [], []
    for ax in axs[[0, 1]]:
        for a, b in zip(*ax.get_legend_handles_labels()):
            if b not in l and b != 'MWPM':
                h.append(a); l.append(b)
    fig.legend(h, l, loc='outside upper center', ncol=3, frameon=False, handlelength=2.2, columnspacing=1.5, borderaxespad=0.1)
    save(fig, 'fig_parity')


BETA_W = 0.75      # Figure 3 has two panels and is set at this share of the text width, centred


def fig_beta():
    """Learning curves of R for each beta at d = 5, one line per beta with beta = 1 in grey, as Peters plots them:
    a higher training error rate helps where its curve falls below the beta = 1 curve. Single-seed sweeps only;
    the four-seed step-count check that revises N = 3e5 is reported in the text."""
    p, e12 = V['p1']['d']['5'], V['ex12']
    style = {1.0: dict(color=GRAY, marker='o'), **{b: dict(color=RAMP[b], marker=MARK[b]) for b in (1.5, 2.0, 3.0)}}
    fig, axs = plt.subplots(1, 2, figsize=(BETA_W * WIDTH, 1.75), sharex=True, sharey=True, constrained_layout=True)
    idx = {b: i for i, b in enumerate(p['betas'])}
    for b in (1.0, 1.5, 2.0, 3.0):
        axs[0].plot(p['ns'], [row[idx[b]] for row in p['ratio']], ms=3.8, lw=1.2, mec='white', mew=0.4, zorder=3,
                    label=r'$\beta=%g$' % b + (' (target rate)' if b == 1 else ''), **style[b])
    for k, b in (('uniform', 1.0), ('knob2', 2.0), ('knob3', 3.0)):
        axs[1].plot(e12['ns5'], e12['d5'][k], ms=3.8, lw=1.2, mec='white', mew=0.4, zorder=3, **style[b])
    for ax, t in ((axs[0], r'(a) $d=5$, $p=0.5\%$'), (axs[1], r'(b) $d=5$, $p=0.3\%$')):
        ax.axhline(1, color=INK, lw=0.9, ls='--', zorder=2, label='MWPM')
        logx(ax, [1e4, 1e5, 1e6], ['$10^{4}$', '$10^{5}$', '$10^{6}$']); logy(ax, [1, 3, 10, 30], ['1', '3', '10', '30'])
        ax.set_xlim(2e4, 4.5e6); ax.set_ylim(0.8, 30)
        ax.set_xlabel('Training set size $N$'); title(ax, t); clean(ax)
    axs[0].set_ylabel('LER / MWPM LER')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside upper center', ncol=5, frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.1)
    save(fig, 'fig_beta')


def fig_targets():
    """Exact targets at d = 3, then the main training targets at d = 5 and 7. Table 4 lists every target."""
    e, B = V['ex17'], V['supp']['B']
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 1.8), constrained_layout=True, gridspec_kw={'width_ratios': [1.1, 1, 0.84]})
    ax = axs[0]
    for k, name, c, mk, ls in (('hard', 'Hard', GRAY, 'o', '-'), ('hard_ls10', r'Smoothing, $\varepsilon=0.1$', ORANGE, 's', (0, (4, 2))), ('exact', 'Exact posterior', TEAL, 'D', (0, (1, 1.2)))):
        s = [B['summary'][f'{k}@{n}'] for n in B['ns']]
        med = [x['median'] for x in s]
        ax.errorbar(B['ns'], med, yerr=[[x['median'] - x['min'] for x in s], [x['max'] - x['median'] for x in s]], color=c, marker=mk, ms=3.8, lw=1.3, ls=ls, mec='white', mew=0.4, capsize=1.6, elinewidth=0.7, zorder=3, label=name)
    ax.axhline(1, color=INK, lw=0.9, ls='--', zorder=2)
    logx(ax, [1e4, 1e5], ['$10^{4}$', '$10^{5}$']); ax.set_xlim(7.5e3, 4e5)
    ax.set_ylim(0.75, 1.9); ax.set_yticks([0.8, 1, 1.2, 1.4, 1.6, 1.8])
    ax.set_xlabel('Training set size $N$'); ax.set_ylabel('LER / MWPM LER'); title(ax, r'(a) $d=3$'); clean(ax)
    ax.legend(frameon=False, loc='upper right', handlelength=1.5, labelspacing=0.2, borderaxespad=0.1)
    rows = [('hard', 'Hard', GRAY), ('mwpm', 'MWPM gap', PALE), ('tess', 'Tesseract gap', BLUE), ('ls10', r'Smoothing, $\varepsilon=0.1$', ORANGE), ('hard2e6', r'Hard, $N=2\times10^{6}$', GRAY)]
    for ax, st, name, top in ((axs[1], e['d5'], r'(b) $d=5$', 2.1), (axs[2], e['d7'], r'(c) $d=7$', 45)):
        use = [r for r in rows if r[0] in st]
        xs = np.arange(len(use))
        bars(ax, xs, [st[k]['median'] for k, _, _ in use], [st[k]['min'] for k, _, _ in use], [st[k]['max'] for k, _, _ in use], [c for _, _, c in use])
        ax.axhline(1, color=INK, lw=0.9, ls='--', zorder=4)
        ax.set_xticks(xs); ax.set_xticklabels([r[1] for r in use], rotation=32, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', length=0, labelcolor=INK)
        ax.set_ylim(0, top); ax.set_xlim(-0.6, len(use) - 0.4); title(ax, name); clean(ax, 'y'); ax.grid(False, axis='x')
    save(fig, 'fig_targets')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_sampling(); fig_parity(); fig_beta(); fig_targets()
    print('figures written to', OUT)
