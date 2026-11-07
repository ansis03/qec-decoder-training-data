# Prose numbers of the manuscript, tied to the computed values. Read by numbers.py (exec), so V is in scope.
# c(name, computed value, value as printed, literal that must appear in the prose). Tables come from tables.py and are not listed.
S, P, E, T, O, U = V['s1'], V['p1'], V['ex12'], V['ex17'], V['overlap'], V['supp']
A, B, C_ = U['A'], U['B'], U['C']
dp, d7p = T['d5_paired'], T['d7_paired']
R5, R7 = T['d5'], T['d7']
t1 = {(r['teacher'], r['beta']): r for r in T['t1']}
t8 = {r['set']: r for r in T['t8']}


def c(name, val, printed, lit=None):
    CLAIMS.append((name, float(val), printed, lit if lit is not None else printed))


def ci(name, pair, a, b, lit):
    c(name + ' lo', pair['ci'][0], a, lit); c(name + ' hi', pair['ci'][1], b, lit)


fail_up = (S['methods']['fail_0.2']['mean'][2] / S['methods']['uniform']['mean'][2] - 1) * 100

# abstract and introduction
c('knob range low', min(list(E['eff_d5'].values()) + list(P['eff_d5'].values())), '1.2', '1.2 to 2.6')
c('beta 1.5 eff', P['eff_d5']['1.5'], '1.27', '1.27 and 1.25')
c('beta 3 eff', P['eff_d5']['3.0'], '1.25', '1.27 and 1.25')
c('knob range high', S['methods']['beta_2']['eff'], '2.6', '1.2 to 2.6')
c('enrichment rise', fail_up, '36', '36\\%')
c('exact factor low', B['factor_exact']['10000'], '6', '6 to 8 times fewer samples')
c('exact factor high', B['factor_exact']['30000'], '8', '6 to 8 times fewer samples')
c('ls vs tess', dp['ls10_vs_tess']['ratio'], '1.00', 'Tesseract is 1.00 [0.96, 1.05]')
c('ls vs tess lo', dp['ls10_vs_tess']['ci'][0], '0.96', 'Tesseract is 1.00 [0.96, 1.05]')
c('ls vs tess hi', dp['ls10_vs_tess']['ci'][1], '1.05', 'Tesseract is 1.00 [0.96, 1.05]')
c('ls vs tess n', dp['ls10_vs_tess']['n'], '16', '16 seeds')
c('floor rule', C_['RC'], '1.5', '1.5 times worse')

# setup
for d, s in (('3', '1.38'), ('5', '3.87'), ('7', '8.72')):
    c('params ' + d, V['params'][d] / 1e6, s, '1.38, 3.87 and 8.72 million')
c('mwpm failures', T['mwpm']['5'] * 1e6, '3240', '3240 failures')

# sampling
c('s1 mwpm', S['mwpm'] * 100, '1.41', '1.41\\%')
c('pool failures', S['fail_mix']['frac'] * 100, '7', 'about 7\\%')
c('s1 factor', S['methods']['beta_2']['eff'], '2.56', 'reaching 2.56')
c('s1 g 1.25', S['methods']['beta_1.25']['eff'], '1.55', '1.55, 1.67, 1.98 and 2.56')
c('s1 g 1.4', S['methods']['highp_0.007']['eff'], '1.67', '1.55, 1.67, 1.98 and 2.56')
c('s1 g 1.5', S['methods']['beta_1.5']['eff'], '1.98', '1.55, 1.67, 1.98 and 2.56')
c('beta2 seeds apart', (max(S['methods']['beta_2']['seeds']['100000']) / min(S['methods']['beta_2']['seeds']['100000']) - 1) * 100, '44', '44\\% above the other')
c('strat factor', S['methods']['strat']['eff'], '1.20', 'data-efficiency gain of stratification is 1.20')
c('keep successes', 100 / S['fail_mix']['boost'], '19', 'about 19\\%')
c('boost', S['fail_mix']['boost'], '5.3', 'm\\approx 5.3')
c('flip threshold', S['fail_mix']['flip_threshold'], '0.16', '\\approx 0.16')
c('overlap d5', O['5']['tess_given_mwpm'] * 100, '50', '50\\% of the shots')
c('overlap d7', O['7']['tess_given_mwpm'] * 100, '37', '37\\% at $d=7$')
c('factor scale', P['eff_d5']['2.0'], '1.63', '1.63 in the scaling study')
c('factor post lo', E['eff_d5']['knob3'], '1.3', '1.3--1.4')
c('factor post hi', E['eff_d5']['knob2'], '1.4', '1.3--1.4')

# scale
c('N* d3', P['d']['3']['Nstar_env'] / 1e5, '1.0', '\\num{1.0e5}')
c('d3 best', P['d']['3']['envelope'][-1], '0.88', 'ratio of 0.88')
c('d5 best', P['d']['5']['envelope'][-1], '1.01', 'best point is 1.01')
c('d7 best', P['d']['7']['envelope'][-1], '2.12', 'best point is 2.12')
c('N* d3 p03', V['scale03']['Nstar3'] / 1e5, '1.1', '\\num{1.1e5}')
c('d5 2e6 mean', R5['hard2e6']['mean'], '1.00', 'mean ratio 1.00 over nine seeds')
c('d7 4e6 median', R7['hard40']['median'], '2.49', 'reaches 2.49')
c('extrap low', P['extrap_d7']['powerlaw'] / 1e7, '0.8', '\\num{0.8e7}')
c('extrap high', P['extrap_d7']['curve'] / 1e7, '1.8', '\\num{1.8e7}')
c('extrap p03', V['scale03']['Nstar7_low'] / 1e6, '8.5', '\\num{8.5e6}')
c('grow35 p05', V['scale05']['grow35'], '29', 'factor of 29 at')
c('grow35 p03', V['scale03']['grow35'], '17', 'and 17 at $p=0.3\\%$')
b4 = {(r['method'], r['N']): r['ratio'] for r in E['b4b']}
c('d7 beta3 ler', P['d']['7']['ler']['3.0'][2] * 100, '35', 'near 35\\%')
c('gain 1e6', P['d']['5']['ratio'][3][2] / P['d']['5']['ratio'][3][0], '0.64', '0.64 at $N')
c('gain 3e6', P['d']['5']['ratio'][4][3] / P['d']['5']['ratio'][4][0], '0.86', '0.86 at $N')
c('missed by', 4 / P['d']['7']['envelope'][-1], '1.9', 'factor of 1.9')
c('d7 hard max N', max(T['d7_curve']['ns']) / 1e6, '4', '\\num{4e6}')
c('s1 3e5', S['methods']['beta_2']['mean'][1] * 100, '2.90', '2.90\\%')
c('p1 3e5', P['d']['5']['ler']['2.0'][2] * 100, '4.97', '4.97\\%')
t2 = {(r['method'], r['N']): r['ratio'] for r in T['t2']}
m = C_['median']
c('C short', m['2/1764/final'] * 100, '2.82', '2.82\\%')
c('C long', m['2/6000/best'] * 100, '4.26', '4.26\\%')
c('C ratio', C_['RC'], '1.51', 'ratio of 1.51')
c('C short best', m['2/1764/best'] * 100, '2.80', '2.80\\%')
c('C long final', m['2/6000/final'] * 100, '4.45', '4.45\\%')
c('C b1 short', m['1/1764/final'] * 100, '3.90', '3.90\\%')
c('C b1 long', m['1/6000/best'] * 100, '5.17', '5.17\\%')
c('s1 1e6', S['methods']['beta_2']['mean'][2] * 100, '1.68', '1.68\\%')
c('p1 1e6', P['d']['5']['ler']['2.0'][3] * 100, '1.63', '1.63\\%')
c('C gain short', C_['beta_gain_short'], '0.72', '0.72 times the median LER')
c('C gain long', C_['beta_gain_long'], '0.82', '0.82 times with 6000')
c('epochs at floor', 6000 / ((300000 + 2047) // 2048), '41', '41 epochs')
ms = P['d']['7']['ratio']

# labels
c('t1 mae low', t1[('mwpm_gap', 3.0)]['mae'] / t1[('tess_gap', 3.0)]['mae'], '3.4', '3.4--4.3')
c('t1 mae high', t1[('mwpm_gap', 1.0)]['mae'] / t1[('tess_gap', 1.0)]['mae'], '4.3', '3.4--4.3')
c('t1 kl low', t1[('mwpm_gap', 3.0)]['kl'] / t1[('tess_gap', 3.0)]['kl'], '11', 'about 11--13')
c('t1 kl high', t1[('mwpm_gap', 1.0)]['kl'] / t1[('tess_gap', 1.0)]['kl'], '13', 'about 11--13')
c('first exact 1', E['exact_eff']['10000'], '4.9', '4.9 and 8.1')
c('first exact 2', E['exact_eff']['30000'], '8.1', '4.9 and 8.1')
c('ctrl exact 1', B['factor_exact']['10000'], '5.8', '5.8 and 7.9')
c('ctrl exact 2', B['factor_exact']['30000'], '7.9', '5.8 and 7.9')
x = B['exact_vs_hard']['300000']
c('exact 3e5', x['ratio'], '1.01', '1.01 [0.96, 1.09]'); ci('exact 3e5', x, '0.96', '1.09', '1.01 [0.96, 1.09]')
c('ls factor 1', B['factor_ls']['10000'], '1.5', 'about 1.5 and 1.6')
c('ls factor 2', B['factor_ls']['30000'], '1.6', 'about 1.5 and 1.6')
for n, s in (('10000', '39'), ('30000', '26'), ('100000', '12')):
    c('ls worse ' + n, (B['ls_vs_exact'][n]['ratio'] - 1) * 100, s, '39\\%, 26\\% and 12\\%')
c('gap 1e4', E['d3']['knob3+gap'][0], '1.15', '1.15 against 1.74')
c('hard 1e4', E['d3']['knob3'][0], '1.74', '1.15 against 1.74')
c('gap stall', E['d3']['knob3+gap'][3], '1.02', 'plateaued at 1.02')
c('hard 3e5', E['d3']['knob3'][3], '0.90', 'continued to improve to 0.90')
c('gap d5', E['d5']['knob3+gap'][3], '1.28', '1.28 against 1.76')
c('hard d5', E['d5']['knob3'][3], '1.76', '1.28 against 1.76')
c('hard 2e6', b4[('knob3', 2_000_000)], '1.01', '1.01 against 1.08')
c('gap 2e6', b4[('knob3+gap', 2_000_000)], '1.08', '1.01 against 1.08')
c('d5 hard', R5['hard']['mean'], '1.35', 'mean ratio of 1.35')
c('d5 mwpm', R5['mwpm']['mean'], '1.23', 'targets give 1.23')
c('d5 tess', R5['tess']['mean'], '1.01', 'targets give 1.01')
x = dp['tess_vs_2e6']
c('tess 2e6', x['ratio'], '0.99', '0.99 [0.93, 1.05]'); ci('tess 2e6', x, '0.93', '1.05', '0.99 [0.93, 1.05]')
c('d5 ls', R5['ls10']['mean'], '1.03', 'gives 1.03')
x = dp['ls10_vs_tess']
c('ls tess body', x['ratio'], '1.00', '1.00 [0.96, 1.05]'); ci('ls tess body', x, '0.96', '1.05', '1.00 [0.96, 1.05]')
c('ls tess wins', x['wins'], '9', '9 of 16')
x = dp['ls10_vs_hard']
c('ls hard', x['ratio'], '0.75', '0.75 [0.69, 0.82]'); ci('ls hard', x, '0.69', '0.82', '0.75 [0.69, 0.82]')
c('ls hard wins', x['wins'], '8', '8 of 8 seeds')
x = dp['ls10_vs_2e6']
c('ls 2e6', x['ratio'], '1.02', '1.02 [0.92, 1.12]'); ci('ls 2e6', x, '0.92', '1.12', '1.02 [0.92, 1.12]')
x = dp['hard9780_vs_hard']
c('T4 pair', x['ratio'], '1.08', '1.08 [1.00, 1.14]'); ci('T4 pair', x, '1.00', '1.14', '1.08 [1.00, 1.14]')
c('d7 hard', R7['hard']['median'], '12.0', 'median of 12.0')
c('d7 mwpm', R7['mwpm']['median'], '9.7', 'targets give 9.7')
c('d7 mwpm wins', d7p['mwpm_vs_hard']['wins'], '8', 'better in 8 of 8')
c('d7 ls', R7['ls10']['median'], '9.2', 'gives 9.2')
c('d7 ls wins', d7p['ls10_vs_hard']['wins'], '4', 'better in 4 of 4')
c('d7 eff m', T['d7_eff']['mwpm'], '1.26', '1.26 and 1.32')
c('d7 eff l', T['d7_eff']['ls10'], '1.32', '1.26 and 1.32')
c('d7 tess', R7['tess']['median'], '12.7', 'give 12.7')
c('d7 tess wins', d7p['tess_vs_hard']['wins'], '2', 'in 2 of 8 seeds')
c('d7 T4', R7['tess_T4']['median'], '10.4', 'they give 10.4')
c('hard fail run', T['d7_hard_failed']['failed'], '1', '1 of 24')
c('soft failed', T['d7_soft_failed']['failed'], '8', '8 of 52')
c('soft runs', T['d7_soft_failed']['n'], '52', '8 of 52')
c('fisher 1e6', T['d7_soft_failed']['fisher_vs_hard_1e6'], '0.58', '$P=0.58$')
c('fisher all', T['d7_soft_failed']['fisher_vs_hard_all'], '0.26', '$P=0.26$')
x = A['paired']
c('A ratio', x['ratio'], '0.90', '0.90 [0.69, 1.34]'); ci('A', x, '0.69', '1.34', '0.90 [0.69, 1.34]')
c('A failed', A['ls10']['max'], '7.56', 'smoothing run at 7.56')
c('d7 ls paired', d7p['ls10_vs_hard']['ratio'], '0.73', 'paired ratio 0.73')
c('A median hard', A['hard']['median'], '2.27', 'from 2.27 to 1.83')
c('A median ls', A['ls10']['median'], '1.83', 'from 2.27 to 1.83')
c('A posthoc', A['posthoc_ratio'], '0.74', 'is 0.74')
c('A vs beta3', dp['ls10_vs_hard']['ratio'], '0.75', 'the 0.75 at $\\beta=3$')

# guidance
c('g cut d5', (1 - dp['ls10_vs_hard']['ratio']) * 100, '25', 'by 25\\%')
c('g cut d7', (1 - d7p['ls10_vs_hard']['ratio']) * 100, '27', 'by 27\\%')
c('g tess hours', (12875 - 129) / 3600, '3.5', '3.5 hours')
c('g range lo', R5['hard']['min'], '1.16', 'from 1.16 to 1.84')
c('g range hi', R5['hard']['max'], '1.84', 'from 1.16 to 1.84')

# step counts and run totals quoted in the prose
d7s = P['d']['7']['steps']
c('steps s1 1e6', 8 * ((1_000_000 + 2047) // 2048), '3912', '3912 steps')
c('steps 2e6', 10 * ((2_000_000 + 2047) // 2048), '9770', '6000 to 9770')
c('steps 20ep', 20 * ((1_000_000 + 2047) // 2048), '9780', '9780 steps')
