# How Training Data Affect a 3D-CNN Surface-Code Decoder

Code and results for the paper *How Training Data Affect a 3D-CNN Surface-Code Decoder* by Hyungsuk Lee, In-Nea Wang, Kang Woo Lee, Yunsik Son, Jin-Woo Jung, and Junho Jeong, submitted to ICCDE 2027 and under review.

A neural decoder learns from simulated shots, so which shots to generate, how many, and which training target each one carries are all choices. The paper compares these choices for one 3D convolutional decoder on rotated surface-code memory experiments at distances 3, 5, and 7 under uniform circuit-level noise, with minimum-weight perfect matching (MWPM) as the reference.

- `experiments/` holds the sampling, labeling, and training scripts of the seven experiments.
- `data/` holds every result file the paper uses, as the scripts wrote them, with the run logs.
- `analysis/` recomputes the numbers, tables, and figures of the paper from `data/`.

## Experiments

| Experiment (Table 1 of the paper) | Code | Results | Shown in |
|---|---|---|---|
| Sampling study | `experiments/sampling/` | `data/s1/` | Figure 1 |
| Scaling study | `experiments/scaling/` | `data/p1/` | Figures 2 and 3(a) |
| Posterior study | `experiments/posterior/` | `data/ex12/` | Figures 2 and 3(b) |
| Teacher study | `experiments/teacher/` | `data/ex17/` | Figures 2, 4(b), and 4(c), Table 2 |
| Step-count check | `experiments/checks/u_runs.py`, part `c` | `data/supp/u_c_*` | text with Figure 3 |
| Exact-posterior check | `experiments/checks/u_runs.py`, part `b` | `data/supp/u_b_*` | Figure 4(a) |
| β = 1 smoothing check | `experiments/checks/u_runs.py`, part `a` | `data/supp/u_a_*` | text after Table 2 |

The result folders keep the working names that the analysis scripts use (`s1`, `p1`, `ex12`, `ex17`, `supp`). The three checks were registered, with their pass criteria, before any of their results were seen; `analysis/numbers.py` evaluates those criteria.

## Reproducing the paper's numbers

```bash
pip install -r requirements.txt
python analysis/numbers.py   # recompute from data/, write data/numbers.json, check the numbers printed in the paper
python analysis/tables.py    # LaTeX rows in out/tables/ (labels.tex is Table 2)
python analysis/figs.py      # Figures 1 to 4 as PDF in out/figs/
```

`analysis/claims.py` ties 140 numbers in the text of the paper to the values they were computed from, and `numbers.py` stops if a printed value does not round from the data. It should print `140 claims checked, 0 failed`. Tables and figures are drawn from `data/numbers.json` directly. Bootstrap intervals use a fixed seed, so the output is the same on every run. `figs.py` sets figure text with pdflatex and needs the `libertine` and `newtx` LaTeX packages. These scripts were checked with Python 3.14.4, numpy 2.2.6, scipy 1.18.1, matplotlib 3.11.2, and stim 1.16.0. `data/MD5SUMS` lists checksums of the result files.

## Rerunning the experiments

Every training run needs a CUDA GPU. The sampling, scaling, and posterior studies ran on a local GPU. The teacher study and the checks ran on a server with four NVIDIA H200 GPUs, one process per GPU, and Tesseract labeling ran in parallel worker processes (`WORKERS`). The first line of each log in `data/ex17/logs/` and `data/supp/logs/` gives the exact command and seeds of that run. Training shots come from fixed sampler seeds, so a rerun draws the same shots.

```bash
# Sampling, scaling, and posterior studies: each line from the repository root (results go to results/ in that folder)
cd experiments/sampling && mkdir -p results && python sanity.py 5 0.005 && python compare.py 0.005 2
cd experiments/scaling && mkdir -p results && python p1_scaling.py
cd experiments/posterior && mkdir -p results && python b3_relabel.py && python b4_teachers.py && python b4_train.py

# Teacher study: run from experiments/ (results go to teacher/results/)
python teacher/t1_teacher.py 50000
python teacher/t2_train.py 2000000                  # also writes the d = 5 labels, t2_labels_<stamp>.npz
python teacher/t3_seeds.py <t2 labels> 1,2          # and 3,4  5,6  7,8
python teacher/t4_compute.py <t2 labels> 1,2        # and 3,4  5,6  7,8
python teacher/t9_d5_controls.py <t2 labels> 1,2    # and 3,4  5,6  7,8
LS_EPS=0.1 WITH_TESS1=1 python teacher/t9_d5_controls.py <t2 labels> 9,10   # and 11,12  13,14  15,16
python teacher/t6_labels7.py 4000000 1000000 10     # d = 7 labels, Tesseract beam 10
python teacher/t7_train7.py <t6 labels> <seeds> <method@N,...>   # one job per log in data/ex17/logs/t7*.log
python teacher/t8_labeldiag.py <t2 labels> <t6 labels>

# Checks: run from the repository root, one process per GPU
python experiments/checks/u_smoke.py
python experiments/checks/u_runs.py a:1,2 c:1 b:1,2   # and a:3,4 c:2 b:3,4 / a:5,6 c:3 b:5,6 / a:7,8 c:4 b:7,8
```

The logs do not record environment variables. The settings for `t9_d5_controls.py` with seeds 9 to 16 are inferred from their result files, which contain only ε = 0.1 smoothing, temperature-2 Tesseract targets, and plain Tesseract targets.

## Changes from the code as run

- **Import paths.** The scripts loaded shared code from absolute paths on the machines they ran on. These paths now point inside this repository.
- **`experiments/exact_posterior.py`.** Four scripts loaded the exact-posterior functions with `exec` from a larger pilot script, of which they used only this part. The module holds those functions unchanged, and the scripts import it.
- **Comments.** Internal folder names were replaced by the study names used here.
- **Output folders.** `analysis/` writes to `out/` instead of the manuscript folder, which is not included.
- **Logs.** In seven logs, the server's directory path in a Python traceback is replaced by `<server>/`.

Nothing else in the code or the result files was changed.

## Notes on the record

- `data/p1/results/p1_20261008_144244_FLAWED_fixed_epochs.json` and `p1_log_FLAWED.txt` come from an earlier scaling run with a fixed number of epochs. It was replaced by `p1_20261008_144832.json`, which sets a floor on the step count. The paper uses only the second run.
- Logs that end in a Python traceback, or stop before `done`, are failed or interrupted starts that were then fixed and run again. They have no result file.
- `t6_labels7_500k.npz` holds the first 500,000 of the d = 7 labels, saved while labeling continued. The `tess_gap@500000` runs used it, and it matches the start of the full label file.
- A few teacher-study runs appear in more than one result file. `numbers.py` checks that the repeats agree exactly.

## Citation

Hyungsuk Lee, In-Nea Wang, Kang Woo Lee, Yunsik Son, Jin-Woo Jung, and Junho Jeong. 2026. How Training Data Affect a 3D-CNN Surface-Code Decoder. Submitted to ICCDE 2027.

## License

MIT, for both the code and the result files. See `LICENSE`.
