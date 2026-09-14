# Sleep-stage transition analysis and memory in older adults

The REM-exit continuity index.

Analysis code for the manuscript *"Sleep-stage transition analysis and memory in older adults:
the REM-exit continuity index."*

The index of interest is **P(REM → light NREM)**: of all the times a REM episode ends, the share that
hand off to light NREM rather than to wake or deep NREM. Everything here runs from a public dataset,
so the analysis can be reproduced end to end without contacting the authors.

---

## 1. Get the data

The RESILIENT dataset is open on Zenodo under CC BY 4.0:

- **Data:** https://doi.org/10.5281/zenodo.16755408
- **Data descriptor:** Céspedes Gómez N, *et al.* *Scientific Data* 2025;12:1675.
  https://doi.org/10.1038/s41597-025-05958-x

Download and unpack it so the layout looks like this:

```
data/raw/
├── Demographics.csv
├── Resilient_metadata/
└── Sleepmat_Watch_Data/
    └── <participant-id>/Sleep_state.csv
```

The scripts locate these by searching for `Demographics.csv` and `Sleep_state.csv`, so the exact
nesting does not matter as long as they sit somewhere under `data/raw/`.

## 2. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy statsmodels scikit-learn matplotlib
```

## 3. Reproduce the analysis

**Figures 1–12 and Tables 1–4** — open `notebook/rem-exit-continuity-visual.ipynb` and run it top to
bottom (~5 min). It writes every figure as PDF and PNG plus the result CSVs. A fully executed copy
with all outputs rendered is on Kaggle:
https://www.kaggle.com/code/alisaremi/rem-exit-continuity-visual-edition

**Figures 13–15** — the analyses added in response to review:

```bash
python analysis/extra_analyses.py          # ~7 min; --reuse to skip the transition pass
```

| Output | What it is |
|---|---|
| `fig13_selection` | Bootstrap of the whole selection procedure (winner's-curse estimate) |
| `fig14_specification` | Specification curve over 320 analytic paths |
| `fig15_diagnostics` | Influence, leverage, normality and heteroscedasticity checks |
| `selection_stability.csv` | How often each transition wins under resampling |
| `specification_curve.csv` | Every specification and its coefficient |
| `night_halves.csv` | Association re-estimated from disjoint night subsets |

`src/make_rem_notebook_v2.py` is the generator the notebook is built from — edit that, not the
`.ipynb`. Running it with `--script` emits a flat `_validate.py` of all code cells, which is how the
notebook is tested (`nbconvert --execute` exits 0 even when a cell raises).
`src/validate_palette.py` checks the figure palette against colour-vision-deficiency thresholds.

## 4. Check you got the same answer

| Quantity | Expected |
|---|---|
| Analysis sample | 52 participants; 50 in age- and sex-adjusted models |
| Nights / transitions / REM terminations | 6,740 / 128,852 / 19,500 |
| P(REM→light) vs ACE-III memory | β = +0.493, p = 0.0004, FDR q = 0.005 |
| Split-half reliability / six-month test–retest | 0.872 / r = 0.774 |
| AUC for ACE-III ≤ 82 | 0.735 (best conventional metric 0.639) |
| Selection optimism / re-selection rate | −0.005 / 86% |
| Specification curve | 320 paths, 98% positive, memory median β = +0.44 |
| Nights needed | AUC 0.60 at 1 night → 0.74 at 28 |

Permutation tests, bootstraps and the night-subsampling procedure are seeded, so these reproduce
exactly. Anything that moves by more than rounding means the data or environment differs.

## 5. Scope

This is an exploratory, cross-sectional analysis of 50–52 people, with sleep stages inferred by a
consumer under-mattress sensor rather than polysomnography. The manuscript treats P(REM → light NREM)
as a candidate index requiring replication, not as an established biomarker, and the code is
published so that claim can be checked rather than taken on trust.
