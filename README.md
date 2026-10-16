# Sleep-stage transition analysis and memory in older adults

The REM-exit continuity index.

Analysis code for the manuscript *"Sleep-stage transition analysis and memory in older adults:
the REM-exit continuity index."*

The index of interest is **P(REM → light NREM)**: of all the times a REM episode ends, the share that
hand off to light NREM rather than to wake or deep NREM. Everything here runs from a public dataset,
so the analysis can be reproduced end to end without contacting the authors.

Figure numbers in this repository match the figure numbers in the manuscript.

---

## 1. Get the data

The RESILIENT dataset is open on Zenodo under CC BY 4.0:

- **Data:** https://doi.org/10.5281/zenodo.16755408 (deposit of 6 August 2025)
- **Data descriptor:** Céspedes Gómez N, *et al.* *Scientific Data* 2025;12:1675.
  https://doi.org/10.1038/s41597-025-05958-x

Download and unpack it so the layout looks like this:

```
data/raw/
├── Demographics.csv
├── Resilient_metadata/
└── Sleepmat_Watch_Data/
    └── <participant-id>/
        ├── Sleep_state.csv      # staging, per minute
        └── Sleep_physio.csv     # heart rate, respiration, SDNN, snoring
```

The scripts search for `Demographics.csv` and `Sleep_state.csv`, so the exact nesting does not
matter as long as they sit under `data/raw/`. To keep the data elsewhere, set `REM_DATA`:

```bash
export REM_DATA=/path/to/resilient/data
```

### Which version matters

Two versions of the deposit are in circulation and they differ in the cognition they carry.

| Version | Deposited | Demographics.csv | Six-month ACE-III |
|---|---|---|---|
| `10.5281/zenodo.15877645` | 14 Jul 2025 | 24,829 B | no |
| **`10.5281/zenodo.16755408`** | **6 Aug 2025** | **56,975 B** | **yes** |

Use **16755408**, the link above. The sleep recordings are byte-identical between the two (same
MD5 for `Sleepmat_Watch_Data.zip`), and every baseline variable used here is identical, so only
the six-month follow-up analysis depends on the choice. Note that 16755408 renames
`gds15_total` to `gds_total`; the loaders accept either.

If you have both, name the newer one `Demographics_v16755408.csv` and the scripts will prefer it.

## 2. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Produced with Python 3.9.6. `requirements.txt` pins the exact versions used; later versions should
give the same numbers.

## 3. Reproduce everything

```bash
./run_all.sh          # ~18 min total
```

Or step by step:

```bash
bash analysis/make_figures.sh          # Figures 1-8, 10, 13, 15, 16 + result tables   ~10 min
python analysis/extra_analyses.py      # Figures 11, 12, 17                            ~7 min
python analysis/followup_and_physio.py # Figure 9 + the physiology tables              ~90 s
python analysis/fig_physio_w3plus.py   # Figure 14                                     ~10 s
```

`extra_analyses.py --reuse` skips the slow transition pass by loading the cache it wrote on a
previous run.

| Script | Produces |
|---|---|
| `analysis/make_figures.sh` | Reruns the notebook's own code with the in-figure titles suppressed, then maps the notebook's figure names onto the manuscript's numbering |
| `analysis/extra_analyses.py` | Selection bootstrap (Fig 11), specification curve over 320 paths (Fig 12), disjoint night halves, regression diagnostics (Fig 17) |
| `analysis/followup_and_physio.py` | Six-month follow-up cognition (Fig 9); nocturnal physiology by cognitive status and the adjusted models |
| `analysis/fig_physio_w3plus.py` | Figure 14, the staging-artefact test. **This is the only producer of `fig14_physio`** — `followup_and_physio.py` used to draw a competing version under the same name, and whichever ran last won |
| `analysis/make_tables.py` | LaTeX table fragments from the result CSVs |
| `src/make_rem_notebook_v2.py` | Generator the notebook is built from — **edit this, not the `.ipynb`** |
| `src/validate_palette.py` | Checks the palette against colour-vision-deficiency thresholds |

The notebook covers the same ground as Sections 1-12 of the manuscript and adds Section 12b, the
staging-artefact test, so Figure 14 can be reproduced inside the notebook as well as from
`analysis/fig_physio_w3plus.py`. A fully executed copy is on Kaggle:
https://www.kaggle.com/code/alisaremi/rem-exit-continuity-visual-edition

Running the generator with `--script` emits a flat `_validate.py` of all code cells, which is how
the notebook is tested — `nbconvert --execute` exits 0 even when a cell raises.

## 4. Check you got the same answer

| Quantity | Expected |
|---|---|
| Analysis sample | 52 participants; 50 in age- and sex-adjusted models |
| Nights / transitions / REM terminations | 6,740 / 128,852 / 19,500 |
| REM terminations per night | mean 2.89, median 3 |
| P(REM→light) vs ACE-III memory | β = +0.493, p = 0.0004, FDR q = 0.005 |
| Split-half reliability / six-month test–retest | 0.872 / r = 0.774 |
| AUC for ACE-III ≤ 82 | 0.735 (best conventional metric 0.639) |
| Selection optimism / re-selection rate | −0.005 / 86% |
| Specification curve | 320 paths, 98% positive, memory median β = +0.44 |
| Nights needed | AUC 0.60 at 1 night → 0.74 at 28 |
| Six-month follow-up, memory level | β = +0.32, p = 0.036, n = 46 of 47 returners |
| Six-month follow-up, memory change | β = +0.02, p = 0.90 (null; MDE ±0.43 at 80% power) |
| Physiology by cognitive status | no difference on any of six channels, all p ≥ 0.21, max \|g\| = 0.35 |
| Association adjusted for all six channels | β = +0.48, p = 0.003 (vs +0.49 unadjusted) |

Permutation tests, bootstraps and the night-subsampling procedure are seeded, so these reproduce
exactly. Anything that moves by more than rounding means the data or environment differs.

## 5. What is not here

No participant-level data. The repository holds code, aggregate figures and the instructions to
obtain the data from Zenodo; `.gitignore` excludes `data/`, any `.csv`, cached `.pkl` files and
credentials. Per-participant outputs written by the scripts (for example
`tables/physio_per_participant.csv`) stay local and are not published.

## 6. Scope

This is an exploratory, cross-sectional analysis of 50–52 people, with sleep stages inferred by a
consumer under-mattress sensor rather than polysomnography. The manuscript treats P(REM → light NREM)
as a candidate index requiring replication, not as an established biomarker, and the code is
published so that claim can be checked rather than taken on trust.
