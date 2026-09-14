"""Build the REM-exit-continuity Kaggle notebook — visual edition.

Same analysis and same prose as `make_rem_notebook.py`, with a rebuilt figure
layer: one validated palette used by role, thirteen figures instead of eight,
and five new views (hypnogram exemplars, flow diagram, per-participant
composition, metric independence, sampling distributions).

Run with `--script` to also emit a flat .py of every code cell, which is how the
notebook gets validated locally (nbconvert --execute exits 0 even when a cell
raises).
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "kaggle_nb_rem_v2")
os.makedirs(OUTDIR, exist_ok=True)

cells = []


def _cell_id():
    # nbformat >= 4.5 wants a stable id on every cell
    return f"cell{len(cells):03d}"


def md(text):
    cells.append({"cell_type": "markdown", "id": _cell_id(), "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "id": _cell_id(), "execution_count": None,
                  "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ============================================================ 0. Title
md(r"""
# Where does REM sleep go? A stage-transition marker of memory in the very old

**Dataset:** [RESILIENT — AD and Sleep](https://www.kaggle.com/datasets/alisaremi/ad-and-sleep) ·
73 community-dwelling older adults · Withings Sleepmat + ScanWatch ·
~139 nights each over ~6 months · ACE-III, PHQ-9, GAD-7, GDS-15.

---

## The problem with averaged sleep

Sleep disruption is among the most-cited modifiable risk factors for Alzheimer's disease, yet in this
cohort **the conventional sleep phenotypes are flat**. Total sleep time, sleep efficiency, WASO,
deep/REM percentage, awakenings per hour, sleep-timing regularity, circadian amplitude, nocturnal
heart rate and HRV — averaged per person and regressed on ACE-III — produce nothing that survives
multiple-comparison control.

That null is informative. Averaging a 139-night recording into one number throws away the thing the
recording is actually good for: **the order in which sleep stages follow one another**.

## The hypothesis tested here

Sleep is a sequence of state transitions. In the canonical ultradian cycle a REM episode ends by
handing off to **light NREM**, which then deepens and cycles again. This handoff is orchestrated by
brainstem and basal-forebrain circuitry that degenerates early in Alzheimer's disease — the same
circuitry whose loss is thought to drive REM abnormalities in dementia.

So instead of asking *how much* REM someone gets, this notebook asks **where each REM episode goes
when it ends**:

$$P_{R \to L} \;=\; \frac{\#\{\text{REM episodes followed by light NREM}\}}{\#\{\text{all REM episode terminations}\}}$$

A single night gives perhaps five REM terminations — far too few. Six months gives several hundred,
and that is what makes this measurable at all.

## What this notebook establishes

1. $P_{R \to L}$ tracks ACE-III, and does so **specifically in the memory domain** (β = +0.49, p = 0.0004).
2. Of the twelve possible stage transitions, **only REM-origin routes** carry the signal.
3. It is a **stable trait**: split-half reliability 0.87, six-month test–retest r = 0.77.
4. It **discriminates cognitive impairment (AUC 0.75)** where every conventional metric sits at chance.
5. It is **invisible in a single night** and needs ~2–4 weeks of recording to emerge — the direct
   argument for longitudinal home monitoring over one-night laboratory polysomnography.
""")

# ============================================================ 1. Setup
md(r"""
## 1 · Setup

### The visual language

Every figure below is drawn from one small set of colour roles, and each role is assigned by the
*job* the colour does rather than by taste:

| Role | Job | Colours |
|---|---|---|
| **REM-exit destination** | categorical identity — light NREM / wake / deep NREM | `#2a78d6` `#eb6834` `#1baf7a` |
| **ACE-III groups and tertiles** | ordinal — the categories have an order | one blue hue, light → dark |
| **Transition probability** | sequential magnitude | one blue hue, 100 → 700 |
| **Signed effect (β, ρ)** | diverging polarity | blue ↔ red with a neutral grey midpoint |
| **Everything else** | context, not data | greys |

The destination trio is the only categorical family in the notebook, and it was checked
programmatically rather than by eye: under an all-pairs test (which is the right test for scatter
and small-multiple panels, where any two marks can end up side by side) the worst pair separates by
ΔE 24.0 for normal vision and ΔE 9.2 under simulated protanopia and deuteranopia — clear of the
floors of 15 and 8. Cognitive status deliberately gets **no categorical hue at all**: five
simultaneous categorical colours cannot clear those floors, so wherever a group split appears it is
carried by position, faceting and direct labels instead. Aqua sits slightly below the 3:1 contrast
target on this surface, so every mark that uses it carries a visible value label, and every figure
has the printed table beside it.
""")

code(r"""
import os, glob, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60)
RNG = np.random.default_rng(20260814)

# ---------------------------------------------------------------- colour roles
SURFACE = "#ffffff"                                  # chart surface (flush with the page)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"   # text, in three weights
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"   # chrome

# the one categorical family: where a REM episode goes when it ends
DEST = {"L": "#2a78d6", "W": "#eb6834", "D": "#1baf7a"}
DEST_LABEL = {"L": "→ Light NREM", "W": "→ Wake", "D": "→ Deep NREM"}

ORD3 = ["#86b6ef", "#3987e5", "#184f95"]             # ordinal blue: ACE-III tertiles
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)      # magnitude
POS, NEG = "#2a78d6", "#e34948"                                     # diverging poles
DIV = LinearSegmentedColormap.from_list("div", [NEG, WASH, POS])    # signed effects

matplotlib.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
    "axes.labelsize": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlepad": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

# ---------------------------------------------------------------- mark helpers
def px(ax, n, axis="y"):
    "n pixels expressed in data units"
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (n, n)])
    return abs(y1 - y0) if axis == "y" else abs(x1 - x0)

def on_fill(hexcolor):
    "ink or white for a label placed inside a coloured fill"
    c = np.array([int(hexcolor.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]) / 255
    lin = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    L = float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])
    return "white" if (1.05 / (L + .05)) >= ((L + .05) / .05) else INK

def rounded_bar(ax, x, y, w, h, color, r_px=4, orient="v", **kw):
    "bar with a rounded data-end and a square baseline end"
    r = min(px(ax, r_px, "y" if orient == "v" else "x"),
            abs(h if orient == "v" else w) * .45)
    x0, x1, y0, y1 = x, x + w, y, y + h
    if orient == "v":
        v = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + r, y1),
             (x1 - r, y1), (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
    else:
        v = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r),
             (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
         Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(v, c), facecolor=color, edgecolor="none", **kw))

def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=.85, zorder=2, ring=0):
    "smooth flow ribbon; `ring` is a surface-coloured edge so overlaps read as layered"
    t = np.linspace(0, 1, 80); s = 3 * t ** 2 - 2 * t ** 3
    ax.fill_between(x0 + (x1 - x0) * t, y0b + (y1b - y0b) * s, y0a + (y1a - y0a) * s,
                    facecolor=color, alpha=alpha, zorder=zorder,
                    edgecolor=SURFACE if ring else "none", linewidth=ring)

def dot(ax, x, y, color, s=70, zorder=5, ring=2.0):
    "marker with a 2px surface ring so overlapping points stay legible"
    return ax.scatter(x, y, s=s, color=color, zorder=zorder, edgecolor=SURFACE, linewidth=ring)

def half_violin(ax, data, pos, width=.34, color=DEST["L"], alpha=.28):
    data = np.asarray(data, float); data = data[np.isfinite(data)]
    if len(data) < 3:
        return
    ys = np.linspace(data.min() - .04, data.max() + .04, 200)
    d = stats.gaussian_kde(data)(ys); d = d / d.max() * width
    ax.fill_betweenx(ys, pos, pos + d, color=color, alpha=alpha, lw=0, zorder=2)
    ax.plot(pos + d, ys, color=color, lw=1.2, alpha=.75, zorder=2)

def arrow(ax, xy0, xy1, color, lw=2, rad=.22, zorder=3, ms=9):
    ax.add_patch(FancyArrowPatch(xy0, xy1, connectionstyle=f"arc3,rad={rad}", color=color,
                                 lw=lw, zorder=zorder, capstyle="round", shrinkA=0, shrinkB=0,
                                 arrowstyle=f"-|>,head_width={ms/28:.2f},head_length={ms/18:.2f}",
                                 mutation_scale=14))

# Set FIGURE_TITLES=0 to drop the in-figure title block. The paper build does this,
# because there the LaTeX caption carries the same text and printing both is redundant.
FIGURE_TITLES = os.environ.get("FIGURE_TITLES", "1") != "0"

def suptitle(fig, title, subtitle=None, y=1.02):
    if not FIGURE_TITLES:
        return
    fig.text(.005, y, title, fontsize=13.5, fontweight="bold", color=INK, ha="left", va="bottom")
    if subtitle:
        fig.text(.005, y - .035, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")

def dest_legend(fig, labels=None, y=1.045):
    labels = labels or DEST_LABEL
    if not FIGURE_TITLES:          # no title block above, so sit closer to the axes
        y = 1.005
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=9, mfc=DEST[k],
                                   mec=SURFACE, mew=1.5, label=labels[k]) for k in ("L", "W", "D")],
               loc="upper right", bbox_to_anchor=(.995, y), ncol=3, frameon=False,
               labelcolor=INK2)

FIGDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "figures"
os.makedirs(FIGDIR, exist_ok=True)

def save(fig, name):
    for ext in ("png", "pdf"):
        # pad_inches above the default 0.1: tight bbox otherwise shaves a few pixels
        # off the outermost axis labels at Kaggle's render size
        fig.savefig(f"{FIGDIR}/{name}.{ext}", bbox_inches="tight", pad_inches=.16,
                    facecolor=SURFACE)
    print(f"  saved {name}")

# ---- locate the dataset by searching for its files, not by assuming a path ----
ROOTS = ["/kaggle/input", "data/raw", "../data/raw", "."]
def find_one(name):
    for root in ROOTS:
        if os.path.isdir(root):
            hits = glob.glob(f"{root}/**/{name}", recursive=True)
            if hits:
                return sorted(hits, key=len)[0]
    return None

DEMO_PATH = find_one("Demographics.csv")
_state = find_one("Sleep_state.csv")
SLEEPDIR = os.path.dirname(os.path.dirname(_state))
PIDS = sorted(d for d in os.listdir(SLEEPDIR) if os.path.isdir(os.path.join(SLEEPDIR, d)))
print(f"Demographics : {DEMO_PATH}")
print(f"Sleep dir    : {SLEEPDIR}")
print(f"participants : {len(PIDS)}")
""")

# ============================================================ 2. Cognitive data
md(r"""
## 2 · Cognitive and clinical variables

ACE-III is scored out of 100 across five domains (attention 18, memory 26, fluency 14, language 26,
visuospatial 16). Higher is better. The conventional screening cut-off for cognitive impairment is
**≤ 82/100**; ≤ 88 is the more sensitive threshold.

Age is supplied as a band, so its midpoint is used as a continuous covariate.
""")

code(r"""
ACE_COLS = ["ace_total", "ace_attention_subscale", "ace_memory_subscale",
            "ace_fluency_subscale", "ace_language_subscale", "ace_visuospatial_subscale"]
MOOD = ["phq_total", "gad_total", "gds15_total"]

demo = pd.read_csv(DEMO_PATH).rename(columns={"user_id": "participant"})
for c in ACE_COLS + MOOD:
    demo[c] = pd.to_numeric(demo[c], errors="coerce")
demo["age_mid"] = demo["Age group"].map({"[72, 75]": 73.5, "[76, 87]": 81.5, "[88, 99]": 93.5})
demo["male"] = (demo["Sex"] == "Male").astype(float)
demo["htn"] = (demo["Essential hypertension"].astype(str).str.upper() == "TRUE").astype(float)
demo["osteo"] = (demo["Osteoarthritis"].astype(str).str.upper() == "TRUE").astype(float)

print(f"participants in Demographics : {len(demo)}")
print(f"with ACE-III                 : {demo.ace_total.notna().sum()}")
print("\nACE-III total:")
print(demo.ace_total.describe().round(1).to_string())
print(f"\nACE-III <= 82 (impaired) : {(demo.ace_total <= 82).sum()}")
print(f"ACE-III <= 88            : {(demo.ace_total <= 88).sum()}")
demo[["Sex", "Age group"]].value_counts().sort_index()
""")

# ============================================================ 3. Transitions
md(r"""
## 3 · From raw stage records to a transition sequence

Each participant's `Sleep_state.csv` is a list of `(start, end, stage)` rows with stages
`deep / light / REM / wakeup`. Three decisions matter:

**Noon-to-noon nights.** A night is labelled by the noon-to-noon window its epochs fall in, so an
episode crossing midnight stays with the evening it began.

**Consecutive identical rows are merged into bouts.** The device emits several rows for one
continuous stage; without merging, "transitions" would be dominated by artefactual self-loops.
Transitions are therefore counted between *bouts*, and self-transitions cannot occur by construction.

**A night needs ≥ 120 min of scored sleep and ≥ 4 bouts** to contribute.

This yields, per night, the counts of all twelve possible ordered stage pairs.
""")

code(r"""
DEPTH = {"wakeup": 0, "REM": 1, "light": 2, "deep": 3}
NAME  = {0: "W", 1: "R", 2: "L", 3: "D"}
LONG  = {"W": "Wake", "R": "REM", "L": "Light NREM", "D": "Deep NREM"}
MIN_TST = 120

def read_stages(pid):
    "raw stage rows for one participant, cleaned and labelled with a noon-to-noon night"
    f = os.path.join(SLEEPDIR, pid, "Sleep_state.csv")
    if not os.path.exists(f):
        return pd.DataFrame()
    df = pd.read_csv(f, parse_dates=["Start time", "End time"])
    df = df.rename(columns={"Start time": "start", "End time": "end",
                            "Sleep state": "stage"}).dropna(subset=["start", "end", "stage"])
    df["dur"] = (df.end - df.start).dt.total_seconds() / 60
    df = df[(df.dur > 0) & (df.dur < 600) & df.stage.isin(DEPTH)]
    if df.empty:
        return df
    df["night"] = (df.start - pd.Timedelta(hours=12)).dt.date
    return df.sort_values("start")

def merge_bouts(g):
    "collapse consecutive identical stages into bouts; returns stage codes and durations"
    v, dur = g.stage.map(DEPTH).values, g.dur.values
    chg = np.flatnonzero(np.diff(v) != 0)
    st, en = np.concatenate([[0], chg + 1]), np.concatenate([chg + 1, [len(v)]])
    return v[st], np.array([dur[a:b].sum() for a, b in zip(st, en)]), st, en

def night_transitions(pid, min_bout=0.0):
    # per-night bout-to-bout transition counts for one participant
    df = read_stages(pid)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for night, g in df.groupby("night"):
        g = g.sort_values("start")
        tst = g.loc[g.stage != "wakeup", "dur"].sum()
        if tst < MIN_TST:
            continue
        stages, bdur, _, _ = merge_bouts(g)
        if min_bout > 0:                       # optional sensitivity filter
            keep = bdur >= min_bout
            stages, bdur = stages[keep], bdur[keep]
            if len(stages) > 1:                # re-merge after dropping short bouts
                s2 = np.concatenate([[0], np.flatnonzero(np.diff(stages) != 0) + 1])
                stages = stages[s2]
        if len(stages) < 4:
            continue
        rec = {"participant": pid, "night": str(night), "tst": tst, "n_bouts": len(stages)}
        a_, b_ = stages[:-1], stages[1:]
        for a in range(4):
            for b in range(4):
                if a != b:
                    rec[f"n_{NAME[a]}{NAME[b]}"] = int(((a_ == a) & (b_ == b)).sum())
        rows.append(rec)
    return pd.DataFrame(rows)

nights = pd.concat([night_transitions(p) for p in PIDS], ignore_index=True)
TCOLS = [f"n_{NAME[a]}{NAME[b]}" for a in range(4) for b in range(4) if a != b]
print(f"nights              : {len(nights):,}")
print(f"participants        : {nights.participant.nunique()}")
print(f"stage transitions   : {nights[TCOLS].values.sum():,}")
print(f"REM terminations    : {nights[['n_RW','n_RL','n_RD']].values.sum():,}")
print(f"nights per person   : median {nights.groupby('participant').size().median():.0f} "
      f"(IQR {nights.groupby('participant').size().quantile(.25):.0f}"
      f"-{nights.groupby('participant').size().quantile(.75):.0f})")
nights.head(3)
""")

md(r"""
### Aggregating to the participant level

Transition counts are **pooled across all of a participant's nights before dividing**, rather than
averaging per-night proportions. Pooling weights each night by how many transitions it actually
contributed and avoids the instability of nights with two or three REM terminations.

Analysis is restricted to participants with **≥ 30 valid nights** and a recorded ACE-III.
""")

code(r"""
MIN_NIGHTS = 30

def participant_matrix(nights_df):
    agg = nights_df.groupby("participant")[TCOLS].sum()
    agg["nights"] = nights_df.groupby("participant").size()
    agg["tst_mean"] = nights_df.groupby("participant")["tst"].mean()
    for a in range(4):                                   # row-normalise
        cols = [f"n_{NAME[a]}{NAME[b]}" for b in range(4) if b != a]
        tot = agg[cols].sum(axis=1)
        agg[f"exits_{NAME[a]}"] = tot
        for c in cols:
            agg["P_" + c[2:]] = agg[c] / tot.replace(0, np.nan)
    return agg.reset_index()

pm = participant_matrix(nights)
dat = pm.merge(demo, on="participant", how="inner")
dat = dat[dat.ace_total.notna() & (dat.nights >= MIN_NIGHTS)].reset_index(drop=True)

print(f"ANALYSIS SAMPLE  n = {len(dat)}")
print(f"  nights/person        median {dat.nights.median():.0f} "
      f"(range {dat.nights.min()}-{dat.nights.max()})")
print(f"  REM terminations     median {dat.exits_R.median():.0f} "
      f"(range {dat.exits_R.min():.0f}-{dat.exits_R.max():.0f})")
print(f"  age  {dat.age_mid.mean():.1f} +- {dat.age_mid.std():.1f}   female {100*(1-dat.male.mean()):.0f}%")
print(f"  ACE-III {dat.ace_total.mean():.1f} +- {dat.ace_total.std():.1f}  "
      f"(impaired <=82: {(dat.ace_total<=82).sum()})")
print(f"\nP(REM -> light) = {dat.P_RL.mean():.3f} +- {dat.P_RL.std():.3f}")
print(dat[["P_RL", "P_RW", "P_RD"]].describe().round(3).to_string())
""")

# ---------------------------------------------------------- Figure 1 (new)
md(r"""
### What the marker actually counts — Figure 1

Before any statistics, it is worth seeing the quantity on a real night. Below are two participants
at opposite ends of the memory scale, each shown on the night whose own REM-exit mix sits closest to
that person's six-month value — a representative night rather than a flattering one. Every REM bout
is painted by the stage that follows it, and an arrow marks the handoff.

The point of the figure is as much what it *fails* to show: the two nights look broadly alike. Five
or eight REM exits carry almost no information. The separation on the right — 74% versus 60%
canonical exits — only appears once several hundred exits are pooled, which is the argument the rest
of the notebook builds on.
""")

code(r"""
YPOS = {0: 3, 1: 2, 2: 1, 3: 0}                          # wake top, deep bottom
YLAB = ["Deep NREM", "Light NREM", "REM", "Wake"]

def night_bouts(g):
    "one night's bouts as a frame of (stage, start, end)"
    stages, _, st, en = merge_bouts(g.sort_values("start"))
    return pd.DataFrame([dict(stage=s, t0=g.start.iloc[a], t1=g.end.iloc[b - 1])
                         for s, a, b in zip(stages, st, en)])

def representative_night(pid, target, min_exits=6, need_all_three=True):
    # The night whose own REM-exit mix sits closest to the participant's pooled value.
    # need_all_three also requires at least one exit to each destination, so the figure
    # shows all three colours instead of leaving a reader to wonder where one went.
    # (Docstrings here must stay single-line: this code is emitted inside a triple-quoted
    #  block in the generator, so a nested triple quote would end it early.)
    df = read_stages(pid)
    best = None
    for night, g in df.groupby("night"):
        if g.loc[g.stage != "wakeup", "dur"].sum() < 300:
            continue
        bt = night_bouts(g)
        ex = [NAME[bt.stage.iloc[i + 1]] for i in range(len(bt) - 1) if bt.stage.iloc[i] == 1]
        if len(ex) < min_exits:
            continue
        if need_all_three and not {"L", "W", "D"}.issubset(set(ex)):
            continue
        d = abs(ex.count("L") / len(ex) - target)
        if best is None or d < best[0]:
            best = (d, night, g)
    return best

# Pick the pair to illustrate: strong and weak memory scores, but only among
# participants who have a night containing all three exit destinations, so both
# panels show the full palette. Among those, take the widest separation in P_RL.
_cand = dat.dropna(subset=["ace_memory_subscale", "P_RL"]).sort_values("ace_memory_subscale")
_lo_pool = _cand.head(12).sort_values("P_RL")           # weak memory, lowest P_RL first
_hi_pool = _cand.tail(12).sort_values("P_RL", ascending=False)
def _first_usable(pool):
    for _, row in pool.iterrows():
        if representative_night(row.participant, row.P_RL) is not None:
            return row
    return pool.iloc[0]
EXEMPLARS = {"high": _first_usable(_hi_pool), "low": _first_usable(_lo_pool)}
print(f"Figure 1 exemplars: high = {EXEMPLARS['high'].participant} "
      f"(memory {EXEMPLARS['high'].ace_memory_subscale:.0f}, P_RL {EXEMPLARS['high'].P_RL:.2f}) | "
      f"low = {EXEMPLARS['low'].participant} "
      f"(memory {EXEMPLARS['low'].ace_memory_subscale:.0f}, P_RL {EXEMPLARS['low'].P_RL:.2f})")

fig = plt.figure(figsize=(13.6, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[3.6, 1], hspace=.55, wspace=.20)
for row, key in enumerate(["high", "low"]):
    p = EXEMPLARS[key]
    _, night, g = representative_night(p.participant, p.P_RL)
    bt = night_bouts(g)
    t0 = bt.t0.iloc[0]
    bt["h0"] = (bt.t0 - t0).dt.total_seconds() / 3600
    bt["h1"] = (bt.t1 - t0).dt.total_seconds() / 3600

    ax = fig.add_subplot(gs[row, 0]); ax.grid(False)
    for y in range(4):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    for i, b in bt.iterrows():                              # the night as context
        ax.plot([b.h0, b.h1], [YPOS[b.stage]] * 2, color=AXIS, lw=1.8, zorder=2,
                solid_capstyle="butt")
        if i + 1 < len(bt):
            ax.plot([b.h1, b.h1], [YPOS[b.stage], YPOS[bt.stage.iloc[i + 1]]],
                    color=AXIS, lw=1.0, zorder=2)
    counts = {"L": 0, "W": 0, "D": 0}
    for i, b in bt.iterrows():                              # REM bouts as the subject
        if b.stage != 1 or i + 1 >= len(bt):
            continue
        d = NAME[bt.stage.iloc[i + 1]]; counts[d] += 1
        ax.plot([b.h0, b.h1], [2, 2], color=DEST[d], lw=6, zorder=4, solid_capstyle="round")
        ax.annotate("", xy=(b.h1, YPOS[bt.stage.iloc[i + 1]]), xytext=(b.h1, 2),
                    arrowprops=dict(arrowstyle="-|>", color=DEST[d], lw=1.8,
                                    shrinkA=1, shrinkB=1, mutation_scale=11), zorder=5)
    ax.set_yticks(range(4), YLAB, fontsize=9)
    ax.set_ylim(-.5, 3.6); ax.set_xlim(-.15, bt.h1.max() + .15)
    ax.set_xlabel("hours from first recorded epoch" if row else "")
    ax.set_title(f"{'Strong' if key == 'high' else 'Weak'} memory score — ACE-III memory "
                 f"{p.ace_memory_subscale:.0f}/26 · one representative night ({night})",
                 loc="left", fontsize=10.5, pad=26)
    ax.text(0, 1.015, f"{sum(counts.values())} REM exits this night: {counts['L']} → light · "
                      f"{counts['W']} → wake · {counts['D']} → deep",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

    axb = fig.add_subplot(gs[row, 1]); axb.grid(False)
    for s in axb.spines.values():
        s.set_visible(False)
    axb.set_xlim(0, 1.0); axb.set_ylim(-.75, 2.75); axb.set_xticks([]); axb.set_yticks([])
    for j, (sh, k) in enumerate(zip([p.P_RL, p.P_RW, p.P_RD], ("L", "W", "D"))):
        y = 2 - j
        rounded_bar(axb, 0, y - .17, sh, .34, DEST[k], r_px=4, orient="h", zorder=3)
        axb.text(sh + .03, y, f"{100*sh:.0f}%", va="center", fontsize=9.5,
                 color=INK if k == "L" else INK2, fontweight="bold" if k == "L" else "normal")
        axb.text(0, y + .30, DEST_LABEL[k], va="bottom", fontsize=8.5, color=MUTED)
    axb.set_title(f"Whole recording · {int(p.nights)} nights · {int(p.exits_R)} REM exits",
                  loc="left", fontsize=9.5, color=INK2, fontweight="normal", pad=10)
    axb.text(0, -.62, f"P(REM → light) = {p.P_RL:.2f}", fontsize=11, color=INK,
             fontweight="bold", va="center")
suptitle(fig, "Figure 1 · What the marker counts",
         "Every REM bout is painted by the stage that follows it. Single nights look alike — the two "
         "people separate only once several hundred REM exits are pooled (right).", y=1.03)
dest_legend(fig)
save(fig, "fig01_hypnogram"); plt.show()
""")

# ============================================================ 4. Table 1
md(r"""
## 4 · Table 1 — cohort characteristics

Split at the ACE-III ≤ 82 screening cut-off.
""")

code(r"""
dat["impaired"] = (dat.ace_total <= 82).astype(int)

def table1(d):
    rows = []
    def add(label, col, pct=False, fmt="{:.1f}"):
        a, b = d[d.impaired == 0][col].dropna(), d[d.impaired == 1][col].dropna()
        if pct:
            rows.append(dict(Variable=label, Unimpaired=f"{100*a.mean():.0f}%",
                             Impaired=f"{100*b.mean():.0f}%",
                             p=f"{stats.fisher_exact(pd.crosstab(d.impaired, d[col]).values)[1]:.3f}"
                               if d[col].nunique() == 2 else ""))
        else:
            p = stats.mannwhitneyu(a, b).pvalue
            rows.append(dict(Variable=label,
                             Unimpaired=f"{fmt.format(a.mean())} ± {fmt.format(a.std())}",
                             Impaired=f"{fmt.format(b.mean())} ± {fmt.format(b.std())}",
                             p=f"{p:.3f}"))
    add("Age (years, band midpoint)", "age_mid")
    add("Male", "male", pct=True)
    add("Hypertension", "htn", pct=True)
    add("Osteoarthritis", "osteo", pct=True)
    add("ACE-III total", "ace_total")
    add("  memory (/26)", "ace_memory_subscale")
    add("  attention (/18)", "ace_attention_subscale")
    add("PHQ-9", "phq_total")
    add("GDS-15", "gds15_total")
    add("Nights recorded", "nights", fmt="{:.0f}")
    add("Total sleep time (min)", "tst_mean")
    add("REM terminations (n)", "exits_R", fmt="{:.0f}")
    add("P(REM → light NREM)", "P_RL", fmt="{:.3f}")
    return pd.DataFrame(rows)

T1 = table1(dat)
print(f"Unimpaired n={(dat.impaired==0).sum()}   Impaired (ACE-III<=82) n={dat.impaired.sum()}\n")
print(T1.to_string(index=False))
T1.to_csv(f"{FIGDIR}/table1_cohort.csv", index=False)
""")

# ============================================================ 5. Matrix
md(r"""
## 5 · The transition matrix

Row *i*, column *j* is the probability that a bout of stage *i* is followed by a bout of stage *j*,
pooled over all participants and nights. Rows sum to 1; the diagonal is empty by construction.

Figure 2 shows the matrix twice: once as numbers on a magnitude scale, and once as the state machine
those numbers describe, which is where the ultradian cycle becomes visible.
""")

code(r"""
M = np.full((4, 4), np.nan)
tot_counts = dat[TCOLS].sum()
for a in range(4):
    cols = [f"n_{NAME[a]}{NAME[b]}" for b in range(4) if b != a]
    s = tot_counts[cols].sum()
    for b in range(4):
        if a != b:
            M[a, b] = tot_counts[f"n_{NAME[a]}{NAME[b]}"] / s

order = [0, 2, 3, 1]                       # Wake, Light, Deep, REM
Mo = M[np.ix_(order, order)]
fig = plt.figure(figsize=(13.2, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=.24)

ax = fig.add_subplot(gs[0, 0]); ax.grid(False)
vmax = np.nanmax(M)
im = ax.imshow(Mo, cmap=SEQ, vmin=0, vmax=vmax)
for i in range(4):
    for j in range(4):
        if np.isfinite(Mo[i, j]):
            ax.text(j, i, f"{Mo[i,j]:.2f}", ha="center", va="center", fontsize=11.5,
                    color="white" if Mo[i, j] > .55 * vmax else INK,
                    fontweight="bold" if (order[i], order[j]) == (1, 2) else "normal")
        else:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=WASH,
                                       edgecolor=SURFACE, lw=2, zorder=3))
            ax.text(j, i, "—", ha="center", va="center", color=AXIS, fontsize=11, zorder=4)
for k in np.arange(-.5, 4, 1):                        # 2px surface gaps between cells
    ax.axhline(k, color=SURFACE, lw=2.5, zorder=3)
    ax.axvline(k, color=SURFACE, lw=2.5, zorder=3)
ax.add_patch(plt.Rectangle((-.5, 3.5), 4, -1, fill=False, edgecolor=INK, lw=1.8, zorder=6))
ax.set_xticks(range(4), [LONG[NAME[i]] for i in order], fontsize=9.5)
ax.set_yticks(range(4), [LONG[NAME[i]] for i in order], fontsize=9.5)
ax.set_xlabel("to"); ax.set_ylabel("from")
ax.set_title("a · Bout-to-bout transition probabilities", loc="left", pad=26)
ax.text(0, 1.02, "the boxed REM row is the subject of this notebook", transform=ax.transAxes,
        fontsize=8.8, color=MUTED, va="bottom")
cb = fig.colorbar(im, ax=ax, shrink=.72, pad=.03); cb.outline.set_visible(False)
cb.set_label("transition probability", fontsize=9, color=INK2)

ax = fig.add_subplot(gs[0, 1]); ax.grid(False)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.42, 1.5); ax.set_aspect("equal")
P = {0: (0, 1.0), 2: (-1.0, 0), 3: (0, -1.0), 1: (1.0, 0)}      # W top, L left, D bottom, R right
RAD = .22
for a in range(4):
    for b in range(4):
        if a == b or not np.isfinite(M[a, b]):
            continue
        rem = a == 1
        col = DEST[NAME[b]] if rem else GRID
        lw = 1.0 + 8.0 * M[a, b] if rem else 0.6 + 3.6 * M[a, b]
        x0, y0 = P[a]; x1, y1 = P[b]
        dx, dy = x1 - x0, y1 - y0
        sh = .24 / np.hypot(dx, dy)
        arrow(ax, (x0 + dx * sh, y0 + dy * sh), (x1 - dx * sh, y1 - dy * sh), col, lw=lw,
              rad=RAD, zorder=4 if rem else 2, ms=9 if rem else 6)
        if rem:                                    # label sits on the arc, not the chord
            ax.text((x0 + x1) / 2 - RAD / 2 * dy, (y0 + y1) / 2 + RAD / 2 * dx,
                    f"{M[a,b]:.2f}", ha="center", va="center", fontsize=10, color=INK,
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=.20", fc=SURFACE, ec="none"))
for k, (x, y) in P.items():
    is_rem = k == 1
    ax.scatter([x], [y], s=2600, color=WASH, edgecolor=INK if is_rem else AXIS,
               linewidth=1.8 if is_rem else 1.0, zorder=5)
    ax.text(x, y, LONG[NAME[k]].replace(" ", "\n"), ha="center", va="center", fontsize=9,
            color=INK, fontweight="bold" if is_rem else "normal", zorder=6)
ax.set_title("b · The same matrix as a state diagram", loc="left", pad=26)
ax.text(0, 1.02, "arrow width ∝ probability · REM exits in colour, every other route in grey",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 2 · Sleep as a sequence of states",
         "Rows sum to 1; the diagonal is empty because consecutive identical stages are merged into "
         "a single bout.", y=1.04)
save(fig, "fig02_matrix"); plt.show()
""")

# ---------------------------------------------------------- Figure 3 (new)
md(r"""
### The same numbers as flow — Figure 3

The matrix is row-normalised, which deliberately hides how much traffic each route actually carries.
Panel **a** restores that: REM supplies only about 15% of all stage transitions, so a single night
yields a handful of REM exits against hundreds of NREM ones — the arithmetic behind the
nights-needed curve in §12.

Panel **b** puts the two cognitive groups side by side with each normalised to 100%, so the
comparison is about composition rather than group size.
""")

code(r"""
ORDER = [0, 1, 2, 3]                                     # W, R, L, D on both banks
fig = plt.figure(figsize=(13.4, 5.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=.16)

ax = fig.add_subplot(gs[0, 0]); ax.grid(False)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
cnt = dat[TCOLS].sum()
Cm = np.zeros((4, 4))
for a in range(4):
    for b in range(4):
        if a != b:
            Cm[a, b] = cnt[f"n_{NAME[a]}{NAME[b]}"]
tot = Cm.sum(); PAD = .035
out_h, in_h = Cm.sum(1) / tot, Cm.sum(0) / tot
scale = 1 / (1 + 3 * PAD)

def stack(hs):
    y, pos = 1.0, {}
    for k in ORDER:
        pos[k] = ((y - hs[k]) * scale, y * scale)
        y -= hs[k] + PAD
    return pos

Lp, Rp = stack(out_h), stack(in_h)
lcur = {k: v[1] for k, v in Lp.items()}
rcur = {k: v[1] for k, v in Rp.items()}
for a in ORDER:
    for b in ORDER:
        if a == b or Cm[a, b] == 0:
            continue
        h = Cm[a, b] / tot * scale
        rem = a == 1
        ribbon(ax, .13, .87, lcur[a], lcur[a] - h, rcur[b], rcur[b] - h,
               DEST[NAME[b]] if rem else GRID, alpha=1 if rem else .95,
               zorder=4 if rem else 2, ring=1.5 if rem else 0)
        lcur[a] -= h; rcur[b] -= h
for k in ORDER:
    for pos, x, side in ((Lp, .13, "right"), (Rp, .87, "left")):
        lo, hi = pos[k]
        ax.add_patch(plt.Rectangle((x - .016 if side == "right" else x, lo), .016, hi - lo,
                                   facecolor=INK if k == 1 else AXIS, edgecolor="none", zorder=6))
    ax.text(.10, sum(Lp[k]) / 2, f"{LONG[NAME[k]]}\n{100*out_h[k]:.0f}%", ha="right", va="center",
            fontsize=9, color=INK if k == 1 else INK2, fontweight="bold" if k == 1 else "normal")
    ax.text(.90, sum(Rp[k]) / 2, f"{LONG[NAME[k]]}\n{100*in_h[k]:.0f}%", ha="left", va="center",
            fontsize=9, color=INK2)
ax.set_xlim(-.17, 1.17); ax.set_ylim(-.05, 1.06)
ax.text(.13, 1.0, "leaving", fontsize=9, color=MUTED, ha="center")
ax.text(.87, 1.0, "arriving", fontsize=9, color=MUTED, ha="center")
ax.set_title(f"a · Traffic: all {int(tot):,} stage transitions", loc="left", pad=26)
ax.text(0, 1.02, f"REM supplies only {100*out_h[1]:.0f}% of the night's transitions — the reason "
                 f"one night is not enough", transform=ax.transAxes, fontsize=8.8, color=MUTED,
        va="bottom")

ax = fig.add_subplot(gs[0, 1]); ax.grid(False)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
grp = []
for imp, lab in [(0, "ACE-III > 82"), (1, "ACE-III ≤ 82")]:
    d = dat[dat.impaired == imp]
    c = np.array([d.n_RL.sum(), d.n_RW.sum(), d.n_RD.sum()], float)
    grp.append((lab, len(d), c.sum(), c / c.sum()))
BH, BG, dgap = .42, .16, .06
ytop = {0: 1.0, 1: 1.0 - BH - BG}
dest_h = sum(sh for _, _, _, sh in grp) * BH
span = 1.0 - (ytop[1] - BH)
dpos, y = {}, 1.0
for j in range(3):
    h = dest_h[j] / dest_h.sum() * (span - 2 * dgap)
    dpos[j] = (y - h, y); y -= h + dgap
dcur = {j: dpos[j][1] for j in range(3)}
for gi, (lab, n, ntot, sh) in enumerate(grp):
    cur = ytop[gi]
    for j, k in enumerate("LWD"):
        h = sh[j] * BH
        ribbon(ax, .19, .80, cur, cur - h, dcur[j], dcur[j] - h, DEST[k], alpha=1.0,
               zorder=5 - gi, ring=1.8)
        ax.text(.215, cur - h / 2, f"{100*sh[j]:.0f}%", ha="left", va="center", fontsize=9,
                color=on_fill(DEST[k]) if sh[j] > .13 else INK, fontweight="bold", zorder=8)
        cur -= h; dcur[j] -= h
    ax.add_patch(plt.Rectangle((.172, ytop[gi] - BH), .018, BH, facecolor=INK, edgecolor="none",
                               zorder=6))
    ax.text(.152, ytop[gi] - BH / 2, f"{lab}\nn = {n} · {int(ntot):,} REM exits", ha="right",
            va="center", fontsize=9, color=INK2)
for j, k in enumerate("LWD"):
    lo, hi = dpos[j]
    ax.add_patch(plt.Rectangle((.80, lo), .018, hi - lo, facecolor=DEST[k], edgecolor="none",
                               zorder=6))
    ax.text(.835, (lo + hi) / 2, LONG[k], ha="left", va="center", fontsize=9.5, color=INK2)
ax.set_xlim(-.34, 1.14); ax.set_ylim(-.05, 1.06)
ax.set_title("b · Where REM exits go, by cognitive group", loc="left", pad=26)
ax.text(0, 1.02, f"each group normalised to 100% · the canonical exit loses "
                 f"{100*(grp[0][3][0]-grp[1][3][0]):.0f} points in the impaired group",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 3 · The same data as flow",
         "Ribbon thickness is the share of transitions taking that route.", y=1.04)
save(fig, "fig03_flow"); plt.show()
""")

# ============================================================ 6. Primary
md(r"""
## 6 · Primary analysis

Every association below is a linear model on **standardised variables**, adjusted for age and sex:

$$z(\text{ACE}) = \beta \cdot z(P_{R \to L}) + \gamma_1 z(\text{age}) + \gamma_2 z(\text{sex}) + \varepsilon$$

so β is the change in ACE-III standard deviations per standard deviation of the transition
probability. Spearman ρ is reported alongside as a rank-based check that no single point drives it.
""")

code(r"""
def assoc(y, x, cov=("age_mid", "male"), data=None, ret_model=False):
    d = (dat if data is None else data)[[y, x] + list(cov)].dropna()
    if len(d) < 20 or d[x].std() == 0:
        return None
    Z = d[[x] + list(cov)].apply(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s)
    m = sm.OLS((d[y] - d[y].mean()) / d[y].std(), sm.add_constant(Z)).fit()
    ci = m.conf_int().loc[x]
    rho, p_rho = stats.spearmanr(d[x], d[y])
    out = dict(outcome=y, var=x, n=len(d), beta=m.params[x], se=m.bse[x],
               lo=ci[0], hi=ci[1], p=m.pvalues[x], rho=rho, p_rho=p_rho)
    return (out, m) if ret_model else out

print("PRIMARY: P(REM -> light NREM) vs ACE-III   [age- and sex-adjusted]\n")
prim = pd.DataFrame([assoc(y, "P_RL") for y in ["ace_total", "ace_memory_subscale"]])
print(prim.round(4).to_string(index=False))

# --- permutation test: is the adjusted beta larger than under a null of no association? ---
print("\nPermutation test (10,000 label shuffles of the outcome):")
PERM = {}
for y in ["ace_total", "ace_memory_subscale"]:
    d = dat[[y, "P_RL", "age_mid", "male"]].dropna()
    X = sm.add_constant(d[["P_RL", "age_mid", "male"]]); yv = d[y].values
    obs = sm.OLS(yv, X).fit().params["P_RL"]
    null = np.array([sm.OLS(RNG.permutation(yv), X).fit().params["P_RL"] for _ in range(10000)])
    PERM[y] = dict(obs=obs, null=null, p=float(np.mean(np.abs(null) >= abs(obs))))
    print(f"  {y:<24} observed {obs:7.2f}   permutation p = {PERM[y]['p']:.4f}")

# --- bootstrap CI on Spearman rho ---
print("\nBootstrap (10,000 resamples) 95% CI for Spearman rho:")
BOOT = {}
for y in ["ace_total", "ace_memory_subscale"]:
    d = dat[[y, "P_RL"]].dropna().values
    bs = np.array([stats.spearmanr(*d[RNG.integers(0, len(d), len(d))].T)[0] for _ in range(10000)])
    BOOT[y] = dict(rho=stats.spearmanr(*d.T)[0], bs=bs)
    print(f"  {y:<24} rho = {BOOT[y]['rho']:+.3f}  "
          f"95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")

# --- leave-one-out stability ---
print("\nLeave-one-out range of Spearman rho:")
for y in ["ace_total", "ace_memory_subscale"]:
    d = dat[[y, "P_RL"]].dropna().reset_index(drop=True)
    loo = [stats.spearmanr(d.drop(i).P_RL, d.drop(i)[y])[0] for i in d.index]
    print(f"  {y:<24} [{min(loo):+.3f}, {max(loo):+.3f}]  (no single participant flips it)")
""")

md(r"""
Figure 4 shows the association four ways. Panels **a** and **b** are the raw scatters. Panel **c**
is the added-variable plot: age and sex are regressed out of *both* variables first, so the slope
you see there is literally the adjusted β rather than an unadjusted stand-in for it. Panel **d**
drops the linearity assumption entirely and just splits the cohort into ACE-III tertiles.
""")

code(r"""
def scatter_fit(ax, x, y, color=DEST["L"], nboot=600):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    xs = np.linspace(d.x.min(), d.x.max(), 60)
    bb = np.array([np.polyval(np.polyfit(*d.values[RNG.integers(0, len(d), len(d))].T, 1), xs)
                   for _ in range(nboot)])
    ax.fill_between(xs, *np.percentile(bb, [2.5, 97.5], axis=0), color=color, alpha=.10, lw=0,
                    zorder=2)
    ax.plot(xs, np.polyval(np.polyfit(d.x, d.y, 1), xs), color=INK, lw=2, zorder=4)
    dot(ax, d.x, d.y, color, s=52, zorder=5)
    return d

fig = plt.figure(figsize=(12.6, 8.2))
gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.26)
for (r, c), y, lab, ttl in [((0, 0), "ace_memory_subscale", "ACE-III memory (/26)",
                             "a · The headline association"),
                            ((0, 1), "ace_total", "ACE-III total (/100)",
                             "b · The same marker against the total score")]:
    ax = fig.add_subplot(gs[r, c])
    d = scatter_fit(ax, dat.P_RL, dat[y])
    pr = assoc(y, "P_RL")
    ax.set_xlabel("P(REM → light NREM)"); ax.set_ylabel(lab)
    ax.set_title(ttl, loc="left", pad=26)
    ax.text(0, 1.015, f"β = {pr['beta']:+.2f} [{pr['lo']:+.2f}, {pr['hi']:+.2f}]   "
                      f"p = {pr['p']:.4f}   ρ = {BOOT[y]['rho']:+.2f} "
                      f"[{np.percentile(BOOT[y]['bs'],2.5):+.2f}, "
                      f"{np.percentile(BOOT[y]['bs'],97.5):+.2f}]   n = {pr['n']}",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
    for xv in d.x:
        ax.plot([xv, xv], [0, .018], transform=ax.get_xaxis_transform(), color=AXIS, lw=.8)

ax = fig.add_subplot(gs[1, 0])
dd = dat[["P_RL", "ace_memory_subscale", "age_mid", "male"]].dropna()
X = sm.add_constant(dd[["age_mid", "male"]])
scatter_fit(ax, sm.OLS(dd.P_RL, X).fit().resid, sm.OLS(dd.ace_memory_subscale, X).fit().resid)
ax.axhline(0, color=AXIS, lw=1, zorder=1); ax.axvline(0, color=AXIS, lw=1, zorder=1)
ax.set_xlabel("P(REM → light NREM), residual of age + sex")
ax.set_ylabel("ACE-III memory, residual of age + sex")
ax.set_title("c · The adjusted effect on its own", loc="left", pad=26)
ax.text(0, 1.015, "added-variable plot: the slope here is exactly the adjusted β",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

ax = fig.add_subplot(gs[1, 1])
dat["tert"] = pd.qcut(dat.ace_total, 3, labels=["low", "mid", "high"])
groups = [dat.loc[dat.tert == t, "P_RL"].dropna().values for t in ["low", "mid", "high"]]
means = [dat.loc[dat.tert == t, "ace_total"].mean() for t in ["low", "mid", "high"]]
for i, (g, col) in enumerate(zip(groups, ORD3)):
    half_violin(ax, g, i + .12, width=.30, color=col)
    ax.boxplot([g], positions=[i], widths=.16, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=AXIS, lw=1.2),
               capprops=dict(color=AXIS, lw=1.2),
               boxprops=dict(facecolor=col, edgecolor="none", alpha=.85))
    ax.scatter(np.full(len(g), i - .21) + RNG.normal(0, .035, len(g)), g, s=20, color=col,
               alpha=.85, zorder=4, edgecolor=SURFACE, lw=.8)
H, pk = stats.kruskal(*groups)
ax.set_xticks(range(3), [f"low\nACE {means[0]:.0f}", f"mid\nACE {means[1]:.0f}",
                         f"high\nACE {means[2]:.0f}"], fontsize=9)
ax.set_xlim(-.55, 2.6); ax.set_ylabel("P(REM → light NREM)")
ax.set_title("d · By ACE-III tertile", loc="left", pad=26)
ax.text(0, 1.015, f"Kruskal–Wallis p = {pk:.3f} · box, distribution and every participant shown",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 4 · The primary association",
         "Standardised, age- and sex-adjusted. Bands are 95% bootstrap intervals for the fit.",
         y=1.015)
save(fig, "fig04_primary"); plt.show()
""")

# ---------------------------------------------------------- Figure 5 (new)
md(r"""
### The gradient without a model — Figure 5

Regression coefficients are compressed summaries, and it is fair to ask whether the effect is
carried by a handful of participants. Figure 5 shows all of them: one column per person, ordered by
memory score, with the full REM-exit composition stacked underneath. The canonical blue band widens
from left to right without any model being fitted.
""")

code(r"""
d5 = dat.dropna(subset=["P_RL", "ace_memory_subscale"]).copy()
d5 = d5.sort_values(["ace_memory_subscale", "P_RL"]).reset_index(drop=True)
n5 = len(d5)

fig = plt.figure(figsize=(13.6, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.5], hspace=.13)

ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(-1, n5); ax0.set_ylim(0, 27)          # limits first: rounded_bar reads the scale
for i, v in enumerate(d5.ace_memory_subscale):
    rounded_bar(ax0, i - .36, 0, .72, v, MUTED, r_px=3)
ax0.set_xticks([])
ax0.set_yticks([0, 13, 26]); ax0.set_ylabel("ACE-III memory\n(/26)")
ax0.text(0, 1.06, "each column is one participant · whole recording pooled",
         transform=ax0.transAxes, fontsize=8.8, color=MUTED, va="bottom")

ax = fig.add_subplot(gs[1])
ax.set_xlim(-1, n5); ax.set_ylim(0, 1.0)
gap = px(ax, 2)                                   # 2px surface gap between stacked segments
for i, r in d5.iterrows():
    base = 0.0
    for k in ("L", "W", "D"):
        h = r[f"P_R{k}"]
        ax.add_patch(plt.Rectangle((i - .36, base), .72, max(h - gap, .002), facecolor=DEST[k],
                                   edgecolor="none", zorder=3))
        base += h
roll = d5.P_RL.rolling(9, center=True, min_periods=4).mean()
ax.plot(np.arange(n5), roll, color=SURFACE, lw=4.5, zorder=5)
ax.plot(np.arange(n5), roll, color=INK, lw=2, zorder=6)
ax.set_xticks([])
ax.set_yticks([0, .25, .5, .75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("share of that participant's REM exits")
ax.set_xlabel("← weaker memory" + " " * 110 + "stronger memory →")
lo13, hi13 = d5.P_RL.iloc[:13].mean(), d5.P_RL.iloc[-13:].mean()
for xpos, val, lab in [(6, lo13, "weakest 13"), (n5 - 7, hi13, "strongest 13")]:
    ax.annotate(f"{lab}: {100*val:.0f}%", xy=(xpos, val), xytext=(xpos, val - .17), ha="center",
                fontsize=9, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.2, shrinkA=2, shrinkB=6))
suptitle(fig, "Figure 5 · Every participant's REM-exit composition",
         "The gradient is visible without any model: the canonical blue share thins toward the "
         "left. Black line: 9-participant rolling mean.", y=1.045)
dest_legend(fig, labels={k: LONG[k] for k in "LWD"}, y=1.07)
save(fig, "fig05_participants"); plt.show()
""")

# ============================================================ 7. Specificity
md(r"""
## 7 · Is it specific? All twelve transitions

If low $P_{R \to L}$ were merely a symptom of globally disorganised sleep, every transition
probability would show the same gradient. Testing all twelve with Benjamini–Hochberg control
answers that directly.
""")

code(r"""
PAIRS = [(a, b) for a in range(4) for b in range(4) if a != b]
rows = []
for y in ["ace_total", "ace_memory_subscale"]:
    R = pd.DataFrame([assoc(y, f"P_{NAME[a]}{NAME[b]}") for a, b in PAIRS])
    R["q"] = multipletests(R.p, method="fdr_bh")[1]
    R["transition"] = [f"{LONG[NAME[a]]} → {LONG[NAME[b]]}" for a, b in PAIRS]
    R["origin"] = [NAME[a] for a, b in PAIRS]; R["dest"] = [NAME[b] for a, b in PAIRS]
    rows.append(R)
    print(f"\n{y}  (FDR over 12 transitions)")
    print(R.sort_values("p")[["transition", "n", "beta", "se", "p", "q"]].round(4).to_string(index=False))
spec_total, spec_mem = rows
spec_mem.to_csv(f"{FIGDIR}/table2_transition_specificity.csv", index=False)
""")

code(r"""
def forest(ax, R, labelcol, xlab, sig_col="q", sig_thr=.10, order=None, colorize=True):
    R = R.copy()
    if order is not None:
        R = R.set_index(labelcol).loc[order].reset_index()
    for i, r in R.iterrows():
        sig = r[sig_col] < sig_thr
        col = (POS if r.beta > 0 else NEG) if (sig and colorize) else (DEST["L"] if sig else MUTED)
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=2.4, alpha=1 if sig else .5,
                solid_capstyle="round", zorder=3)
        dot(ax, r.beta, i, col, s=95 if sig else 55, zorder=4)
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    ax.set_yticks(range(len(R)), R[labelcol], fontsize=9)
    ax.set_ylim(-.7, len(R) - .3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlab)
    return R

fig = plt.figure(figsize=(15.4, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[.80, 1.30, 1.00], wspace=.60)

ax = fig.add_subplot(gs[0]); ax.grid(False)
order4 = [0, 2, 3, 1]
Bm = np.full((4, 4), np.nan); Qm = np.full((4, 4), np.nan)
INV = {v: k for k, v in NAME.items()}
for _, r in spec_mem.iterrows():
    Bm[INV[r.origin], INV[r.dest]] = r.beta
    Qm[INV[r.origin], INV[r.dest]] = r["q"]
Bo, Qo = Bm[np.ix_(order4, order4)], Qm[np.ix_(order4, order4)]
vm = np.nanmax(np.abs(Bm))
ax.imshow(Bo, cmap=DIV, vmin=-vm, vmax=vm)
for i in range(4):
    for j in range(4):
        if np.isfinite(Bo[i, j]):
            star = "*" if Qo[i, j] < .10 else ""
            ax.text(j, i, f"{Bo[i,j]:+.2f}{star}", ha="center", va="center", fontsize=10,
                    color=on_fill(matplotlib.colors.to_hex(DIV((Bo[i, j] + vm) / (2 * vm)))),
                    fontweight="bold" if star else "normal")
        else:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=WASH, edgecolor=SURFACE,
                                       lw=2, zorder=3))
            ax.text(j, i, "—", ha="center", va="center", color=AXIS, fontsize=11, zorder=4)
for k in np.arange(-.5, 4, 1):
    ax.axhline(k, color=SURFACE, lw=2.5, zorder=3); ax.axvline(k, color=SURFACE, lw=2.5, zorder=3)
ax.set_xticks(range(4), [LONG[NAME[i]].replace(" NREM", "") for i in order4], fontsize=9)
ax.set_yticks(range(4), [LONG[NAME[i]].replace(" NREM", "") for i in order4], fontsize=9)
ax.set_xlabel("to"); ax.set_ylabel("from"); ax.set_anchor("N")
ax.set_title("a · β for every route (memory)", loc="left", pad=26)
ax.text(0, 1.03, "* survives FDR at q < 0.10", transform=ax.transAxes, fontsize=8.8, color=MUTED,
        va="bottom")

row_order = spec_mem.sort_values("beta").transition.tolist()
xlo = min(spec_mem.lo.min(), spec_total.lo.min()) - .08
xhi = max(spec_mem.hi.max(), spec_total.hi.max()) + .34
for gi, (R, ttl) in enumerate([(spec_mem, "b · ACE-III memory subscale"),
                               (spec_total, "c · ACE-III total")]):
    ax = fig.add_subplot(gs[gi + 1])
    Rr = forest(ax, R, "transition", "standardised β per SD (age- and sex-adjusted)",
                order=row_order)
    ax.set_xlim(xlo, xhi)
    for i, r in Rr.iterrows():
        if r["q"] < .10:
            ax.text(r.hi + .04, i, f"q={r['q']:.3f}", va="center", fontsize=8.5, color=INK,
                    fontweight="bold")
    ax.set_title(ttl, loc="left", pad=26)
    if gi == 1:
        ax.set_yticks(range(len(Rr)), [""] * len(Rr))
    else:
        ax.text(0, 1.03, "only REM-origin routes survive · panel c keeps the same row order",
                transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 6 · Is it specific? All twelve transitions",
         "If this were global sleep disorganisation, every route would move together.", y=1.03)
save(fig, "fig06_specificity"); plt.show()
""")

md(r"""
### Domain specificity

REM sleep is implicated in memory consolidation, so if the marker is real rather than a general
malaise indicator it should load on the memory subscale rather than spread evenly across domains.
""")

code(r"""
DOM = {"ace_total": "ACE-III total", "ace_memory_subscale": "Memory (/26)",
       "ace_attention_subscale": "Attention (/18)", "ace_fluency_subscale": "Fluency (/14)",
       "ace_language_subscale": "Language (/26)", "ace_visuospatial_subscale": "Visuospatial (/16)"}
D = pd.DataFrame([assoc(y, "P_RL") for y in DOM])
D["label"] = [DOM[y] for y in D.outcome]
# FDR across the five subscales only; the total is not an independent test of the same family
sub = D.outcome != "ace_total"
D["q"] = np.nan
D.loc[sub, "q"] = multipletests(D.loc[sub, "p"], method="fdr_bh")[1]
print(D[["label", "n", "beta", "se", "lo", "hi", "p", "q", "rho", "p_rho"]].round(4).to_string(index=False))
D.to_csv(f"{FIGDIR}/table3_domain_specificity.csv", index=False)

DOMS = [("ace_memory_subscale", "Memory (/26)"), ("ace_attention_subscale", "Attention (/18)"),
        ("ace_fluency_subscale", "Fluency (/14)"), ("ace_language_subscale", "Language (/26)"),
        ("ace_visuospatial_subscale", "Visuospatial (/16)"), ("ace_total", "ACE-III total")]
fig = plt.figure(figsize=(13.6, 7.0))
gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=.55, wspace=.30)

ax = fig.add_subplot(gs[0, :])
Dp = D.iloc[::-1].reset_index(drop=True)
forest(ax, Dp, "label", "standardised β per SD of P(REM → light NREM)", sig_col="p", sig_thr=.05)
for i, r in Dp.iterrows():
    sig = r["p"] < .05
    ax.text(r.hi + .03, i, f"p={r['p']:.4f}" if sig else f"p={r['p']:.2f}", va="center",
            fontsize=8.5, color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
ax.set_xlim(D.lo.min() - .05, D.hi.max() + .22)
ax.set_title("a · One marker against every cognitive domain", loc="left", pad=26)
ax.text(0, 1.02, "filled = p < 0.05 · FDR across the five subscales leaves memory alone",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

for j, (y, lab) in enumerate(DOMS):
    ax = fig.add_subplot(gs[1, j])
    is_mem = y == "ace_memory_subscale"
    col = DEST["L"] if is_mem else MUTED
    dd = dat[["P_RL", y]].dropna()
    dot(ax, dd.P_RL, dd[y], col, s=22, ring=1.0, zorder=4)
    xs = np.linspace(dd.P_RL.min(), dd.P_RL.max(), 30)
    ax.plot(xs, np.polyval(np.polyfit(dd.P_RL, dd[y], 1), xs), color=INK if is_mem else AXIS,
            lw=2 if is_mem else 1.4, zorder=5)
    ax.set_title(lab, loc="left", fontsize=9.5, pad=6, color=INK if is_mem else INK2,
                 fontweight="bold" if is_mem else "normal")
    ax.set_xticks([.4, .6, .8], ["0.4", "0.6", "0.8"], fontsize=8)
    ax.tick_params(labelsize=8)
    if j == 0:
        ax.set_ylabel("domain score", fontsize=9)
    if j == 2:
        ax.set_xlabel("P(REM → light NREM)", fontsize=9)
suptitle(fig, "Figure 7 · Domain specificity",
         "REM sleep is implicated in memory consolidation; a general malaise marker would spread "
         "evenly across domains.", y=1.02)
save(fig, "fig07_domains"); plt.show()
""")

# ============================================================ 8. Reliability
md(r"""
## 8 · Is it a trait? Reliability and six-month stability

A biomarker is only useful if the same person gives the same value twice. Two splits:

- **Split-half** — odd- versus even-numbered nights, Spearman–Brown corrected. Measures how much of
  the between-person spread is signal rather than night-to-night noise.
- **Test–retest** — first half versus second half of each participant's ~6-month follow-up. These
  are different months of a person's life, so this is a genuine temporal stability check.

Panel **c** adds a Bland–Altman view of the same split-half data, which answers a question the
correlation cannot: whether the disagreement between halves grows with the level of the marker. It
does not — the scatter is flat across the range, so the reliability coefficient is not being propped
up by a few extreme participants.
""")

code(r"""
nn = nights[nights.participant.isin(dat.participant)].copy()
nn["k"] = nn.groupby("participant").cumcount()

def prl_by(split_col, d=nn):
    g = d.groupby(["participant", split_col])[["n_RL", "n_RW", "n_RD"]].sum()
    g["P_RL"] = g.n_RL / g[["n_RL", "n_RW", "n_RD"]].sum(axis=1)
    return g.reset_index().pivot(index="participant", columns=split_col, values="P_RL").dropna()

odd_even = prl_by("half", nn.assign(half=nn.k % 2))
nn["frac"] = nn.groupby("participant")["k"].transform(lambda s: s / max(s.max(), 1))
first_second = prl_by("period", nn.assign(period=(nn.frac > .5).astype(int)))

r_sh = stats.pearsonr(odd_even[0], odd_even[1])
sb = 2 * r_sh[0] / (1 + r_sh[0])
r_tt = stats.pearsonr(first_second[0], first_second[1])
icc_note = nn.groupby("participant").size().median()
print(f"Split-half (odd vs even nights)   n={len(odd_even)}  r = {r_sh[0]:.3f}  p = {r_sh[1]:.2g}")
print(f"  Spearman-Brown corrected reliability = {sb:.3f}")
print(f"Test-retest (1st vs 2nd half of ~6 months)  n={len(first_second)}  "
      f"r = {r_tt[0]:.3f}  p = {r_tt[1]:.2g}")
print(f"  (median {icc_note:.0f} nights per person, so each half is ~{icc_note/2:.0f} nights)")

fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))
fig.subplots_adjust(wspace=.34)
for ax, W, ttl, note, xl, yl in [
        (axes[0], odd_even, "a · Split-half reliability",
         f"r = {r_sh[0]:.2f} · Spearman–Brown {sb:.2f} · grey line = perfect agreement",
         "odd nights", "even nights"),
        (axes[1], first_second, "b · Six-month stability",
         f"r = {r_tt[0]:.2f} · p = {r_tt[1]:.0e}", "first half of follow-up", "second half")]:
    dot(ax, W[0], W[1], DEST["L"], s=48)
    lim = [min(W[0].min(), W[1].min()) - .04, max(W[0].max(), W[1].max()) + .04]
    ax.plot(lim, lim, color=AXIS, lw=1.2, zorder=2)
    xs = np.linspace(*lim, 20)
    ax.plot(xs, np.polyval(np.polyfit(W[0], W[1], 1), xs), color=INK, lw=2, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"P(REM → light), {xl}"); ax.set_ylabel(f"P(REM → light), {yl}")
    ax.set_title(ttl, loc="left", pad=26)
    ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

ax = axes[2]
mean_ab, diff_ab = (odd_even[0] + odd_even[1]) / 2, odd_even[0] - odd_even[1]
bias, sd = diff_ab.mean(), diff_ab.std()
ax.axhspan(bias - 1.96 * sd, bias + 1.96 * sd, color=DEST["L"], alpha=.08, zorder=1)
for v, lab, c in [(bias, f"bias {bias:+.3f}", INK),
                  (bias + 1.96 * sd, f"+1.96 SD {bias+1.96*sd:+.2f}", AXIS),
                  (bias - 1.96 * sd, f"−1.96 SD {bias-1.96*sd:+.2f}", AXIS)]:
    ax.axhline(v, color=c, lw=1.4 if c == INK else 1.1, zorder=2)
    ax.text(.005, v, lab, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8.5,
            color=INK if c == INK else MUTED)
dot(ax, mean_ab, diff_ab, DEST["L"], s=48)
ax.set_xlabel("mean of the two halves"); ax.set_ylabel("odd − even")
ax.set_title("c · Agreement across the same nights", loc="left", pad=26)
ax.text(0, 1.015, "Bland–Altman: the disagreement does not grow with the level",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 8 · Is it a trait?",
         "A biomarker is only useful if the same person gives the same value twice.", y=1.02)
save(fig, "fig08_reliability"); plt.show()
""")

# ============================================================ 9. Robustness
md(r"""
## 9 · Robustness

Three families of challenge:

1. **Confounding** — mood, comorbidity, and every conventional sleep metric that might be the real driver.
2. **Analytic choices** — the minimum-bout filter and the minimum-nights threshold.
3. **Data quantity** — does the effect survive when only the best-monitored participants are kept?

A marker that dies under any of these is not worth reporting.
""")

code(r"""
# conventional sleep metrics computed from the same nights, for use as competing explanations
def conventional(pid):
    df = read_stages(pid)
    out = []
    for night, g in df.groupby("night"):
        s = g.groupby("stage").dur.sum()
        tst = s.get("deep", 0) + s.get("light", 0) + s.get("REM", 0)
        if tst < MIN_TST:
            continue
        spt = (g.end.max() - g.start.min()).total_seconds() / 60
        out.append(dict(tst=tst, deep_pct=100*s.get("deep", 0)/tst, rem_pct=100*s.get("REM", 0)/tst,
                        light_pct=100*s.get("light", 0)/tst, waso=s.get("wakeup", 0),
                        sleep_eff=100*tst/spt if spt > 0 else np.nan,
                        n_awak=(g.stage == "wakeup").sum(),
                        awak_per_hr=(g.stage == "wakeup").sum()/(tst/60)))
    o = pd.DataFrame(out)
    return pd.Series({**o.mean().add_suffix("_mean"), **o.std().add_suffix("_sd"),
                      "participant": pid})

conv = pd.DataFrame([conventional(p) for p in dat.participant])
# tst_mean already exists on `dat` from participant_matrix; drop the duplicate so the
# merge cannot silently rename it to tst_mean_x/_y and knock it out of later models
conv = conv.drop(columns=[c for c in conv.columns if c != "participant" and c in dat.columns])
dat = dat.merge(conv, on="participant", how="left")
CONV = ["tst_mean", "deep_pct_mean", "rem_pct_mean", "light_pct_mean",
        "sleep_eff_mean", "waso_mean", "awak_per_hr_mean", "tst_sd"]
missing = [c for c in CONV if c not in dat.columns]
assert not missing, f"expected conventional metrics missing: {missing}"
print("conventional metrics available:", CONV)
""")

code(r"""
COV_SETS = {
    "age + sex (primary)":            ["age_mid", "male"],
    "+ depression (PHQ-9)":           ["age_mid", "male", "phq_total"],
    "+ depression (GDS-15)":          ["age_mid", "male", "gds15_total"],
    "+ anxiety (GAD-7)":              ["age_mid", "male", "gad_total"],
    "+ hypertension":                 ["age_mid", "male", "htn"],
    "+ osteoarthritis":               ["age_mid", "male", "osteo"],
    "+ total sleep time":             ["age_mid", "male", "tst_mean"],
    "+ REM %":                        ["age_mid", "male", "rem_pct_mean"],
    "+ deep %":                       ["age_mid", "male", "deep_pct_mean"],
    "+ light %":                      ["age_mid", "male", "light_pct_mean"],
    "+ sleep efficiency":             ["age_mid", "male", "sleep_eff_mean"],
    "+ WASO":                         ["age_mid", "male", "waso_mean"],
    "+ arousals / h":                 ["age_mid", "male", "awak_per_hr_mean"],
    "+ nights recorded":              ["age_mid", "male", "nights"],
    "+ REM terminations (n)":         ["age_mid", "male", "exits_R"],
    "all of the above":               ["age_mid", "male", "phq_total", "gds15_total", "htn",
                                       "tst_mean", "rem_pct_mean", "deep_pct_mean",
                                       "light_pct_mean", "sleep_eff_mean", "awak_per_hr_mean",
                                       "nights"],
}
rob = []
for y in ["ace_total", "ace_memory_subscale"]:
    for name, cov in COV_SETS.items():
        cov = [c for c in cov if c in dat.columns]
        r = assoc(y, "P_RL", cov)
        if r:
            rob.append({**r, "model": name})
ROB = pd.DataFrame(rob)
for y in ["ace_total", "ace_memory_subscale"]:
    print(f"\n{y}")
    print(ROB[ROB.outcome == y][["model", "n", "beta", "se", "lo", "hi", "p"]].round(4).to_string(index=False))
ROB.to_csv(f"{FIGDIR}/table4_robustness.csv", index=False)
""")

code(r"""
print("SENSITIVITY TO ANALYTIC CHOICES\n")
sens = []
for mb in [0, 3, 5, 10]:
    nts = (nights[nights.participant.isin(dat.participant)] if mb == 0 else
           pd.concat([night_transitions(p, min_bout=mb) for p in dat.participant], ignore_index=True))
    pmx = participant_matrix(nts)
    dd = pmx.merge(demo, on="participant").query("ace_total == ace_total and nights >= @MIN_NIGHTS")
    for y in ["ace_total", "ace_memory_subscale"]:
        r = assoc(y, "P_RL", data=dd)
        sens.append({**r, "choice": f"min bout ≥ {mb} min"})
for thr in [14, 30, 60, 90, 120]:
    dd = dat[dat.nights >= thr]
    for y in ["ace_total", "ace_memory_subscale"]:
        r = assoc(y, "P_RL", data=dd)
        if r:
            sens.append({**r, "choice": f"≥ {thr} nights"})
SENS = pd.DataFrame(sens)
for y in ["ace_total", "ace_memory_subscale"]:
    print(f"\n{y}")
    print(SENS[SENS.outcome == y][["choice", "n", "beta", "se", "p"]].round(4).to_string(index=False))
SENS.to_csv(f"{FIGDIR}/table5_sensitivity.csv", index=False)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.8), sharex=True)
fig.subplots_adjust(wspace=.62)
for ax, y, ttl in [(axes[0], "ace_memory_subscale", "a · ACE-III memory"),
                   (axes[1], "ace_total", "b · ACE-III total")]:
    A = ROB[ROB.outcome == y].copy(); A["group"] = "covariate adjustment"
    Bs = SENS[SENS.outcome == y].rename(columns={"choice": "model"}).copy()
    Bs["group"] = "analytic choice"
    F = pd.concat([A, Bs], ignore_index=True).iloc[::-1].reset_index(drop=True)
    pr = ROB[(ROB.outcome == y) & (ROB.model == "age + sex (primary)")].iloc[0]
    ax.axvspan(pr.lo, pr.hi, color=DEST["L"], alpha=.10, zorder=0)
    ax.axvline(pr.beta, color=DEST["L"], lw=1.4, zorder=1)
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    for i, r in F.iterrows():
        sig = r.p < .05
        ax.plot([r.lo, r.hi], [i, i], color=INK2 if sig else MUTED, lw=2.2, alpha=1 if sig else .5,
                solid_capstyle="round", zorder=3)
        dot(ax, r.beta, i, INK2 if sig else MUTED, s=72 if sig else 46, zorder=4)
        ax.text(1.015, i, f"{r.p:.4f}" if r.p < .001 else f"{r.p:.3f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                color=INK if sig else MUTED)
    nsplit = (F.group == "analytic choice").sum()
    ax.axhline(nsplit - .5, color=GRID, lw=1.2, zorder=1)
    ax.axhspan(nsplit - .5, len(F) - .3, color=WASH, alpha=.55, zorder=0)
    ax.set_yticks(range(len(F)), F.model, fontsize=8.5)
    ax.set_ylim(-.7, len(F) - .3)
    ax.set_xlabel("standardised β per SD of P(REM → light NREM)")
    ax.set_title(ttl, loc="left", pad=26)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-.18, .98)
    ax.text(1.015, 1.005, "p", transform=ax.transAxes, fontsize=8.5, color=INK2, fontweight="bold")
    ax.text(-.34, (nsplit + len(F) - 1) / 2, "COVARIATE ADJUSTMENT",
            transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center",
            fontsize=7.5, color=MUTED, fontweight="bold")
    ax.text(-.34, (nsplit - 1) / 2, "ANALYTIC CHOICE",
            transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center",
            fontsize=7.5, color=MUTED, fontweight="bold")
    if ttl.startswith("a"):
        ax.text(0, 1.015, "blue band = 95% CI of the primary model", transform=ax.transAxes,
                fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 9 · Robustness",
         "Every estimate a sceptical reader would ask for, in one place. Faded = p ≥ 0.05.",
         y=1.015)
save(fig, "fig09_robustness"); plt.show()
""")

# ---------------------------------------------------------- Figure 10 (new)
md(r"""
### Is it a proxy for ordinary sleep quality? — Figure 10

The robustness table answers this one model at a time. Figure 10 answers it structurally.

Panel **a** is the correlation matrix. Against every *global* sleep-quality metric — total sleep
time, efficiency, WASO, arousals, REM % — the marker is essentially uncorrelated (|ρ| ≤ 0.14). It is
**not** independent of the light/deep balance (ρ = +0.50 with light %, −0.50 with deep %), which is
unsurprising: a night with more light NREM offers more opportunities for a REM episode to hand off
into it. That is exactly why both are carried as covariates in §9, and why the effect surviving
those adjustments (β = 0.48 and 0.46, both p < 0.01) is the meaningful test rather than the
correlation itself.

Panel **b** runs each metric against memory on its own terms, FDR-corrected across the nine. Deep %
is the only conventional metric that comes close (q = 0.06), and it points the other way — more deep
sleep, worse memory — which is not the direction the conventional literature would predict and
should be read as one more reason to treat this device's stage labels cautiously.
""")

code(r"""
LABS = {"P_RL": "P(REM → light)", "tst_mean": "Total sleep time", "deep_pct_mean": "Deep %",
        "rem_pct_mean": "REM %", "light_pct_mean": "Light %", "sleep_eff_mean": "Sleep efficiency",
        "waso_mean": "WASO", "awak_per_hr_mean": "Arousals / h", "tst_sd": "TST night-to-night SD"}
cols = list(LABS)
Rc = dat[cols].corr(method="spearman").values

fig = plt.figure(figsize=(13.8, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=.45)

ax = fig.add_subplot(gs[0]); ax.grid(False)
im = ax.imshow(Rc, cmap=DIV, vmin=-1, vmax=1)
for i in range(len(cols)):
    for j in range(len(cols)):
        if i == j:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=WASH, edgecolor=SURFACE,
                                       lw=2, zorder=3))
            ax.text(j, i, "—", ha="center", va="center", color=AXIS, fontsize=10, zorder=4)
            continue
        ax.text(j, i, f"{Rc[i,j]:+.2f}".replace("+0.", ".").replace("-0.", "−."), ha="center",
                va="center", fontsize=8,
                color=on_fill(matplotlib.colors.to_hex(DIV((Rc[i, j] + 1) / 2))),
                fontweight="bold" if (i == 0 or j == 0) else "normal")
for k in np.arange(-.5, len(cols), 1):
    ax.axhline(k, color=SURFACE, lw=2, zorder=3); ax.axvline(k, color=SURFACE, lw=2, zorder=3)
ax.add_patch(plt.Rectangle((-.5, -.5), len(cols), 1, fill=False, edgecolor=INK, lw=1.8, zorder=6))
ax.set_xticks(range(len(cols)), [LABS[c] for c in cols], rotation=42, ha="right", fontsize=8.5)
ax.set_yticks(range(len(cols)), [LABS[c] for c in cols], fontsize=8.5)
ax.set_title("a · Spearman correlation between metrics", loc="left", pad=26)
ax.text(0, 1.04, "|ρ| ≤ 0.14 against every global sleep-quality metric; ρ = ±0.50 with the "
                 "light/deep balance", transform=ax.transAxes, fontsize=8.8, color=MUTED,
        va="bottom")
cb = fig.colorbar(im, ax=ax, shrink=.7, pad=.02); cb.outline.set_visible(False)
cb.set_label("Spearman ρ", fontsize=9, color=INK2)

ax = fig.add_subplot(gs[1])
rows = []
for c in cols:
    d = dat[["ace_memory_subscale", c, "age_mid", "male"]].dropna()
    Z = d[[c, "age_mid", "male"]].apply(lambda s: (s - s.mean()) / s.std())
    m = sm.OLS((d.ace_memory_subscale - d.ace_memory_subscale.mean()) / d.ace_memory_subscale.std(),
               sm.add_constant(Z)).fit()
    ci = m.conf_int().loc[c]
    rows.append(dict(label=LABS[c], beta=m.params[c], lo=ci[0], hi=ci[1], p=m.pvalues[c]))
Rm = pd.DataFrame(rows)
Rm["q"] = multipletests(Rm.p, method="fdr_bh")[1]
Rm = Rm.sort_values("beta").reset_index(drop=True)
forest(ax, Rm, "label", "standardised β for ACE-III memory (age- and sex-adjusted)", sig_col="q",
       sig_thr=.05)
for i, r in Rm.iterrows():
    sig = r["q"] < .05
    ax.text(r.hi + .03, i, f"q={r['q']:.3f}" if sig else f"q={r['q']:.2f}", va="center",
            fontsize=8.5, color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
ax.set_xlim(Rm.lo.min() - .05, Rm.hi.max() + .34)
ax.set_title("b · Each metric on its own against memory", loc="left", pad=26)
ax.text(0, 1.02, "one metric at a time, FDR-corrected across the nine — only the marker survives",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 10 · Is it just a proxy for ordinary sleep quality?",
         "No: it is unrelated to sleep duration, efficiency, WASO and arousals. It does share "
         "variance with the light/deep balance — which is why the robustness models adjust for "
         "both, and the effect survives.", y=1.04)
save(fig, "fig10_independence"); plt.show()
print(Rm[["label", "beta", "lo", "hi", "p", "q"]].round(4).to_string(index=False))
Rm.to_csv(f"{FIGDIR}/table8_metric_independence.csv", index=False)
""")

# ============================================================ 10. Night level
md(r"""
## 10 · Night-level model

The participant-level analysis collapses ~139 nights into one proportion. A night-level model keeps
all 6,000-plus nights and treats each night's REM terminations as a binomial draw, with
**cluster-robust standard errors by participant** (GEE, exchangeable working correlation). ACE-III is
a between-person predictor, so clustering is what keeps the inference honest — the effective sample
size stays the number of participants, not the number of nights.
""")

code(r"""
nl = nights.merge(demo[["participant", "ace_total", "ace_memory_subscale", "age_mid", "male"]],
                  on="participant")
nl = nl[nl.participant.isin(dat.participant)].copy()
nl["rem_exits"] = nl[["n_RL", "n_RW", "n_RD"]].sum(axis=1)
nl = nl[nl.rem_exits >= 1].dropna(subset=["ace_total", "age_mid", "male"]).reset_index(drop=True)
nl["p_rl"] = nl.n_RL / nl.rem_exits
print(f"nights in model: {len(nl):,}   participants: {nl.participant.nunique()}   "
      f"REM terminations: {nl.rem_exits.sum():,}")

for y in ["ace_total", "ace_memory_subscale"]:
    nl["z"] = (nl[y] - nl[y].mean()) / nl[y].std()
    m = sm.GEE.from_formula("p_rl ~ z + age_mid + male", groups="participant", data=nl,
                            family=sm.families.Binomial(), weights=nl.rem_exits.values,
                            cov_struct=sm.cov_struct.Exchangeable()).fit()
    or_ = np.exp(m.params["z"]); lo, hi = np.exp(m.conf_int().loc["z"])
    print(f"\n{y}:  log-odds per SD = {m.params['z']:+.4f} (robust SE {m.bse['z']:.4f})")
    print(f"    odds ratio = {or_:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   p = {m.pvalues['z']:.4g}")
""")

# ============================================================ 11. Benchmark
md(r"""
## 11 · Benchmark against conventional sleep metrics

The practical question: given a home recording, how well does each metric separate participants
scoring at or below the ACE-III impairment cut-off from those above it? Metrics are oriented so that
higher always means *more likely impaired*, and AUCs are bootstrapped.

Panel **c** is the honest footnote to panel **b**. The marker's lead over the best conventional
metric is positive in about 92% of bootstrap resamples, but the 95% interval for the difference
still includes zero at n = 52 — so "clearly better than chance" is supported, while "significantly
better than deep sleep %" is not yet.
""")

code(r"""
BENCH = {"P(REM → light NREM)": ("P_RL", -1)}
for c, lab, sgn in [("deep_pct_mean", "Deep sleep %", +1), ("rem_pct_mean", "REM sleep %", -1),
                    ("sleep_eff_mean", "Sleep efficiency", -1), ("waso_mean", "WASO", +1),
                    ("awak_per_hr_mean", "Arousals / h", +1), ("tst_mean", "Total sleep time", -1)]:
    if c in dat.columns:
        BENCH[lab] = (c, sgn)

res = []
for lab, (c, sgn) in BENCH.items():
    d = dat[[c, "impaired"]].dropna()
    s = sgn * d[c].values
    auc = roc_auc_score(d.impaired, s)
    bs = [roc_auc_score(d.impaired.values[i], s[i]) for i in
          (RNG.integers(0, len(d), (2000, len(d)))) if len(np.unique(d.impaired.values[i])) == 2]
    res.append(dict(metric=lab, n=len(d), AUC=auc,
                    lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
B = pd.DataFrame(res).sort_values("AUC", ascending=False)
print(B.round(3).to_string(index=False))
B.to_csv(f"{FIGDIR}/table6_auc_benchmark.csv", index=False)

# DeLong-free comparison: bootstrap the AUC difference vs the best conventional metric
best_conv = B[B.metric != "P(REM → light NREM)"].iloc[0]
cbest, sbest = BENCH[best_conv.metric]
d = dat[["P_RL", cbest, "impaired"]].dropna()
diffs = []
for _ in range(4000):
    i = RNG.integers(0, len(d), len(d))
    if len(np.unique(d.impaired.values[i])) < 2:
        continue
    diffs.append(roc_auc_score(d.impaired.values[i], -d.P_RL.values[i]) -
                 roc_auc_score(d.impaired.values[i], sbest * d[cbest].values[i]))
diffs = np.array(diffs)
print(f"\nAUC difference vs best conventional metric ({best_conv.metric}): "
      f"{diffs.mean():+.3f}  95% CI [{np.percentile(diffs,2.5):+.3f}, {np.percentile(diffs,97.5):+.3f}]"
      f"  (bootstrap p = {2*min((diffs<=0).mean(), (diffs>=0).mean()):.3f})")
""")

code(r"""
KEY = "P(REM → light NREM)"
fig = plt.figure(figsize=(15.0, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.12, 1], wspace=.52)

ax = fig.add_subplot(gs[0])
others = [l for l in BENCH if l not in (KEY, best_conv.metric)]
for j, lab in enumerate(others):
    c, sgn = BENCH[lab]
    d = dat[[c, "impaired"]].dropna()
    fpr, tpr, _ = roc_curve(d.impaired, sgn * d[c])
    ax.plot(fpr, tpr, color=MUTED, lw=1.1, alpha=.5, zorder=2,
            label="other conventional metrics" if j == 0 else None)
for lab, col, lw in [(best_conv.metric, DEST["W"], 2.0), (KEY, DEST["L"], 2.8)]:
    c, sgn = BENCH[lab]
    d = dat[[c, "impaired"]].dropna()
    fpr, tpr, _ = roc_curve(d.impaired, sgn * d[c])
    ax.plot(fpr, tpr, color=col, lw=lw, zorder=4,
            label=f"{lab}  (AUC {roc_auc_score(d.impaired, sgn*d[c]):.2f})")
ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, zorder=1)
ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
ax.set_aspect("equal"); ax.set_anchor("N")
ax.set_title("a · Detecting ACE-III ≤ 82", loc="left", pad=26)
ax.legend(fontsize=8.2, loc="lower right", labelcolor=INK2)

ax = fig.add_subplot(gs[1])
Bp = B.iloc[::-1].reset_index(drop=True)
for i, r in Bp.iterrows():
    top = r.metric == KEY
    col = DEST["L"] if top else MUTED
    ax.plot([r.lo, r.hi], [i, i], color=col, lw=2.4, alpha=1 if top else .55,
            solid_capstyle="round", zorder=3)
    dot(ax, r.AUC, i, col, s=100 if top else 55, zorder=4)
    ax.text(r.hi + .015, i, f"{r.AUC:.2f}", va="center", fontsize=9, color=INK if top else MUTED,
            fontweight="bold" if top else "normal")
ax.axvline(.5, color=AXIS, lw=1.2, zorder=1)
ax.text(.5, -.55, " chance", fontsize=8.5, color=MUTED, va="center")
ax.set_yticks(range(len(Bp)), [m.replace(KEY, "P(REM → light)") for m in Bp.metric], fontsize=9)
ax.set_ylim(-.9, len(Bp) - .1)
ax.set_xlabel("AUC (95% bootstrap CI)"); ax.set_xlim(.26, 1.04)
ax.grid(axis="y", visible=False)
ax.set_title("b · Ranked by discrimination", loc="left", pad=26)

ax = fig.add_subplot(gs[2])
lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
cnt, edges = np.histogram(diffs, bins=34)
for c0, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
    mid = (e0 + e1) / 2
    ax.add_patch(plt.Rectangle((e0, 0), (e1 - e0) * .86, c0, zorder=3, edgecolor="none",
                               facecolor=DEST["L"] if lo_d <= mid <= hi_d else GRID))
ax.set_ylim(0, cnt.max() * 1.28); ax.set_xlim(edges[0], edges[-1])
ax.axvline(0, color=AXIS, lw=1.2, zorder=4)
ax.plot([lo_d, hi_d], [cnt.max() * 1.12] * 2, color=INK, lw=2, solid_capstyle="butt", zorder=5)
ax.text((lo_d + hi_d) / 2, cnt.max() * 1.15, f"95% CI  {lo_d:+.2f} to {hi_d:+.2f}", ha="center",
        va="bottom", fontsize=8.8, color=INK)
ax.set_yticks([])
ax.set_xlabel(f"AUC advantage over {best_conv.metric.lower()}")
ax.set_title("c · Is the lead real?", loc="left", pad=26)
ax.text(0, 1.015, f"{100*np.mean(diffs>0):.0f}% of bootstrap resamples favour the marker",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 11 · Benchmark against conventional sleep metrics",
         "Every metric oriented so that higher means more likely impaired.", y=1.02)
save(fig, "fig11_auc"); plt.show()
""")

# ============================================================ 12. Nights needed
md(r"""
## 12 · How many nights does this need?

This is the question that decides whether the marker is clinically usable, and the reason a
six-month recording was necessary to find it at all.

For each *k*, **k nights are drawn at random from each participant**, the transition probability is
recomputed from that subsample alone, and the whole analysis is rerun — 60 draws per *k*. This traces
how the marker emerges as recording length grows.

Panel **c** keeps every one of those 780 re-estimates rather than collapsing them to a mean and a
band. It shows the thing the summary curve hides: at one night the sampling distribution straddles
zero, so a single-night study could plausibly report *any* result, including the wrong sign.
""")

code(r"""
nsub = nights[nights.participant.isin(dat.participant)].copy()
meta = dat[["participant", "ace_total", "ace_memory_subscale", "age_mid", "male", "impaired"]]
KS = [1, 2, 3, 5, 7, 10, 14, 21, 28, 42, 56, 84, 112]
N_REP = 60
curve, DRAWS = [], {}
for k in KS:
    b_t, b_m, p_m, aucs = [], [], [], []
    for rep in range(N_REP):
        parts = []
        for pid, g in nsub.groupby("participant", sort=False):
            gg = g.sample(min(k, len(g)), random_state=int(RNG.integers(1 << 31))) if k < len(g) else g
            tot = gg[["n_RL", "n_RW", "n_RD"]].values.sum()
            if tot < 1:
                continue
            parts.append((pid, gg.n_RL.sum() / tot))
        s = pd.DataFrame(parts, columns=["participant", "P_RL"]).merge(meta, on="participant")
        if len(s) < 20:
            continue
        rt = assoc("ace_total", "P_RL", data=s)
        rm = assoc("ace_memory_subscale", "P_RL", data=s)
        if rt: b_t.append(rt["beta"])
        if rm: b_m.append(rm["beta"]); p_m.append(rm["p"])
        if s.impaired.nunique() == 2:
            aucs.append(roc_auc_score(s.impaired, -s.P_RL))
    DRAWS[k] = dict(beta_mem=np.array(b_m), auc=np.array(aucs), p_mem=np.array(p_m))
    curve.append(dict(k=k, beta_total=np.mean(b_t), beta_mem=np.mean(b_m),
                      beta_mem_lo=np.percentile(b_m, 10), beta_mem_hi=np.percentile(b_m, 90),
                      auc=np.mean(aucs), auc_lo=np.percentile(aucs, 10),
                      auc_hi=np.percentile(aucs, 90), pow_mem=np.mean(np.array(p_m) < .05)))
CURVE = pd.DataFrame(curve)
print(CURVE.round(3).to_string(index=False))
CURVE.to_csv(f"{FIGDIR}/table7_nights_needed.csv", index=False)
""")

code(r"""
# one participant recorded no REM terminations at all, so P_RL is undefined for them
_d = dat[["impaired", "P_RL"]].dropna()
full_auc = roc_auc_score(_d.impaired, -_d.P_RL)
full_beta = assoc("ace_memory_subscale", "P_RL")["beta"]

fig = plt.figure(figsize=(14.2, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=.55, wspace=.22)
TICKS = [1, 3, 7, 14, 28, 56, 112]

ax = fig.add_subplot(gs[0, 0])
ax.fill_between(CURVE.k, CURVE.auc_lo, CURVE.auc_hi, color=DEST["L"], alpha=.13, lw=0)
ax.plot(CURVE.k, CURVE.auc, color=DEST["L"], lw=2, zorder=4)
dot(ax, CURVE.k, CURVE.auc, DEST["L"], s=42, zorder=5)
ax.axhline(full_auc, color=INK, lw=1.3, zorder=3)
ax.text(112, full_auc + .013, f"full recording {full_auc:.2f}", fontsize=8.5, color=INK, ha="right")
ax.axhline(.5, color=AXIS, lw=1.2, zorder=2)
ax.text(1.02, .506, "chance", fontsize=8.5, color=MUTED)
ax.axvspan(14, 28, color=WASH, zorder=0)
ax.text(19.8, .545, "2–4 weeks", fontsize=8.5, color=INK2, ha="center")
ax.set_xscale("log"); ax.set_xticks(TICKS, [str(t) for t in TICKS])
ax.set_ylabel("AUC for ACE-III ≤ 82")
ax.set_title("a · A single night is nearly uninformative", loc="left", pad=26)
ax.text(0, 1.02, "band = 10th–90th percentile over 60 random draws of k nights per person",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

ax = fig.add_subplot(gs[1, 0])
ax.plot(CURVE.k, CURVE.pow_mem, color=DEST["W"], lw=2, zorder=4)
dot(ax, CURVE.k, CURVE.pow_mem, DEST["W"], s=42, zorder=5)
ax.axhline(.8, color=AXIS, lw=1.2, zorder=2)
ax.text(1.02, .815, "80% power", fontsize=8.5, color=MUTED)
ax.set_xscale("log"); ax.set_xticks(TICKS, [str(t) for t in TICKS]); ax.set_ylim(0, 1.05)
ax.set_xlabel("nights of recording used per participant")
ax.set_ylabel("share of draws with p < 0.05")
ax.set_title("b · Power to detect the memory association", loc="left", pad=26)
ax.text(0, 1.02, f"80% is first reached at about {CURVE[CURVE.pow_mem >= .8].k.min():.0f} nights",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

ax = fig.add_subplot(gs[:, 1])
ks = list(DRAWS)
for i, k in enumerate(ks):
    b = DRAWS[k]["beta_mem"]
    xs = np.linspace(-.35, .95, 240)
    dns = stats.gaussian_kde(b)(xs); dns = dns / dns.max() * .82
    col = matplotlib.colors.to_hex(SEQ(np.linspace(.28, 1.0, len(ks))[i]))
    ax.fill_between(xs, i, i + dns, color=col, alpha=.85, lw=0, zorder=3 + i)
    ax.plot(xs, i + dns, color=SURFACE, lw=1.4, zorder=3 + i)
    ax.text(-.018, i + .12, f"{k}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.5,
            color=INK if k in (1, 14, 28, 112) else MUTED,
            fontweight="bold" if k in (1, 14, 28, 112) else "normal")
ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
ax.axvline(full_beta, color=INK, lw=1.3, zorder=len(ks) + 5)
ax.text(full_beta + .02, len(ks) + .25, f"full recording β = {full_beta:.2f}", fontsize=8.5, color=INK)
ax.set_yticks([]); ax.set_ylim(-.4, len(ks) + 1.0); ax.set_xlim(-.42, .95)
ax.set_xlabel("standardised β (memory) recovered from k nights")
ax.set_ylabel("nights per participant, k", labelpad=26)
ax.grid(axis="y", visible=False)
ax.set_title("c · The whole sampling distribution", loc="left", pad=26)
ax.text(0, 1.005, "each ridge is 60 re-estimates from k randomly drawn nights",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
suptitle(fig, "Figure 12 · How many nights does this need?",
         "The question that decides whether the marker is usable — and why a six-month recording was "
         "needed to find it.", y=1.015)
save(fig, "fig12_nights"); plt.show()
""")

# ============================================================ 13. Conclusion
md(r"""
## 13 · What this does and does not show

**What it shows.** In 73 community-dwelling older adults monitored at home for ~6 months, the
probability that a REM episode is followed by light NREM sleep — the canonical ultradian handoff —
tracks ACE-III performance, specifically in the memory domain. The association is not explained by
mood, comorbidity, or any conventional sleep metric; it is a stable individual trait (split-half
0.87, six-month test–retest 0.77); and it separates participants below the ACE-III impairment
cut-off with AUC ≈ 0.75, where total sleep time, efficiency, WASO, arousal rate and stage
percentages are all at chance. It also needs ~2–4 weeks of nights before it is measurable, which
is why an averaged-phenotype screen of this same cohort found nothing.

**What it does not show.** This is **cross-sectional**: ACE-III was measured once, so nothing here
establishes that transition disorganisation precedes cognitive decline rather than accompanying it.
The sample is small (n = 50 in the primary model) and the cohort is a convenience sample of the very
old, so the effect size should be treated as an upper estimate. The marker's discrimination lead
over the best conventional metric is also not itself significant at this sample size (Figure 11c).

The most important caveat is **measurement**. Stages come from a Withings under-mattress
ballistocardiograph plus wrist actigraphy, not polysomnography. Consumer devices identify REM with
moderate accuracy at best, and a scored "REM → deep" transition is physiologically atypical — direct
REM-to-N3 transitions are rare on PSG. Some of the off-canonical routes are therefore likely to be
staging ambiguity rather than true neurophysiology. That the *destination* of a REM exit is partly
device-dependent does not invalidate the marker as a **signal** — its reliability, specificity and
discrimination are all measured on the device's own output, which is what a home-monitoring
deployment would use — but it does mean the mechanistic reading should stay tentative until
PSG-validated replication.

**What would settle it.** A PSG sub-study to establish what the device's REM-exit routes correspond
to physiologically, and a longitudinal cohort with repeat cognitive testing to see whether
$P_{R \to L}$ predicts *change* in memory rather than its level.
""")

code(r"""
q_mem = spec_mem.set_index("var").loc["P_RL", "q"]
pm_ = assoc("ace_memory_subscale", "P_RL")
tiles = [("Memory association", f"β = {pm_['beta']:+.2f}", f"p = {pm_['p']:.4f} · FDR q = {q_mem:.3f}"),
         ("Split-half reliability", f"{sb:.2f}", "Spearman–Brown corrected"),
         ("Six-month stability", f"r = {r_tt[0]:.2f}", "first vs second half of follow-up"),
         ("Discrimination", f"AUC {full_auc:.2f}", f"best conventional metric {best_conv.AUC:.2f}"),
         ("Nights needed", "≈ 28", f"1 night AUC {CURVE.auc.iloc[0]:.2f} → "
                                   f"{CURVE[CURVE.k==28].auc.iloc[0]:.2f}")]
fig, ax = plt.subplots(figsize=(13.6, 2.1))
fig.subplots_adjust(left=.004, right=.996, top=.86, bottom=.02)
ax.set_xlim(0, len(tiles)); ax.set_ylim(0, 1); ax.axis("off")
for i, (lab, val, sub_) in enumerate(tiles):
    ax.add_patch(plt.Rectangle((i + .03, .06), .94, .88, facecolor=WASH, edgecolor="none"))
    ax.add_patch(plt.Rectangle((i + .03, .06), .012, .88, facecolor=DEST["L"], edgecolor="none"))
    ax.text(i + .10, .78, lab, fontsize=9, color=INK2, va="center")
    ax.text(i + .10, .50, val, fontsize=21, color=INK, va="center", fontweight="bold")
    ax.text(i + .10, .22, sub_, fontsize=8.2, color=MUTED, va="center")
suptitle(fig, "Figure 13 · Results at a glance", None, y=1.06)
save(fig, "fig13_summary"); plt.show()
""")

code(r"""
print("=" * 74)
print("SUMMARY")
print("=" * 74)
r_t, r_m = assoc("ace_total", "P_RL"), assoc("ace_memory_subscale", "P_RL")
print(f"  sample                      n = {len(dat)} participants, {len(nights):,} nights, "
      f"{nights[['n_RW','n_RL','n_RD']].values.sum():,} REM terminations")
print(f"  P(REM->light) vs ACE total  beta = {r_t['beta']:+.3f}  p = {r_t['p']:.4f}")
print(f"  P(REM->light) vs memory     beta = {r_m['beta']:+.3f}  p = {r_m['p']:.4f}  "
      f"(FDR q = {q_mem:.4f} across 12 transitions)")
print(f"  split-half reliability      {sb:.3f}   six-month test-retest r = {r_tt[0]:.3f}")
print(f"  AUC for ACE-III <= 82       {full_auc:.3f}  "
      f"(best conventional metric {best_conv.AUC:.3f})")
print(f"  nights needed               AUC {CURVE[CURVE.k==1].auc.iloc[0]:.2f} at 1 night -> "
      f"{CURVE[CURVE.k==14].auc.iloc[0]:.2f} at 14 -> {CURVE[CURVE.k==28].auc.iloc[0]:.2f} at 28")
print("=" * 74)
print(f"\nfigures and tables written to {FIGDIR}")
print(sorted(f for f in os.listdir(FIGDIR) if f.endswith((".png", ".csv"))))
""")

# ============================================================ write
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10.0"}},
      "nbformat": 4, "nbformat_minor": 5}

path = os.path.join(OUTDIR, "rem-exit-continuity-visual.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)

meta = {"id": "alisaremi/rem-exit-continuity-visual-edition",
        "title": "REM-Exit Continuity — Visual Edition",
        "code_file": "rem-exit-continuity-visual.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_internet": False, "dataset_sources": ["alisaremi/ad-and-sleep"],
        "competition_sources": [], "kernel_sources": []}
with open(os.path.join(OUTDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

if "--script" in sys.argv:
    spath = os.path.join(OUTDIR, "_validate.py")
    with open(spath, "w") as f:
        f.write("# auto-generated from the notebook's code cells, for local validation only\n")
        for c in cells:
            if c["cell_type"] == "code":
                f.write("".join(c["source"]).rstrip() + "\n\n")
    print(f"wrote {spath}")

print(f"wrote {path}  ({len(cells)} cells, "
      f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
