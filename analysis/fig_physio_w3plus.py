"""The staging-artefact figure, built as a split-violin comparison by cognitive status.

Enhancements over the first draft of this design:
  - mirrored half-violins sharing a baseline, so the two groups are compared directly rather
    than as two stacked shapes the eye has to align
  - median and interquartile range drawn inside each half
  - a stats column carrying the raw means in their own units, Hedges g with a 95% interval,
    and the p-value: an effect size with an interval says far more than p alone about how
    much difference the data could have hidden
  - panel b reoriented so the model labels read horizontally instead of at 38 degrees, with
    an interval on every model rather than a single continuous band

Writes figures/fig14_physio.{pdf,png} plus a preview copy under figures/variants/.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle

RNG = np.random.default_rng(20261015)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB, FIG = os.path.join(HERE, "tables"), os.path.join(HERE, "figures")
VAR = os.path.join(FIG, "variants")
os.makedirs(VAR, exist_ok=True)

SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f4f3f0"
SLATE, AMBER, SAGE, BRICK = "#374E55", "#DF8F44", "#79AF97", "#B24745"

matplotlib.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9.4,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "axes.labelcolor": INK2,
    "axes.labelsize": 9.0, "axes.titlesize": 9.8, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.grid": False, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8.4,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.solid_capstyle": "round",
})

# ───────────────────────────────────────────────────────────────── data
NIGHTS, _ = pickle.load(open(f"{TAB}/_nights_cache.pkl", "rb"))
NAME = {0: "W", 1: "R", 2: "L", 3: "D"}
TC = [f"n_{NAME[a]}{NAME[b]}" for a in range(4) for b in range(4) if a != b]
n0 = NIGHTS[0]
agg = n0.groupby("participant")[TC].sum()
agg["nights"] = n0.groupby("participant").size()
ex = agg[["n_RL", "n_RW", "n_RD"]].sum(axis=1)
agg["P_RL"] = agg["n_RL"] / ex.replace(0, np.nan)
demo = pd.read_csv(os.path.join(os.path.dirname(HERE), "data/raw/Demographics_v16755408.csv")
                   ).rename(columns={"user_id": "participant"})
ph = pd.read_csv(f"{TAB}/physio_per_participant.csv")
D = (agg.reset_index().merge(demo, on="participant").query("nights>=30 and ace_total==ace_total")
     .merge(ph, on="participant", how="inner").reset_index(drop=True))
D["impaired"] = (D.ace_total <= 82).astype(int)

CH = [("rr_mean", "Respiration rate", "breaths/min", "{:.1f}"),
      ("rr_sd", "Respiration irregularity", "within-night SD", "{:.2f}"),
      ("rr_btw_sd", "Respiration, night-to-night", "SD of nightly means", "{:.2f}"),
      ("sdnn_mean", "SDNN", "ms", "{:.1f}"),
      ("hr_mean", "Heart rate", "bpm", "{:.1f}"),
      ("snore_pct", "Snoring", "% of minutes", "{:.1f}")]
ADJ = pd.read_csv(f"{TAB}/physio_adjusted.csv")
ADJ_LAB = ["Age and sex only", "+ Respiration rate", "+ Respiration irregularity",
           "+ Respiration, night-to-night", "+ SDNN", "+ Heart rate", "+ Snoring",
           "+ all six channels"]


def hedges_g(a, b):
    """g for b - a, with a 95% interval."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (b.mean() - a.mean()) / sp
    g *= 1 - 3 / (4 * (na + nb) - 9)                       # small-sample correction
    se = np.sqrt((na + nb) / (na * nb) + g**2 / (2 * (na + nb)))
    return g, g - 1.96 * se, g + 1.96 * se


def half_violin(ax, vals, row, side, color):
    """A half kernel density on one side of the row baseline, with points, median and IQR."""
    v = np.asarray(vals, float)
    kde = stats.gaussian_kde(v)
    lo, hi = v.min() - .45, v.max() + .45
    xs = np.linspace(lo, hi, 220)
    dens = kde(xs)
    dens = dens / dens.max() * 0.38
    ax.fill_between(xs, row, row + side * dens, color=color, alpha=.26, zorder=2, linewidth=0)
    ax.plot(xs, row + side * dens, color=color, linewidth=1.1, alpha=.65, zorder=3)
    jit = RNG.uniform(.06, .20, len(v)) * side
    ax.scatter(v, row + jit, s=11, color=color, alpha=.85, edgecolor="white",
               linewidth=.45, zorder=5)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [row + side * .028] * 2, color=color, linewidth=3.4,
            alpha=.95, zorder=6, solid_capstyle="butt")
    ax.plot([med, med], [row, row + side * .30], color="white", linewidth=2.6, zorder=7)
    ax.plot([med, med], [row, row + side * .30], color=color, linewidth=1.5, zorder=8)


def build():
    fig = plt.figure(figsize=(7.7, 8.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.00, .62], height_ratios=[1.00, .54],
                          wspace=.10, hspace=.34)

    # ───────────────────────────── panel a: the distributions
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(CH))[::-1]
    g0 = D[D.impaired == 0]
    g1 = D[D.impaired == 1]
    stat_rows = []
    for yy, (k, lab, unit, fmt) in zip(y, CH):
        a = g0[k].dropna().to_numpy()
        b = g1[k].dropna().to_numpy()
        pool = np.concatenate([a, b])
        m, s = pool.mean(), pool.std(ddof=1)
        ax.axhline(yy, color=GRID, linewidth=.9, zorder=1)
        half_violin(ax, (a - m) / s, yy, +1, SLATE)
        half_violin(ax, (b - m) / s, yy, -1, AMBER)
        g, glo, ghi = hedges_g(a, b)
        pv = stats.mannwhitneyu(a, b)[1]
        stat_rows.append(dict(y=yy, mean0=fmt.format(a.mean()), mean1=fmt.format(b.mean()),
                              g=g, glo=glo, ghi=ghi, p=pv))
    ax.axvline(0, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n({u})" for _, l, u, _ in CH], fontsize=7.8)
    ax.set_ylim(-.62, len(CH) - .38)
    ax.set_xlim(-2.75, 4.65)
    ax.set_xlabel("Standardised value  (SD from the cohort mean)")
    ax.set_title("a   The two groups are drawn from the same physiology",
                 loc="left", pad=20)
    ax.legend(handles=[Line2D([], [], color=SLATE, marker="o", linestyle="none",
                              markersize=7, label=f"ACE-III > 82   ($n$ = {len(g0)})"),
                       Line2D([], [], color=AMBER, marker="o", linestyle="none",
                              markersize=7, label=f"ACE-III $\\leq$ 82   ($n$ = {len(g1)})")],
              loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, handletextpad=.4,
              columnspacing=2.2)

    # ───────────────────────────── stats column
    axs = fig.add_subplot(gs[0, 1])
    axs.set_axis_off()
    axs.set_xlim(-.02, 1.02)
    axs.set_ylim(-.62, len(CH) - .38)
    COLS = [(.0, "> 82"), (.155, "$\\leq$ 82"), (.34, "Hedges $g$ (95% CI)"), (1.0, "$p$")]
    hdr_y = len(CH) - .50
    for xx, lab in COLS:
        axs.annotate(lab, xy=(xx, hdr_y), fontsize=8.1, color=MUTED, fontweight="bold",
                     ha="right" if lab == "$p$" else "left")
    axs.annotate("group mean", xy=(.0, hdr_y + .20), fontsize=7.1, color=MUTED,
                 style="italic")
    axs.plot([-.02, 1.0], [hdr_y - .14] * 2, color=GRID, linewidth=.9, clip_on=False)
    for r in stat_rows:
        axs.annotate(r["mean0"], xy=(.0, r["y"]), va="center", fontsize=7.7, color=INK2)
        axs.annotate(r["mean1"], xy=(.155, r["y"]), va="center", fontsize=7.7, color=INK2)
        axs.annotate(f"{r['g']:+.2f}  ({r['glo']:+.2f}, {r['ghi']:+.2f})",
                     xy=(.34, r["y"]), va="center", fontsize=7.4, color=INK2)
        axs.annotate(f"{r['p']:.2f}", xy=(1.0, r["y"]), va="center", ha="right",
                     fontsize=7.7, color=INK2)

    # ───────────────────────────── panel b: the association under adjustment
    ax2 = fig.add_subplot(gs[1, :])
    yb = np.arange(len(ADJ))[::-1]
    p0 = ADJ.iloc[0]
    ax2.axvspan(p0.lo, p0.hi, color=WASH, zorder=1)
    ax2.axvline(p0.beta, color=AXIS, linewidth=1.1, linestyle=(0, (5, 4)), zorder=2)
    ax2.plot(ADJ.beta, yb, color=SLATE, linewidth=1.0, alpha=.5, zorder=3)
    for yy, (i, r) in zip(yb, ADJ.iterrows()):
        col = INK if i == 0 else SLATE
        ax2.plot([r.lo, r.hi], [yy] * 2, color=col, linewidth=1.9, zorder=4)
        ax2.plot([r.beta], [yy], "o", color=col, markersize=8 if i == 0 else 6.8,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax2.annotate(f"{r.beta:+.2f}", xy=(r.hi + .016, yy), va="center", fontsize=8.0,
                     color=INK if i == 0 else INK2,
                     fontweight="bold" if i == 0 else "normal")
    ax2.set_yticks(yb)
    ax2.set_yticklabels(ADJ_LAB, fontsize=8.2)
    ax2.get_yticklabels()[0].set_fontweight("bold")
    ax2.get_yticklabels()[-1].set_fontweight("bold")
    ax2.set_ylim(-.62, len(ADJ) - .38)
    ax2.set_xlim(0, ADJ.hi.max() + .13)
    ax2.set_xticks([0, .2, .4, .6, .8])
    ax2.set_xlabel("Standardised $\\beta$ on ACE-III memory")
    ax2.set_title("b   And adjusting for them changes nothing", loc="left", pad=12)
    ax2.annotate("shaded: 95% CI of the primary estimate", xy=(0, -.22),
                 xycoords="axes fraction", fontsize=7.6, color=MUTED)

    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/fig14_physio.{ext}", bbox_inches="tight", pad_inches=.2,
                    facecolor=SURFACE)
    fig.savefig(f"{VAR}/physio_w3plus.png", bbox_inches="tight", pad_inches=.2,
                facecolor=SURFACE)
    plt.close(fig)
    print("saved fig14_physio (pdf, png) and variants/physio_w3plus.png")


build()
