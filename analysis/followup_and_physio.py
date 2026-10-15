"""Two analyses added for the second review round, both using data already in the public release.

  A. Six-month follow-up cognition. The cited Zenodo version (16755408) carries ACE-III at
     baseline *and* at six months; the copy this project was built on (15877645) carried only
     baseline. Baseline scores are byte-identical between the two versions, so nothing already
     reported changes -- the follow-up is purely additive. This part asks whether P_RL relates to
     the *level* of memory at a second, independent assessment, and whether it relates to *change*.

  B. Physiological confounding of device staging. The reviewer's central unresolved objection is
     that a ballistocardiograph may stage sleep less accurately in people with worse cognition,
     because irregular breathing, autonomic change and movement travel with neurodegeneration --
     and that this cannot be adjusted for. The mattress in fact records respiration rate, SDNN
     and snoring every minute, and those channels were never analysed. This part tests whether
     impaired participants differ on them, whether they relate to P_RL, and whether the primary
     association survives adjustment for them.

Reads the cached transition pass from extra_analyses.py (tables/_nights_cache.pkl); run that
script first if the cache is absent. Run time ~90 s, dominated by reading the physiology CSVs.
"""
import os, glob, pickle, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20261014)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
TABDIR = os.path.join(HERE, "tables")
os.makedirs(FIGDIR, exist_ok=True); os.makedirs(TABDIR, exist_ok=True)

# ---------------------------------------------------------------- house style (JAMA, as fig01-15)
SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA, RED = "#374E55", "#DF8F44", "#79AF97", "#B24745"
matplotlib.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
    "axes.labelsize": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlepad": 22, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}", bbox_inches="tight", pad_inches=.16,
                    facecolor=SURFACE)
    plt.close(fig); print("  saved", name)

# ---------------------------------------------------------------- data layer
# Data root. Searched in order so the same script runs from the paper's project layout,
# from a fresh clone of the public repository (put the Zenodo files in data/raw/), or on
# Kaggle. Override with the REM_DATA environment variable.
ROOTS = [r for r in [os.environ.get("REM_DATA"),
                     os.path.join(HERE, "data/raw"),
                     os.path.join(os.path.dirname(HERE), "data/raw"),
                     "/kaggle/input", "data/raw", "../data/raw", "."] if r]
def find_one(name):
    for root in ROOTS:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits: return sorted(hits, key=len)[0]
    return None

# The versioned file is the one the manuscript cites. Fall back to the plain name so the
# script still runs on a checkout that only has the older release (part A will then be empty).
DEMO_PATH = find_one("Demographics_v16755408.csv") or find_one("Demographics.csv")
SLEEPDIR = os.path.dirname(os.path.dirname(find_one("Sleep_state.csv")))
print("demographics:", os.path.basename(DEMO_PATH))

DEPTH = {"wakeup": 0, "REM": 1, "light": 2, "deep": 3}
NAME = {0: "W", 1: "R", 2: "L", 3: "D"}
TCOLS = [f"n_{NAME[a]}{NAME[b]}" for a in range(4) for b in range(4) if a != b]
ACE = ["ace_total", "ace_attention_subscale", "ace_memory_subscale", "ace_fluency_subscale",
       "ace_language_subscale", "ace_visuospatial_subscale"]
FOLLOW = ["ace_total_6months", "memory_subscale_6months", "attention_subscale_6months",
          "fluency_subscale_6months", "language_subscale_6months", "visuospatial_subscale_6months"]

demo = pd.read_csv(DEMO_PATH).rename(columns={"user_id": "participant"})
# the newer release renamed gds15_total -> gds_total; accept either
if "gds_total" in demo.columns and "gds15_total" not in demo.columns:
    demo["gds15_total"] = demo["gds_total"]
for c in ACE + [c for c in FOLLOW if c in demo.columns] + ["phq_total", "gad_total", "gds15_total"]:
    demo[c] = pd.to_numeric(demo[c], errors="coerce")
demo["age_mid"] = demo["Age group"].map({"[72, 75]": 73.5, "[76, 87]": 81.5, "[88, 99]": 93.5})
demo["male"] = (demo["Sex"] == "Male").astype(float)
HAS_FOLLOW = "memory_subscale_6months" in demo.columns

CACHE = os.path.join(TABDIR, "_nights_cache.pkl")
if not os.path.exists(CACHE):
    raise SystemExit("missing %s -- run scripts/extra_analyses.py first" % CACHE)
NIGHTS, CONV = pickle.load(open(CACHE, "rb"))
print("loaded cached transitions")


def participant_matrix(nights_df):
    agg = nights_df.groupby("participant")[TCOLS].sum()
    agg["nights"] = nights_df.groupby("participant").size()
    for a in range(4):
        cols = [f"n_{NAME[a]}{NAME[b]}" for b in range(4) if b != a]
        tot = agg[cols].sum(axis=1); agg[f"exits_{NAME[a]}"] = tot
        for c in cols: agg["P_" + c[2:]] = agg[c] / tot.replace(0, np.nan)
    return agg.reset_index()


def build(min_nights=30):
    pm = participant_matrix(NIGHTS[0])
    d = pm.merge(demo, on="participant", how="inner").merge(CONV, on="participant", how="left")
    d = d[d.ace_total.notna() & (d.nights >= min_nights)].reset_index(drop=True)
    d["impaired"] = (d.ace_total <= 82).astype(int)
    return d


def assoc(d, y, x, cov=("age_mid", "male")):
    """Standardised OLS coefficient for x on y, adjusted for cov. Returns None if too thin."""
    cov = [c for c in cov if c in d.columns]
    dd = d[[y, x] + cov].dropna()
    if len(dd) < 20 or dd[x].std() == 0: return None
    Z = dd[[x] + cov].apply(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s)
    m = sm.OLS((dd[y] - dd[y].mean()) / dd[y].std(), sm.add_constant(Z)).fit()
    ci = m.conf_int().loc[x]
    return dict(n=len(dd), beta=m.params[x], se=m.bse[x], lo=ci[0], hi=ci[1], p=m.pvalues[x])


DAT = build()
print("analysis sample n=%d" % len(DAT))
P = assoc(DAT, "ace_memory_subscale", "P_RL")
print("baseline primary: beta %+.4f  95%% CI %+.2f to %+.2f  p %.5f  (n=%d)"
      % (P["beta"], P["lo"], P["hi"], P["p"], P["n"]))
OUT = {"baseline_primary": P}

# ================================================================ A. six-month follow-up
print("\n" + "=" * 70 + "\nA. SIX-MONTH FOLLOW-UP COGNITION\n" + "=" * 70)
if not HAS_FOLLOW:
    print("follow-up columns absent from", os.path.basename(DEMO_PATH), "-- skipping part A")
    FU = pd.DataFrame()
else:
    d = DAT.copy()
    d["mem_base"] = d.ace_memory_subscale
    d["mem_6mo"] = d.memory_subscale_6months
    d["mem_chg"] = d.mem_6mo - d.mem_base
    d["tot_base"] = d.ace_total
    d["tot_6mo"] = d.ace_total_6months
    d["tot_chg"] = d.tot_6mo - d.tot_base
    FU = d[d.mem_6mo.notna()].copy()
    print("with 6-month ACE-III: n=%d of %d in the analysis sample" % (len(FU), len(d)))

    dt = (pd.to_datetime(FU.ace_date_6months, errors="coerce", dayfirst=True, utc=True)
          - pd.to_datetime(FU.ace_date, errors="coerce", dayfirst=True, utc=True)).dt.days
    print("assessment interval: median %.0f days (range %.0f-%.0f)"
          % (dt.median(), dt.min(), dt.max()))
    print("memory  baseline %.1f+-%.1f -> 6mo %.1f+-%.1f, change %+.2f+-%.2f, retest r=%.3f"
          % (FU.mem_base.mean(), FU.mem_base.std(), FU.mem_6mo.mean(), FU.mem_6mo.std(),
             FU.mem_chg.mean(), FU.mem_chg.std(), FU.mem_base.corr(FU.mem_6mo)))

    # --- attrition: are the participants who returned different from those who did not?
    print("\nattrition check (returned vs not, within the analysis sample):")
    att = []
    for v, lab in [("P_RL", "P_RL"), ("mem_base", "baseline memory"),
                   ("age_mid", "age"), ("nights", "nights")]:
        a = d.loc[d.mem_6mo.notna(), v].dropna(); b = d.loc[d.mem_6mo.isna(), v].dropna()
        if len(b) < 2: continue
        u, p = stats.mannwhitneyu(a, b)
        print("  %-16s returned %.3f   lost %.3f   p=%.3f" % (lab, a.mean(), b.mean(), p))
        att.append(dict(variable=lab, returned=a.mean(), lost=b.mean(), p=p, n_lost=len(b)))
    pd.DataFrame(att).to_csv(f"{TABDIR}/followup_attrition.csv", index=False)

    # --- three questions, in order of what they can support
    rows = []
    # 1. cross-sectional at the second assessment: an independent measurement of the outcome
    for y, lab in [("mem_6mo", "Memory at 6 months"), ("tot_6mo", "ACE-III total at 6 months")]:
        r = assoc(FU, y, "P_RL")
        if r: rows.append(dict(model="cross-sectional, 6-month assessment", outcome=lab, **r))
    # 2. ANCOVA: 6-month level adjusted for baseline level = residualised change (preferred)
    for y, base, lab in [("mem_6mo", "mem_base", "Memory at 6 months"),
                         ("tot_6mo", "tot_base", "ACE-III total at 6 months")]:
        r = assoc(FU, y, "P_RL", cov=("age_mid", "male", base))
        if r: rows.append(dict(model="ANCOVA, adjusted for baseline", outcome=lab, **r))
    # 3. raw change score
    for y, lab in [("mem_chg", "Change in memory"), ("tot_chg", "Change in ACE-III total")]:
        r = assoc(FU, y, "P_RL")
        if r: rows.append(dict(model="raw change score", outcome=lab, **r))
    fu = pd.DataFrame(rows)
    fu.to_csv(f"{TABDIR}/followup_models.csv", index=False)
    print("\n%-38s %-28s %4s %8s %18s %9s" % ("MODEL", "OUTCOME", "n", "beta", "95% CI", "p"))
    for _, r in fu.iterrows():
        print("%-38s %-28s %4d %+8.3f   [%+.2f, %+.2f] %9.4f"
              % (r.model, r.outcome, r.n, r.beta, r.lo, r.hi, r.p))
    OUT["followup"] = fu.to_dict("records")

    # --- what change could we have detected? minimum detectable effect at 80% power
    n = len(FU)
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) / np.sqrt(n - 4)
    print("\nminimum detectable standardised beta at 80%% power, n=%d: +-%.2f" % (n, mde))
    OUT["followup_mde"] = float(mde)

# ================================================================ B. physiological confounding
print("\n" + "=" * 70 + "\nB. PHYSIOLOGICAL CONFOUNDING OF DEVICE STAGING\n" + "=" * 70)
MIN_PHYSIO_NIGHTS = 10


def physio(pid, keep_nights):
    """Per-participant nocturnal physiology, over the same nights the transitions came from.

    rr_sd is the within-night SD of respiration rate averaged across nights: the breathing
    irregularity the reviewer names. snore_pct is the share of recorded minutes with snoring.
    """
    f = os.path.join(SLEEPDIR, pid, "Sleep_physio.csv")
    if not os.path.exists(f): return None
    p = pd.read_csv(f, parse_dates=["Timestamp"])
    p["night"] = (p.Timestamp - pd.Timedelta(hours=12)).dt.date.astype(str)
    p = p[p.night.isin(keep_nights)]
    if p.empty: return None
    g = p.groupby("night").agg(rr_mean=("Respiration Rate", "mean"),
                               rr_sd=("Respiration Rate", "std"),
                               sdnn=("SDNN_1", "mean"),
                               hr=("Heart Rate", "mean"),
                               snore=("Snoring", lambda s: 100 * (s > 0).mean()),
                               nmin=("Heart Rate", "size"))
    g = g[g.nmin >= 60]                       # at least an hour of physiology that night
    if len(g) < MIN_PHYSIO_NIGHTS: return None
    return dict(participant=pid, physio_nights=len(g),
                rr_mean=g.rr_mean.mean(), rr_sd=g.rr_sd.mean(), rr_btw_sd=g.rr_mean.std(),
                sdnn_mean=g.sdnn.mean(), hr_mean=g.hr.mean(), snore_pct=g.snore.mean())


print("computing nocturnal physiology over the analysis nights ...")
nn = NIGHTS[0]
recs = [physio(pid, set(nn[nn.participant == pid].night)) for pid in DAT.participant]
PH = pd.DataFrame([r for r in recs if r])
print("physiology available for %d of %d participants (>=%d nights)"
      % (len(PH), len(DAT), MIN_PHYSIO_NIGHTS))
PH.to_csv(f"{TABDIR}/physio_per_participant.csv", index=False)
D2 = DAT.merge(PH, on="participant", how="left")

PHVARS = [("rr_mean", "Respiration rate, mean (breaths/min)"),
          ("rr_sd", "Respiration irregularity (within-night SD)"),
          ("rr_btw_sd", "Respiration rate, night-to-night SD"),
          ("sdnn_mean", "SDNN, mean (ms)"),
          ("hr_mean", "Heart rate, mean (bpm)"),
          ("snore_pct", "Snoring (% of recorded minutes)")]

# --- B1. do impaired participants differ on these channels at all?
print("\nB1. physiology by cognitive status (the premise of the artefact account)")
rows = []
for v, lab in PHVARS:
    a = D2.loc[D2.impaired == 0, v].dropna(); b = D2.loc[D2.impaired == 1, v].dropna()
    if len(a) < 5 or len(b) < 5: continue
    u, p = stats.mannwhitneyu(a, b)
    rows.append(dict(variable=lab, unimpaired=a.mean(), unimpaired_sd=a.std(),
                     impaired=b.mean(), impaired_sd=b.std(), p=p,
                     n_unimpaired=len(a), n_impaired=len(b)))
    print("  %-44s %6.2f vs %6.2f   p=%.3f" % (lab, a.mean(), b.mean(), p))
b1 = pd.DataFrame(rows); b1.to_csv(f"{TABDIR}/physio_by_status.csv", index=False)
OUT["physio_by_status"] = b1.to_dict("records")

# --- B2. do they relate to P_RL, and to memory?
print("\nB2. association of each channel with P_RL and with memory")
rows = []
for v, lab in PHVARS:
    dd = D2[[v, "P_RL", "ace_memory_subscale"]].dropna()
    if len(dd) < 20: continue
    r1, p1 = stats.spearmanr(dd[v], dd.P_RL)
    r2, p2 = stats.spearmanr(dd[v], dd.ace_memory_subscale)
    rows.append(dict(variable=lab, n=len(dd), rho_PRL=r1, p_PRL=p1,
                     rho_memory=r2, p_memory=p2))
    print("  %-44s rho(P_RL)=%+.3f p=%.3f | rho(memory)=%+.3f p=%.3f"
          % (lab, r1, p1, r2, p2))
b2 = pd.DataFrame(rows); b2.to_csv(f"{TABDIR}/physio_correlations.csv", index=False)
OUT["physio_correlations"] = b2.to_dict("records")

# --- B3. does the primary association survive adjustment?
print("\nB3. primary association adjusted for each channel, then for all of them")
rows = [dict(model="Age and sex only (primary)", **assoc(D2, "ace_memory_subscale", "P_RL"))]
print("  %-46s n=%2d beta %+.3f [%+.2f, %+.2f] p=%.4f"
      % (rows[0]["model"], rows[0]["n"], rows[0]["beta"], rows[0]["lo"], rows[0]["hi"],
         rows[0]["p"]))
for v, lab in PHVARS:
    r = assoc(D2, "ace_memory_subscale", "P_RL", cov=("age_mid", "male", v))
    if not r: continue
    rows.append(dict(model="+ " + lab, **r))
    print("  %-46s n=%2d beta %+.3f [%+.2f, %+.2f] p=%.4f"
          % ("+ " + lab, r["n"], r["beta"], r["lo"], r["hi"], r["p"]))
allv = tuple(["age_mid", "male"] + [v for v, _ in PHVARS])
r = assoc(D2, "ace_memory_subscale", "P_RL", cov=allv)
if r:
    rows.append(dict(model="+ all six physiology channels", **r))
    print("  %-46s n=%2d beta %+.3f [%+.2f, %+.2f] p=%.4f"
          % ("+ all six physiology channels", r["n"], r["beta"], r["lo"], r["hi"], r["p"]))
b3 = pd.DataFrame(rows); b3.to_csv(f"{TABDIR}/physio_adjusted.csv", index=False)
OUT["physio_adjusted"] = b3.to_dict("records")

json.dump(OUT, open(f"{TABDIR}/followup_physio_results.json", "w"), indent=1, default=float)

# ================================================================ figures
print("\nfigures ...")

# ---- fig16: six-month follow-up
if HAS_FOLLOW and len(FU) >= 20:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    fig.subplots_adjust(wspace=.42)

    ax = axes[0]
    ax.scatter(FU.P_RL, FU.mem_6mo, s=46, facecolor=BLUE, edgecolor="white", linewidth=.9,
               zorder=3)
    s, i, rr, pp, _ = stats.linregress(FU.P_RL, FU.mem_6mo)
    xs = np.linspace(FU.P_RL.min(), FU.P_RL.max(), 50)
    ax.plot(xs, i + s * xs, color=INK, linewidth=1.6, zorder=4)
    ax.set_xlabel("REM-exit continuity index, $P_{R\\rightarrow L}$")
    ax.set_ylabel("ACE-III memory at 6 months")
    ax.set_title("a   Second, independent assessment", loc="left")
    ax.annotate("unadjusted $r$ = %.2f, $p$ = %.3f\n($n$ = %d)" % (rr, pp, len(FU)),
                xy=(.04, .95), xycoords="axes fraction", va="top", fontsize=8.5, color=INK2)

    ax = axes[1]
    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax.scatter(FU.P_RL, FU.mem_chg, s=46, facecolor=ORANGE, edgecolor="white", linewidth=.9,
               zorder=3)
    s, i, rr, pp, _ = stats.linregress(FU.P_RL, FU.mem_chg)
    ax.plot(xs, i + s * xs, color=INK, linewidth=1.6, zorder=4)
    ax.set_xlabel("REM-exit continuity index, $P_{R\\rightarrow L}$")
    ax.set_ylabel("Change in ACE-III memory over 6 months")
    ax.set_title("b   No relation to six-month change", loc="left")
    ax.annotate("unadjusted $r$ = %.2f, $p$ = %.3f" % (rr, pp),
                xy=(.04, .95), xycoords="axes fraction", va="top", fontsize=8.5, color=INK2)

    ax = axes[2]
    sub = fu[fu.outcome.str.contains("[Mm]emory")].reset_index(drop=True)
    y = np.arange(len(sub))[::-1]
    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
    for yy, (_, r) in zip(y, sub.iterrows()):
        c = BLUE if r.p < .05 else MUTED
        ax.plot([r.lo, r.hi], [yy, yy], color=c, linewidth=2.4, solid_capstyle="round", zorder=3)
        ax.plot([r.beta], [yy], "o", color=c, markersize=8, markeredgecolor="white",
                markeredgewidth=1.1, zorder=4)
        ax.annotate("%+.2f  (p = %.3f)" % (r.beta, r.p), xy=(r.hi + .06, yy), va="center",
                    fontsize=8.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([m.replace(", ", ",\n") for m in sub.model], fontsize=8.5)
    ax.set_xlabel("Standardised $\\beta$ for $P_{R\\rightarrow L}$")
    ax.set_title("c   What the follow-up supports", loc="left")
    ax.set_xlim(min(-.55, sub.lo.min() - .1), max(sub.hi.max() + .95, 1.3))
    ax.set_xticks([-0.5, 0, 0.5, 1.0])          # sparse: the default ticks ran into each other
    ax.set_ylim(-.6, len(sub) - .4)
    ax.grid(axis="y", visible=False)
    save(fig, "fig09_followup")

# ---- fig17: physiological confounds
if len(PH) >= 20:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                             gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.subplots_adjust(wspace=.5)

    ax = axes[0]
    sub = b2.copy().reset_index(drop=True)
    y = np.arange(len(sub))[::-1]
    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
    h = .17
    for yy, (_, r) in zip(y, sub.iterrows()):
        ax.barh(yy + h, r.rho_PRL, height=.3, color=BLUE, zorder=3)
        ax.barh(yy - h, r.rho_memory, height=.3, color=AQUA, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([v.replace(" (", "\n(") for v in sub.variable], fontsize=8.5)
    ax.set_xlabel("Spearman $\\rho$")
    ax.set_title("a   Do the physiology channels track either variable?", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-.55, .72)                      # headroom on the right for the key
    ax.set_ylim(-.62, len(sub) - .38)
    # a real legend, inside the empty right-hand region: the earlier annotation at the
    # axes origin overdrew the bottom tick label and the x-axis numbers
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, label="with $P_{R\\rightarrow L}$"),
                       Patch(facecolor=AQUA, label="with ACE-III memory")],
              loc="upper right", fontsize=8.5, handlelength=1.1, handleheight=1.0,
              borderpad=.5, labelspacing=.5)

    ax = axes[1]
    sub = b3.reset_index(drop=True)
    y = np.arange(len(sub))[::-1]
    ax.axvspan(sub.lo.iloc[0], sub.hi.iloc[0], color=WASH, zorder=1)
    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax.axvline(sub.beta.iloc[0], color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    for yy, (_, r) in zip(y, sub.iterrows()):
        c = INK if yy == y[0] else (BLUE if r.p < .05 else RED)
        ax.plot([r.lo, r.hi], [yy, yy], color=c, linewidth=2.4, solid_capstyle="round", zorder=3)
        ax.plot([r.beta], [yy], "o", color=c, markersize=7.5, markeredgecolor="white",
                markeredgewidth=1.1, zorder=4)
        ax.annotate("%+.2f" % r.beta, xy=(r.hi + .035, yy), va="center", fontsize=8.5,
                    color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace(" (", "\n(") for m in sub.model], fontsize=8.5)
    ax.set_xlabel("Standardised $\\beta$ for $P_{R\\rightarrow L}$ on ACE-III memory")
    ax.set_title("b   The association under physiological adjustment", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, sub.lo.min() - .06), sub.hi.max() + .22)
    save(fig, "fig14_physio")

print("\ndone. tables ->", TABDIR)
