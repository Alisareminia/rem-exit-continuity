"""Analyses added for paper 4, all answering reviewer concerns within the fixed dataset.

  1. Selection ("winner's curse") analysis  - how much of beta is selection optimism,
     and how stable the choice of P_RL is under resampling.
  2. Specification curve                    - every defensible analytic path, not one.
  3. Disjoint-night-half association        - does the association hold when the exposure
                                              is estimated from either half of the nights?
  4. Regression diagnostics                 - influence, leverage, heteroscedasticity, VIF.

Nothing here needs data we do not have. Run time ~6 min (the min-bout variants dominate).
"""
import os, glob, warnings, itertools, json, pickle, sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260913)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
TABDIR = os.path.join(HERE, "tables")
os.makedirs(FIGDIR, exist_ok=True); os.makedirs(TABDIR, exist_ok=True)

# ---------------------------------------------------------------- house style
SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
matplotlib.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
    "axes.labelsize": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlepad": 26, "axes.grid": True,
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
ROOTS = [os.path.join(os.path.dirname(HERE), "data/raw")]
def find_one(name):
    for root in ROOTS:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits: return sorted(hits, key=len)[0]
DEMO_PATH = find_one("Demographics.csv")
SLEEPDIR = os.path.dirname(os.path.dirname(find_one("Sleep_state.csv")))
PIDS = sorted(d for d in os.listdir(SLEEPDIR) if os.path.isdir(os.path.join(SLEEPDIR, d)))

DEPTH = {"wakeup": 0, "REM": 1, "light": 2, "deep": 3}
NAME = {0: "W", 1: "R", 2: "L", 3: "D"}
LONG = {"W": "Wake", "R": "REM", "L": "Light NREM", "D": "Deep NREM"}
MIN_TST = 120
TCOLS = [f"n_{NAME[a]}{NAME[b]}" for a in range(4) for b in range(4) if a != b]

ACE = ["ace_total", "ace_attention_subscale", "ace_memory_subscale", "ace_fluency_subscale",
       "ace_language_subscale", "ace_visuospatial_subscale"]
demo = pd.read_csv(DEMO_PATH).rename(columns={"user_id": "participant"})
for c in ACE + ["phq_total", "gad_total", "gds15_total"]:
    demo[c] = pd.to_numeric(demo[c], errors="coerce")
demo["age_mid"] = demo["Age group"].map({"[72, 75]": 73.5, "[76, 87]": 81.5, "[88, 99]": 93.5})
demo["male"] = (demo["Sex"] == "Male").astype(float)
demo["htn"] = (demo["Essential hypertension"].astype(str).str.upper() == "TRUE").astype(float)

def read_stages(pid):
    f = os.path.join(SLEEPDIR, pid, "Sleep_state.csv")
    df = pd.read_csv(f, parse_dates=["Start time", "End time"]).rename(
        columns={"Start time": "start", "End time": "end", "Sleep state": "stage"}).dropna(
        subset=["start", "end", "stage"])
    df["dur"] = (df.end - df.start).dt.total_seconds() / 60
    df = df[(df.dur > 0) & (df.dur < 600) & df.stage.isin(DEPTH)]
    if df.empty: return df
    df["night"] = (df.start - pd.Timedelta(hours=12)).dt.date
    return df.sort_values("start")

def night_transitions(pid, min_bout=0.0):
    df = read_stages(pid)
    if df.empty: return pd.DataFrame()
    rows = []
    for night, g in df.groupby("night"):
        g = g.sort_values("start")
        if g.loc[g.stage != "wakeup", "dur"].sum() < MIN_TST: continue
        v, dur = g.stage.map(DEPTH).values, g.dur.values
        chg = np.flatnonzero(np.diff(v) != 0)
        st, en = np.concatenate([[0], chg + 1]), np.concatenate([chg + 1, [len(v)]])
        stages = v[st]; bdur = np.array([dur[a:b].sum() for a, b in zip(st, en)])
        if min_bout > 0:
            keep = bdur >= min_bout
            stages = stages[keep]
            if len(stages) > 1:
                stages = stages[np.concatenate([[0], np.flatnonzero(np.diff(stages) != 0) + 1])]
        if len(stages) < 4: continue
        rec = {"participant": pid, "night": str(night),
               "tst": g.loc[g.stage != "wakeup", "dur"].sum()}
        a_, b_ = stages[:-1], stages[1:]
        for a in range(4):
            for b in range(4):
                if a != b: rec[f"n_{NAME[a]}{NAME[b]}"] = int(((a_ == a) & (b_ == b)).sum())
        rows.append(rec)
    return pd.DataFrame(rows)

def participant_matrix(nights_df):
    agg = nights_df.groupby("participant")[TCOLS].sum()
    agg["nights"] = nights_df.groupby("participant").size()
    agg["tst_mean"] = nights_df.groupby("participant")["tst"].mean()
    for a in range(4):
        cols = [f"n_{NAME[a]}{NAME[b]}" for b in range(4) if b != a]
        tot = agg[cols].sum(axis=1); agg[f"exits_{NAME[a]}"] = tot
        for c in cols: agg["P_" + c[2:]] = agg[c] / tot.replace(0, np.nan)
    return agg.reset_index()

def conventional(pid):
    df = read_stages(pid); out = []
    for night, g in df.groupby("night"):
        s = g.groupby("stage").dur.sum()
        tst = s.get("deep", 0) + s.get("light", 0) + s.get("REM", 0)
        if tst < MIN_TST: continue
        spt = (g.end.max() - g.start.min()).total_seconds() / 60
        out.append(dict(tst=tst, deep_pct=100*s.get("deep",0)/tst, rem_pct=100*s.get("REM",0)/tst,
                        light_pct=100*s.get("light",0)/tst, waso=s.get("wakeup",0),
                        sleep_eff=100*tst/spt if spt > 0 else np.nan,
                        awak_per_hr=(g.stage == "wakeup").sum()/(tst/60)))
    o = pd.DataFrame(out)
    return pd.Series({**o.mean().add_suffix("_mean"), "participant": pid})

# The transition pass is the only slow step (~6 min). Cache it so that redrawing a
# figure does not mean recomputing the analysis; pass --reuse to load it.
CACHE = os.path.join(TABDIR, "_nights_cache.pkl")
if "--reuse" in sys.argv and os.path.exists(CACHE):
    NIGHTS, CONV = pickle.load(open(CACHE, "rb"))
    print("loaded cached transitions")
else:
    print("computing transitions for each min-bout variant ...")
    NIGHTS = {}
    for mb in (0, 3, 5, 10):
        NIGHTS[mb] = pd.concat([night_transitions(p, mb) for p in PIDS], ignore_index=True)
        print(f"  min_bout={mb:>2}: {len(NIGHTS[mb]):,} nights")
    CONV = pd.DataFrame([conventional(p) for p in PIDS])
    pickle.dump((NIGHTS, CONV), open(CACHE, "wb"))

def build(min_bout=0, min_nights=30, pooling="pooled"):
    nts = NIGHTS[min_bout]
    if pooling == "pooled":
        pm = participant_matrix(nts)
    else:                                   # mean of nightly proportions
        n = nts.copy()
        n["ex"] = n[["n_RL", "n_RW", "n_RD"]].sum(axis=1)
        n = n[n.ex >= 1]
        pm = participant_matrix(nts)
        pm = pm.drop(columns=["P_RL"]).merge(
            (n.n_RL / n.ex).groupby(n.participant).mean().rename("P_RL").reset_index(),
            on="participant", how="left")
    d = pm.merge(demo, on="participant", how="inner").merge(CONV, on="participant", how="left")
    d = d[d.ace_total.notna() & (d.nights >= min_nights)].reset_index(drop=True)
    d["impaired"] = (d.ace_total <= 82).astype(int)
    return d

def assoc(d, y, x, cov=("age_mid", "male")):
    cov = [c for c in cov if c in d.columns]
    dd = d[[y, x] + cov].dropna()
    if len(dd) < 20 or dd[x].std() == 0: return None
    Z = dd[[x] + cov].apply(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s)
    m = sm.OLS((dd[y] - dd[y].mean()) / dd[y].std(), sm.add_constant(Z)).fit()
    ci = m.conf_int().loc[x]
    return dict(n=len(dd), beta=m.params[x], se=m.bse[x], lo=ci[0], hi=ci[1], p=m.pvalues[x])

DAT = build()
print(f"primary sample n={len(DAT)}")
PRIMARY = assoc(DAT, "ace_memory_subscale", "P_RL")
print("primary beta %.4f  p %.5f" % (PRIMARY["beta"], PRIMARY["p"]))
RESULTS = {"primary": PRIMARY}
PAIRS = [(a, b) for a in range(4) for b in range(4) if a != b]
TR = [f"P_{NAME[a]}{NAME[b]}" for a, b in PAIRS]
TRLAB = {f"P_{NAME[a]}{NAME[b]}": f"{LONG[NAME[a]]} → {LONG[NAME[b]]}" for a, b in PAIRS}

# ============================================================ 1. selection analysis
# P_RL was chosen because it was the strongest of twelve transitions. A coefficient
# selected that way is biased upward. Bootstrap the *whole* selection procedure to
# estimate how much, and how often the same transition wins.
print("\nselection / winner's-curse analysis ...")
B_SEL = 2000
orig = {t: assoc(DAT, "ace_memory_subscale", t) for t in TR}
orig_beta = {t: (r["beta"] if r else np.nan) for t, r in orig.items()}
winner = min([t for t in TR if orig[t]], key=lambda t: orig[t]["p"])
print("  winner in the real data:", TRLAB[winner])

idx = np.arange(len(DAT))
sel_counts, optimism, boot_prl = {t: 0 for t in TR}, [], []
for b in range(B_SEL):
    d = DAT.iloc[RNG.choice(idx, len(idx), replace=True)]
    best, best_p, best_beta = None, np.inf, np.nan
    for t in TR:
        r = assoc(d, "ace_memory_subscale", t)
        if r and r["p"] < best_p:
            best, best_p, best_beta = t, r["p"], r["beta"]
    if best is None: continue
    sel_counts[best] += 1
    optimism.append(best_beta - orig_beta[best])      # inflation attributable to selection
    r_prl = assoc(d, "ace_memory_subscale", "P_RL")
    if r_prl: boot_prl.append(r_prl["beta"])
optimism = np.array(optimism); boot_prl = np.array(boot_prl)
bias = float(np.mean(optimism))
beta_corr = PRIMARY["beta"] - bias
stability = sel_counts[winner] / max(sum(sel_counts.values()), 1)
RESULTS["selection"] = dict(winner=TRLAB[winner], bias=bias, beta_naive=PRIMARY["beta"],
                            beta_corrected=beta_corr, stability=stability,
                            boot_prl_lo=float(np.percentile(boot_prl, 2.5)),
                            boot_prl_hi=float(np.percentile(boot_prl, 97.5)))
print(f"  selection optimism {bias:+.3f} -> corrected beta {beta_corr:+.3f} "
      f"(naive {PRIMARY['beta']:+.3f}); P_RL re-selected in {100*stability:.0f}% of resamples")
pd.DataFrame([{"transition": TRLAB[t], "selected_pct": 100*sel_counts[t]/B_SEL,
               "beta_full_sample": orig_beta[t]} for t in TR]
             ).sort_values("selected_pct", ascending=False).to_csv(
    f"{TABDIR}/selection_stability.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.3),
                         gridspec_kw={"width_ratios": [1, 1.15], "wspace": .46})
ax = axes[0]
ax.hist(boot_prl, bins=38, color=BLUE, alpha=.85, edgecolor=SURFACE, linewidth=.6)
# the two estimates are nearly identical, so the labels go in a stack rather than
# beside their lines, where they would overlap
ax.axvline(PRIMARY["beta"], color=INK, lw=1.8, zorder=5)
ax.axvline(beta_corr, color=RED, lw=1.8, ls=(0, (4, 2)), zorder=6)
for i, (lab, c) in enumerate([(f"naive  β = {PRIMARY['beta']:.2f}", INK),
                              (f"selection-adjusted  β = {beta_corr:.2f}", RED)]):
    ax.text(.03, .955 - i*.085, lab, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.2, color=c, fontweight="bold")
ax.axvline(0, color=AXIS, lw=1.2)
ax.set_xlabel("bootstrap β for P(REM → light NREM), memory subscale")
ax.set_ylabel("resamples"); ax.set_yticks([])
ax.set_title("a · What selection buys", loc="left")
ax.text(0, 1.02, f"2,000 resamples · the two differ by "
                 f"{abs(beta_corr-PRIMARY['beta']):.3f}",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

ax = axes[1]
sel = sorted(((100*sel_counts[t]/B_SEL, TRLAB[t]) for t in TR), reverse=True)[:8][::-1]
ypos = np.arange(len(sel))
for i, (pct, lab) in enumerate(sel):
    col = BLUE if lab == TRLAB[winner] else MUTED
    ax.barh(i, pct, height=.62, color=col, edgecolor="none")
    ax.text(pct + 1.2, i, f"{pct:.0f}%", va="center", fontsize=8.8,
            color=INK if lab == TRLAB[winner] else MUTED,
            fontweight="bold" if lab == TRLAB[winner] else "normal")
ax.set_yticks(ypos, [l for _, l in sel], fontsize=8.8)
ax.set_xlim(0, max(p for p, _ in sel) * 1.22)
ax.grid(axis="y", visible=False)
ax.set_xlabel("share of bootstrap resamples in which this transition is the strongest")
ax.set_title("b · How stable the choice is", loc="left")
ax.text(0, 1.02, "an arbitrary winner would scatter across routes",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
save(fig, "fig13_selection")

# ============================================================ 2. specification curve
print("\nspecification curve ...")
COVSETS = {
    "age + sex": ["age_mid", "male"],
    "+ mood": ["age_mid", "male", "phq_total", "gds15_total", "gad_total"],
    "+ sleep metrics": ["age_mid", "male", "tst_mean", "rem_pct_mean", "deep_pct_mean",
                        "light_pct_mean", "sleep_eff_mean", "waso_mean", "awak_per_hr_mean"],
    "+ everything": ["age_mid", "male", "phq_total", "gds15_total", "gad_total", "htn",
                     "tst_mean", "rem_pct_mean", "deep_pct_mean", "light_pct_mean",
                     "sleep_eff_mean", "waso_mean", "awak_per_hr_mean", "nights"],
}
specs = []
for mb in (0, 3, 5, 10):
    for pooling in ("pooled", "nightly mean"):
        for mn in (14, 30, 60, 90, 120):
            d = build(mb, mn, pooling)
            for outcome, olab in (("ace_memory_subscale", "memory"), ("ace_total", "total")):
                for cname, cov in COVSETS.items():
                    r = assoc(d, outcome, "P_RL", cov)
                    if r:
                        specs.append({**r, "min_bout": mb, "min_nights": mn, "pooling": pooling,
                                      "outcome": olab, "covariates": cname})
SPEC = pd.DataFrame(specs).sort_values("beta").reset_index(drop=True)
SPEC.to_csv(f"{TABDIR}/specification_curve.csv", index=False)
RESULTS["spec"] = dict(n=len(SPEC), pos=int((SPEC.beta > 0).sum()),
                       sig=int((SPEC.p < .05).sum()), med=float(SPEC.beta.median()),
                       lo=float(SPEC.beta.min()), hi=float(SPEC.beta.max()),
                       med_mem=float(SPEC[SPEC.outcome == "memory"].beta.median()),
                       sig_mem=float((SPEC[SPEC.outcome == "memory"].p < .05).mean()))
print(f"  {len(SPEC)} specifications | {100*(SPEC.beta>0).mean():.0f}% positive | "
      f"{100*(SPEC.p<.05).mean():.0f}% significant | median β {SPEC.beta.median():+.3f}")

CHOICES = [("outcome", ["memory", "total"]), ("covariates", list(COVSETS)),
           ("pooling", ["pooled", "nightly mean"]), ("min_bout", [0, 3, 5, 10]),
           ("min_nights", [14, 30, 60, 90, 120])]
rows = sum(len(v) for _, v in CHOICES)
fig = plt.figure(figsize=(13.4, 7.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.25], hspace=.11)
ax = fig.add_subplot(gs[0])
x = np.arange(len(SPEC))
sig = SPEC.p < .05
ax.fill_between(x, SPEC.lo, SPEC.hi, color=BLUE, alpha=.13, lw=0)
ax.scatter(x[~sig.values], SPEC.beta[~sig.values], s=9, color=MUTED, alpha=.7, zorder=3)
ax.scatter(x[sig.values], SPEC.beta[sig.values], s=9, color=BLUE, zorder=4)
ax.axhline(0, color=AXIS, lw=1.2)
ax.axhline(PRIMARY["beta"], color=INK, lw=1.3, zorder=5)
# left end: the curve rises into the right side, so a label there gets overdrawn
ax.text(len(SPEC)*.012, PRIMARY["beta"]+.035, f"reported specification  β = {PRIMARY['beta']:.2f}",
        ha="left", va="bottom", fontsize=8.8, color=INK, zorder=7,
        bbox=dict(boxstyle="round,pad=.25", fc=SURFACE, ec="none"))
ax.set_xlim(-3, len(SPEC)+3); ax.set_xticks([])
ax.set_ylabel("standardised β per SD of P(REM → light NREM)")
ax.set_title("a · Every defensible analytic path", loc="left")
ax.text(0, 1.02, f"{len(SPEC)} specifications, ordered by effect size · filled = p < 0.05 · "
                 f"band is each model's 95% CI", transform=ax.transAxes, fontsize=8.8,
        color=MUTED, va="bottom")
ax2 = fig.add_subplot(gs[1], sharex=ax)
ypos, ylabels = {}, []
r = 0
for cname, vals in CHOICES:
    for v in vals:
        ypos[(cname, v)] = r; ylabels.append(f"{v}"); r += 1
r = 0                                   # alternating bands tie values to their group
for gi, (cname, vals) in enumerate(CHOICES):
    if gi % 2 == 0:
        ax2.axhspan(r - .5, r + len(vals) - .5, color="#f4f4f2", zorder=0)
    r += len(vals)
for i, row in SPEC.iterrows():
    for cname, _ in CHOICES:
        ax2.plot([i], [ypos[(cname, row[cname])]], marker="s", ms=1.7,
                 color=BLUE if row.p < .05 else MUTED, alpha=.85, zorder=3)
r = 0
for cname, vals in CHOICES:
    ax2.axhline(r - .5, color=AXIS, lw=1.1, zorder=2)
    ax2.text(-0.138, r + (len(vals)-1)/2, cname.replace("_", " "),
             transform=ax2.get_yaxis_transform(), ha="right", va="center",
             fontsize=8.5, color=INK, fontweight="bold")
    # a bracket from the group name to the rows it covers
    ax2.plot([-0.131, -0.131], [r - .35, r + len(vals) - .65],
             transform=ax2.get_yaxis_transform(), color=AXIS, lw=1.2,
             clip_on=False, solid_capstyle="butt")
    r += len(vals)
ax2.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
ax2.set_ylim(-.7, len(ylabels) - .3); ax2.invert_yaxis()
ax2.grid(False); ax2.set_xticks([])
ax2.set_xlabel("specifications, ordered as above")
ax2.text(0, 1.03, "b · the analytic choices behind each specification", transform=ax2.transAxes,
         fontsize=9.5, color=INK, fontweight="bold", va="bottom")
for s in ("top", "right", "bottom"): ax2.spines[s].set_visible(False)
save(fig, "fig14_specification")

# ============================================================ 3. disjoint night halves
# The exposure is estimated twice from non-overlapping nights. This does not make the
# outcome independent, so it tests measurement robustness, not discovery bias.
print("\ndisjoint night halves ...")
nn = NIGHTS[0][NIGHTS[0].participant.isin(DAT.participant)].copy()
nn["k"] = nn.groupby("participant").cumcount()
nn["frac"] = nn.groupby("participant")["k"].transform(lambda s: s / max(s.max(), 1))
halves = {}
for lab, mask in [("odd nights", nn.k % 2 == 1), ("even nights", nn.k % 2 == 0),
                  ("first half of follow-up", nn.frac <= .5),
                  ("second half of follow-up", nn.frac > .5)]:
    g = nn[mask].groupby("participant")[["n_RL", "n_RW", "n_RD"]].sum()
    g["P_RL"] = g.n_RL / g[["n_RL", "n_RW", "n_RD"]].sum(axis=1)
    d = DAT.drop(columns=["P_RL"]).merge(g[["P_RL"]].reset_index(), on="participant")
    halves[lab] = assoc(d, "ace_memory_subscale", "P_RL")
    print(f"  {lab:<26} beta {halves[lab]['beta']:+.3f}  p {halves[lab]['p']:.4f}  "
          f"n {halves[lab]['n']}")
RESULTS["halves"] = halves
pd.DataFrame([{"subset": k, **v} for k, v in halves.items()]).to_csv(
    f"{TABDIR}/night_halves.csv", index=False)

# ============================================================ 4. diagnostics
print("\nregression diagnostics ...")
d = DAT[["ace_memory_subscale", "P_RL", "age_mid", "male"]].dropna()
Z = d[["P_RL", "age_mid", "male"]].apply(lambda s: (s - s.mean()) / s.std())
X = sm.add_constant(Z)
yz = (d.ace_memory_subscale - d.ace_memory_subscale.mean()) / d.ace_memory_subscale.std()
m = sm.OLS(yz, X).fit()
inf = OLSInfluence(m)
cooks = inf.cooks_distance[0]; hat = inf.hat_matrix_diag
stud = inf.resid_studentized_external
dfb = inf.dfbetas[:, list(X.columns).index("P_RL")]
bp = het_breuschpagan(m.resid, X)
sw = stats.shapiro(m.resid)
vif = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}
cook_thr = 4 / len(d)
RESULTS["diag"] = dict(n=len(d), cook_max=float(cooks.max()), cook_thr=float(cook_thr),
                       n_cook=int((cooks > cook_thr).sum()),
                       hat_mean=float(hat.mean()), n_hat=int((hat > 2*hat.mean()).sum()),
                       dfbeta_max=float(np.abs(dfb).max()), dfbeta_thr=float(2/np.sqrt(len(d))),
                       n_dfbeta=int((np.abs(dfb) > 2/np.sqrt(len(d))).sum()),
                       bp_p=float(bp[1]), sw_p=float(sw.pvalue),
                       vif_max=float(max(vif.values())), vif=vif)
print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in RESULTS["diag"].items() if k != "vif"}))

# refit dropping the most influential point, to show the estimate does not depend on it
worst = int(np.argmax(cooks))
m_drop = assoc(DAT.drop(DAT.index[worst]), "ace_memory_subscale", "P_RL")
RESULTS["diag"]["beta_drop_worst"] = m_drop["beta"]; RESULTS["diag"]["p_drop_worst"] = m_drop["p"]
print(f"  dropping the most influential participant: beta {m_drop['beta']:+.3f} p {m_drop['p']:.4f}")

fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.7))
ax = axes[0]
ax.scatter(m.fittedvalues, m.resid, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.4)
ax.axhline(0, color=AXIS, lw=1.2)
lo = sm.nonparametric.lowess(m.resid, m.fittedvalues, frac=.8)
ax.plot(lo[:, 0], lo[:, 1], color=INK, lw=1.6)
ax.set_xlabel("fitted value"); ax.set_ylabel("residual")
ax.set_title("a · Residuals vs fitted", loc="left")
ax.text(0, 1.02, f"Breusch–Pagan p = {bp[1]:.2f}", transform=ax.transAxes, fontsize=8.8,
        color=MUTED, va="bottom")

ax = axes[1]
(osm, osr), (sl, ic, _) = stats.probplot(m.resid, dist="norm")
ax.scatter(osm, osr, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.4)
ax.plot(osm, sl*osm + ic, color=INK, lw=1.6)
ax.set_xlabel("theoretical quantile"); ax.set_ylabel("residual quantile")
ax.set_title("b · Normal Q–Q", loc="left")
ax.text(0, 1.02, f"Shapiro–Wilk p = {sw.pvalue:.2f}", transform=ax.transAxes, fontsize=8.8,
        color=MUTED, va="bottom")

ax = axes[2]
ax.vlines(np.arange(len(cooks)), 0, cooks, color=BLUE, lw=1.8)
ax.axhline(cook_thr, color=RED, lw=1.3)
ax.text(len(cooks)*.98, cook_thr*1.08, "4/n", ha="right", fontsize=8.5, color=RED)
ax.set_xlabel("participant"); ax.set_ylabel("Cook's distance")
ax.set_title("c · Influence", loc="left")
ax.text(0, 1.02, f"max {cooks.max():.2f} · {int((cooks>cook_thr).sum())} above 4/n",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

ax = axes[3]
ax.scatter(hat, stud, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.4)
ax.axhline(0, color=AXIS, lw=1.2)
for v in (-2, 2): ax.axhline(v, color=GRID, lw=1)
ax.axvline(2*hat.mean(), color=GRID, lw=1)
ax.set_xlabel("leverage"); ax.set_ylabel("studentised residual")
ax.set_title("d · Leverage", loc="left")
ax.text(0, 1.02, f"max VIF {max(vif.values()):.2f}", transform=ax.transAxes, fontsize=8.8,
        color=MUTED, va="bottom")
save(fig, "fig15_diagnostics")

with open(f"{TABDIR}/extra_results.json", "w") as f:
    json.dump(RESULTS, f, indent=1, default=float)
print("\nwrote tables/extra_results.json")
