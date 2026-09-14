"""Turn the visual-edition notebook's result CSVs into LaTeX table fragments."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(HERE), "kaggle_nb_rem_v2", "figures")
DST = os.path.join(HERE, "tables")
os.makedirs(DST, exist_ok=True)
NL = chr(10)


def esc(s):
    s = str(s)
    for a, b in [("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("±", r"$\pm$"),
                 ("≤", r"$\leq$"), ("→", r"$\rightarrow$"), ("≥", r"$\geq$")]:
        s = s.replace(a, b)
    return s


def pfmt(p):
    p = float(p)
    if p < 0.0001:
        return "$<$0.0001"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


# ------------------------------------------------------------------ Table 1
t1 = pd.read_csv(f"{SRC}/table1_cohort.csv")
rows = []
for _, r in t1.iterrows():
    lab = esc(r.Variable)
    if str(r.Variable).startswith("  "):
        lab = r"\quad " + lab.strip()
    p = pfmt(r.p) if str(r.p) not in ("nan", "") else ""
    rows.append(f"{lab} & {esc(r.Unimpaired)} & {esc(r.Impaired)} & {p} \\\\")
open(f"{DST}/table1.tex", "w").write(rf"""
\begin{{table}}[htbp]
\centering
\begin{{threeparttable}}
\caption{{\textbf{{Cohort characteristics}}, split at the ACE-III screening threshold for cognitive
impairment. Values are mean $\pm$ SD or per cent.}}
\label{{tab:cohort}}
\small
\begin{{tabular}}{{lccc}}
\toprule
 & ACE-III $>$ 82 & ACE-III $\leq$ 82 & $p$ \\
\midrule
{NL.join(rows)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item Continuous variables compared with the Mann--Whitney $U$ test
\citep{{mann1947test}}, binary variables with Fisher's exact test \citep{{fisher1922interpretation}}. $P_{{\mathrm{{R}}\rightarrow\mathrm{{L}}}}$ is the REM-exit continuity index.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip() + NL)

# ------------------------------------------------------------------ Table 2 (domains)
t3 = pd.read_csv(f"{SRC}/table3_domain_specificity.csv")
rows = []
for _, r in t3.iterrows():
    q = "" if str(r["q"]) == "nan" else pfmt(r["q"])
    rows.append(f"{esc(r.label)} & {int(r.n)} & {r.beta:+.3f} & "
                f"[{r.lo:+.2f}, {r.hi:+.2f}] & {pfmt(r.p)} & {q} & {r.rho:+.3f} \\\\")
    if r.outcome == "ace_total":
        rows.append(r"\midrule")
open(f"{DST}/table_domains.tex", "w").write(rf"""
\begin{{table}}[htbp]
\centering
\begin{{threeparttable}}
\caption{{\textbf{{REM-exit continuity across ACE-III domains.}} Standardised coefficients from
age- and sex-adjusted linear models.}}
\label{{tab:domains}}
\small
\begin{{tabular}}{{lcccccc}}
\toprule
Outcome & $n$ & $\beta$ & 95\% CI & $p$ & $q$ & Spearman $\rho$ \\
\midrule
{NL.join(rows)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item $\beta$ is the change in outcome SD per SD of $P_{{\mathrm{{R}}\rightarrow\mathrm{{L}}}}$.
$q$ is the Benjamini--Hochberg value across the five subscales; the total score is not an
independent member of that family and is left uncorrected. Subscale maxima: attention 18,
memory 26, fluency 14, language 26, visuospatial 16.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip() + NL)

# ------------------------------------------------------------------ Table 3 (AUC)
t6 = pd.read_csv(f"{SRC}/table6_auc_benchmark.csv")
rows = [f"{esc(r.metric)} & {int(r.n)} & {r.AUC:.3f} & [{r.lo:.2f}, {r.hi:.2f}] \\\\"
        for _, r in t6.iterrows()]
rows.insert(1, r"\midrule")
open(f"{DST}/table_auc.tex", "w").write(rf"""
\begin{{table}}[htbp]
\centering
\begin{{threeparttable}}
\caption{{\textbf{{Discrimination of cognitive impairment}} (ACE-III $\leq$ 82) by
$P_{{\mathrm{{R}}\rightarrow\mathrm{{L}}}}$ and by conventional sleep metrics computed from the same
nights.}}
\label{{tab:auc}}
\small
\begin{{tabular}}{{lccc}}
\toprule
Metric & $n$ & AUC & 95\% CI \\
\midrule
{NL.join(rows)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item Each metric is oriented so that higher values indicate greater likelihood of impairment.
Confidence intervals are from 2{{,}}000 bootstrap resamples.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip() + NL)

# ------------------------------------------------- Table 4 (metric independence)
p8 = f"{SRC}/table8_metric_independence.csv"
if os.path.exists(p8):
    t8 = pd.read_csv(p8).sort_values("q")
    rows = [f"{esc(r.label)} & {r.beta:+.3f} & [{r.lo:+.2f}, {r.hi:+.2f}] & {pfmt(r.p)} & "
            f"{pfmt(r['q'])} \\\\" for _, r in t8.iterrows()]
    rows.insert(1, r"\midrule")
    open(f"{DST}/table_independence.tex", "w").write(rf"""
\begin{{table}}[htbp]
\centering
\begin{{threeparttable}}
\caption{{\textbf{{Each sleep metric against the ACE-III memory subscale}}, one at a time, in the
same age- and sex-adjusted model.}}
\label{{tab:independence}}
\small
\begin{{tabular}}{{lcccc}}
\toprule
Metric & $\beta$ & 95\% CI & $p$ & $q$ \\
\midrule
{NL.join(rows)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item $q$ is the Benjamini--Hochberg value across the nine metrics tested.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip() + NL)

# --------------------------------------------------- Appendix: robustness
t4 = pd.read_csv(f"{SRC}/table4_robustness.csv")
lab = {"ace_total": "ACE-III total", "ace_memory_subscale": "ACE-III memory"}
rows = []
for y in ["ace_total", "ace_memory_subscale"]:
    rows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{lab[y]}}}}} \\")
    for _, r in t4[t4.outcome == y].iterrows():
        rows.append(f"\\quad {esc(r.model)} & {int(r.n)} & {r.beta:+.3f} & "
                    f"[{r.lo:+.2f}, {r.hi:+.2f}] & {pfmt(r.p)} \\\\")
    if y == "ace_total":
        rows.append(r"\midrule")
open(f"{DST}/table_robust.tex", "w").write(rf"""
\begin{{table}}[htbp]
\centering
\begin{{threeparttable}}
\caption{{\textbf{{Robustness to covariate adjustment.}} Standardised coefficient for
$P_{{\mathrm{{R}}\rightarrow\mathrm{{L}}}}$ under progressively richer models; all include age and sex.}}
\label{{tab:robust}}
\small
\begin{{tabular}}{{lcccc}}
\toprule
Model & $n$ & $\beta$ & 95\% CI & $p$ \\
\midrule
{NL.join(rows)}
\bottomrule
\end{{tabular}}
\end{{threeparttable}}
\end{{table}}
""".strip() + NL)

# --------------------------------------------------- Nights needed
p7 = f"{SRC}/table7_nights_needed.csv"
if os.path.exists(p7):
    t7 = pd.read_csv(p7)
    rows = [f"{int(r.k)} & {r.auc:.3f} & [{r.auc_lo:.2f}, {r.auc_hi:.2f}] & "
            f"{r.beta_mem:+.3f} & {100*r.pow_mem:.0f}\\% \\\\" for _, r in t7.iterrows()]
    open(f"{DST}/table_nights.tex", "w").write(rf"""
\begin{{table}}[htbp]
\centering
\begin{{threeparttable}}
\caption{{\textbf{{Effect of recording length.}} $P_{{\mathrm{{R}}\rightarrow\mathrm{{L}}}}$
recomputed from $k$ nights drawn at random from each participant, 60 draws per $k$.}}
\label{{tab:nights}}
\small
\begin{{tabular}}{{ccccc}}
\toprule
Nights $k$ & AUC & 10th--90th pct & $\beta$ (memory) & Draws with $p<0.05$ \\
\midrule
{NL.join(rows)}
\bottomrule
\end{{tabular}}
\end{{threeparttable}}
\end{{table}}
""".strip() + NL)

print("wrote:", sorted(os.listdir(DST)))
