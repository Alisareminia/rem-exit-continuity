#!/bin/bash
# Regenerate the paper's figures from the notebook's own code, with the in-figure
# title block suppressed (FIGURE_TITLES=0) so the LaTeX caption is the only caption.
#
# Source of truth is src/make_rem_notebook_v2.py: this runs the flattened copy of
# its code cells, so the paper's figures cannot drift from the published notebook.
# Takes ~10 minutes (it reruns the full analysis, permutations included).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=../.venv-ad/bin/python
"$PY" ../src/make_rem_notebook_v2.py --script >/dev/null

FIGURE_TITLES=0 MPLBACKEND=Agg "$PY" ../kaggle_nb_rem_v2/_validate.py > .figures.log 2>&1

echo "regenerated $(ls figures/fig*.pdf | wc -l | tr -d ' ') figures into $(pwd)/figures"

# The notebook numbers its figures in the order the notebook presents them; the paper
# presents several of them in a different order. Map notebook names onto paper numbers
# here, so both artefacts stay internally consistent: figure N in the PDF is figNN_*.
cd figures
for pair in "fig09_robustness:fig10_robustness" "fig10_independence:fig13_independence" \
            "fig11_auc:fig15_auc" "fig12_nights:fig16_nights"; do
  old="${pair%%:*}"; new="${pair##*:}"
  for ext in pdf png; do
    [ -f "$old.$ext" ] && mv -f "$old.$ext" "__m_$new.$ext"
  done
done
for f in __m_*; do [ -e "$f" ] && mv -f "$f" "${f#__m_}"; done
cd ..
echo "mapped notebook figure names onto the paper's numbering"
