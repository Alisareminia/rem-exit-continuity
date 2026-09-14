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
