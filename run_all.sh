#!/bin/bash
# Reproduce every number and figure in the manuscript, in order.
# Expects the RESILIENT data unzipped into data/raw/ (see README).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "[1/3] notebook figures and result tables  (~10 min)"
bash analysis/make_figures.sh

echo "[2/3] selection bootstrap, specification curve, diagnostics  (~7 min)"
"$PY" analysis/extra_analyses.py

echo "[3/3] six-month follow-up and physiological adjustment  (~90 s)"
"$PY" analysis/followup_and_physio.py
"$PY" analysis/fig_physio_w3plus.py

echo
echo "Done. Figures in figures/, result tables in tables/."
echo "Check the headline numbers against the table in README.md."
