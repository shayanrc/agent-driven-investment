#!/usr/bin/env bash
# v4 post-canonical orchestrator: one-shot the step-9 deliverables for any
# v4 experiment (B1, A2.1, B5, ...).
#
# Given a walk-forward run dir and an experiment label, produces:
#   - results/analog_mc/data/fat_tail_<label>.json           (eval payload)
#   - results/analog_mc/data/fat_tail_<label>_diff.json      (diff vs v2.4 baseline)
#   - docs/analog_mc/experiments/_<label>_fat_tail.md        (markdown summary)
#   - docs/analog_mc/experiments/figs/<label>_fat_tail/*.png (15-chart panel)
#
# Usage:
#   bash scripts/analog_mc/v4_post_canonical.sh runs/analog_mc/<TIMESTAMP> <label> "<display title>"
#
# Example:
#   bash scripts/analog_mc/v4_post_canonical.sh runs/analog_mc/20260520T155220Z \
#       b1_local_linear "B1 (Platzer local-linear)"

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <run-dir> <label-slug> <display-title>" >&2
    echo "  label-slug:    used in filenames and dir name (e.g. b1_local_linear)" >&2
    echo "  display-title: shown in plot titles (e.g. 'B1 (Platzer local-linear)')" >&2
    exit 2
fi

RUN_DIR="$1"
LABEL="$2"
TITLE="$3"

BASELINE_JSON="results/analog_mc/data/fat_tail_baseline_v24.json"
PANEL_DIR="docs/analog_mc/experiments/figs/${LABEL}_fat_tail"

if [ ! -d "$RUN_DIR/folds" ]; then
    echo "ERROR: $RUN_DIR has no folds/ subdir" >&2
    exit 1
fi

echo "[1/2] Computing fat-tail eval + diff vs v2.4 baseline..."
uv run python scripts/analog_mc/compute_fat_tail_eval.py \
    --run-dir "$RUN_DIR" \
    --label "$LABEL" \
    --baseline-json "$BASELINE_JSON"

echo
echo "[2/2] Rendering 15-anchor fat-tail panel..."
uv run python scripts/analog_mc/render_fat_tail_panel.py \
    --run-dir "$RUN_DIR" \
    --label "$TITLE" \
    --out-dir "$PANEL_DIR" \
    --prefix "$LABEL"

echo
echo "All step-9 deliverables produced for $LABEL"
echo "  eval JSON:   results/analog_mc/data/fat_tail_${LABEL}.json"
echo "  diff JSON:   results/analog_mc/data/fat_tail_${LABEL}_diff.json"
echo "  markdown:    docs/analog_mc/experiments/_${LABEL}_fat_tail.md"
echo "  panel dir:   $PANEL_DIR/"
