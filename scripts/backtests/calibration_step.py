"""Calibration fit + Stage-5 checkpoint for the cell-5 back-test (plan §9.5).

Reusable by the Stage-8 runner: ``fit_calibrator`` fits the Bayesian
calibrator on the cell's VAL split (plan D5) and ``run_checkpoint`` emits
the checkpoint JSON + reliability figure and reports whether any Stage-5
gate condition tripped (which the plan says must be surfaced to the user
before Stage 6).

Input-column selection (plan R9): read ``metrics.json::calibration.decision``.
``"native"`` means gbdt's internal isotonic was pass-through, so
``p_calibrated == p_raw`` (within float tolerance on cell-5) and we consume
``p_calibrated``. When isotonic IS active, ``p_calibrated`` is the already-
shrunk probability — still the correct input (we recalibrate on top, the
value-add being the credible band).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.bayesian import BetaBinomialBucketed
from calibration.diagnostics import (
    expected_calibration_error,
    reliability_diagram,
)

# Stage-5 gate thresholds (plan §9 step 5).
GATE_MIN_EFFECTIVE_BINS = 3
GATE_ECE_EVAL_MAX = 0.10
GATE_ECE_DELTA_MAX = 0.05
GATE_MAX_BAND_WIDTH = 0.15


def pick_input_column(metrics: dict) -> str:
    """Return the prediction column to calibrate on (plan R9)."""
    decision = metrics.get("calibration", {}).get("decision")
    # Both 'native' (pass-through) and an active isotonic feed p_calibrated;
    # p_calibrated is the gbdt module's canonical post-calibration column.
    # We always consume it — on 'native' it equals p_raw to float tolerance.
    return "p_calibrated"


def load_split(artifact_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(artifact_dir / "predictions" / f"{split}.csv", parse_dates=["date"])
    return df


def fit_calibrator(
    artifact_dir: Path,
    *,
    n_bins: int = 10,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> tuple[BetaBinomialBucketed, str]:
    """Fit BetaBinomialBucketed on the cell's VAL split (D5).

    Returns the fitted calibrator and the input column name consumed.
    """
    metrics = json.loads((artifact_dir / "metrics.json").read_text())
    col = pick_input_column(metrics)
    val = load_split(artifact_dir, "val")
    sw = val["sample_weight"].to_numpy() if "sample_weight" in val else None
    cal = BetaBinomialBucketed(
        n_bins=n_bins, alpha_prior=alpha_prior, beta_prior=beta_prior
    ).fit(val[col].to_numpy(), val["y_true"].to_numpy(), sample_weight=sw)
    return cal, col


def _per_bin_band_width(
    p_mean: np.ndarray,
    p_low: np.ndarray,
    p_high: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Max over equal-width bins of the mean (p_high − p_low) in that bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges[1:-1], p_mean, side="right"), 0, n_bins - 1)
    widths = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            widths.append(float(np.mean(p_high[mask] - p_low[mask])))
    return max(widths) if widths else 0.0


@dataclass
class CheckpointResult:
    effective_n_bins: int
    ece_val: float
    ece_eval: float
    ece_delta: float
    max_band_width: float
    input_column: str
    base_rate_val: float
    base_rate_eval: float
    gate_triggered: bool
    gate_reasons: list[str]


def run_checkpoint(
    artifact_dir: Path,
    out_dir: Path,
    *,
    n_bins: int = 10,
) -> CheckpointResult:
    """Fit on VAL, transform VAL + EVAL, emit checkpoint JSON + figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibrator").mkdir(exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)

    cal, col = fit_calibrator(artifact_dir, n_bins=n_bins)
    val = load_split(artifact_dir, "val")
    ev = load_split(artifact_dir, "eval")

    val_w = val["sample_weight"].to_numpy() if "sample_weight" in val else None
    ev_w = ev["sample_weight"].to_numpy() if "sample_weight" in ev else None

    out_val = cal.transform(val[col].to_numpy())
    out_ev = cal.transform(ev[col].to_numpy())

    ece_val = expected_calibration_error(
        out_val.p_mean, val["y_true"].to_numpy(), sample_weight=val_w
    )
    ece_eval = expected_calibration_error(
        out_ev.p_mean, ev["y_true"].to_numpy(), sample_weight=ev_w
    )
    max_band = _per_bin_band_width(out_ev.p_mean, out_ev.p_low, out_ev.p_high)

    eff_bins = int(cal.fit_diagnostics_["effective_n_bins"])
    ece_delta = abs(ece_val - ece_eval)

    reasons: list[str] = []
    if eff_bins < GATE_MIN_EFFECTIVE_BINS:
        reasons.append(f"effective_n_bins={eff_bins} < {GATE_MIN_EFFECTIVE_BINS}")
    if ece_eval > GATE_ECE_EVAL_MAX:
        reasons.append(f"ece_eval={ece_eval:.4f} > {GATE_ECE_EVAL_MAX}")
    if ece_delta > GATE_ECE_DELTA_MAX:
        reasons.append(
            f"|ece_val-ece_eval|={ece_delta:.4f} > {GATE_ECE_DELTA_MAX}"
        )
    if max_band > GATE_MAX_BAND_WIDTH:
        reasons.append(f"max_band_width={max_band:.4f} > {GATE_MAX_BAND_WIDTH}")

    result = CheckpointResult(
        effective_n_bins=eff_bins,
        ece_val=float(ece_val),
        ece_eval=float(ece_eval),
        ece_delta=float(ece_delta),
        max_band_width=float(max_band),
        input_column=col,
        base_rate_val=float(cal.fit_diagnostics_["base_rate"]),
        base_rate_eval=float(
            np.average(
                ev["y_true"].to_numpy(),
                weights=ev_w if ev_w is not None else np.ones(len(ev)),
            )
        ),
        gate_triggered=bool(reasons),
        gate_reasons=reasons,
    )

    # Persist calibrator artifact + bins.csv + checkpoint JSON + figure.
    with open(out_dir / "calibrator" / "artifact.pkl", "wb") as f:
        pickle.dump(cal, f)
    bins = pd.DataFrame(
        {
            "bin_lo": cal.fit_diagnostics_["bin_lo"],
            "bin_hi": cal.fit_diagnostics_["bin_hi"],
            "n": cal.fit_diagnostics_["bin_n"],
            "posterior_mean": cal.fit_diagnostics_["posterior_mean"],
            "alpha": cal.alphas_.tolist(),
            "beta": cal.betas_.tolist(),
        }
    )
    bins.to_csv(out_dir / "calibrator" / "bins.csv", index=False)
    (out_dir / "checkpoint.json").write_text(json.dumps(asdict(result), indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    reliability_diagram(
        out_ev.p_mean,
        ev["y_true"].to_numpy(),
        n_bins=10,
        p_low=out_ev.p_low,
        p_high=out_ev.p_high,
        sample_weight=ev_w,
        title="Cell-5 calibrator — EVAL reliability (val-fit)",
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "reliability.png", dpi=130)
    plt.close(fig)

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage-5 calibrator checkpoint")
    ap.add_argument(
        "--artifact-dir",
        default="results/gbdt/experiments/"
        "nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen",
    )
    ap.add_argument("--out-dir", default="results/backtests/_001_cell5_bayesian_kelly")
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    res = run_checkpoint(
        Path(args.artifact_dir), Path(args.out_dir), n_bins=args.n_bins
    )
    print(json.dumps(asdict(res), indent=2))
    if res.gate_triggered:
        print("\n[STAGE-5 GATE TRIPPED] surface to user before Stage 6:")
        for r in res.gate_reasons:
            print(f"  - {r}")
    else:
        print("\n[STAGE-5 OK] no gate condition tripped; clear to proceed.")
