"""Centralized configuration for the analog_mc pipeline.

A single ``Config`` dataclass holds every parameter that the pipeline reads.
Pass it through the pipeline explicitly; do not read from globals.

Round-trips through YAML so experiments can be reproduced from a file. Tuples
become lists in YAML; the loader coerces them back so downstream code can rely
on hashability.

Invariants are validated in ``__post_init__`` so misconfiguration is caught at
construction time rather than mid-run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # ---- Asset ----------------------------------------------------------
    ticker: str = "NASDAQ100"
    data_path: str = "data/NASDAQ100.csv"
    date_col: str = "observation_date"
    close_col: str = "NASDAQ100"

    # ---- Horizons (fully configurable per asset / use case) -------------
    forecast_horizon: int = 60
    block_length: int = 10
    n_blocks: int = 6
    n_paths: int = 1000

    # ---- Z-score horizons -----------------------------------------------
    # Must be length-3; grid search in search.py is built for 3 weights.
    zscore_horizons: tuple[int, ...] = (20, 50, 200)

    # ---- EWMA vol -------------------------------------------------------
    ewma_halflife: int = 20

    # ---- Walk-forward ---------------------------------------------------
    train_initial_size: int = 1000
    val_size: int = 60
    test_size: int = 60

    # ---- Hyperparameter search -----------------------------------------
    weight_grid_resolution: float = 0.1
    n_eff_values: tuple[int, ...] = (15, 30, 50, 80, 150)
    local_refine_top_k: int = 5
    nelder_mead_xatol: float = 0.01
    nelder_mead_maxiter: int = 50

    # ---- Volatility scaling --------------------------------------------
    vol_clip_lower: float = 0.5
    vol_clip_upper: float = 3.0
    drift_mode: str = "zero"  # "zero" (v1) or "trailing_momentum" (v2)
    momentum_lookback: int = 20
    momentum_shrinkage: float = 0.5

    # ---- Diagnostics ----------------------------------------------------
    pit_n_bins: int = 20
    acf_lags: tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    vol_regime_quantiles: tuple[float, ...] = (0.33, 0.67)

    # ---- Reproducibility ------------------------------------------------
    random_seed: int = 42

    # ---- Output ---------------------------------------------------------
    runs_dir: str = "runs/analog_mc"

    def __post_init__(self) -> None:
        # Coerce list -> tuple for sequence fields (YAML always loads lists).
        object.__setattr__(self, "zscore_horizons", tuple(self.zscore_horizons))
        object.__setattr__(self, "n_eff_values", tuple(self.n_eff_values))
        object.__setattr__(self, "acf_lags", tuple(self.acf_lags))
        object.__setattr__(self, "vol_regime_quantiles", tuple(self.vol_regime_quantiles))

        self._validate()

    def _validate(self) -> None:
        if self.forecast_horizon != self.n_blocks * self.block_length:
            raise ValueError(
                f"forecast_horizon ({self.forecast_horizon}) must equal "
                f"n_blocks * block_length ({self.n_blocks} * {self.block_length} "
                f"= {self.n_blocks * self.block_length})"
            )
        if len(self.zscore_horizons) != 3:
            raise ValueError(
                f"zscore_horizons must have length 3 (got {len(self.zscore_horizons)}); "
                "the grid search is built for 3 weights."
            )
        if max(self.zscore_horizons) >= self.train_initial_size:
            raise ValueError(
                f"max(zscore_horizons) ({max(self.zscore_horizons)}) must be < "
                f"train_initial_size ({self.train_initial_size}); early Train dates "
                "would otherwise have undefined long-horizon z-scores."
            )
        if not (self.vol_clip_lower < 1.0 < self.vol_clip_upper):
            raise ValueError(
                f"vol_clip_lower ({self.vol_clip_lower}) < 1.0 < "
                f"vol_clip_upper ({self.vol_clip_upper}) required."
            )
        if self.n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if self.drift_mode not in {"zero", "trailing_momentum", "scale_with_vol"}:
            raise ValueError(f"drift_mode must be one of zero|trailing_momentum|scale_with_vol; got {self.drift_mode}")

    # ---- I/O ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)
