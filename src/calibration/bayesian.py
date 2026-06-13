"""BetaBinomialBucketed — bucketed Bayesian calibrator (plan §5.1 / §6.1).

Bin ``p_raw`` into quantile (equal-mass) buckets; per bucket, place a
``Beta(α₀, β₀)`` prior on the bucket's true positive rate and update with
the observed counts to a ``Beta(α₀+k, β₀+n−k)`` posterior. ``transform``
maps each raw probability to its bucket's posterior mean plus a 95%
credible interval from the Beta ppf — closed-form, no MCMC.

Tie robustness (plan R8): tiny tree models emit very few distinct ``p_raw``
values, so naive quantile edges collapse. We drop duplicate edges, merge
adjacent under-populated bins (``min_bin_size``), and raise if fewer than
``min_effective_bins`` survive — the caller then falls back to a simpler
scheme.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

from calibration import CalibrationOutput

_CRED_LO = 0.025
_CRED_HI = 0.975


class BetaBinomialBucketed:
    """Quantile-bucketed Beta-Binomial calibrator with credible bands."""

    def __init__(
        self,
        n_bins: int = 10,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        min_bin_size: int = 20,
        min_effective_bins: int = 3,
    ) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1; got {n_bins}")
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError(
                f"prior params must be > 0; got alpha={alpha_prior}, "
                f"beta={beta_prior}"
            )
        if min_effective_bins < 1:
            raise ValueError(
                f"min_effective_bins must be >= 1; got {min_effective_bins}"
            )
        self.n_bins = n_bins
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.min_bin_size = min_bin_size
        self.min_effective_bins = min_effective_bins

        # Set by fit().
        self.edges_: np.ndarray | None = None
        self.alphas_: np.ndarray | None = None
        self.betas_: np.ndarray | None = None
        self.fit_diagnostics_: dict[str, Any] | None = None

    # -- fit -----------------------------------------------------------------
    def fit(
        self,
        p_raw: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> "BetaBinomialBucketed":
        """Fit per-bucket Beta posteriors on ``(p_raw, y_true)``.

        ``sample_weight`` (if given) weights the per-bucket count and
        positive-count, mirroring gbdt's weighted-base-rate convention; the
        Beta update then uses the weighted ``(k, n)``.
        """
        p = np.asarray(p_raw, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if p.shape != y.shape:
            raise ValueError(
                f"p_raw shape {p.shape} != y_true shape {y.shape}"
            )
        if p.size == 0:
            raise ValueError("BetaBinomialBucketed.fit: empty input")
        if sample_weight is None:
            w = np.ones_like(p)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != p.shape:
                raise ValueError(
                    f"sample_weight shape {w.shape} != p_raw shape {p.shape}"
                )
            if np.any(w < 0.0):
                raise ValueError("sample_weight must be non-negative")
            # Normalize weights to preserve the true sample size: the Beta
            # posterior's concentration is its pseudo-count, so it must
            # reflect the actual number of observations, not the arbitrary
            # scale of the loss-weighting vector. gbdt emits a constant
            # down-weight (~0.0101/row on cell-5) that would otherwise
            # deflate every bin's effective n ~100x and balloon the credible
            # bands. Rescaling to sum=N recovers unweighted counts when
            # weights are constant and keeps relative weighting otherwise.
            # (A stricter Kish effective-sample-size = (Σw)²/Σw² is a future
            # refinement; sum=N is the conventional, plan-consistent choice.)
            total = w.sum()
            if total <= 0.0:
                raise ValueError("sample_weight sums to zero")
            w = w * (w.size / total)

        # 1. Quantile edges, dropping duplicate boundaries (R8). We bin with
        #    pd.qcut to get equal-mass buckets; duplicates='drop' collapses
        #    ties at quantile boundaries into fewer effective bins.
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        raw_edges = np.quantile(p, quantiles)
        edges = np.unique(raw_edges)
        # Guard the outer edges so transform's searchsorted never escapes the
        # fitted range (clip handles out-of-range p at transform time).
        n_raw_distinct = int(np.unique(p).size)

        # bin index per row using interior edges; np.searchsorted with the
        # interior cut points gives bin ids in [0, len(edges)-2].
        interior = edges[1:-1]
        bin_idx = np.searchsorted(interior, p, side="right")
        n_bins_initial = len(edges) - 1
        if n_bins_initial < 1:
            # All p_raw identical → one bin.
            n_bins_initial = 1
            bin_idx = np.zeros_like(p, dtype=int)
            edges = np.array([p[0], p[0]])

        # 2. Merge adjacent under-populated bins (weighted count < min_bin_size)
        #    left-to-right, then re-fold the final bin if still short.
        merged_labels = self._merge_small_bins(
            bin_idx, w, n_bins_initial
        )
        effective_n_bins = int(np.unique(merged_labels).size)

        # 3. Effective-bin floor.
        if effective_n_bins < self.min_effective_bins:
            raise ValueError(
                f"calibrator: only {effective_n_bins} bins survived "
                f"dedup+merge; check p_raw distribution "
                f"(n_raw_distinct={n_raw_distinct})"
            )

        # 4. Per surviving bin: weighted k, n → posterior (α, β). Also record
        #    each bin's [lo, hi) raw-p extent so transform can map by value.
        labels = np.unique(merged_labels)
        alphas = np.empty(labels.size, dtype=float)
        betas = np.empty(labels.size, dtype=float)
        bin_lo = np.empty(labels.size, dtype=float)
        bin_hi = np.empty(labels.size, dtype=float)
        bin_n = np.empty(labels.size, dtype=float)
        for i, lab in enumerate(labels):
            mask = merged_labels == lab
            n_i = float(w[mask].sum())
            k_i = float((w[mask] * y[mask]).sum())
            alphas[i] = self.alpha_prior + k_i
            betas[i] = self.beta_prior + (n_i - k_i)
            bin_lo[i] = float(p[mask].min())
            bin_hi[i] = float(p[mask].max())
            bin_n[i] = n_i

        # Transform-time mapping uses contiguous boundaries: midpoints between
        # adjacent bins' raw-p extents. Build cut points from sorted bin_hi.
        sort = np.argsort(bin_lo)
        alphas, betas = alphas[sort], betas[sort]
        bin_lo, bin_hi, bin_n = bin_lo[sort], bin_hi[sort], bin_n[sort]
        # Boundaries between bins: midpoint of (prev_hi, next_lo).
        cut_points = (bin_hi[:-1] + bin_lo[1:]) / 2.0

        self.edges_ = cut_points
        self.alphas_ = alphas
        self.betas_ = betas
        self.fit_diagnostics_ = {
            "effective_n_bins": effective_n_bins,
            "n_bins_requested": self.n_bins,
            "n_raw_distinct": n_raw_distinct,
            "n_rows": int(p.size),
            "bin_n": bin_n.tolist(),
            "bin_lo": bin_lo.tolist(),
            "bin_hi": bin_hi.tolist(),
            "posterior_mean": (alphas / (alphas + betas)).tolist(),
            "base_rate": float((w * y).sum() / w.sum()),
        }
        return self

    def _merge_small_bins(
        self, bin_idx: np.ndarray, w: np.ndarray, n_bins_initial: int
    ) -> np.ndarray:
        """Merge adjacent bins until each has weighted count >= min_bin_size.

        Returns a relabeled array (contiguous group ids by ascending bin).
        Merges left-to-right; a trailing short bin folds into its left
        neighbour.
        """
        # Weighted count per initial bin.
        counts = np.zeros(n_bins_initial, dtype=float)
        for b in range(n_bins_initial):
            counts[b] = w[bin_idx == b].sum()

        # Assign each initial bin to a group; walk left-to-right accumulating.
        group_of = np.empty(n_bins_initial, dtype=int)
        g = 0
        acc = 0.0
        for b in range(n_bins_initial):
            group_of[b] = g
            acc += counts[b]
            if acc >= self.min_bin_size:
                g += 1
                acc = 0.0
        # If the last group is short (acc>0 left over and g not advanced past
        # it), fold it into the previous group.
        n_groups = int(group_of.max()) + 1
        if n_groups > 1 and acc > 0.0 and acc < self.min_bin_size:
            last_group = group_of.max()
            group_of[group_of == last_group] = last_group - 1

        # Relabel groups to be contiguous from 0.
        uniq = np.unique(group_of)
        remap = {old: new for new, old in enumerate(uniq)}
        group_of = np.array([remap[gg] for gg in group_of])
        return group_of[bin_idx]

    # -- transform -----------------------------------------------------------
    def transform(self, p_raw: np.ndarray) -> CalibrationOutput:
        """Map raw probabilities to posterior mean + 95% credible interval."""
        if self.alphas_ is None:
            raise RuntimeError("BetaBinomialBucketed.fit() must be called first")
        p = np.asarray(p_raw, dtype=float)
        bin_idx = np.searchsorted(self.edges_, p, side="right")
        bin_idx = np.clip(bin_idx, 0, len(self.alphas_) - 1)
        a = self.alphas_[bin_idx]
        b = self.betas_[bin_idx]
        p_mean = a / (a + b)
        p_low = _beta.ppf(_CRED_LO, a, b)
        p_high = _beta.ppf(_CRED_HI, a, b)
        return CalibrationOutput(p_mean=p_mean, p_low=p_low, p_high=p_high)
