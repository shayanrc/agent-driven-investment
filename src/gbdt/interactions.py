"""Feature-interaction methodology for the gbdt XGBoost backend (V1.2 Phase 4).

This is the **measurement library** for "how do feature interactions drive the
categorical-outcome models" — the V1.2 raison d'être
(``docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`` § 3 / § 8 Phase 4). It
ships three things, in the priority order the plan's decision **D1** mandates:

1. :func:`interaction_strength` — the public entry point. ``kind="shap"`` is the
   **headline**: it uses XGBoost's *native* TreeSHAP
   (``booster.predict(dmatrix, pred_interactions=True)``, *not* the external
   ``shap`` package — D1 / R4) to get per-row signed pairwise interaction values,
   then **streams** the pair aggregation in row-batches so the dense
   ``(rows, F+1, F+1)`` tensor is never materialised (§ 3.3 feasibility budget).
   ``kind="cooccurrence"`` is the near-free native split-pair cross-check
   (O(trees·depth²); ranking only, no per-row signed values).
2. :class:`InteractionResult` — the small JSON-serialisable dataclass that carries
   the ranked pairs + per-feature involvement (Σ|off-diagonal|) + per-feature
   main-effect (mean |diagonal SHAP|, ``shap`` kind only) + ``sign_consistency``
   + which ``kind`` produced it. ``per_feature_main_effect`` is carried alongside
   ``per_feature_involvement`` so a future agent-loop bundle (D7, Phase 5) can
   apply the *drop-only-if-low-main-effect-AND-low-interaction-load* pruning rule
   verbatim (``project-xgboost-interaction-analysis`` § 3).
3. :func:`ablate_interactions` — the ``interaction_constraints`` causal-ablation
   tool (§ 3.2 / plan § 8 Phase 4): retrain an :class:`~gbdt.model.XGBoostModel`
   with the top-K SHAP pairs **forbidden**, so a caller can confirm the measured
   interaction drops ~0 and Brier degrades in proportion to SHAP magnitude. This
   is an *intervention experiment* — explicitly **never** on the hot FS+HP loop
   (``project-xgboost-training-essentials`` § 3).

**Scope discipline (Phase 4 = methodology + unit tests only):** nothing here is
wired into ``/gbdt-diagnose``, ``diagnose.json``, the diagnostic bundle,
``train.py`` or the runner — that is Phase 5. No feature-interaction experiment is
run here — that is Phase 8. The ``shap`` package is **not** imported anywhere on
the core path (native TreeSHAP via the booster is the headline; the optional
``shap``-based viz helper :func:`shap_interaction_summary_plot` lazily imports it
and the core path never requires it).

The module is intentionally self-contained — it reads the fitted model through
:class:`~gbdt.model.XGBoostModel`'s public surface (``get_booster``-equivalent via
the wrapped estimator) and never touches ``model.py``'s validation/determinism
internals.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gbdt.model import BaseGBDTModel, XGBoostModel


# Default row cap for the SHAP-interaction pass. A few thousand rows rank pairs
# stably (§ 3.3 mitigation 2); larger panels sub-sample down to this.
_DEFAULT_MAX_ROWS = 5_000

# Sign-consistency on a near-zero interaction is meaningless (it is noise around
# zero); below this absolute-magnitude floor we report sign_consistency = nan so
# downstream readers do not over-interpret a "consistent" sign on a dead pair.
_SIGN_CONSISTENCY_FLOOR = 1e-9


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InteractionResult:
    """Ranked feature-pair interaction strengths for one fitted model.

    JSON-serialisable (all fields are built-in types). Produced by
    :func:`interaction_strength`; consumed by ``/gbdt-diagnose`` and the agent
    loop bundle in a later phase. Reported with the no-lift-column convention
    (CLAUDE.md § "Reporting conventions") — raw magnitudes + a base reference,
    never a lift column.

    Attributes
    ----------
    top_pairs:
        Ranked list of ``(feature_a, feature_b, strength, sign_consistency)``,
        descending by ``strength`` (the aggregated pairwise interaction
        magnitude). For ``kind="shap"`` ``strength`` is the mean absolute
        interaction value over the rows used and ``sign_consistency`` is the
        fraction of rows on which the (signed) interaction points the dominant
        direction (∈ [0.5, 1.0]; ``nan`` for a pair whose magnitude is below the
        floor). For ``kind="cooccurrence"`` ``strength`` is the gain-weighted
        split co-occurrence count and ``sign_consistency`` is ``nan`` (the cheap
        cross-check has no signed per-row notion).
    per_feature_involvement:
        ``{feature: total_interaction_load}`` — Σ over pairs of the pairwise
        strength a feature participates in (the XGBoost analog of
        ``diagnose.py::interaction_involvement``). Mean |off-diagonal| for
        ``shap``.
    per_feature_main_effect:
        ``{feature: mean_abs_main_effect}`` — the mean |diagonal SHAP| per
        feature (``shap`` kind only; empty dict for ``cooccurrence``). Carried so
        the agent-loop pruning rule (drop only if low main **and** low
        interaction) is applicable straight off the result (D7).
    method:
        The ``kind`` that produced this result (``"shap"`` | ``"cooccurrence"``).
    n_rows_used:
        Number of rows the SHAP pass aggregated over (after the ``max_rows`` cap
        / sub-sample). ``0`` for ``cooccurrence`` (no rows are scored).
    n_features:
        Number of model features the ranking ranges over.
    """

    top_pairs: list[tuple[str, str, float, float]]
    per_feature_involvement: dict[str, float]
    per_feature_main_effect: dict[str, float] = field(default_factory=dict)
    method: str = "shap"
    n_rows_used: int = 0
    n_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (tuples → lists)."""
        d = asdict(self)
        d["top_pairs"] = [list(p) for p in self.top_pairs]
        return d

    def pair_strength(self, feat_a: str, feat_b: str) -> float:
        """Look up the aggregated strength for an unordered ``(feat_a, feat_b)``
        pair, or ``0.0`` if the pair is not in :attr:`top_pairs`."""
        key = frozenset((feat_a, feat_b))
        for a, b, strength, _sign in self.top_pairs:
            if frozenset((a, b)) == key:
                return float(strength)
        return 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def interaction_strength(
    model: "BaseGBDTModel",
    X: pd.DataFrame | np.ndarray,
    *,
    kind: str = "shap",
    top_n: int = 15,
    max_rows: int = _DEFAULT_MAX_ROWS,
    batch_size: int = 512,
    random_seed: int = 42,
) -> InteractionResult:
    """Rank pairwise feature interactions for a fitted XGBoost ``model``.

    Parameters
    ----------
    model:
        A fitted :class:`~gbdt.model.XGBoostModel` (the headline backend). The
        function reads the underlying booster through the model's public
        surface; it never touches ``model.py`` internals.
    X:
        The in-sample matrix to score. Pruned to the model's active feature set
        is the expected input (§ 3.3 mitigation 1); the full 279-pool dense
        tensor is infeasible. A ``DataFrame`` carries the feature names; a bare
        array falls back to the model's ``feature_names``.
    kind:
        ``"shap"`` (default, headline) — native TreeSHAP ``pred_interactions``,
        streamed pair aggregation (§ 3.3). ``"cooccurrence"`` — the near-free
        split-pair gain cross-check from the tree dump (ranking only).
    top_n:
        Number of top pairs to keep in :attr:`InteractionResult.top_pairs`.
    max_rows:
        Row cap for the SHAP pass (§ 3.3 mitigation 2). If ``len(X) > max_rows``
        the rows are sub-sampled (seeded by ``random_seed``) before scoring.
        Ignored for ``kind="cooccurrence"``.
    batch_size:
        Row-block size for the streamed SHAP aggregation. The dense
        ``(rows, F+1, F+1)`` tensor is never materialised — only one
        ``(batch_size, F+1, F+1)`` block at a time, plus the small
        ``(F+1, F+1)`` accumulators (§ 3.3 mitigation 3).
    random_seed:
        Seeds the row sub-sample so the ranking is reproducible.

    Returns
    -------
    InteractionResult
    """
    booster = _booster_of(model)
    feat_names = _feature_names_of(model, X)

    if kind == "shap":
        return _shap_interaction_strength(
            booster,
            X,
            feat_names,
            top_n=top_n,
            max_rows=max_rows,
            batch_size=batch_size,
            random_seed=random_seed,
        )
    if kind == "cooccurrence":
        return _cooccurrence_interaction_strength(
            booster, feat_names, top_n=top_n
        )
    raise ValueError(
        f"unknown interaction kind {kind!r}; expected 'shap' or 'cooccurrence'."
    )


# ---------------------------------------------------------------------------
# kind="shap": native TreeSHAP pred_interactions, streamed aggregation
# ---------------------------------------------------------------------------


def _shap_interaction_strength(
    booster,
    X: pd.DataFrame | np.ndarray,
    feat_names: list[str],
    *,
    top_n: int,
    max_rows: int,
    batch_size: int,
    random_seed: int,
) -> InteractionResult:
    """Streamed native-TreeSHAP pairwise interaction ranking.

    Uses ``booster.predict(dmatrix, pred_interactions=True)`` per row-block; the
    block is a ``(b, F+1, F+1)`` tensor (diagonal = main effects, off-diagonal =
    pairwise interaction attributions, last index = bias). We accumulate, per
    unordered pair ``(i<j)``: Σ|interaction| and — bucketed by the **sign
    quadrant** of the two interacting features — Σsigned and Σ|.| — never holding
    the full ``(rows, F+1, F+1)`` array (§ 3.3 mitigation 3). The pairwise SHAP
    interaction matrix is symmetric, so the ``(i,j)`` and ``(j,i)`` entries are
    summed (they are equal; together they give the conventional total pairwise
    attribution).

    **Sign-consistency** (plan § 3.2): the metric must distinguish a *stable*
    interaction from a *noisy averaging-to-zero* one. A globally-signed mean is
    the wrong primitive for a symmetric interaction like XOR, whose direction
    genuinely flips by quadrant (it averages to ~0.5 globally even though it is
    perfectly reproducible *given* the inputs). We therefore condition on the
    sign of the two features: within each of the four ``(sign x_i, sign x_j)``
    quadrants we compute ``|Σsigned| / Σ|.|`` (1.0 if the interaction always
    points the same way in that quadrant, 0.0 if it cancels), then report the
    **magnitude-weighted mean** across quadrants mapped onto ``[0.5, 1.0]``. A
    true XOR scores ~1.0 (perfectly directional per quadrant); white noise scores
    ~0.5. Pairs below the magnitude floor report ``nan``.
    """
    import xgboost as xgb

    X2 = _to_2d(X)
    n_total = X2.shape[0]
    F = len(feat_names)

    # Sub-sample rows down to the cap (seeded — reproducible). Sort the chosen
    # index so the stream order is deterministic regardless of permutation.
    if n_total > max_rows:
        rng = np.random.default_rng(random_seed)
        sel = np.sort(rng.choice(n_total, size=max_rows, replace=False))
        X2 = X2[sel]
    n_rows = X2.shape[0]

    # Accumulators over the (F+1)x(F+1) interaction matrix (incl. the bias row/col).
    dim = F + 1
    abs_sum = np.zeros((dim, dim), dtype=np.float64)
    # Per-(i,j)-pair, per-feature-sign-quadrant accumulators for sign-consistency.
    # quadrant index q ∈ {0,1,2,3} = 2*(x_i >= 0) + (x_j >= 0). Memory: 8·F²·4
    # bytes — small on the pruned active set; never the dense row tensor.
    q_signed = np.zeros((F, F, 4), dtype=np.float64)
    q_abs = np.zeros((F, F, 4), dtype=np.float64)

    for start in range(0, n_rows, batch_size):
        block = X2[start : start + batch_size]
        dm = xgb.DMatrix(block, feature_names=feat_names)
        # (b, F+1, F+1) — the only dense per-row object held; one block at a time.
        inter = np.asarray(
            booster.predict(dm, pred_interactions=True), dtype=np.float64
        )
        if inter.ndim != 3:
            raise RuntimeError(
                "booster.predict(pred_interactions=True) did not return a 3-D "
                f"tensor (got shape {inter.shape}); is this an XGBoost booster?"
            )
        abs_sum += np.abs(inter).sum(axis=0)

        # Sign-quadrant bucketing for this block (NaNs treated as "non-negative").
        block_sign = np.nan_to_num(block, nan=0.0) >= 0.0  # (b, F) bool
        for i in range(F):
            si = block_sign[:, i].astype(np.int64)  # (b,)
            for j in range(i + 1, F):
                sj = block_sign[:, j].astype(np.int64)
                quad = 2 * si + sj  # (b,) ∈ {0,1,2,3}
                # symmetric → (i,j)+(j,i) is the total pairwise attribution
                pair_vals = inter[:, i, j] + inter[:, j, i]  # (b,)
                np.add.at(q_signed[i, j], quad, pair_vals)
                np.add.at(q_abs[i, j], quad, np.abs(pair_vals))

    # Diagonal (excluding bias) = main effects; off-diagonal = pairwise.
    main_effect_mean = np.diag(abs_sum)[:F] / max(n_rows, 1)
    per_feature_main_effect = {
        feat_names[i]: float(main_effect_mean[i]) for i in range(F)
    }

    pairs: list[tuple[str, str, float, float]] = []
    involvement = {name: 0.0 for name in feat_names}
    for i in range(F):
        for j in range(i + 1, F):
            # symmetric matrix → (i,j) + (j,i) is the total pairwise attribution
            abs_total = abs_sum[i, j] + abs_sum[j, i]
            strength = abs_total / max(n_rows, 1)
            sign_consistency = _sign_consistency(q_signed[i, j], q_abs[i, j])
            pairs.append(
                (feat_names[i], feat_names[j], float(strength), float(sign_consistency))
            )
            involvement[feat_names[i]] += strength
            involvement[feat_names[j]] += strength

    pairs.sort(key=lambda p: p[2], reverse=True)
    return InteractionResult(
        top_pairs=pairs[:top_n],
        per_feature_involvement=involvement,
        per_feature_main_effect=per_feature_main_effect,
        method="shap",
        n_rows_used=n_rows,
        n_features=F,
    )


def _sign_consistency(q_signed: np.ndarray, q_abs: np.ndarray) -> float:
    """Magnitude-weighted, quadrant-conditioned sign-consistency for one pair.

    ``q_signed`` / ``q_abs`` are the length-4 per-sign-quadrant Σsigned / Σ|.|
    accumulators. Within each quadrant the directional purity is
    ``|Σsigned| / Σ|.|`` ∈ [0, 1] (1 = always same direction, 0 = cancels). The
    pair's consistency is the magnitude-weighted mean of the per-quadrant purity,
    mapped to ``[0.5, 1.0]`` so it reads on the same scale as a "fraction of rows
    pointing the dominant direction" statistic. Returns ``nan`` if the pair's
    total magnitude is below the floor (a dead pair — sign is meaningless noise).
    """
    total_abs = float(q_abs.sum())
    if total_abs <= _SIGN_CONSISTENCY_FLOOR:
        return math.nan
    purity = np.zeros(4, dtype=np.float64)
    nz = q_abs > _SIGN_CONSISTENCY_FLOOR
    purity[nz] = np.abs(q_signed[nz]) / q_abs[nz]
    weighted_purity = float((purity * q_abs).sum() / total_abs)
    return 0.5 * (1.0 + weighted_purity)


def shap_interaction_dense_reference(
    model: "BaseGBDTModel",
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Dense reference: the full mean-absolute pairwise-interaction matrix.

    Returns ``(M, feature_names)`` where ``M`` is the ``(F, F)`` matrix of
    ``mean_over_rows |interaction(i, j)| + |interaction(j, i)|`` for the model's
    features (bias row/col dropped). This is the **un-streamed** computation —
    it materialises the full ``(rows, F+1, F+1)`` tensor in one shot — and exists
    as the cross-check the streamed aggregator must match on a small matrix
    (plan § 8 Phase-4 test). Do **not** call it on a large panel; it is the thing
    the streamed path exists to avoid. The diagonal of ``M`` is zeroed (it is the
    main effect, not a pairwise interaction).
    """
    import xgboost as xgb

    booster = _booster_of(model)
    feat_names = _feature_names_of(model, X)
    X2 = _to_2d(X)
    dm = xgb.DMatrix(X2, feature_names=feat_names)
    inter = np.asarray(
        booster.predict(dm, pred_interactions=True), dtype=np.float64
    )  # (n, F+1, F+1) — dense, on purpose
    mean_abs = np.abs(inter).mean(axis=0)  # (F+1, F+1)
    F = len(feat_names)
    M = mean_abs[:F, :F] + mean_abs[:F, :F].T
    np.fill_diagonal(M, 0.0)
    return M, feat_names


# ---------------------------------------------------------------------------
# kind="cooccurrence": tree-dump split-pair gain cross-check (near-free)
# ---------------------------------------------------------------------------


def _cooccurrence_interaction_strength(
    booster,
    feat_names: list[str],
    *,
    top_n: int,
) -> InteractionResult:
    """Native split-pair co-occurrence ranking from the tree dump.

    For each tree, walk every root→leaf ancestor chain; every (ancestor,
    descendant) split-feature pair that co-occurs on a path is credited with the
    descendant split's gain (the analog of CatBoost ``type="Interaction"``). This
    is O(trees·depth²) and needs no scoring rows — the cheap cross-check (§ 3.2 /
    D1). Returns a ranking only; ``sign_consistency`` is ``nan`` and
    ``per_feature_main_effect`` is empty (no signed per-row notion).
    """
    df = booster.trees_to_dataframe()
    booster_names = list(booster.feature_names or [])
    # Map booster feature names ("f0".. or real names) → our display names.
    name_for = _booster_name_resolver(booster_names, feat_names)

    pair_gain: dict[frozenset, float] = {}
    involvement = {name: 0.0 for name in feat_names}

    for _tree_id, tree_df in df.groupby("Tree", sort=False):
        # Build the parent map for this tree from Yes/No child IDs.
        node_feature: dict[str, str] = {}
        parent_of: dict[str, str] = {}
        node_gain: dict[str, float] = {}
        for row in tree_df.itertuples(index=False):
            nid = row.ID
            feat = row.Feature
            node_feature[nid] = feat
            node_gain[nid] = float(row.Gain) if feat != "Leaf" else 0.0
            for child in (row.Yes, row.No):
                if isinstance(child, str):
                    parent_of[child] = nid

        # For each internal split node, credit the pair (this feature, ancestor
        # split feature) with this node's gain, walking up the ancestor chain.
        for nid, feat in node_feature.items():
            if feat == "Leaf":
                continue
            gain = node_gain[nid]
            seen_ancestor_feats: set[str] = set()
            anc = parent_of.get(nid)
            while anc is not None:
                anc_feat = node_feature.get(anc, "Leaf")
                if anc_feat != "Leaf" and anc_feat != feat:
                    if anc_feat not in seen_ancestor_feats:
                        seen_ancestor_feats.add(anc_feat)
                        a = name_for(anc_feat)
                        b = name_for(feat)
                        if a is not None and b is not None and a != b:
                            key = frozenset((a, b))
                            pair_gain[key] = pair_gain.get(key, 0.0) + gain
                anc = parent_of.get(anc)

    for key, g in pair_gain.items():
        a, b = tuple(key)
        involvement[a] += g
        involvement[b] += g

    pairs: list[tuple[str, str, float, float]] = []
    for key, g in pair_gain.items():
        a, b = tuple(key)
        pairs.append((a, b, float(g), math.nan))
    pairs.sort(key=lambda p: p[2], reverse=True)

    return InteractionResult(
        top_pairs=pairs[:top_n],
        per_feature_involvement=involvement,
        per_feature_main_effect={},
        method="cooccurrence",
        n_rows_used=0,
        n_features=len(feat_names),
    )


# ---------------------------------------------------------------------------
# interaction_constraints causal ablation (Phase-4 intervention experiment)
# ---------------------------------------------------------------------------


def ablate_interactions(
    model: "XGBoostModel",
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray | pd.Series,
    forbidden_pairs: list[tuple[str, str]],
    *,
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: np.ndarray | pd.Series | None = None,
    train_weight: np.ndarray | pd.Series | None = None,
    val_weight: np.ndarray | pd.Series | None = None,
    early_stopping_rounds: int | None = None,
    feature_names: list[str] | None = None,
    random_seed: int = 42,
) -> "XGBoostModel":
    """Retrain an :class:`~gbdt.model.XGBoostModel` with ``forbidden_pairs``
    forbidden from co-splitting, via XGBoost ``interaction_constraints``.

    The Phase-4 *causal-ablation* tool (plan § 3.2 / § 8): take the top-K SHAP
    interaction pairs, forbid the model from combining them in a single tree
    path, retrain, and confirm (in the caller) that the measured interaction on
    those pairs drops ~0 and Brier degrades — closing the loop between *measured*
    interaction strength and *causal* contribution to predictability.

    **This is an intervention experiment, never a hot-loop knob**
    (``project-xgboost-training-essentials`` § 3): ``interaction_constraints``
    stays out of the FS+HP decision schema. This helper is tooling so a caller /
    test can run the ablation; it does **not** run any real experiment.

    The returned model is a fresh :class:`~gbdt.model.XGBoostModel` carrying the
    same HP as ``model`` plus the derived ``interaction_constraints``, refit on
    ``(X_train, y_train)``. ``interaction_constraints`` is **not** a tunable HP,
    so it is injected on the fresh wrapper after construction (bypassing the HP
    validation table) — by design, since the ablation is an out-of-band
    intervention. The constraint groups are emitted as a JSON string of
    **integer feature indices** (not names), so the constraint resolves
    correctly even though ``XGBoostModel.fit`` feeds the booster a name-less
    numpy matrix.

    Parameters
    ----------
    model:
        The fitted reference :class:`~gbdt.model.XGBoostModel` whose HP +
        features the ablation reuses.
    X_train, y_train:
        Training data to retrain on (same split the reference saw).
    forbidden_pairs:
        Unordered ``(feature_a, feature_b)`` pairs to forbid. Feature names must
        be in the model's feature set.
    X_val, y_val, train_weight, val_weight, early_stopping_rounds:
        Forwarded to :meth:`~gbdt.model.XGBoostModel.fit` (same semantics).
    feature_names:
        Override the feature-name list (else taken from ``model`` / the training
        ``DataFrame``).
    random_seed:
        Seed for the fresh model (kept equal to the reference for a controlled
        comparison).

    Returns
    -------
    XGBoostModel
        The retrained, interaction-constrained model.
    """
    from gbdt.model import XGBoostModel

    feat_names = feature_names or _feature_names_of(model, X_train)
    feat_index = {name: i for i, name in enumerate(feat_names)}

    forbidden_set: set[frozenset] = set()
    for a, b in forbidden_pairs:
        if a not in feat_index or b not in feat_index:
            raise ValueError(
                f"forbidden pair ({a!r}, {b!r}) references a feature not in the "
                f"model's feature set {feat_names!r}."
            )
        if a != b:
            forbidden_set.add(frozenset((a, b)))

    constraints = _interaction_constraints_from_forbidden(feat_names, forbidden_set)

    # Build a fresh wrapper carrying the reference HP (drop the seed spellings
    # so the fresh constructor re-derives them from random_seed). interaction_
    # constraints is injected post-construction — it is an out-of-band
    # intervention, not a validated HP.
    base_hp = {
        k: v
        for k, v in model.hp.items()
        if k not in ("seed", "random_state", "verbosity")
    }
    fresh = XGBoostModel(
        base_hp, feature_names=list(feat_names), random_seed=random_seed
    )
    fresh._hp["interaction_constraints"] = constraints

    fresh.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        early_stopping_rounds=early_stopping_rounds,
        train_weight=train_weight,
        val_weight=val_weight,
    )
    return fresh


def _interaction_constraints_from_forbidden(
    feat_names: list[str],
    forbidden_set: set[frozenset],
) -> str:
    """Build an XGBoost ``interaction_constraints`` spec that forbids exactly the
    pairs in ``forbidden_set`` from co-splitting (all other pairs stay free).

    XGBoost ``interaction_constraints`` is a list of *allowed* groups: two
    features may share a root→leaf path iff they appear together in **some**
    group. The safe, always-correct construction is the **allowed-edge list** —
    emit one 2-feature group per *allowed* pair (every pair except the forbidden
    ones). A forbidden pair then appears in no group and XGBoost will not
    co-split it, while every other pair has its own enabling group. (A
    per-feature "everything I may interact with" group is *wrong* — feature C's
    group ``[C, A, B, D]`` would silently re-permit the forbidden ``(A, B)``
    pair because they co-occur inside C's group.) Any feature that ends up in no
    allowed pair gets a singleton group so it can still be used as a standalone
    split.

    Groups are emitted as a **JSON string of integer feature indices** (not
    names): ``XGBoostModel.fit`` feeds the booster a name-less numpy matrix
    (``_to_2d``), and XGBoost's name-based constraint resolution fails on a
    name-less booster — the integer-index JSON form is resolved positionally and
    works regardless.
    """
    n = len(feat_names)
    if not forbidden_set:
        # No forbidden pairs → one all-features group (unconstrained).
        return json.dumps([list(range(n))])

    index_of = {name: i for i, name in enumerate(feat_names)}
    groups: list[list[int]] = []
    in_a_group = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            pair = frozenset((feat_names[i], feat_names[j]))
            if pair in forbidden_set:
                continue
            groups.append([i, j])
            in_a_group[i] = True
            in_a_group[j] = True
    # Singletons for any feature isolated by the forbidden set (so it can still
    # be split on, just never co-split with the feature(s) it is forbidden from).
    for i in range(n):
        if not in_a_group[i]:
            groups.append([i])
    return json.dumps(groups)


# ---------------------------------------------------------------------------
# Optional viz helper — soft `shap` import, NEVER on the core path
# ---------------------------------------------------------------------------


def shap_interaction_summary_plot(*args, **kwargs):  # pragma: no cover - viz only
    """Optional visualisation via the external ``shap`` package.

    The ``shap`` package is deliberately **not** a hard dependency (V1.2 plan R4
    / D1): native TreeSHAP via the booster is the headline measurement and the
    core path of this module never imports ``shap``. This helper exists only for
    a future visualisation surface and lazily imports ``shap`` — raising a clear
    error if it is not installed. No core function calls it.
    """
    try:
        import shap  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "shap is an optional viz-only dependency and is not installed; "
            "the core interaction measurement uses native TreeSHAP via the "
            "XGBoost booster and never needs it. Install `shap` only if you "
            "want the package's plotting surface."
        ) from exc
    return shap.summary_plot(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers — model/booster access through the public surface
# ---------------------------------------------------------------------------


def _booster_of(model: "BaseGBDTModel"):
    """Return the fitted XGBoost ``Booster`` behind an :class:`XGBoostModel`.

    Reads through the model's public surface only — ``XGBoostModel`` wraps an
    ``xgboost.XGBClassifier`` (``model._model``) whose ``get_booster()`` is the
    booster the native-TreeSHAP / tree-dump methods need. Raises a clear error if
    ``model`` is not a fitted XGBoost backend (the headline path is XGBoost-only;
    CatBoost SHAP-interaction parity is a later phase, not Phase 4).
    """
    if not getattr(model, "fitted", False):
        raise RuntimeError("model is not fitted")
    estimator = getattr(model, "_model", None)
    get_booster = getattr(estimator, "get_booster", None)
    if get_booster is None:
        raise TypeError(
            "interaction_strength requires a fitted XGBoostModel (the native "
            "TreeSHAP backend); got a model whose estimator has no "
            "get_booster(). CatBoost SHAP-interaction parity is out of Phase-4 "
            "scope."
        )
    return get_booster()


def _feature_names_of(
    model: "BaseGBDTModel", X: pd.DataFrame | np.ndarray
) -> list[str]:
    """Resolve display feature names: prefer ``X``'s columns, then the model's
    ``feature_names``, else positional ``f0..fN``."""
    if isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    names = getattr(model, "feature_names", None)
    if names:
        return [str(n) for n in names]
    n = _to_2d(X).shape[1]
    return [f"f{i}" for i in range(n)]


def _booster_name_resolver(booster_names: list[str], feat_names: list[str]):
    """Return a function mapping a booster split-feature label to a display name.

    XGBoost's ``trees_to_dataframe`` labels splits by the booster's own feature
    names — either the real names (if the model was fit on a named ``DataFrame``)
    or positional ``f0..fN`` (if fit on a bare array). Map both back to our
    ``feat_names`` display list.
    """
    booster_to_display: dict[str, str] = {}
    for i, name in enumerate(feat_names):
        if i < len(booster_names):
            booster_to_display[booster_names[i]] = name
        booster_to_display.setdefault(f"f{i}", name)
        booster_to_display.setdefault(name, name)

    def resolve(label: str) -> str | None:
        if label in booster_to_display:
            return booster_to_display[label]
        # Positional fallback "f<idx>".
        if label.startswith("f") and label[1:].isdigit():
            idx = int(label[1:])
            if 0 <= idx < len(feat_names):
                return feat_names[idx]
        return None

    return resolve


def _to_2d(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


__all__ = [
    "InteractionResult",
    "interaction_strength",
    "shap_interaction_dense_reference",
    "ablate_interactions",
    "shap_interaction_summary_plot",
]
