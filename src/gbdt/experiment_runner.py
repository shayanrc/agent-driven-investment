"""The gbdt experiment orchestrator — ``run_experiment`` plus its
provenance, preflight, and emission helpers.

Extracted verbatim from ``gbdt.__main__`` (the v1 Stage 8 orchestrator) by
the runner split — behavior unchanged. ``gbdt.__main__`` stays the CLI and
re-exports these names; ``gbdt.experiment`` stays the documented
per-experiment entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss

from gbdt import checkpoint as gbdt_checkpoint
from gbdt import data as gbdt_data
from gbdt import feature_cache as gbdt_feature_cache
from gbdt import features as gbdt_features
from gbdt import loop_observability
from gbdt import loop_protocol
from gbdt import preflight as gbdt_preflight
from gbdt import scout_io as gbdt_scout_io
from gbdt import universe_calendar as gbdt_universe_calendar
from gbdt import universe_feature_cache as gbdt_universe_feature_cache
from gbdt.agent_cycles import (
    _build_scout_metrics_blocks,
    _handle_scout_cycles_agent_mode,
    _load_and_apply_resume,
    _resolve_callback,
)
from gbdt.heartbeat import Heartbeat
from gbdt.model import count_nonfinite, model_filename
from gbdt.report import compute_segment_diagnostics, emit_figures, render_report
from gbdt.spec import (
    _DEFAULT_CALLBACK_MODE,
    _DEFAULT_DEGENERATE_SINK_THRESHOLD,
    _DEFAULT_SWEEP_CSV_RELPATH,
    _HP_SEARCH_ITER_THRESHOLD,
    _TEST_ROWS_WARNING_THRESHOLD,
    _VALID_CALLBACK_MODES,
    _spec_hash,
    load_spec,
)
from gbdt.sweep_lookup import (
    cell_key_to_experiment_name,
    lookup_sweep_row,
)
from gbdt.targets import build_target
from gbdt.train import SplitSpec, walk_forward_train
from gbdt.uniqueness import (
    compute_uniqueness_weights,
    effective_sample_size,
    weighted_auc,
    weighted_brier,
)


def _has_runner_artifacts(out_dir: Path) -> bool:
    """Return True if ``out_dir`` contains anything the runner would treat
    as a prior artifact (so a fresh launch without ``--overwrite`` must
    refuse to clobber it).

    Dotfile entries are ignored: they belong to supervising tools (e.g.
    the ``.wrapper/`` sidecar dir written by
    ``scripts/gbdt/run_agent_loop_resumable.sh`` post-#193 bug 2), not to
    the runner. Treating a dir as "empty" when only dotfiles are present
    lets the wrapper write its state before invoking the runner without
    forcing callers to pass ``--overwrite``.
    """
    return any(p for p in out_dir.iterdir() if not p.name.startswith("."))


def _clear_stale_loop_decisions(out_dir: Path) -> list[Path]:
    """Remove agent-authored decision files left over from a prior run.

    On ``--overwrite`` a fresh run is intended: the runner re-trains iter 0 and
    rewrites the loop ``checkpoint.json`` + ``iter_0_request.json``, but it never
    rewrites the **decision** files — those are agent-authored, so nothing on the
    runner side overwrites them. A leftover
    ``loop/iter_<N>_decision.json`` (or ``scout/{combine,iter_0}_decision.json``)
    would then be read by the next ``--resume`` and silently re-applied — its
    old ``hp_changes`` / ``prune_features`` contaminating a run that was meant to
    start clean. Deleting them makes the loop dir pristine so a bare ``--resume``
    fails fast ("decision file not found") until the agent writes a fresh
    decision. ``--resume`` itself never clears — it depends on the decision the
    agent just authored.

    All ``iter_*_decision.json`` are removed (not just ``iter_0``): a prior run
    that reached iter N left decisions 0..N, and the loop pauses again at each
    iteration, so a surviving ``iter_1_decision.json`` would re-contaminate the
    resume one iteration later.

    Returns the list of paths actually removed. Best-effort per file: a removal
    error is logged, not raised — an otherwise-fine overwrite must not crash over
    a decision file we couldn't unlink.
    """
    loop_dir = loop_protocol.decision_path(out_dir, 0).parent
    stale = sorted(loop_dir.glob("iter_*_decision.json"))
    for path in (
        gbdt_scout_io.combine_decision_path(out_dir),
        gbdt_scout_io.iter_0_decision_path(out_dir),
    ):
        if path.exists():
            stale.append(path)

    removed: list[Path] = []
    for path in stale:
        try:
            path.unlink()
            removed.append(path)
            print(
                f"[overwrite] cleared stale decision file "
                f"{path.parent.name}/{path.name}",
                flush=True,
            )
        except OSError as exc:
            print(
                f"[overwrite] WARNING: could not remove stale decision file "
                f"{path} ({exc!r}); a subsequent --resume may re-apply it",
                flush=True,
            )
    return removed


def _project_test_rows(
    panel: pd.DataFrame,
    *,
    test_rows_per_ticker: int,
    horizon_days: int,
) -> dict:
    """Estimate the number of usable rows the test segment will yield.

    The walk-forward driver carves the trailing ``test_rows`` positional
    rows per ticker as the test segment. The target builder produces
    ``NaN`` for the last ``horizon_days`` rows per ticker (forward window
    incomplete), and the driver drops NaN-target rows before scoring. So
    the expected usable test row count per ticker is
    ``max(0, test_rows - horizon_days)``, summed across kept tickers.

    Returns a dict with ``expected_test_rows``, ``per_ticker_usable``,
    and ``n_tickers`` so the caller can compose a human-readable warning.
    """
    n_tickers = int(panel.index.get_level_values("ticker").nunique())
    per_ticker_usable = max(0, int(test_rows_per_ticker) - int(horizon_days))
    expected = per_ticker_usable * n_tickers
    return {
        "expected_test_rows": int(expected),
        "per_ticker_usable": int(per_ticker_usable),
        "n_tickers": int(n_tickers),
        "test_rows_per_ticker": int(test_rows_per_ticker),
        "horizon_days": int(horizon_days),
    }


def _format_test_split_warning(projection: dict, threshold: int) -> str:
    """One-line, human-readable explanation of the test_rows projection."""
    per = projection["per_ticker_usable"]
    h = projection["horizon_days"]
    tpt = projection["test_rows_per_ticker"]
    n = projection["n_tickers"]
    exp = projection["expected_test_rows"]
    if per == 0:
        return (
            f"Test segment expected to be EMPTY: horizon_days={h} >= "
            f"split.test_rows={tpt}, so every ticker's trailing {tpt} rows "
            f"have NaN targets (forward window incomplete). "
            f"headline_test will be {{}} and predictions/test.csv will be "
            f"header-only. Eval segment is still measured. "
            f"(threshold={threshold})"
        )
    return (
        f"Test segment will be SMALL: per-ticker usable = "
        f"max(0, test_rows={tpt} - horizon_days={h}) = {per}; "
        f"{n} kept ticker(s) -> expected ~{exp} rows < threshold={threshold}. "
        f"Headline_test will be computed but may be unreliable."
    )


def _collect_preflight(repo_root: Path) -> dict:
    """Capture a fingerprint of the cache + code state at run start.

    Six fields, captured BEFORE any data is loaded so the fingerprint
    survives even when the data stage fails:

    - ``cache_db``: resolved absolute path to ``<repo_root>/data/processed.db``.
      Symlinks are followed (via ``os.path.realpath``) so a run reading
      from a tmpfs-backed cache mounted at ``data/`` records the real
      target path. Empty string if the file does not exist.
    - ``cache_db_size``: size in bytes; ``0`` if missing.
    - ``cache_db_mtime``: UTC ISO-8601 mtime; empty string if missing.
    - ``data_root``: resolved absolute path to ``<repo_root>/data``
      (the directory data_pipelines was rooted at). Same realpath rule.
    - ``code_commit``: ``git rev-parse HEAD`` output, or ``"unknown"``
      when git is unavailable / repo_root is not a git checkout.
    - ``code_dirty``: ``True`` when ``git status --porcelain`` is
      non-empty, else ``False``. Defaults to ``False`` when git is
      unavailable (matches the ``code_commit="unknown"`` fail-safe).

    Motivated by the ``/tmp/exp_data`` tmpfs wipe (May 27): two runs on
    consecutive dates reading from a transient cache looked identical
    in their artifacts but in fact consumed different cache snapshots.
    Persisting these six fields into the artifact (``metrics.json``)
    makes archived runs self-describing post-hoc.
    """
    data_root = Path(repo_root) / "data"
    cache_db = data_root / "processed.db"

    data_root_resolved = os.path.realpath(data_root)
    if cache_db.exists():
        cache_db_resolved = os.path.realpath(cache_db)
        try:
            st = os.stat(cache_db_resolved)
            cache_db_size = int(st.st_size)
            cache_db_mtime = datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc,
            ).isoformat()
        except OSError:
            cache_db_size = 0
            cache_db_mtime = ""
    else:
        cache_db_resolved = ""
        cache_db_size = 0
        cache_db_mtime = ""

    code_commit = "unknown"
    code_dirty = False
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if rev.returncode == 0:
            commit = rev.stdout.strip()
            if commit:
                code_commit = commit
                status = subprocess.run(
                    ["git", "-C", str(repo_root), "status", "--porcelain"],
                    capture_output=True, text=True, check=False, timeout=5,
                )
                if status.returncode == 0:
                    code_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # ``git`` binary missing, perms denied, hung, etc. — preserve the
        # ``unknown``/``False`` fail-safe and continue. Never crash a run.
        pass

    return {
        "cache_db": cache_db_resolved,
        "cache_db_size": cache_db_size,
        "cache_db_mtime": cache_db_mtime,
        "data_root": data_root_resolved,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
    }


def _sanitize_path_for_emission(p: str | os.PathLike, repo_root: Path) -> str:
    """Rewrite an absolute path as repo-relative for emission to logs / metrics.

    Paths inside the repo become repo-relative (``"results/gbdt/experiments/..."``,
    ``"data/processed.db"``). Paths outside the repo — typically the scratch
    cache under ``/mnt/<UUID>/cache_data`` per the worktree-symlink contract —
    collapse to ``"<external>/<basename>"``. The basename is preserved so
    post-hoc audits can still distinguish e.g. ``processed.db`` from a sibling,
    but the host-specific prefix never reaches committed artifacts.

    Apply at every ``_milestone(...)`` call site that interpolates a path so
    ``loop/progress.log`` (a committed artifact under
    ``results/gbdt/experiments/<cell>/loop/``) stays host-agnostic. Empty /
    ``None`` input returns ``""`` without raising.
    """
    if not p:
        return ""
    p_str = str(p)
    try:
        rel = os.path.relpath(p_str, repo_root)
    except ValueError:
        rel = ".."  # different drive on Windows; fall through to external
    if rel.startswith(".."):
        return "<external>/" + os.path.basename(p_str)
    return rel


def _sanitize_preflight_for_emission(pf: dict, repo_root: Path) -> dict:
    """Return a copy of ``pf`` with absolute paths rewritten as repo-relative.

    The raw ``preflight`` dict carries ``os.path.realpath``-resolved paths
    for ``cache_db`` and ``data_root`` (downstream consumers — universe
    feature cache, FS-prefit cache — pass them to ``Path()``). Those
    realpaths point inside the per-machine cache mount (e.g.
    ``/mnt/<UUID>/cache_data``) and would leak the host's partition layout
    into committed ``metrics.json`` files and run logs.

    This helper produces an emission-only view via
    :func:`_sanitize_path_for_emission`. The raw ``pf`` continues to carry
    realpaths for live consumers.

    Apply at every serialization site (the ``[preflight]`` log line and
    the ``metrics.json::preflight`` write).
    """
    out = dict(pf)
    for key in ("cache_db", "data_root"):
        if out.get(key, ""):
            out[key] = _sanitize_path_for_emission(out[key], repo_root)
    return out


def _format_preflight_line(pf: dict, repo_root: Path) -> str:
    pf = _sanitize_preflight_for_emission(pf, repo_root)
    return (
        f"[preflight] cache_db={pf['cache_db']} "
        f"cache_db_size={pf['cache_db_size']} "
        f"cache_db_mtime={pf['cache_db_mtime']} "
        f"data_root={pf['data_root']} "
        f"code_commit={pf['code_commit']} "
        f"code_dirty={pf['code_dirty']}"
    )


def _data_hash(panel: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(panel.shape).encode())
    h.update(str(panel.index[:5].tolist()).encode())
    h.update(str(panel.index[-5:].tolist()).encode())
    return "sha256:" + h.hexdigest()


def _compute_headline(pred_df: pd.DataFrame | None) -> dict:
    """Headline metrics on a prediction segment.

    Uses LdP §4.4 sample weights from the ``sample_weight`` column when
    present (uniform-1.0 fallback collapses to unweighted metrics so
    legacy callers and the opt-out path produce numerically-identical
    outputs).
    """
    if pred_df is None or pred_df.empty:
        return {}
    y = pred_df["y_true"].values.astype(int)
    p = pred_df["p_calibrated"].values
    if "sample_weight" in pred_df.columns:
        w = pred_df["sample_weight"].values.astype(float)
    else:
        w = np.ones_like(y, dtype=float)

    total_w = float(w.sum())
    base = float(np.sum(w * y) / total_w) if total_w > 0 else float(np.mean(y))
    brier = weighted_brier(y, p, w)
    brier_base = weighted_brier(y, np.full_like(y, base, dtype=float), w)
    # Unweighted variants kept for backward-compat / cross-check.
    brier_unw = float(brier_score_loss(y, p))
    brier_base_unw = float(brier_score_loss(
        y, np.full_like(y, float(np.mean(y)), dtype=float),
    ))
    # log_loss has a sample_weight kwarg.
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=w))
    out = {
        "brier": float(brier),
        "brier_baseline_baserate": float(brier_base),
        "brier_improvement_vs_baseline": float(brier_base - brier),
        "log_loss": ll,
        "brier_unweighted": brier_unw,
        "brier_baseline_baserate_unweighted": brier_base_unw,
        "brier_improvement_vs_baseline_unweighted": brier_base_unw - brier_unw,
        "effective_sample_size_kish": float(effective_sample_size(w)),
        "sum_weights": float(w.sum()),
        "n_rows": int(len(y)),
        "weighted_prevalence": base,
    }
    out["roc_auc"] = weighted_auc(y, p, w)
    return out



def run_experiment(spec_path: Path, *, overwrite: bool = False,
                    callback_mode_override: str | None = None,
                    resume: str | None = None,
                    snapshot_end: date | None = None,
                    repo_root: Path | None = None) -> Path:
    """Run the experiment end-to-end. Returns the artifact dir path.

    ``callback_mode_override`` (CLI ``--callback-mode``): when set, overrides
    ``backend.fs_hp_loop.callback_mode`` from the spec. Validated against
    :data:`_VALID_CALLBACK_MODES`.

    ``resume`` (CLI ``--resume <run_id>``): V1.1 Phase 1 scaffolding only.
    Accepted + logged; the exit-and-resume control flow lands in Phase 2.

    ``snapshot_end`` (CLI ``--snapshot-end YYYY-MM-DD``): when set, pins the
    spec's ``date_range.end`` for the lifetime of this run. Sweep
    orchestrators MUST pass the SAME value to every cell so the universe-
    level feature cache key stays stable across cells (an auto-fetch between
    cells otherwise drifts ``panel_signature`` and forces a cold rebuild on
    every sibling). Persisted into ``metrics.json::preflight.snapshot_end_override``
    for post-hoc audit. See bug #226.
    """
    spec_path = Path(spec_path).resolve()
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

    # Pre-flight fingerprint — captured BEFORE spec load / artifact dir
    # checks / data load so even an aborted run leaves a trail in stdout.
    # Persisted into ``metrics.json::preflight`` below for post-hoc audit.
    preflight = _collect_preflight(repo_root)
    print(_format_preflight_line(preflight, repo_root), flush=True)

    spec = load_spec(spec_path, default_path=repo_root / "configs/gbdt/default.yaml")
    name = spec_path.stem

    # V1.1 — CLI ``--callback-mode`` overrides the spec's
    # ``backend.fs_hp_loop.callback_mode`` (and the snapshotted value, since we
    # mutate the merged spec in place before the snapshot below). Validated the
    # same way as the spec-level field.
    if callback_mode_override is not None:
        if callback_mode_override not in _VALID_CALLBACK_MODES:
            raise ValueError(
                f"--callback-mode must be in {sorted(_VALID_CALLBACK_MODES)}, "
                f"got {callback_mode_override!r}"
            )
        spec.setdefault("backend", {}).setdefault("fs_hp_loop", {})[
            "callback_mode"
        ] = callback_mode_override
        # Mirror the override into the per-experiment snapshot source (issue
        # #30) so the persisted spec.yaml reflects the *effective* callback
        # mode this run actually used, not the on-disk default.
        per_exp = spec.get("__per_experiment_spec__")
        if isinstance(per_exp, dict):
            per_exp.setdefault("backend", {}).setdefault("fs_hp_loop", {})[
                "callback_mode"
            ] = callback_mode_override

    # Bug #226 — CLI ``--snapshot-end`` pins ``date_range.end`` for the
    # lifetime of this run. Applied BEFORE Phase 0b (cache_currency_check)
    # and BEFORE ``load_panel`` so the universe-level feature cache key is
    # computed against the pinned snapshot (every sweep cell sees the same
    # ``panel_signature`` ⇒ shared universe-cache hit). Mirrored into the
    # per-experiment snapshot source so ``spec.yaml`` reflects the
    # *effective* end-date this run actually used. ``snapshot_end_override``
    # is also persisted into ``metrics.json::preflight`` below for audit;
    # ``None`` ⇒ no override (the spec's value is used unchanged).
    snapshot_end_override_iso: str | None = None
    if snapshot_end is not None:
        snapshot_end_override_iso = snapshot_end.isoformat()
        prior_end = (spec.get("date_range") or {}).get("end")
        prior_end_repr = (
            prior_end.isoformat() if isinstance(prior_end, date)
            else (str(prior_end) if prior_end is not None else None)
        )
        spec.setdefault("date_range", {})["end"] = snapshot_end
        per_exp = spec.get("__per_experiment_spec__")
        if isinstance(per_exp, dict):
            per_exp.setdefault("date_range", {})["end"] = snapshot_end
        print(
            f"[data] snapshot-end pinned to {snapshot_end_override_iso} "
            f"(override; spec said: {prior_end_repr})",
            flush=True,
        )
    preflight["snapshot_end_override"] = snapshot_end_override_iso

    out_root = repo_root / spec.get("artifacts", {}).get(
        "experiment_dir", "results/gbdt/experiments"
    )
    out_dir = Path(out_root) / name
    # On --resume the artifact dir is EXPECTED to exist (it holds the prior
    # iteration's loop/ request + checkpoint), so the non-empty-dir guard is
    # bypassed. A fresh run still refuses to clobber a non-empty dir.
    if (
        resume is None
        and out_dir.exists()
        and _has_runner_artifacts(out_dir)
        and not overwrite
    ):
        print(f"[experiment] artifact dir already exists at {out_dir}", file=sys.stderr)
        print("[experiment] pass --overwrite to replace", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --overwrite means a FRESH run: purge any agent-authored loop/scout
    # decision files left behind by a prior run of this same cell so a later
    # --resume can't silently re-apply the old HP schedule (the runner rewrites
    # checkpoint.json + iter_0_request.json, but never the decision files).
    # Gated on ``resume is None`` because a --resume run legitimately depends on
    # the decision the agent just wrote; only a fresh overwrite clears.
    if overwrite and resume is None:
        _clear_stale_loop_decisions(out_dir)

    # Persistent agent-loop observability (task #177): an APPEND-only
    # ``loop/progress.log`` (survives across the separate-process resume model)
    # + an OVERWRITE ``loop/status.json`` (single-shot machine-readable
    # position + liveness). The progress log + a ``_milestone`` helper mirror
    # the runner's existing milestone prints to BOTH stdout (unchanged) and the
    # log; the heartbeat is teed into the log and refreshes status.json's
    # ``last_heartbeat_utc`` each tick. Both are best-effort — a write error
    # never crashes the run. See ``gbdt.loop_observability``.
    progress_log = loop_observability.ProgressLog(out_dir)
    status = loop_observability.StatusFile(out_dir, run_id=name)

    def _milestone(message: str) -> None:
        """Print to stdout (unchanged) AND append to ``loop/progress.log``."""
        print(message, flush=True)
        progress_log.append(message)

    _milestone(
        f"[experiment] start spec={spec_path.name} -> "
        f"{_sanitize_path_for_emission(out_dir, repo_root)}"
    )

    # V1.3 Option B (P4) — agent_file_protocol scout cycles use
    # ``scout/`` files (not the V1.1 loop checkpoint). When a ``--resume``
    # lands in cycle 2 (combine_decision present, no loop checkpoint yet)
    # or cycle 3 (iter_0_decision present, still no checkpoint), we MUST
    # NOT enter the V1.1 ``_load_and_apply_resume`` path — it would refuse
    # for a missing checkpoint. Distinguish by checking whether the V1.1
    # checkpoint exists; if not, we're in a scout cycle.
    _spec_backend = spec.get("backend", {}) or {}
    _spec_loop = _spec_backend.get("fs_hp_loop", {}) or {}
    _spec_scout = _spec_backend.get("scout", {}) or {}
    _is_agent_mode = (
        _spec_loop.get("callback_mode", _DEFAULT_CALLBACK_MODE)
        == "agent_file_protocol"
    )
    _scout_enabled_in_spec = bool(_spec_scout.get("enabled", False))
    _v1_1_ckpt_exists = gbdt_checkpoint.read_checkpoint(out_dir) is not None
    _in_scout_cycle = (
        resume is not None
        and _is_agent_mode
        and _scout_enabled_in_spec
        and not _v1_1_ckpt_exists
    )

    # PR #122 contract — `--resume` MUST re-pass `--snapshot-end` when the
    # spec uses agent_file_protocol mode AND has scout enabled (V1.3 Option B
    # scout cycles MUST keep universe-cache keys stable across exits). The
    # pre-V1.3 Option B agent loop doesn't strictly require it (existing
    # tests resume without --snapshot-end), so we only enforce when scout
    # is part of the picture.
    if (
        resume is not None
        and _is_agent_mode
        and _scout_enabled_in_spec
        and snapshot_end is None
    ):
        progress_log.close()
        raise ValueError(
            "[resume] --snapshot-end is REQUIRED when --resume targets an "
            "agent_file_protocol spec with scout enabled (V1.3 Option B "
            "P4 contract). Pass the same --snapshot-end value the fresh "
            "run used so universe-cache keys stay stable across cycles."
        )

    # V1.1 exit-and-resume (plan § 0): load the prior checkpoint + the agent's
    # decision, validate + apply it, and build the ``resume_state`` that seeds
    # ``walk_forward_train`` at iter N+1 (without re-training 0..N). ``None``
    # when this is a fresh run.
    resume_state: dict | None = None
    if resume is not None and not _in_scout_cycle:
        try:
            resume_state = _load_and_apply_resume(
                out_dir, spec, run_id=resume, progress_log=progress_log,
            )
        except Exception:
            # A rejected/missing decision (DecisionError) etc. surfaces to the
            # caller. Close the log handle first so the file is flushed + not
            # leaked — state on disk (checkpoint/request) is untouched, so the
            # user fixes the decision file and relaunches --resume (which
            # re-opens progress.log in append mode and continues the same file).
            progress_log.close()
            raise
        status.update(
            iter_idx=int(resume_state["iter_idx"]),
            phase="resume",
            awaiting_decision=False,
        )
    t0 = time.time()

    # Liveness heartbeat: emits [heartbeat] lines on a fixed cadence so a
    # stalled run is detectable (a stale heartbeat = wedged process) without a
    # tight timeout. Daemon thread — dies with the process; stopped explicitly
    # on the normal path below. Disable via GBDT_HEARTBEAT_INTERVAL=0.
    #
    # task #177: the heartbeat's ``stream`` is a TeeStream so every [heartbeat]
    # line also lands in progress.log, and ``on_tick`` refreshes status.json's
    # last_heartbeat_utc. When disabled (interval=0) the thread never runs, so
    # neither side-effect fires — the file heartbeat no-ops automatically.
    heartbeat = Heartbeat.from_env(
        stream=loop_observability.TeeStream(sys.stdout, progress_log),
        on_tick=status.heartbeat,
    ).start()

    # -------- Phase 1: data --------
    target = spec["target"]
    dr = spec.get("date_range", {}) or {}
    split_d = spec.get("split", {}) or {}
    # V1.4 (date_aligned splits, plan §3 D1/D2/D7): mode defaults to
    # 'trailing' (existing specs unchanged). 'date_aligned' anchors every
    # segment to universe-calendar dates from ``train_start`` (default
    # 2019-01-01 per D2). ``min_train_rows_per_ticker`` is the per-ticker
    # validity gate on the train segment (= max(lookback_windows) = 200
    # per D1).
    split_mode = str(split_d.get("mode", "trailing"))
    if split_mode not in ("trailing", "date_aligned"):
        raise ValueError(
            f"spec.split.mode must be 'trailing' or 'date_aligned', got "
            f"{split_mode!r}"
        )
    train_start_raw = split_d.get("train_start")
    train_start_val: date | None = None
    if train_start_raw is not None:
        if isinstance(train_start_raw, date):
            train_start_val = train_start_raw
        else:
            train_start_val = date.fromisoformat(str(train_start_raw)[:10])
    if split_mode == "date_aligned" and train_start_val is None:
        # V1.4 D2: canonical anchor for new date-aligned cells.
        train_start_val = date(2019, 1, 1)
    if "mode" not in split_d:
        # D7: info-log nudge per run when mode is unspecified.
        print(
            "[split] info: spec.split.mode unspecified, defaulting to "
            "'trailing'. New work should prefer 'date_aligned' (V1.4) — "
            "see docs/gbdt/V1.4_date_aligned_splits_plan.md.",
            flush=True,
        )
    split = SplitSpec(
        train_rows=split_d.get("train_rows", 800),
        val_rows=split_d.get("val_rows", 400),
        eval_rows=split_d.get("eval_rows", 200),
        test_rows=split_d.get("test_rows", 100),
        mode=split_mode,
        train_start=train_start_val,
        min_train_rows_per_ticker=int(split_d.get(
            "min_train_rows_per_ticker", 200,
        )),
    )
    min_rows = split_d.get("min_rows_per_ticker", split.total)

    # -------- Phase 0a: universe trading calendar (V1.4 date-aligned mode) --------
    # Resolved BEFORE the heavy data-load below so the Phase 0b cache-currency
    # check can REFUSE without wasting panel-build wall-clock. Only built when
    # ``split.mode == "date_aligned"``; trailing carves have no calendar dep.
    universe_cal: "pd.DatetimeIndex | None" = None
    if split.mode == "date_aligned":
        universes_cfg = spec.get("universes") or {}
        uni_block = universes_cfg.get(target["universe"])
        # Span: from a year before train_start through today + a year, so
        # ``searchsorted`` boundaries in carve_universe_aligned never run off
        # either end. The exact ``end`` is irrelevant beyond ``test_end``.
        cal_start = (
            split.train_start.replace(year=split.train_start.year - 1)
            if split.train_start is not None else date(2018, 1, 1)
        )
        universe_cal = gbdt_universe_calendar.get_calendar(
            target["universe"], uni_block, start=cal_start,
        )
        _milestone(
            f"[split] date_aligned: train_start={split.train_start.isoformat()} "
            f"calendar={gbdt_universe_calendar.resolve_calendar_name(target['universe'], uni_block)} "
            f"({len(universe_cal)} trading days available from {cal_start.isoformat()})"
        )

        # -------- Phase 0b: cache-currency check (V1.4 P5, plan §6) --------
        # Verify the cache covers test_end + horizon for every ticker; auto-
        # fetch with 3-retry exponential backoff on transient errors. REFUSE
        # with a worked-example table when the deficient set is non-empty
        # after retries. Sub-case A (universe-level shortfall) raises
        # CacheCurrencyError inline. Runs BEFORE load_panel so REFUSE skips
        # the heavy panel build entirely.
        days_train_start = int(universe_cal.searchsorted(
            pd.Timestamp(split.train_start), side="left",
        ))
        days_test_end = (
            days_train_start
            + split.train_rows + split.val_rows + split.eval_rows + split.test_rows
            - 1
        )
        universe_tickers = gbdt_data.resolve_universe(
            target["universe"], repo_root=repo_root,
        )
        from data_pipelines import fetch as _dp_fetch  # local import — heavy
        try:
            deficient = gbdt_preflight.cache_currency_check(
                universe_tickers=universe_tickers,
                universe_calendar=universe_cal,
                days_test_end=days_test_end,
                horizon_days=int(target["horizon_days"]),
                today=date.today(),
                cache_latest_date=lambda t: gbdt_data._cache_last_date(
                    t, repo_root=repo_root,
                ),
                fetcher=_dp_fetch,
            )
        except gbdt_preflight.CacheCurrencyError as cce:
            print(str(cce), file=sys.stderr)
            heartbeat.stop()
            progress_log.close()
            sys.exit(2)
        if deficient:
            table = gbdt_preflight.format_deficient_table(deficient)
            msg = (
                "[preflight] REFUSE: cache cannot satisfy test_end + horizon "
                f"for {len(deficient)} ticker(s) after 3 retries. "
                "Remediation: run /fetch-data on each deficient ticker "
                "(or extend train_start so test_end falls within the cache).\n"
                f"{table}"
            )
            print(msg, file=sys.stderr)
            heartbeat.stop()
            progress_log.close()
            sys.exit(2)
        _milestone(
            f"[preflight] cache currency OK: {len(universe_tickers)} ticker(s) "
            f"covered through {universe_cal[days_test_end + int(target['horizon_days'])].date().isoformat()}"
        )

    heartbeat.set_phase("data")
    status.update(phase="data")
    _milestone(f"[data] start universe={target['universe']}")
    t1 = time.time()
    data_cfg = spec.get("data", {}) or {}
    staleness_days = int(data_cfg.get(
        "staleness_days", gbdt_data.DEFAULT_STALENESS_DAYS,
    ))
    panel_obj = gbdt_data.load_panel(
        target["universe"],
        start=dr.get("start"),
        end=dr.get("end"),
        min_rows=min_rows,
        repo_root=repo_root,
        staleness_days=staleness_days,
    )
    if panel_obj.stale_tickers:
        print(
            f"[data] warning: {len(panel_obj.stale_tickers)} stale ticker(s) "
            f"(cache > {staleness_days}d old): "
            f"{panel_obj.stale_tickers[:5]}{'...' if len(panel_obj.stale_tickers) > 5 else ''}",
            flush=True,
        )
    _milestone(f"[data] complete in {time.time()-t1:.1f}s rows={len(panel_obj.panel)} "
               f"tickers_kept={len(panel_obj.tickers_kept)}")

    # -------- Phase 1a: project test segment size + warn if structurally slim --------
    # Issue #31 — the walk-forward driver silently emits an empty test
    # segment when ``horizon_days >= split.test_rows``: every ticker's
    # trailing ``test_rows`` rows have NaN targets (forward window
    # incomplete) and get dropped before scoring. The user saw
    # ``headline_test={}`` + ``predictions/test.csv`` with only a header
    # and no warning. Project the row count here and emit a clear warning
    # to the run log; the same string is persisted into
    # ``metrics.json::data.test_split_warning`` and surfaced in
    # ``report.md`` so downstream readers can't miss it. We do NOT
    # auto-shift the split (deferred to V2 per the issue).
    test_split_projection = _project_test_rows(
        panel_obj.panel,
        test_rows_per_ticker=split.test_rows,
        horizon_days=int(target["horizon_days"]),
    )
    test_split_warning: str | None = None
    if test_split_projection["expected_test_rows"] < _TEST_ROWS_WARNING_THRESHOLD:
        test_split_warning = _format_test_split_warning(
            test_split_projection, _TEST_ROWS_WARNING_THRESHOLD,
        )
        print(f"[data] WARNING: {test_split_warning}", flush=True)

    # -------- Phase 2: features --------
    # Two-level feature-matrix cache:
    #
    #   1. **Per-run / per-cell cache** (task #181 — :mod:`gbdt.feature_cache`)
    #      lives in this run's artifact dir. Keyed on everything that determines
    #      the cell — including the target tuple. Hit ⇒ skip build entirely
    #      (the --resume case: same cell, same spec, same data).
    #
    #   2. **Universe-level shared cache** (task #183 —
    #      :mod:`gbdt.universe_feature_cache`) lives under ``<data_root>/
    #      gbdt_feature_cache/``. Keyed identically to #1 EXCEPT the target
    #      tuple is dropped — every sibling cell in a same-universe sweep
    #      (e.g. russell1000 across 20 ``(threshold, horizon, drawdown)``)
    #      hashes to the same key and shares the build. Turns N×build into
    #      1×build + N×label-derivation.
    #
    # Flow: try (1) → try (2) → build. After a build, write BOTH so the next
    # resume of THIS cell hits the cheaper per-cell layer and the next sibling
    # cell of this universe hits the shared layer.
    #
    # Correctness is paramount in both layers: the loaded matrix is byte-
    # identical to what the build would produce, so results + determinism +
    # the finalization-retrain contract are unchanged. On any read error /
    # corruption / key mismatch / sidecar disagreement, we treat it as a miss
    # and rebuild — the conservative, correctness-preserving fallback. The
    # universe cache deliberately reuses the per-cell cache's
    # ``panel_signature`` + ``feature_code_signature`` helpers so a code or
    # data change invalidates both layers at once.
    heartbeat.set_phase("features")
    status.update(phase="features")
    fcfg = spec.get("features", {}) or {}
    lookbacks = tuple(fcfg.get("lookback_windows", gbdt_features.DEFAULT_LOOKBACKS))
    families = fcfg.get("candidates", "all")
    exclude = fcfg.get("exclude") or []
    # F17 macro is opt-in: only the "all_macro" token or an explicit "F17" in a
    # families list triggers the (cache-only) FRED read below. Every existing
    # spec uses "all" → no FRED read, no behaviour change.
    _macro_selected = (
        families == "all_macro"
        or (not isinstance(families, str) and "F17" in set(families))
    )
    # Macro-data content signature for the cache key (which FRED series are
    # cached + their coverage) — None for non-macro runs so their keys are
    # byte-identical. Cheap metadata lookup; the macro panel itself loads later.
    _macro_sig = (
        gbdt_data.macro_panel_signature(
            gbdt_features.MACRO_SERIES, repo_root=repo_root,
        )
        if _macro_selected else None
    )
    # F18/F19 fundamentals are opt-in the same way: "all_fundamentals"
    # (F18), "all_fundamentals2" (F18+F19), "all_fundamentals_vwap" (F18+F20),
    # or an explicit "F18"/"F19" triggers the (cache-only) valuation-panel read
    # below; every existing spec uses "all" → no read, no behaviour change. The
    # panel-artifact signature is folded into the cache key so a rebuilt panel
    # invalidates. NB "all_vwap" (F20 only) is pure-panel — no fund read.
    _fund_selected = (
        families in ("all_fundamentals", "all_fundamentals2",
                     "all_fundamentals_calendar2",
                     "all_fundamentals_vwap", "all_fundamentals_vwap_calendar2")
        or (not isinstance(families, str)
            and bool({"F18", "F19"} & set(families)))
    )
    _fund_sig = (
        gbdt_data.fundamentals_panel_signature(repo_root=repo_root)
        if _fund_selected else None
    )

    panel_sig = gbdt_feature_cache.panel_signature(
        panel_obj.panel, panel_obj.index_series,
    )
    # Note: ``code_commit`` + ``code_dirty`` are still gathered into the
    # ``preflight`` dict (kept as run metadata in metrics.json — useful for
    # post-hoc archival fingerprinting), but they are NO LONGER passed to the
    # cache-key computation. Task #190: the per-commit invalidator was too
    # coarse (every unrelated commit forced a ~5 h cold rebuild). The
    # feature-code signature now carries a targeted SHA-256 of
    # ``gbdt.features`` source, so only edits to features.py invalidate.
    matrix_key = gbdt_feature_cache.compute_key(
        universe=target["universe"],
        target=target,
        split=split_d,
        lookbacks=lookbacks,
        families=families,
        exclude=exclude,
        random_seed=spec.get("random_seed", 42),
        panel_sig=panel_sig,
        macro_sig=_macro_sig,
        fund_sig=_fund_sig,
    )
    universe_key = gbdt_universe_feature_cache.compute_key(
        universe=target["universe"],
        split=split_d,
        lookbacks=lookbacks,
        families=families,
        exclude=exclude,
        random_seed=spec.get("random_seed", 42),
        panel_sig=panel_sig,
        macro_sig=_macro_sig,
        fund_sig=_fund_sig,
    )
    universe_cache_root = preflight.get("data_root") or str(Path("data").resolve())

    # Bug #226 (diagnostic): mirror the exact dicts that compute_key hashed,
    # so we can persist them into the sidecar JSONs alongside the keys. Without
    # this, two cache files with different keys are uninvestigable post-hoc
    # (the input payload was lost the moment compute_key returned its digest).
    # No behaviour change: compute_key's hash inputs are unchanged; we are
    # only persisting the inputs separately for observability.
    _cell_payload = {
        "schema_version": gbdt_feature_cache.SCHEMA_VERSION,
        "universe": target["universe"],
        "target": {
            "direction": target.get("direction"),
            "threshold_pct": target.get("threshold_pct"),
            "horizon_days": target.get("horizon_days"),
            "max_drawdown": target.get("max_drawdown"),
            "uniqueness_weighting": bool(target.get("uniqueness_weighting", True)),
        },
        "split": {
            "train_rows": split_d.get("train_rows"),
            "val_rows": split_d.get("val_rows"),
            "eval_rows": split_d.get("eval_rows"),
            "test_rows": split_d.get("test_rows"),
            "min_rows_per_ticker": split_d.get("min_rows_per_ticker"),
        },
        "features": {
            "lookbacks": list(lookbacks),
            "families": (
                families if isinstance(families, str) else sorted(families)
            ),
            "exclude": sorted(exclude or []),
            "code_signature": gbdt_feature_cache.feature_code_signature(),
        },
        "random_seed": int(spec.get("random_seed", 42)),
        "panel_signature": panel_sig,
    }
    _universe_payload = {
        "schema_version": gbdt_universe_feature_cache.SCHEMA_VERSION,
        "universe": target["universe"],
        "split": {
            "train_rows": split_d.get("train_rows"),
            "val_rows": split_d.get("val_rows"),
            "eval_rows": split_d.get("eval_rows"),
            "test_rows": split_d.get("test_rows"),
            "min_rows_per_ticker": split_d.get("min_rows_per_ticker"),
        },
        "features": {
            "lookbacks": list(lookbacks),
            "families": (
                families if isinstance(families, str) else sorted(families)
            ),
            "exclude": sorted(exclude or []),
            "code_signature": gbdt_feature_cache.feature_code_signature(),
        },
        "random_seed": int(spec.get("random_seed", 42)),
        "panel_signature": panel_sig,
    }
    # Bug #226: track which layer (if any) served the matrix, for metrics.json.
    _matrix_hit = False
    _universe_hit = False

    t1 = time.time()
    X = None
    try:
        X = gbdt_feature_cache.load_cache(out_dir, matrix_key)
    except Exception as exc:  # never let a cache read crash the run
        print(f"[features] per-cell cache read failed ({exc!r}); falling through",
              flush=True)
        X = None

    if X is not None:
        _matrix_hit = True
        _milestone(
            f"[features] loaded from per-cell cache (key match) in "
            f"{time.time()-t1:.1f}s shape={X.shape}"
        )
    else:
        # Try the shared universe-level cache before paying for a full build.
        # A hit here saves the ~5 min (nifty50) / ~3 h (sp500/russell1000)
        # build for every sibling cell after the first in a sweep.
        try:
            X = gbdt_universe_feature_cache.load_cache(
                universe_cache_root, universe_key,
            )
        except Exception as exc:
            print(f"[features] universe cache read failed ({exc!r}); rebuilding",
                  flush=True)
            X = None

        if X is not None:
            _universe_hit = True
            _milestone(
                f"[features] loaded from universe cache (key match) in "
                f"{time.time()-t1:.1f}s shape={X.shape}"
            )
            # Mirror into the per-cell cache so a subsequent --resume of THIS
            # cell hits the cheaper local layer (no shared-cache touch needed).
            try:
                gbdt_feature_cache.write_cache(
                    out_dir, X, matrix_key, payload=_cell_payload,
                )
            except Exception as exc:
                print(f"[features] per-cell cache write failed ({exc!r}); continuing",
                      flush=True)
        else:
            _milestone("[features] start (no cache hit — building)")
            # Fetch the FRED macro panel (cache-only) only on a real build and
            # only when the spec opted into macro features.
            macro_df = None
            if _macro_selected:
                _pdates = panel_obj.panel.index.get_level_values("date")
                macro_df = gbdt_data.load_macro_panel(
                    gbdt_features.MACRO_SERIES,
                    _pdates.min(), _pdates.max(), repo_root=repo_root,
                )
                _milestone(
                    f"[features] macro panel: {macro_df.shape[1]} FRED series × "
                    f"{len(macro_df)} dates"
                )
            # F18: load the point-in-time valuation panel (cache-only) only when
            # the spec opted into fundamentals features.
            fund_df = None
            if _fund_selected:
                _pdates = panel_obj.panel.index.get_level_values("date")
                # Route the valuation panel by the universe's calendar: NSE
                # universes read the INR in_fundamentals panel, everything
                # else the US panel. Keeps the F18 feature token universe-
                # agnostic (no new token; byte-identity of `all` preserved).
                _uni_block = (spec.get("universes") or {}).get(target["universe"])
                _fund_path = (
                    gbdt_data.VALUATION_PANEL_NSE_PATH
                    if gbdt_universe_calendar.resolve_calendar_name(
                        target["universe"], _uni_block
                    ) == "NSE"
                    else None  # None → load_fundamentals_panel's US default
                )
                fund_df = gbdt_data.load_fundamentals_panel(
                    _pdates.min(), _pdates.max(), repo_root=repo_root,
                    path=_fund_path,
                )
                _milestone(
                    f"[features] fundamentals panel: {len(fund_df)} (date,symbol) "
                    f"rows × {fund_df.shape[1]} cols"
                )
            X = gbdt_features.build_feature_matrix(
                panel_obj.panel, panel_obj.index_series,
                lookbacks=lookbacks,
                annualization=panel_obj.annualization_factor,
                families=families, exclude=exclude,
                macro_df=macro_df,
                fund_df=fund_df,
            )
            # Drop all-NaN columns (some features may produce no values on a
            # short-history ticker). This is part of what the loop consumes,
            # so BOTH caches MUST persist the post-dropna matrix to stay
            # byte-identical.
            X = X.dropna(axis=1, how="all")
            _milestone(f"[features] complete in {time.time()-t1:.1f}s shape={X.shape}")
            try:
                gbdt_feature_cache.write_cache(
                    out_dir, X, matrix_key, payload=_cell_payload,
                )
                _milestone(f"[features] per-cell cache written (key={matrix_key[:12]}…)")
            except Exception as exc:  # a cache write failure must not fail the run
                print(f"[features] per-cell cache write failed ({exc!r}); continuing",
                      flush=True)
            try:
                gbdt_universe_feature_cache.write_cache(
                    universe_cache_root, X, universe_key,
                    payload=_universe_payload,
                )
                _universe_cache_dir = (
                    f"{universe_cache_root}/"
                    f"{gbdt_universe_feature_cache.DEFAULT_CACHE_SUBDIR}"
                )
                _milestone(
                    f"[features] universe cache written (key={universe_key[:12]}…) "
                    f"at {_sanitize_path_for_emission(_universe_cache_dir, repo_root)}/"
                )
            except Exception as exc:
                print(f"[features] universe cache write failed ({exc!r}); continuing",
                      flush=True)

    # Fail-fast non-finite audit (runs in seconds, BEFORE the multi-hour FS+HP
    # loop). ``±inf`` from ratio/division families (e.g. ``v / v.shift(n) - 1``
    # on a zero prior-period value for a sparse / halted ticker) crashes
    # XGBoost's DMatrix construction at iter-0 fit; CatBoost tolerates it. The
    # XGBoost backend sanitizes ``±inf`` → ``NaN`` at fit/predict time, but we
    # surface it here so a future run sees the offending columns + counts up
    # front rather than after a wasted feature build. NaN is the legitimate
    # missing sentinel and is intentionally NOT flagged.
    inf_offenders = count_nonfinite(X)
    if inf_offenders:
        total_inf = sum(inf_offenders.values())
        top = list(inf_offenders.items())[:10]
        top_str = ", ".join(f"{c}={n}" for c, n in top)
        more = f" (+{len(inf_offenders) - len(top)} more cols)" if len(inf_offenders) > 10 else ""
        _milestone(
            f"[features] WARNING: {total_inf} non-finite (±inf) value(s) across "
            f"{len(inf_offenders)} column(s) in the feature matrix; the XGBoost "
            f"backend sanitizes these to NaN (missing) at fit/predict, CatBoost "
            f"routes them to its missing bucket. Top offenders: {top_str}{more}"
        )

    # -------- Phase 3: target --------
    heartbeat.set_phase("target")
    status.update(phase="target")
    _milestone("[target] start")
    t1 = time.time()
    y = build_target(
        panel_obj.panel,
        direction=target["direction"],
        threshold_pct=target["threshold_pct"],
        horizon_days=target["horizon_days"],
        max_drawdown=target.get("max_drawdown"),
    )
    _milestone(f"[target] complete in {time.time()-t1:.1f}s "
               f"positive_prevalence={float(y.dropna().mean()):.3f}")

    # -------- Phase 3b: sample-uniqueness weights (LdP §4.4) --------
    # ON by default. Opt-out reproduces the legacy (biased) behavior
    # where every row enters the loss with weight 1.0 — useful only for
    # reproducing pre-PR results / measuring the overlap-bias delta.
    uniqueness_on = bool(target.get("uniqueness_weighting", True))
    if uniqueness_on:
        heartbeat.set_phase("uniqueness")
        status.update(phase="uniqueness")
        _milestone("[uniqueness] start")
        t1 = time.time()
        sample_weights = compute_uniqueness_weights(
            panel_obj.panel, horizon=int(target["horizon_days"]),
        )
        # Effective-sample-size summary across the full panel (pre-segment)
        ess_full = float(effective_sample_size(sample_weights.values))
        _milestone(
            f"[uniqueness] complete in {time.time()-t1:.1f}s "
            f"horizon={target['horizon_days']} rows={len(sample_weights)} "
            f"ESS={ess_full:.0f} inflation={len(sample_weights)/max(ess_full,1):.2f}x"
        )
    else:
        sample_weights = None
        print("[uniqueness] disabled by spec (target.uniqueness_weighting=false)",
              flush=True)

    # -------- Phase 4: walk-forward + FS+HP loop --------
    backend = spec.get("backend", {}) or {}
    backend_library = backend.get("library", "catboost")
    hp_starting = backend.get("hp_starting", {}) or {}
    loop_cfg = backend.get("fs_hp_loop", {}) or {}
    cal_method = backend.get("calibration_method", "conditional_isotonic")
    cal_z_thr = backend.get("calibration_z_threshold", 2.0)
    seed = spec.get("random_seed", 42)

    callback_mode = loop_cfg.get("callback_mode", _DEFAULT_CALLBACK_MODE)
    max_iter = loop_cfg.get("max_iterations", 8)
    # V1.3 Option A (plan § 3.3 / D3) — look up the canonical sweep row for this
    # cell ONCE at iter_0; cached for the run and threaded into every
    # build_diagnostic_bundle call so anti_auc_flag is constant. None when no
    # matching row exists (new cell) → flag stays "unknown" → auto-disables
    # (best_checkpoint L1 + inner_stop_check val_brier plateau) safely default
    # to NOT firing on cells without sweep evidence yet.
    sweep_csv_path = repo_root / _DEFAULT_SWEEP_CSV_RELPATH
    cell_experiment_name = cell_key_to_experiment_name(
        universe=target["universe"],
        direction=target["direction"],
        threshold_pct=target["threshold_pct"],
        horizon_days=target["horizon_days"],
        max_drawdown=target.get("max_drawdown"),
    )
    sweep_row = lookup_sweep_row(cell_experiment_name, sweep_csv_path)
    if sweep_row is not None:
        _milestone(
            f"[loop] V1.3 sweep_row matched cell='{cell_experiment_name}' "
            f"(AUC={sweep_row.get('AUC')}, R-p@10={sweep_row.get('R_precision_at_10')}, "
            f"base_rate={sweep_row.get('base_rate')})"
        )
    else:
        _milestone(
            f"[loop] V1.3 no sweep_row for cell='{cell_experiment_name}' "
            f"(csv={_sanitize_path_for_emission(sweep_csv_path, repo_root)}); "
            f"anti_auc_flag will be 'unknown'"
        )
    degenerate_sink_threshold = float(
        loop_cfg.get(
            "degenerate_sink_threshold", _DEFAULT_DEGENERATE_SINK_THRESHOLD
        )
    )
    # V1.3 Option B (P4) — agent_file_protocol scout cycles.
    # In agent mode with scout enabled, we run cycles 1-3 OUTSIDE of
    # walk_forward_train: cycle 1 produces scout_results + combine_request,
    # cycle 2 fits the agent's combine_decision configs, cycle 3 reads the
    # agent's iter_0_decision and uses it as iter_0's hp_starting. The
    # walk_forward_train call below sees scout-disabled + the agent's HP
    # already on hp_starting, so it behaves like the V1.1 agent loop from
    # iter_0 onwards.
    scout_cfg = backend.get("scout", {}) or {}
    fs_prefit_cfg = backend.get("fs_prefit", {}) or {}
    scout_enabled = bool(scout_cfg.get("enabled", False))
    fs_prefit_enabled = bool(fs_prefit_cfg.get("enabled", scout_enabled))
    iter_0_features = list(X.columns)
    cycle_outcome: dict | None = None
    if scout_enabled and callback_mode == "agent_file_protocol":
        cycle_outcome = _handle_scout_cycles_agent_mode(
            out_dir=out_dir,
            spec=spec,
            run_id=name, spec_path=spec_path,
            panel=panel_obj.panel, X=X, y=y, sample_weights=sample_weights,
            split=split, universe_calendar=universe_cal,
            backend_library=backend_library,
            hp_starting=hp_starting,
            calibration_method=cal_method,
            calibration_z_threshold=cal_z_thr,
            random_seed=seed,
            scout_cfg=scout_cfg, fs_prefit_cfg=fs_prefit_cfg,
            milestone=_milestone, heartbeat=heartbeat, status=status,
            progress_log=progress_log,
            # D6.2.A — FS-prefit cache inputs.
            fs_prefit_universe=target["universe"],
            fs_prefit_cache_root=preflight.get("data_root"),
            fs_prefit_features_source_sha256=(
                gbdt_feature_cache.feature_code_signature().get("source_sha256")
            ),
            fs_prefit_snapshot_end_iso=(
                snapshot_end_override_iso
                or panel_sig.get("panel_date_max")
            ),
        )
        if cycle_outcome is None:
            # Cycle paused — runner already wrote the relevant files +
            # logged the resume hint. Exit cleanly.
            heartbeat.stop()
            progress_log.close()
            return out_dir
        # cycle 3: proceed with the agent's HP + cliff-cut feature set.
        hp_starting = cycle_outcome["hp_starting"]
        iter_0_features = cycle_outcome["features"]
        # The scout report is stitched into the metrics later via
        # ``cycle_outcome['scout_report']``.
    elif scout_enabled and callback_mode == "default":
        # Default mode — scout runs INSIDE walk_forward_train. We don't
        # touch hp_starting / iter_0_features here; the in-train helper
        # will mutate them based on the lex auto-compose winner.
        pass
    else:
        cycle_outcome = None

    # V1.1 — resolve the FS+HP callback from the (possibly CLI-overridden) spec.
    # ``default`` resolves to None so walk_forward_train keeps using its built-in
    # default_fs_hp_callback (v1 behaviour preserved byte-for-byte). The
    # agent_file_protocol callback gets the artifact dir + a live loop-state
    # sink so it can write a complete resume checkpoint before pausing.
    loop_state_sink: dict | None = (
        {} if callback_mode == "agent_file_protocol" else None
    )
    fs_hp_callback = _resolve_callback(
        loop_cfg, run_id=name,
        artifact_dir=out_dir,
        loop_state_sink=loop_state_sink,
        max_iterations=int(max_iter),
        cell={
            k: target.get(k)
            for k in ("universe", "direction", "threshold_pct",
                      "horizon_days", "max_drawdown")
        },
    )

    heartbeat.set_phase("loop")
    status.update(
        phase="loop",
        iter_idx=int(resume_state["iter_idx"]) if resume_state else 0,
        awaiting_decision=False,
    )
    resume_note = (
        f" (resume from iter {resume_state['iter_idx']})" if resume_state else ""
    )
    _milestone(
        f"[loop] start max_iter={max_iter} callback_mode={callback_mode}{resume_note}"
    )
    t1 = time.time()
    # Task #204: in agent_file_protocol mode the runner defers loop-continuation
    # to the agent. The val_brier plateau gate is too coarse a stop signal
    # there — the agent should be free to pivot to a structurally-different
    # knob (e.g. ``colsample`` after ``min_child_weight`` plateaued) instead of
    # being auto-stopped on one knob's flatline. ``degradation`` + ``cap``
    # remain active in both modes (regression is a real stop signal; cap bounds
    # the loop). Default (sweep) mode keeps the plateau gate — fully algorithmic,
    # no agent to defer to.
    disable_plateau = (callback_mode == "agent_file_protocol")
    try:
        result = walk_forward_train(
            panel=panel_obj.panel, X=X, y=y,
            features=list(iter_0_features), hp=dict(hp_starting), split=split,
            calibration_method=cal_method,
            calibration_z_threshold=cal_z_thr,
            max_iterations=max_iter,
            plateau_threshold=loop_cfg.get("plateau_threshold", 0.005),
            degradation_gate=loop_cfg.get("degradation_gate", 0.01),
            tie_band=loop_cfg.get("tie_band"),
            fs_hp_callback=fs_hp_callback,
            random_seed=seed,
            sample_weights=sample_weights,
            resume_state=resume_state,
            loop_state_sink=loop_state_sink,
            backend=backend_library,
            disable_plateau=disable_plateau,
            # V1.3 Option A — canonical sweep CSV row (anti_auc_flag source)
            # + degenerate-sink warning threshold. Both constant for the run.
            sweep_row=sweep_row,
            degenerate_sink_threshold=degenerate_sink_threshold,
            # V1.4 (date-aligned splits): None for trailing carves (the
            # default path); the resolved universe calendar otherwise.
            universe_calendar=universe_cal,
            # V1.3 Option B (P3) — scout/prefit hooks. Active only in default
            # mode (the in-train helper skips agent_file_protocol mode since
            # P4's runner-side cycles already handled it).
            scout_spec=scout_cfg if scout_enabled else None,
            fs_prefit_spec=fs_prefit_cfg if fs_prefit_enabled else None,
            callback_mode=callback_mode,
            # V1.3 Option B D6.2.A — FS-prefit kept-feature cache key inputs.
            # Default mode flows through ``_maybe_run_scout_and_prefit``; the
            # cache is the cross-cell reuse layer for sibling cells sharing
            # the universe + snapshot + feature-source + default HP.
            fs_prefit_universe=target["universe"],
            fs_prefit_cache_root=preflight.get("data_root"),
            fs_prefit_features_source_sha256=(
                gbdt_feature_cache.feature_code_signature().get("source_sha256")
            ),
            fs_prefit_snapshot_end_iso=(
                snapshot_end_override_iso
                or panel_sig.get("panel_date_max")
            ),
        )
    except loop_protocol.PauseForAgentDecision as pause:
        # Exit half of exit-and-resume (plan § 0): the callback wrote the
        # request bundle + checkpoint and handed control back. Log a
        # copy-pasteable resume hint and return cleanly (NOT an error). The
        # agent reads the request, writes loop/iter_<N>_decision.json, then
        # relaunches `--resume <run_id>` to continue at iter N+1.
        heartbeat.stop()
        # task #177: record the pause as the run's terminal milestone +
        # mark status awaiting_decision so a monitor reading status.json knows
        # the loop is parked on the agent, not wedged. best_val_brier is the
        # best val Brier observed so far (from the live loop-state sink).
        paused_best_brier = None
        if loop_state_sink:
            briers = [b for b in (loop_state_sink.get("val_briers") or [])
                      if b is not None]
            if briers:
                paused_best_brier = float(min(briers))
        status.update(
            iter_idx=int(pause.iter_n),
            phase="loop",
            awaiting_decision=True,
            best_val_brier=paused_best_brier,
        )
        _milestone(
            f"[loop] PAUSED iter {pause.iter_n} awaiting decision "
            f"(best_val_brier={paused_best_brier})"
        )
        _milestone(
            f"[loop] paused at iter {pause.iter_n} — request written: "
            f"{pause.request_path}"
        )
        _milestone(f"[loop] checkpoint written: {pause.checkpoint_path}")
        _milestone(
            f"[loop] paused at iter {pause.iter_n} — resume with: "
            f"uv run python -m gbdt experiment {spec_path.name} "
            f"--resume {pause.run_id}"
        )
        progress_log.close()
        return out_dir
    _milestone(
        f"[loop] LOOP COMPLETE in {time.time()-t1:.1f}s "
        f"best_iter={result.best_iteration} "
        f"val_brier={result.best_val_brier:.4f} "
        f"reason={result.inner_stop_signal}"
    )
    status.update(
        phase="loop",
        iter_idx=int(result.best_iteration),
        awaiting_decision=False,
        best_val_brier=float(result.best_val_brier),
    )

    # -------- Phase 5: artifact emit --------
    heartbeat.set_phase("artifact")
    status.update(phase="artifact")
    _milestone("[artifact] start")
    t1 = time.time()

    # Issue #30 — the snapshot at ``spec.yaml`` MUST be the per-experiment
    # spec as authored on disk, NOT the merge of defaults+spec. Dumping
    # the merged dict (the pre-fix behaviour) buried the actual target
    # under hundreds of lines of universe-registry content from defaults,
    # which made archived artifacts look corrupted (see the
    # nasdaq100_up_10pct_100d_dd5pct_pre_uniqueness_fix archive: the
    # snapshot's first 30 lines listed nifty50/nifty100/... while the
    # actual target was buried at the bottom).
    per_exp_spec = spec.get("__per_experiment_spec__") or {
        k: v for k, v in spec.items()
        if k in ("target", "date_range", "split", "features", "backend",
                  "random_seed", "artifacts", "data")
    }
    # Defence-in-depth: refuse to write a snapshot whose target.universe
    # disagrees with the run we just executed. This is the fail-loud
    # regression assertion called for in issue #30 — even if a future
    # refactor regresses the snapshot path, the user will not silently
    # get an artifact pointing at the wrong universe.
    snap_universe = (per_exp_spec.get("target") or {}).get("universe")
    run_universe = target["universe"]
    if snap_universe is not None and snap_universe != run_universe:
        raise RuntimeError(
            f"spec.yaml snapshot universe mismatch: snapshot says "
            f"{snap_universe!r} but the run actually executed against "
            f"{run_universe!r}. This is a runner bug — refusing to "
            f"persist a misleading artifact (see issue #30)."
        )
    (out_dir / "spec.yaml").write_text(
        yaml.safe_dump(per_exp_spec, sort_keys=False)
    )

    # Backend-determined model filename (V1.2 plan § 4.4): catboost → model.cbm,
    # xgboost → model.ubj. The single source of truth is gbdt.model.model_filename,
    # which the /gbdt-diagnose loader also consults so the two always agree.
    result.best_model.save(out_dir / model_filename(backend_library))
    # Always write a pickle. When no calibrator is needed (native pass) we
    # still pickle ``None`` so downstream ``pickle.load`` is uniform — see
    # PR #8 review (Minor 2): a plaintext-vs-pickle mix produced
    # ``UnpicklingError: invalid load key, '#'.``
    with open(out_dir / "calibration.pkl", "wb") as f:
        pickle.dump(result.calibration.calibrator, f)

    # YAML artifacts are written as explicit top-level-keyed dicts (not
    # bare collections) so they are self-describing and merge/diff cleanly
    # in the cross-experiment table — see PR #8 review (Minor 3).
    (out_dir / "features.yaml").write_text(
        yaml.safe_dump({"features": list(result.best_features)}, sort_keys=False)
    )
    (out_dir / "hp.yaml").write_text(
        yaml.safe_dump({"hp": dict(result.best_hp)}, sort_keys=False)
    )

    with open(out_dir / "iterations.jsonl", "w") as f:
        last_idx = len(result.iterations) - 1
        for i, b in enumerate(result.iterations):
            d = b.to_dict()
            d["inner_stop_signal"] = (
                result.inner_stop_signal if i == last_idx else None
            )
            f.write(json.dumps(d, default=str) + "\n")

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for seg, df in result.predictions.items():
        df.to_csv(pred_dir / f"{seg}.csv", index=False)

    headline_eval = _compute_headline(result.predictions.get("eval"))
    headline_test = _compute_headline(result.predictions.get("test"))
    train_pred = result.predictions.get("train")
    val_pred = result.predictions.get("val")

    # Per-fold ESS — single-fold for v1 (carve_single_fold), so this is a
    # one-entry dict keyed by ``"fold_0"``. Multi-fold mode (V1.1_TBD)
    # will populate one entry per fold.
    #
    # We report two distinct quantities and they answer different questions:
    #   ess_kish (Kish):           (Σw)² / Σw² — variance-effective sample
    #                              size; reduces to ``n`` for uniform
    #                              weights regardless of scale. Use for
    #                              confidence intervals on weighted means.
    #   sum_weights (independent): Σw — approximate count of *independent*
    #                              forward events the panel encodes. For
    #                              uniqueness weights this is ≈ n/(2H-1)
    #                              when n >> H. Use for "how much
    #                              information is actually here".
    def _seg_ess(seg: str) -> dict[str, float | int | None]:
        df = result.predictions.get(seg)
        if df is None or df.empty:
            return {
                "ess_kish": None,
                "sum_weights": None,
                "n_rows": 0,
                "overlap_inflation_ratio": None,
            }
        w = df["sample_weight"].values.astype(float) if "sample_weight" in df.columns \
            else np.ones(len(df), dtype=float)
        ess_kish = float(effective_sample_size(w))
        s = float(w.sum())
        n = int(len(df))
        ratio = float(n / max(s, 1.0))
        return {
            "ess_kish": ess_kish,
            "sum_weights": s,
            "n_rows": n,
            "overlap_inflation_ratio": ratio,
        }

    ess_summary = {
        "uniqueness_weighting": uniqueness_on,
        "horizon_days": int(target["horizon_days"]),
        "effective_sample_size_per_fold": {
            "fold_0": {
                "train": _seg_ess("train"),
                "val": _seg_ess("val"),
                "eval": _seg_ess("eval"),
                "test": _seg_ess("test"),
            },
        },
    }

    # V1.4 (plan §5.2): segment-date envelope + per-ticker sidecars.
    #
    # For date_aligned carves, result.segment_dates is already the
    # universe-calendar window (ISO strings) from carve_universe_aligned.
    # For trailing carves, compute the calendar UNION per segment from the
    # predictions DataFrame: MIN(start) and MAX(end) across all tickers in
    # the segment. Either way, an empty segment yields ``None`` endpoints.
    def _trailing_segment_dates() -> dict[str, dict[str, str | None]]:
        sd: dict[str, dict[str, str | None]] = {}
        for seg in ("train", "val", "eval", "test"):
            df = result.predictions.get(seg)
            if df is None or len(df) == 0:
                sd[seg] = {"start": None, "end": None}
                continue
            dts = pd.to_datetime(df["date"])
            sd[seg] = {
                "start": dts.min().date().isoformat(),
                "end": dts.max().date().isoformat(),
            }
        return sd

    if result.segment_dates is not None:
        segment_dates = result.segment_dates
    else:
        segment_dates = _trailing_segment_dates()

    # Per-ticker sidecars (D4).
    def _per_ticker_sidecars() -> dict[str, object]:
        tickers_per_segment: dict[str, list[str]] = {}
        n_tickers_per_segment: dict[str, int] = {}
        row_counts: dict[str, dict[str, int]] = {}
        for seg in ("train", "val", "eval", "test"):
            df = result.predictions.get(seg)
            if df is None or len(df) == 0:
                tickers_per_segment[seg] = []
                n_tickers_per_segment[seg] = 0
                continue
            grp = df.groupby("ticker", sort=True)["y_true"].count()
            seg_tickers = grp.index.tolist()
            tickers_per_segment[seg] = seg_tickers
            n_tickers_per_segment[seg] = len(seg_tickers)
            for t, n in grp.items():
                row_counts.setdefault(t, {"train": 0, "val": 0, "eval": 0, "test": 0})[
                    seg
                ] = int(n)
        # Ensure every ticker that appears anywhere has all 4 keys (D4's
        # nested int dict has 0 for absent segments — explicit not implicit).
        return {
            "n_tickers_per_segment": n_tickers_per_segment,
            "tickers_per_segment": tickers_per_segment,
            "row_counts_per_segment_per_ticker": row_counts,
        }

    sidecars = _per_ticker_sidecars()

    metrics = {
        "experiment_name": name,
        # Which GBDT backend produced this artifact (V1.2 plan § 4.4) — so a
        # post-hoc reader knows whether model.cbm or model.ubj sits beside it,
        # and which model class /gbdt-diagnose must load.
        "backend": {
            "library": backend_library,
            "model_filename": model_filename(backend_library),
        },
        "spec_hash": _spec_hash(spec),
        "data_hash": _data_hash(panel_obj.panel),
        # Pre-flight cache + code fingerprint (see ``_collect_preflight``).
        # Six fields populated even when git is unavailable. Paths sanitized
        # to repo-relative for emission (see ``_sanitize_preflight_for_emission``);
        # live downstream consumers continue to use the raw realpath copy.
        "preflight": _sanitize_preflight_for_emission(preflight, repo_root),
        # V1.4 (plan §5.2): top-level segment-date envelope + carve mode.
        # ``split_mode`` mirrors the spec; ``split_train_start`` is the
        # ISO date used for the date-aligned anchor (None for trailing).
        # ``segment_dates`` is the universe-calendar window for
        # date-aligned carves, or the calendar UNION across tickers for
        # trailing carves (MIN start, MAX end per segment).
        "split_mode": split.mode,
        "split_train_start": (
            split.train_start.isoformat() if split.train_start is not None else None
        ),
        "segment_dates": segment_dates,
        "data": {
            "n_tickers_in_universe": len(panel_obj.statuses),
            "n_tickers_used": len(panel_obj.tickers_kept),
            "tickers_excluded": panel_obj.tickers_excluded,
            # Cache freshness / NaN-row drop telemetry (PR #8 review, Minor 1+4).
            "staleness_days_threshold": panel_obj.staleness_days_threshold,
            "stale_tickers": panel_obj.stale_tickers,
            "n_tickers_stale": len(panel_obj.stale_tickers),
            "cache_age_days_by_ticker": {
                s.ticker: s.cache_age_days
                for s in panel_obj.statuses
                if s.kept and s.cache_age_days is not None
            },
            "nan_rows_dropped_by_ticker": {
                s.ticker: s.nan_rows_dropped
                for s in panel_obj.statuses
                if s.nan_rows_dropped > 0
            },
            "n_rows_train": int(len(train_pred)) if train_pred is not None else 0,
            "n_rows_val": int(len(val_pred)) if val_pred is not None else 0,
            "n_rows_eval": int(len(result.predictions.get("eval", pd.DataFrame()))),
            "n_rows_test": int(len(result.predictions.get("test", pd.DataFrame()))),
            "positive_prevalence_train": (
                float(train_pred["y_true"].mean())
                if train_pred is not None and len(train_pred) else None
            ),
            "positive_prevalence_eval": (
                float(result.predictions.get("eval", pd.DataFrame({"y_true": []}))["y_true"].mean())
                if len(result.predictions.get("eval", pd.DataFrame())) else None
            ),
            # Issue #31 — surfaces structurally-thin/empty test segments
            # (horizon eats the test window). Absent/None when the test
            # segment is expected to be normally sized; a human-readable
            # string explaining the calculation when below threshold.
            "test_split_warning": test_split_warning,
            "test_split_projection": test_split_projection,
            # V1.4 D4 (plan §5.2): per-ticker sidecars for the carve.
            # ``n_tickers_per_segment`` is the count of tickers contributing
            # to each segment; ``tickers_per_segment`` is the full list;
            # ``row_counts_per_segment_per_ticker`` is the nested int dict
            # keyed by ticker → {segment: n_rows}. Useful for downstream
            # reproducibility audits + the cache-growth invariance check.
            "n_tickers_per_segment": sidecars["n_tickers_per_segment"],
            "tickers_per_segment": sidecars["tickers_per_segment"],
            "row_counts_per_segment_per_ticker": sidecars[
                "row_counts_per_segment_per_ticker"
            ],
        },
        "loop": {
            # Issue #251: total iterations seen across the loop's full
            # history (prior resume-seeded iters + in-process iters). On
            # the default callback path ``len(result.iterations)`` equals
            # this total because there are no prior iters; on the
            # agent_file_protocol exit-and-resume path the prior iters
            # live in the resume checkpoint and ``len(result.iterations)``
            # covers ONLY the in-process bundles built in this finalize
            # call — which undercounts (e.g. 0 on a should_stop resume
            # where the loop body is skipped). ``result.n_iterations_total``
            # is the source of truth in both modes; ``getattr`` keeps
            # back-compat for tests constructing a ``WalkForwardResult``
            # without the field.
            "n_iterations_run": int(
                getattr(result, "n_iterations_total", len(result.iterations))
            ),
            "best_iteration": int(result.best_iteration),
            # Issue #251: surface the loop's best val Brier (the val
            # Brier at ``best_iteration``) alongside the iter index so
            # the metrics block self-describes the chosen checkpoint
            # without forcing a cross-reference to status.json or
            # iterations.jsonl. Already on ``WalkForwardResult`` —
            # previously surfaced into status.json only.
            "best_val_brier": float(result.best_val_brier),
            "inner_stop_signal": result.inner_stop_signal,
            # V1.4 P2 — which branch in ``fs_hp_loop.best_checkpoint``
            # produced the ``best_iteration``. Surfaced into ``report.md``
            # so readers can tell "L1 fired but fell back to eval R-p@1-best"
            # apart from "classic L1 (gap+|z|) winner". See
            # ``src/gbdt/fs_hp_loop.TiebreakPath`` for the 5 label values.
            "tiebreak_path": str(
                getattr(result, "tiebreak_path", "strict_val_brier")
            ),
            # Issue #32 — the default FS+HP callback only nudges HPs in
            # response to overfit/cap signals; when ``max_iterations`` is
            # small (sweep mode = 3) the loop typically reuses the
            # starting HP unchanged across every iteration, so the
            # ``hp_history`` field is honest about FS-only behaviour.
            # We flag this here so artifact readers don't misinterpret
            # the "FS+HP loop" name as evidence of real HP search.
            "hp_search_active": bool(
                int(loop_cfg.get("max_iterations", 8))
                >= _HP_SEARCH_ITER_THRESHOLD
            ),
            "hp_search_iter_threshold": int(_HP_SEARCH_ITER_THRESHOLD),
            "max_iterations": int(loop_cfg.get("max_iterations", 8)),
        },
        "calibration": {
            "method": cal_method,
            "decision": result.calibration.method,
            "spiegelhalter_z": result.calibration.spiegelhalter_z,
            "spiegelhalter_p": result.calibration.spiegelhalter_p,
        },
        "sample_uniqueness": ess_summary,
        # Bug #226 (diagnostic): persist the cache keys + which layer (if any)
        # served the matrix, plus the panel_signature that determined them.
        # This is what lets a sweep-level post-hoc audit answer "did sibling
        # cells share the universe cache, or did each rebuild from scratch?"
        # without having to grep the per-run log. ``panel_signature`` is the
        # most likely discriminator (its index hash captures every (date,
        # ticker) tuple in the panel), so surfacing it directly here makes
        # cross-cell diffs trivial.
        "cache": {
            "matrix_key": matrix_key,
            "universe_key": universe_key,
            "matrix_hit": bool(_matrix_hit),
            "universe_hit": bool(_universe_hit),
            "panel_signature": panel_sig,
        },
        "headline_eval": headline_eval,
        "headline_test": headline_test,
        # Per-segment top-K + per-ticker + per-quarter + pred-range
        # diagnostics. Operate on the same (date, ticker, p_calibrated,
        # y_true) row schema the headline metrics consume; covers eval
        # and test segments (empty-shaped block for missing/empty).
        "segment_diagnostics": compute_segment_diagnostics(result.predictions),
        "wall_time_total_sec": time.time() - t0,
    }

    # V1.3 Option B (P5) — metrics.json::scout + metrics.json::combine.
    # Two data sources:
    # - Default mode: walk_forward_train's ``scout_report`` (the in-train
    #   helper).
    # - Agent mode: the runner-side scout cycles wrote to ``scout/`` files;
    #   the cycle 3 outcome stitched ``scout_bundle.json`` payload into
    #   ``cycle_outcome``.
    scout_block, combine_block = _build_scout_metrics_blocks(
        result=result, out_dir=out_dir, cycle_outcome=cycle_outcome,
        callback_mode=callback_mode, scout_enabled=scout_enabled,
    )
    if scout_block is not None:
        metrics["scout"] = scout_block
    if combine_block is not None:
        metrics["combine"] = combine_block

    # In default mode the in-train scout produced its raw rows in memory;
    # write them to scout/scout_results.jsonl + scout/scout_bundle.json on
    # disk so the artifact layout matches agent mode (P5 § D7.2).
    if (
        callback_mode == "default"
        and scout_enabled
        and result.scout_report is not None
    ):
        try:
            raw_rows = result.scout_report.get("_scout_results_raw") or []
            gbdt_scout_io.write_scout_results(out_dir, raw_rows)
            gbdt_scout_io.write_scout_bundle(out_dir, scout_block or {})
        except Exception as exc:    # noqa: BLE001 — best-effort
            print(f"[scout] failed to write scout/ subdir files: {exc!r}",
                  flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    emit_figures(out_dir, result.iterations, result.predictions)
    render_report(out_dir)

    _milestone(
        f"[artifact] complete in {time.time()-t1:.1f}s -> "
        f"{_sanitize_path_for_emission(out_dir, repo_root)}"
    )
    heartbeat.stop()
    # task #177: terminal status — the loop finished (or the agent stopped it).
    # stop_reason is the inner-stop signal (e.g. ``plateau``, ``agent_should_stop``,
    # ``max_iterations``). A monitor reading a non-null stop_reason knows the run
    # is DONE, not wedged.
    status.update(
        phase="complete",
        awaiting_decision=False,
        stop_reason=str(result.inner_stop_signal),
    )
    _milestone(
        f"[loop] STOPPED reason={result.inner_stop_signal}"
    )
    _milestone(f"[experiment] complete in {time.time()-t0:.1f}s")
    progress_log.close()
    return out_dir
