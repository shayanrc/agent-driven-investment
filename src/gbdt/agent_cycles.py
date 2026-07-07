"""Agent-file-protocol callback resolution and V1.3 Option B scout /
FS-prefit cycle orchestration for the gbdt runner.

Extracted verbatim from ``gbdt.__main__`` by the runner split — behavior
unchanged. ``run_experiment`` (``gbdt.experiment_runner``) is the only
runtime caller; ``gbdt.__main__`` re-exports these names for back-compat.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gbdt import checkpoint as gbdt_checkpoint
from gbdt import fs_prefit as gbdt_fs_prefit
from gbdt import loop_observability
from gbdt import loop_protocol
from gbdt import scout as gbdt_scout
from gbdt import scout_io as gbdt_scout_io
from gbdt.spec import _DEFAULT_CALLBACK_MODE, _VALID_CALLBACK_MODES
from gbdt.train import SplitSpec


def _resolve_callback(
    loop_cfg: dict,
    run_id: str,
    *,
    artifact_dir: Path | None = None,
    loop_state_sink: dict | None = None,
    max_iterations: int = 8,
    cell: dict | None = None,
):
    """Select the FS+HP iteration callback from ``backend.fs_hp_loop`` config.

    Returns either ``None`` (so ``walk_forward_train`` keeps using its built-in
    ``default_fs_hp_callback`` — v1 behaviour, byte-for-byte preserved) or a
    callable matching the ``(bundle, current_features) -> (keep, next_hp,
    rationale)`` signature.

    - ``callback_mode == "default"`` (or absent) → ``None``.
    - ``callback_mode == "agent_file_protocol"`` → the exit-and-resume callback
      (plan § 0): at the end of iteration N it writes
      ``loop/iter_<N>_request.json`` (the minimal request bundle) + a resume
      checkpoint, then raises :class:`~gbdt.loop_protocol.PauseForAgentDecision`
      to hand control back to the agent. ``run_experiment`` catches the pause,
      logs the ``--resume`` hint, and exits cleanly.

    The agent-file-protocol callback needs the artifact dir (where loop files
    land) and the live loop history (``loop_state_sink``, populated by
    ``walk_forward_train`` before each callback invocation). When those are not
    wired (e.g. a Phase 1-style resolution-only call) it raises a clear error
    instead of silently mis-writing.
    """
    mode = (loop_cfg or {}).get("callback_mode", _DEFAULT_CALLBACK_MODE)
    if mode == "default":
        return None
    if mode == "agent_file_protocol":
        return _make_agent_file_protocol_callback(
            run_id=run_id,
            artifact_dir=artifact_dir,
            loop_state_sink=loop_state_sink,
            max_iterations=max_iterations,
            cell=cell,
        )
    # Unreachable when the spec passed _validate_spec, but defend anyway.
    raise ValueError(
        f"unknown callback_mode: {mode!r} (expected one of {sorted(_VALID_CALLBACK_MODES)})"
    )


def _make_agent_file_protocol_callback(
    *,
    run_id: str,
    artifact_dir: Path | None,
    loop_state_sink: dict | None,
    max_iterations: int,
    cell: dict | None = None,
):
    """Build the exit-and-resume callback (plan § 0).

    On invocation (end of iteration N, loop continuing) it: (1) builds + writes
    the minimal request bundle to ``loop/iter_<N>_request.json``; (2) writes a
    resume checkpoint capturing exactly what's needed to seed iter N+1 without
    re-training 0..N (iteration index, accumulated history, current features +
    HP, run/spec identity — NO model blobs, plan § 0.2); (3) raises
    :class:`PauseForAgentDecision`. ``run_experiment`` catches it and exits 0.
    """
    def _cb(bundle, current_features):
        if artifact_dir is None or loop_state_sink is None:
            raise RuntimeError(
                "agent_file_protocol callback invoked without artifact_dir / "
                "loop_state_sink wired — resolve it via run_experiment, not the "
                "Phase-1 resolution-only path."
            )
        state = dict(loop_state_sink)
        iter_n = int(state["iter_idx"])
        # (1) request bundle (Phase 3: the diagnose.json-shaped payload built
        # in-memory from this iteration's DiagnosticBundle — reuses the
        # /gbdt-diagnose pure helpers, no matrix rebuild). ``artifact_dir`` +
        # ``cell`` are surfaced so the agent can run the full on-disk diagnose
        # for the matrix-dependent analyses (plan § 0.5).
        req_payload = loop_protocol.build_request_bundle(
            bundle,
            iter_n=iter_n,
            run_id=run_id,
            max_iterations=int(state.get("max_iterations", max_iterations)),
            available_features=list(current_features),
            artifact_dir=artifact_dir,
            cell=cell,
        )
        req_path = loop_protocol.write_request(artifact_dir, iter_n, req_payload)
        # (2) resume checkpoint — full loop state, no model blobs.
        ckpt_state = {
            "run_id": run_id,
            "iter_idx": iter_n,
            "max_iterations": int(state.get("max_iterations", max_iterations)),
            "current_features": list(state["current_features"]),
            "current_hp": dict(state["current_hp"]),
            "val_briers": list(state["val_briers"]),
            "hp_history": list(state["hp_history"]),
            "feature_history": list(state["feature_history"]),
            "hp_lists": list(state["hp_lists"]),
            "delta_attributions": list(state["delta_attributions"]),
            # V1.3 Option A — gap/Z + eval R-p@1 per-iter series + the
            # run-level anti_auc_flag + audit auto_disabled mapping. These
            # are populated by walk_forward_train into loop_state_sink (the
            # ``state`` dict) and persisted here so the resume-side
            # best_checkpoint sees the full history across the boundary.
            # Missing fields default to safe values on older checkpoints —
            # the read side tolerates absence (see _load_and_apply_resume).
            "train_val_gaps": list(state.get("train_val_gaps", [])),
            "spiegelhalter_zs": list(state.get("spiegelhalter_zs", [])),
            "eval_r_precision_at_1s": list(
                state.get("eval_r_precision_at_1s", [])
            ),
            "anti_auc_flag": str(state.get("anti_auc_flag", "unknown")),
            "auto_disabled": dict(state.get("auto_disabled", {})),
        }
        ckpt_path = gbdt_checkpoint.write_checkpoint(artifact_dir, ckpt_state)
        # (3) hand control back to the agent.
        raise loop_protocol.PauseForAgentDecision(
            iter_n=iter_n,
            request_path=req_path,
            checkpoint_path=ckpt_path,
            run_id=run_id,
        )
    return _cb


def _handle_scout_cycles_agent_mode(
    *,
    out_dir: Path,
    spec: dict,
    run_id: str,
    spec_path: Path,
    panel,
    X,
    y,
    sample_weights,
    split: SplitSpec,
    universe_calendar,
    backend_library: str,
    hp_starting: dict,
    calibration_method: str,
    calibration_z_threshold: float,
    random_seed: int,
    scout_cfg: dict,
    fs_prefit_cfg: dict,
    milestone,
    heartbeat,
    status,
    progress_log,
    # V1.3 Option B D6.2.A — FS-prefit kept-feature cache key inputs.
    # All four optional; when any is None the cache is bypassed and the
    # cycle-1 FS-prefit runs cold every time (back-compat / tests).
    fs_prefit_universe: str | None = None,
    fs_prefit_cache_root: str | None = None,
    fs_prefit_features_source_sha256: str | None = None,
    fs_prefit_snapshot_end_iso: str | None = None,
) -> dict | None:
    """V1.3 Option B P4 — runner-side scout cycles in agent_file_protocol mode.

    Detects which cycle we're in via the presence of scout/ files; runs the
    appropriate work; PAUSES (returns None) when there's nothing to fit yet.
    Returns ``{"hp_starting": dict, "features": list[str], "scout_report":
    dict}`` when cycle 3 is reached (iter_0_decision present); the caller
    proceeds to walk_forward_train with those values.

    Cycle 1 (no scout_results.jsonl yet) — fresh run path:
    - Run FS-prefit, then scout, on a single segment carve.
    - Write scout_results.jsonl + scout_bundle.json + combine_request.json.
    - PAUSE (return None); the runner exits cleanly and the agent picks up.

    Cycle 2 (combine_decision.json exists, no combine_results.json):
    - Load combine_decision.json (≤50 configs per D3b.A).
    - Fit each config in-process, score val/eval metrics, write
      combine_results.json.
    - PAUSE; the agent picks an iter_0 winner.

    Cycle 3 (iter_0_decision.json exists):
    - Load the agent's HP overlay; return it as ``hp_starting`` + the cliff-
      cut feature pool as ``features``. Caller proceeds to walk_forward_train.
    """
    cycle_state = gbdt_scout_io.detect_cycle_state(out_dir)

    def _translate_agent_overlay(
        overlay: dict, *, source_label: str,
    ) -> dict:
        """Apply scout._translate_for_backend to agent-supplied HP overlays
        (V1.3 Option B PR #125 Medium 2 fix). The speed-biased combine
        prompt uses XGBoost-canonical knob names; the agent will echo
        those names; for CatBoost specs the XGBoost-named keys would be
        silently dropped by make_model. Translate-then-fit gives consistent
        semantics with the runner's own grid-build path (which already
        translates XGBoost → CatBoost).

        Logs a one-line warning for every key the translation drops (e.g.
        ``gamma`` on CatBoost — no analog), so the agent gets feedback in
        the next request bundle.
        """
        if not overlay:
            return {}
        translated = gbdt_scout._translate_for_backend(
            dict(overlay), backend_library,
        )
        # XGBoost path: translation is pass-through, no drops to log.
        if backend_library == "xgboost":
            return translated
        # Detect dropped keys (in source but not in translation). Both
        # ``gamma`` (explicit drop) and any unknown-vocab key surface here.
        translated_xgb_names = set()
        for k_in in overlay.keys():
            if k_in == "gamma":
                continue    # known-drop; logged separately below
            if k_in in gbdt_scout._XGB_TO_CATBOOST:
                translated_xgb_names.add(k_in)
            elif k_in == "scale_pos_weight":
                translated_xgb_names.add(k_in)
        dropped = [
            k for k in overlay.keys()
            if k != "gamma" and k not in translated_xgb_names
        ]
        if "gamma" in overlay:
            milestone(
                f"[scout] {source_label}: dropped 'gamma' "
                f"(no CatBoost analog) — translation per V1.3 Option B D1"
            )
        if dropped:
            milestone(
                f"[scout] {source_label}: dropped unknown-vocab key(s) "
                f"{dropped} for backend={backend_library!r}"
            )
        return translated

    # Cycle 3 — iter_0 decision is ready; the agent has chosen the HP.
    if cycle_state.has_iter_0_decision:
        try:
            iter_0_decision = gbdt_scout_io.read_iter_0_decision(out_dir)
        except gbdt_scout_io.Iter0DecisionError:
            progress_log.close()
            raise
        raw_agent_hp = dict(iter_0_decision.get("hp") or {})
        agent_hp = _translate_agent_overlay(
            raw_agent_hp, source_label="cycle 3 iter_0_decision",
        )
        # Cliff-cut feature pool comes from cycle 1's scout_bundle (saved at
        # the cycle-1 exit). Fall back to all of X.columns if unavailable
        # (shouldn't happen for a clean run).
        try:
            bundle_payload = json.loads(
                gbdt_scout_io.scout_bundle_path(out_dir).read_text()
            )
            kept = list(bundle_payload.get("fs_prefit", {}).get(
                "kept_features", list(X.columns),
            ))
        except Exception:
            kept = list(X.columns)
        # Apply the agent overlay on top of the spec's hp_starting.
        composed_hp = dict(hp_starting)
        composed_hp.update(agent_hp)
        milestone(
            f"[scout] cycle 3 — iter_0_decision applied "
            f"(features={len(kept)}, hp_overlay_keys="
            f"{sorted(agent_hp.keys())})"
        )
        return {
            "hp_starting": composed_hp,
            "features": kept,
            "scout_report": bundle_payload,
        }

    # Cycle 2-complete pause — combine_results.json is on disk but the agent
    # has not yet written iter_0_decision.json. Bug #248 — without this
    # explicit branch the conditional ladder fell through to cycle 1 every
    # --resume, re-running FS-prefit + the scout response curves from scratch
    # (~5 min per resume) and clobbering combine_request.json / scout_bundle
    # / scout_results.jsonl. Pause cleanly here instead.
    if cycle_state.has_combine_results and not cycle_state.has_iter_0_decision:
        status.update(phase="scout_cycle_2", awaiting_decision=True)
        milestone(
            "[scout] cycle 2 already complete; awaiting iter_0_decision.json "
            "in scout/. Agent writes scout/iter_0_decision.json and reruns "
            "--resume."
        )
        return None

    # Carve once for cycles 1 + 2 (re-uses the same segments iter_0 will use).
    from gbdt.train import _carve_X_y
    parts = _carve_X_y(
        X, y, panel, split, list(X.columns), sample_weights,
        universe_calendar=universe_calendar,
    )
    X_tr_full, y_tr, _, w_tr = parts["train"]
    X_val_full, y_val, _, w_val = parts["val"]
    X_ev_full, y_ev, mi_ev, w_ev = parts["eval"]
    if len(y_tr) == 0:
        # Nothing to scout — let the caller fall through; the loop will
        # raise the empty-train error there.
        milestone(
            "[scout] WARNING: empty training segment; skipping scout cycles."
        )
        return {
            "hp_starting": dict(hp_starting),
            "features": list(X.columns),
            "scout_report": None,
        }

    # Cycle 2 — combine_decision exists, results pending.
    if cycle_state.has_combine_decision and not cycle_state.has_combine_results:
        try:
            decision = gbdt_scout_io.read_combine_decision(out_dir)
        except gbdt_scout_io.CombineDecisionError:
            progress_log.close()
            raise
        configs = decision.get("configs", [])
        milestone(
            f"[scout] cycle 2 — fitting {len(configs)} combine config(s) "
            f"(D3b.A cap = {gbdt_scout_io.COMBINE_DECISION_MAX_CONFIGS})"
        )
        # Reuse cycle 1's cliff-cut feature pool from scout_bundle.json.
        try:
            bundle_payload = json.loads(
                gbdt_scout_io.scout_bundle_path(out_dir).read_text()
            )
            kept = list(bundle_payload.get("fs_prefit", {}).get(
                "kept_features", list(X.columns),
            ))
        except Exception:
            kept = list(X.columns)
        X_tr = X_tr_full[kept]
        X_val = X_val_full[kept]
        X_ev = X_ev_full[kept]
        fit_one_cb = _build_combine_fit_one(
            backend=backend_library,
            hp_starting=hp_starting,
            feature_names=kept,
            random_seed=random_seed,
            calibration_method=calibration_method,
            calibration_z_threshold=calibration_z_threshold,
        )
        combine_results: list[dict] = []
        for i, cfg in enumerate(configs):
            t_cfg = time.time()
            raw_overlay = dict(cfg.get("hp") or {})
            # Translate agent-supplied XGBoost-vocab keys to the spec's
            # backend (V1.3 Option B PR #125 Medium 2). For XGBoost specs
            # this is pass-through; for CatBoost specs ``gamma`` is dropped
            # + the XGBoost-canonical keys map to their CatBoost
            # equivalents (per scout._translate_for_backend).
            overlay = _translate_agent_overlay(
                raw_overlay,
                source_label=f"cycle 2 combine cfg[{i}] ({cfg.get('label')})",
            )
            try:
                metrics = fit_one_cb(
                    hp_overlay=overlay,
                    X_train=X_tr, y_train=y_tr, w_train=w_tr,
                    X_val=X_val, y_val=y_val, w_val=w_val,
                    X_eval=X_ev, y_eval=y_ev, w_eval=w_ev,
                    mi_eval=mi_ev,
                )
                row = {
                    "index": i,
                    "label": cfg.get("label"),
                    # Preserve BOTH the agent's original keys (audit) AND
                    # the translated keys the model was actually fit on.
                    "hp_overlay_raw": raw_overlay,
                    "hp_overlay": overlay,
                    "metrics": metrics,
                    "fit_seconds": float(time.time() - t_cfg),
                    "status": "ok",
                }
            except Exception as exc:    # noqa: BLE001
                row = {
                    "index": i,
                    "label": cfg.get("label"),
                    "hp_overlay_raw": raw_overlay,
                    "hp_overlay": overlay,
                    "metrics": None,
                    "fit_seconds": float(time.time() - t_cfg),
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}"[:512],
                }
            combine_results.append(row)
        results_path = gbdt_scout_io.write_combine_results(
            out_dir, combine_results,
        )
        status.update(phase="scout_cycle_2", awaiting_decision=True)
        milestone(
            f"[scout] cycle 2 complete; wrote {results_path}. "
            f"Agent writes iter_0_decision.json + reruns --resume "
            f"--snapshot-end <DATE>."
        )
        return None

    # Cycle 1 — fresh run, no scout/ files yet. Run prefit + scout.
    milestone("[scout] cycle 1 — starting FS-prefit + scout response curves")
    heartbeat.set_phase("scout")
    status.update(phase="scout_cycle_1", awaiting_decision=False)

    # FS-prefit
    fs_prefit_enabled = bool(fs_prefit_cfg.get("enabled", True))
    kept_features = list(X.columns)
    fs_prefit_summary: dict = {"enabled": False}
    if fs_prefit_enabled:
        cliff_pct = float(fs_prefit_cfg.get("cliff_pct", 0.01))
        # D6.2.A cache key — only when all four inputs are supplied (runner
        # supplies them; tests may not).
        cache_key: str | None = None
        if (
            fs_prefit_universe is not None
            and fs_prefit_features_source_sha256 is not None
            and fs_prefit_snapshot_end_iso is not None
            and fs_prefit_cache_root is not None
        ):
            cache_key = gbdt_fs_prefit.fs_prefit_cache_key(
                universe=str(fs_prefit_universe),
                features_source_sha256=str(fs_prefit_features_source_sha256),
                snapshot_end=str(fs_prefit_snapshot_end_iso),
                default_hp_sha256=gbdt_fs_prefit.hp_sha256(
                    {"hp": dict(hp_starting), "cliff_pct": float(cliff_pct),
                     "backend": str(backend_library)},
                ),
            )

        prefit_result = None
        cache_hit = False
        if cache_key is not None:
            cached = gbdt_fs_prefit.load_fs_prefit_cache(
                fs_prefit_cache_root, cache_key,
            )
            if cached is not None:
                prefit_result = cached
                cache_hit = True
                milestone(
                    f"[scout] FS-prefit cache HIT (key={cache_key[:12]}…) — "
                    f"skipping fit; kept={len(prefit_result.kept_features)}"
                )

        if prefit_result is None:
            prefit_fit_one = _build_fs_prefit_runner_fit_one(
                backend=backend_library, random_seed=random_seed,
                feature_names=list(X.columns),
            )
            try:
                prefit_result = gbdt_fs_prefit.run_fs_prefit(
                    X_train=X_tr_full, y_train=y_tr, w_train=w_tr,
                    X_val=X_val_full, y_val=y_val, w_val=w_val,
                    fit_one=prefit_fit_one,
                    backend=backend_library,
                    default_hp=dict(hp_starting),
                    cliff_pct=cliff_pct,
                )
                if cache_key is not None:
                    try:
                        gbdt_fs_prefit.save_fs_prefit_cache(
                            fs_prefit_cache_root, cache_key, prefit_result,
                        )
                    except OSError:
                        # Non-fatal — cycle proceeds with the freshly-fit
                        # result; the next sibling cell will miss too.
                        pass
                milestone(
                    f"[scout] FS-prefit: kept {len(prefit_result.kept_features)}, "
                    f"dropped {len(prefit_result.dropped_features)} "
                    f"(cliff_pct={cliff_pct}, "
                    f"top_importance={prefit_result.top_importance:.4g}, "
                    f"cache_key={cache_key[:12] + '…' if cache_key else 'none'})"
                )
            except Exception as exc:    # noqa: BLE001
                fs_prefit_summary = {
                    "enabled": True, "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}"[:512],
                }
                milestone(
                    f"[scout] FS-prefit ERROR: {exc!r}; using full feature set"
                )
                prefit_result = None

        if prefit_result is not None:
            kept_features = list(prefit_result.kept_features)
            fs_prefit_summary = {
                "enabled": True,
                "cliff_pct": cliff_pct,
                "n_kept": len(prefit_result.kept_features),
                "n_dropped": len(prefit_result.dropped_features),
                "top_importance": prefit_result.top_importance,
                "cliff_threshold": prefit_result.cliff_threshold,
                "fit_seconds": prefit_result.fit_seconds,
                "kept_features": kept_features,
                "cache_hit": cache_hit,
                "cache_key": cache_key,
            }

    # Scout fits on the cliff-cut pool.
    X_tr = X_tr_full[kept_features]
    X_val = X_val_full[kept_features]
    X_ev = X_ev_full[kept_features]
    n_pos = int(np.sum(np.asarray(y_tr) == 1))
    n_neg = int(np.sum(np.asarray(y_tr) == 0))
    scout_fit_one = _build_scout_runner_fit_one(
        backend=backend_library, hp_starting=hp_starting,
        feature_names=kept_features, random_seed=random_seed,
        calibration_method=calibration_method,
        calibration_z_threshold=calibration_z_threshold,
    )
    spec_shim = {"backend": {"scout": scout_cfg}}
    t_scout = time.time()
    scout_results = gbdt_scout.run_scout(
        X_train=X_tr, y_train=y_tr, w_train=w_tr,
        X_val=X_val, y_val=y_val, w_val=w_val,
        X_eval=X_ev, y_eval=y_ev, w_eval=w_ev,
        mi_eval=mi_ev,
        fit_one=scout_fit_one,
        backend=backend_library,
        spec=spec_shim,
        per_config_timeout_seconds=scout_cfg.get("per_config_timeout_seconds"),
        soft_wall_clock_seconds=scout_cfg.get("wall_clock_cap_seconds"),
        n_positive=n_pos, n_negative=n_neg,
    )
    scout_runtime = time.time() - t_scout
    n_ok = sum(1 for r in scout_results if r.status == "ok")
    milestone(
        f"[scout] cycle 1 complete; {n_ok}/{len(scout_results)} configs ok "
        f"in {scout_runtime:.1f}s"
    )

    # Build the lex auto-compose to prefill into combine_request.
    lex_winner = gbdt_scout.lexicographic_winner(scout_results)
    per_knob = gbdt_scout.per_knob_winners(scout_results)
    defaults_metrics = next(
        (r.to_dict() for r in scout_results
          if r.config.knob_name == "defaults"),
        None,
    )

    # Write scout/ files (atomic temp+rename).
    scout_results_path = gbdt_scout_io.write_scout_results(
        out_dir, [r.to_dict() for r in scout_results],
    )
    bundle_payload = {
        "run_id": run_id,
        "backend": backend_library,
        "n_configs_total": len(scout_results),
        "n_configs_completed": n_ok,
        "runtime_seconds": scout_runtime,
        "defaults_metrics": defaults_metrics,
        "per_knob_winner": per_knob,
        "lexicographic_auto_compose": {
            "hp_overlay": dict(lex_winner.hp_overlay),
        },
        "fs_prefit": fs_prefit_summary,
        "grid_spec": dict(scout_cfg.get("grid", {})),
    }
    bundle_path = gbdt_scout_io.write_scout_bundle(out_dir, bundle_payload)
    combine_request_payload = {
        "run_id": run_id,
        "backend": backend_library,
        "prompt": gbdt_scout.SPEED_BIASED_COMBINE_PROMPT,
        "cap_n_configs": gbdt_scout_io.COMBINE_DECISION_MAX_CONFIGS,
        "lex_auto_compose_overlay": dict(lex_winner.hp_overlay),
        # Recommend the agent include the lex auto-compose as the zeroth
        # candidate so combine_results always has a "vs auto-compose"
        # comparator visible (plan § 4 D3b).
        "zeroth_candidate_recommendation": {
            "label": "lex_auto_compose",
            "hp": dict(lex_winner.hp_overlay),
        },
        "speed_biased_prompt": gbdt_scout.SPEED_BIASED_COMBINE_PROMPT,
        "scout_results_jsonl": str(scout_results_path),
        "scout_bundle_json": str(bundle_path),
    }
    request_path = gbdt_scout_io.write_combine_request(
        out_dir, combine_request_payload,
    )
    status.update(phase="scout_cycle_1", awaiting_decision=True)
    milestone(
        f"[scout] cycle 1 paused — wrote {request_path}. "
        f"Agent writes combine_decision.json and reruns: "
        f"uv run python -m gbdt experiment {spec_path.name} "
        f"--resume {run_id} --snapshot-end <DATE>"
    )
    return None


def _build_scout_runner_fit_one(
    *, backend: str, hp_starting: dict, feature_names: list[str],
    random_seed: int, calibration_method: str, calibration_z_threshold: float,
):
    """Runner-side fit closure for scout fits (cycle 1).

    Mirrors :func:`gbdt.train._build_scout_fit_one` so the runner can fit
    configs without going through ``walk_forward_train``. Returns the
    metrics dict ``run_scout`` expects from its ``fit_one`` callable.
    """
    from gbdt.train import _build_scout_fit_one
    return _build_scout_fit_one(
        backend=backend, current_hp=hp_starting,
        current_features=feature_names,
        random_seed=random_seed,
        calibration_method=calibration_method,
        calibration_z_threshold=calibration_z_threshold,
    )


def _build_fs_prefit_runner_fit_one(
    *, backend: str, random_seed: int, feature_names: list[str],
):
    from gbdt.train import _build_fs_prefit_fit_one
    return _build_fs_prefit_fit_one(
        backend=backend, random_seed=random_seed,
        feature_names=feature_names,
    )


def _build_scout_metrics_blocks(
    *,
    result,
    out_dir: Path,
    cycle_outcome: dict | None,
    callback_mode: str,
    scout_enabled: bool,
) -> tuple[dict | None, dict | None]:
    """V1.3 Option B P5 — build metrics.json::scout + metrics.json::combine.

    Returns ``(scout_block, combine_block)``; either or both may be None
    when scout was disabled / no data was produced.

    Data sources:
    - Default mode: ``result.scout_report`` (the in-train helper's payload).
    - Agent mode: ``cycle_outcome['scout_report']`` (the scout_bundle.json
      payload from cycle 3 — already in dict form).
    - Combine data in agent mode also reads ``scout/combine_results.json``
      if present, for the per-config fit metrics.
    """
    if not scout_enabled:
        return None, None

    scout_block: dict | None = None
    combine_block: dict | None = None

    # Default mode: in-train report.
    if callback_mode == "default" and result is not None and result.scout_report:
        rep = result.scout_report
        sb = rep.get("scout", {}) or {}
        cb = rep.get("combine", {}) or {}
        scout_block = {
            "enabled": True,
            "backend": sb.get("backend"),
            "n_configs_total": sb.get("n_configs_total"),
            "n_configs_completed": sb.get("n_configs_completed"),
            "runtime_seconds": sb.get("runtime_seconds"),
            "defaults_metrics": sb.get("defaults_metrics"),
            "per_knob_winner": sb.get("per_knob_winner"),
            "lexicographic_auto_compose": sb.get("lexicographic_auto_compose"),
            "status": sb.get("status"),
            "degenerate_sink_fallback": sb.get("degenerate_sink_fallback"),
            "grid_spec": sb.get("grid_spec"),
            "fs_prefit": rep.get("fs_prefit"),
        }
        combine_block = {
            "mode": "default",
            "status": cb.get("status"),
            "n_mix_configs_proposed": cb.get("n_mix_configs_completed", 0),
            "n_mix_configs_completed": cb.get("n_mix_configs_completed", 0),
            "exit_resume_rounds": 0,
            "agent_winner": None,
            "composed_overlay": cb.get("composed_overlay"),
            "vs_lexicographic_auto_compose": None,
        }
        return scout_block, combine_block

    # Agent mode: read from cycle_outcome + scout/ files.
    if callback_mode == "agent_file_protocol":
        bundle_payload = (
            (cycle_outcome or {}).get("scout_report") or {}
        )
        # Fall back to reading the file directly if the cycle outcome
        # didn't stash it (defense-in-depth).
        if not bundle_payload:
            bundle_path = gbdt_scout_io.scout_bundle_path(out_dir)
            if bundle_path.exists():
                try:
                    bundle_payload = json.loads(bundle_path.read_text())
                except Exception:    # noqa: BLE001
                    bundle_payload = {}
        scout_block = {
            "enabled": True,
            "backend": bundle_payload.get("backend"),
            "n_configs_total": bundle_payload.get("n_configs_total"),
            "n_configs_completed": bundle_payload.get("n_configs_completed"),
            "runtime_seconds": bundle_payload.get("runtime_seconds"),
            "defaults_metrics": bundle_payload.get("defaults_metrics"),
            "per_knob_winner": bundle_payload.get("per_knob_winner"),
            "lexicographic_auto_compose": bundle_payload.get(
                "lexicographic_auto_compose",
            ),
            "status": "agent_combine",
            "degenerate_sink_fallback": False,
            "grid_spec": bundle_payload.get("grid_spec"),
            "fs_prefit": bundle_payload.get("fs_prefit"),
        }

        # Combine — read combine_results.json + the agent's iter_0 decision.
        combine_results_path = gbdt_scout_io.combine_results_path(out_dir)
        agent_combine_results: list[dict] = []
        if combine_results_path.exists():
            try:
                payload = json.loads(combine_results_path.read_text())
                agent_combine_results = list(payload.get("configs") or [])
            except Exception:    # noqa: BLE001
                agent_combine_results = []
        iter_0_decision_path = gbdt_scout_io.iter_0_decision_path(out_dir)
        agent_winner_hp: dict | None = None
        if iter_0_decision_path.exists():
            try:
                payload = json.loads(iter_0_decision_path.read_text())
                agent_winner_hp = dict(payload.get("hp") or {})
            except Exception:    # noqa: BLE001
                agent_winner_hp = None

        n_completed = sum(
            1 for c in agent_combine_results
            if (c.get("status") or "").lower() == "ok"
        )
        lex_overlay = (
            (bundle_payload.get("lexicographic_auto_compose") or {})
            .get("hp_overlay")
        )
        vs_lex = (
            (agent_winner_hp == lex_overlay)
            if (agent_winner_hp is not None and lex_overlay is not None)
            else None
        )
        combine_block = {
            "mode": "agent_file_protocol",
            "status": (
                "agent_combine"
                if agent_winner_hp is not None
                else "scout_in_progress"
            ),
            "n_mix_configs_proposed": len(agent_combine_results),
            "n_mix_configs_completed": n_completed,
            "exit_resume_rounds": 2,
            "agent_winner": agent_winner_hp,
            "composed_overlay": agent_winner_hp,
            "vs_lexicographic_auto_compose": (
                {"is_lex": vs_lex, "lex_overlay": lex_overlay}
                if lex_overlay is not None else None
            ),
        }
        return scout_block, combine_block

    return None, None


def _build_combine_fit_one(
    *, backend: str, hp_starting: dict, feature_names: list[str],
    random_seed: int, calibration_method: str, calibration_z_threshold: float,
):
    """Runner-side fit closure for combine configs (cycle 2).

    Same shape as the scout fit closure — just a thin alias for clarity.
    """
    return _build_scout_runner_fit_one(
        backend=backend, hp_starting=hp_starting,
        feature_names=feature_names,
        random_seed=random_seed,
        calibration_method=calibration_method,
        calibration_z_threshold=calibration_z_threshold,
    )


def _load_and_apply_resume(
    out_dir: Path, spec: dict, *, run_id: str,
    progress_log: "loop_observability.ProgressLog | None" = None,
) -> dict:
    """Load the checkpoint + the agent's decision, validate + apply it.

    Returns the ``resume_state`` dict ``walk_forward_train`` seeds the loop with
    at iteration N+1 (plan § 0). The checkpoint (written by the
    agent-file-protocol callback when it paused at iter N) carries the iteration
    index, the accumulated history, and the iter-N features/HP. The decision at
    ``loop/iter_<N>_decision.json`` is validated against the spec
    (bounds / pinned / known features — :func:`loop_protocol.validate_decision`)
    and applied: ``prune_features`` removed + ``hp_changes`` merged → the
    features/HP that seed iter N+1.

    Raises a clear error (caller surfaces it) when the checkpoint is missing,
    the decision is missing/malformed, or the decision violates a constraint —
    the user fixes the decision file + relaunches ``--resume``.
    """
    def _log(message: str) -> None:
        # Mirror the [resume] milestones to stdout (unchanged) + progress.log.
        print(message, flush=True)
        if progress_log is not None:
            progress_log.append(message)

    ckpt = gbdt_checkpoint.read_checkpoint(out_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"[resume] no checkpoint at "
            f"{gbdt_checkpoint.checkpoint_path(out_dir)} — cannot --resume a run "
            f"that never paused. Launch the experiment first (without --resume)."
        )
    iter_n = int(ckpt["iter_idx"])
    _log(f"[resume] loaded checkpoint at iter {iter_n} (run_id={run_id})")

    decision = loop_protocol.read_decision(out_dir, iter_n)

    # Validate against the spec: HP bounds, no pinned-HP changes, prune_features
    # ⊆ the active feature set the checkpoint paused on. The HP names are
    # validated against the spec's backend table (V1.2 Phase 2 / plan § 6.2):
    # an xgboost spec validates against the *_XGB tables, a catboost spec (the
    # default when unset) against the CatBoost tables.
    known_features = list(ckpt["current_features"])
    backend = ((spec or {}).get("backend", {}) or {}).get("library", "catboost")
    loop_protocol.validate_decision(
        decision, spec, known_features, backend=backend,
    )

    next_features, next_hp, should_stop = loop_protocol.apply_decision(
        decision, known_features, dict(ckpt["current_hp"]),
    )
    n_pruned = len(known_features) - len(next_features)
    hp_changed = sorted((decision.get("hp_changes") or {}).keys())
    _log(
        f"[resume] RESUMED iter {iter_n + 1} (decision: pruned {n_pruned}, "
        f"hp_changes={hp_changed}, should_stop={should_stop})"
    )

    # The applied decision becomes iter N's recorded delta_attribution.
    prior_deltas = list(ckpt.get("delta_attributions", []))
    prior_deltas.append(decision.get("rationale", "agent decision (no rationale)"))

    return {
        "iter_idx": iter_n + 1,
        "current_features": next_features,
        "current_hp": next_hp,
        "val_briers": list(ckpt.get("val_briers", [])),
        "hp_history": list(ckpt.get("hp_history", [])),
        "feature_history": list(ckpt.get("feature_history", [])),
        "hp_lists": list(ckpt.get("hp_lists", [])),
        "delta_attributions": prior_deltas,
        "force_stop": should_stop,
        # V1.3 Option A — thread the per-iter signals + run-level flag back
        # to walk_forward_train so the resume-side finalization
        # (best_checkpoint) sees the full history. Older checkpoints
        # predating these fields default to empty/unknown, behavior unchanged.
        "train_val_gaps": list(ckpt.get("train_val_gaps", [])),
        "spiegelhalter_zs": list(ckpt.get("spiegelhalter_zs", [])),
        "eval_r_precision_at_1s": list(ckpt.get("eval_r_precision_at_1s", [])),
        "anti_auc_flag": str(ckpt.get("anti_auc_flag", "unknown")),
    }

