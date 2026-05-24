"""forecasters dispatcher CLI — backs `/forecast`, `/tune-preset`,
`/list-presets`, and a thin `/fetch-data` helper for use in tests.

Each subcommand exits non-zero on errors, with the error class + message
echoed to stderr. The stdout convention is small and structured:
  - `forecast`: prints the output directory path on a single line.
  - `tune`: prints the produced preset YAML path.
  - `fetch`: prints the cache identifier and row count.
  - `list-presets`: prints a table (or JSON with --json).

Per docs/forecasters/V1_PLAN.md §"Anti-goals", this CLI does NOT expose
backend-internal flags like --n-eff or --weights. Override hyperparameters
via --config-overrides path-to-yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402  — must be set before pyplot import below
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from forecasters.cache import cache_key, read_cached, write_cached
from forecasters.data import prepare_data
from forecasters.dispatch import dispatch_forecast, dispatch_tune
from forecasters.errors import (
    PresetSchemaError,
    ResultContractError,
    UnknownBackendError,
    UnknownPresetError,
)
from forecasters.presets import list_presets, load_preset


log = logging.getLogger("forecasters.run")


# ----------------------------------------------------------------------------
# Subcommand: forecast
# ----------------------------------------------------------------------------


def _cmd_forecast(args: argparse.Namespace) -> int:
    # Resolve preset.
    preset = load_preset(args.preset)

    # Resolve data.
    df = prepare_data(
        identifier=args.identifier,
        data_path=args.data_path,
        start=args.start,
        end=args.end,
    )

    # Merge hyperparameter overrides.
    hp = dict(preset["hyperparameters"])
    if args.config_overrides:
        overrides = yaml.safe_load(Path(args.config_overrides).read_text()) or {}
        if not isinstance(overrides, dict):
            print(
                f"--config-overrides must be a YAML mapping; got {type(overrides).__name__}",
                file=sys.stderr,
            )
            return 2
        hp.update(overrides)

    # Cache key + lookup.
    key = cache_key(
        preset_name=preset["name"],
        preset_content_hash=preset["__content_hash__"],
        identifier=args.identifier,
        data_path=args.data_path,
        start=args.start,
        end=args.end,
        origin=args.origin,
        horizon=args.horizon,
        seed=args.seed,
    )
    cache_root = Path(args.cache_path) if args.cache_path else None
    if not args.no_cache:
        cached = read_cached(key, cache_root=cache_root)
        if cached is not None:
            out_dir = (Path(args.cache_path) if args.cache_path else Path("results/forecasters/forecasts")) / key
            print(str(out_dir.resolve()))
            return 0

    # Dispatch.
    input_dict = {
        "data": df,
        "origin": args.origin,
        "horizon": args.horizon,
        "hyperparameters": hp,
        "seed": args.seed,
    }
    result = dispatch_forecast(preset, input_dict)

    # Write to cache (or to --output-dir if specified — for callers that want
    # a known on-disk location even with --no-cache).
    if args.output_dir:
        out_root = Path(args.output_dir)
        out_dir = out_root / key
        # Skip the cache, but still write the artifacts to the requested dir.
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(exist_ok=True)
        _write_artifacts(out_dir, result)
    elif args.no_cache:
        # No on-disk artifacts; just print "stdout" of nothing meaningful.
        # To stay useful, dump a tempdir.
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="forecast_nocache_")) / key
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_artifacts(out_dir, result)
    else:
        out_dir = write_cached(key, result, cache_root=cache_root)
        _augment_with_fan_chart(out_dir, result)

    print(str(out_dir.resolve()))
    return 0


def _write_artifacts(out_dir: Path, result: dict[str, Any]) -> None:
    """Mirror cache write to an explicit output dir (no atomic-rename needed)."""
    summary = {k: v for k, v in result.items() if k != "paths"}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    np.savez_compressed(out_dir / "paths.npz", paths=result["paths"])
    (out_dir / "warnings.json").write_text(json.dumps(result.get("warnings", []), indent=2))
    _augment_with_fan_chart(out_dir, result)


def _augment_with_fan_chart(out_dir: Path, result: dict[str, Any]) -> None:
    """Render a small fan chart PNG into the output directory."""
    try:
        s = result["summary"]
        h = len(s["median"])
        x = np.arange(1, h + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(x, s["p05"], s["p95"], alpha=0.2, label="5–95% band")
        ax.fill_between(x, s["p25"], s["p75"], alpha=0.35, label="25–75% band")
        ax.plot(x, s["median"], color="black", lw=1.5, label="median")
        ax.set_xlabel("forecast step (trading days)")
        ax.set_ylabel("price")
        title_meta = result["metadata"]
        ax.set_title(
            f"{title_meta.get('preset_name', 'preset')} — origin "
            f"{result['anchors']['origin_date']} — backend "
            f"{title_meta.get('backend_name', '?')}"
        )
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fan_chart.png", dpi=110)
        plt.close(fig)
    except Exception as e:  # pragma: no cover - chart rendering is best-effort
        log.warning("fan chart rendering failed: %s", e)


# ----------------------------------------------------------------------------
# Subcommand: tune
# ----------------------------------------------------------------------------


def _cmd_tune(args: argparse.Namespace) -> int:
    df = prepare_data(
        identifier=args.identifier,
        data_path=args.data_path,
        start=args.start,
        end=args.end,
    )
    search_config: dict[str, Any] = {}
    if args.search_config:
        loaded = yaml.safe_load(Path(args.search_config).read_text()) or {}
        if not isinstance(loaded, dict):
            print(f"--search-config must be a YAML mapping; got {type(loaded).__name__}", file=sys.stderr)
            return 2
        search_config = loaded
    identifier = args.identifier or args.data_path or "<unknown>"
    input_dict = {
        "data": df,
        "identifier": identifier,
        "range": (args.start, args.end),
        "search_config": search_config,
        "seed": args.seed,
        "output_name": args.output_preset,
    }
    preset = dispatch_tune(args.backend, input_dict)
    out_root = Path(args.output_root) if args.output_root else Path("results/forecasters/presets")
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.output_preset}.yaml"
    # Round-trip via YAML — strip any private/loader-only keys before writing.
    serializable = {k: v for k, v in preset.items() if not k.startswith("__")}
    out_path.write_text(yaml.safe_dump(serializable, sort_keys=False, default_flow_style=False))
    print(str(out_path.resolve()))
    return 0


# ----------------------------------------------------------------------------
# Subcommand: fetch
# ----------------------------------------------------------------------------


def _cmd_fetch(args: argparse.Namespace) -> int:
    # Thin convenience wrapper — the real /fetch-data skill is owned by
    # data_pipelines and ships in Stage 8. This subcommand exists so the
    # forecasters CLI is callable end-to-end during stage tests.
    from data_pipelines import fetch_with_meta
    df, meta = fetch_with_meta(args.identifier, start=args.start, end=args.end)
    print(json.dumps({
        "rows": int(len(df)),
        "identifier": meta.identifier,
        "domain": meta.domain,
        "range": meta.range,
        "cache_was_cold": meta.cache_was_cold,
        "providers_failed": meta.providers_failed,
    }, indent=2))
    return 0


# ----------------------------------------------------------------------------
# Subcommand: list-presets
# ----------------------------------------------------------------------------


def _cmd_list_presets(args: argparse.Namespace) -> int:
    rows = list_presets(backend=args.backend)
    if args.json:
        print(json.dumps(rows, indent=2, default=_json_default))
        return 0
    if not rows:
        print("(no presets found)")
        return 0
    cols = ["name", "source", "backend", "fitted_on_identifier",
            "fitted_on_start", "fitted_on_end", "fitted_at", "crps_mean", "error"]
    widths = {c: max(len(c), max(len(str(r.get(c) or "")) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c) or "").ljust(widths[c]) for c in cols))
    return 0


# ----------------------------------------------------------------------------
# Argparse
# ----------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.forecasters.run")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast", help="run a saved preset against data")
    src = f.add_mutually_exclusive_group(required=True)
    src.add_argument("--identifier", help="data_pipelines identifier, e.g. NASDAQ:AAPL")
    src.add_argument("--data-path", help="local CSV/parquet path")
    f.add_argument("--preset", required=True)
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--origin", required=True)
    f.add_argument("--horizon", type=int, required=True)
    f.add_argument("--config-overrides", help="YAML file with hyperparameter overrides")
    f.add_argument("--seed", type=int, default=None)
    f.add_argument("--no-cache", action="store_true")
    f.add_argument("--cache-path", help="override the cache root directory")
    f.add_argument("--output-dir", help="explicit output dir (skips cache write)")
    f.set_defaults(fn=_cmd_forecast)

    t = sub.add_parser("tune", help="tune a backend on data; write a preset YAML")
    tsrc = t.add_mutually_exclusive_group(required=True)
    tsrc.add_argument("--identifier")
    tsrc.add_argument("--data-path")
    t.add_argument("--backend", required=True)
    t.add_argument("--start", required=True)
    t.add_argument("--end", required=True)
    t.add_argument("--output-preset", required=True)
    t.add_argument("--search-config", help="YAML file with the backend's search-grid spec")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--output-root", help="override results/forecasters/presets/")
    t.set_defaults(fn=_cmd_tune)

    g = sub.add_parser("fetch", help="thin /fetch-data wrapper (data_pipelines)")
    g.add_argument("--identifier", required=True)
    g.add_argument("--start", required=True)
    g.add_argument("--end", required=True)
    g.set_defaults(fn=_cmd_fetch)

    lp = sub.add_parser("list-presets", help="enumerate canonical + user-tuned presets")
    lp.add_argument("--json", action="store_true")
    lp.add_argument("--backend", help="filter by backend name")
    lp.set_defaults(fn=_cmd_list_presets)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        return int(args.fn(args))
    except (PresetSchemaError, UnknownPresetError, UnknownBackendError,
            ResultContractError, FileNotFoundError, ValueError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
