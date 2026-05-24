"""data_pipelines skill runner — backs `/fetch-data` and `/data-health`.

Two subcommands lifted onto the agent-callable surface (per
docs/forecasters/V1_PLAN.md §"Stage 8"). Ownership of the implementation
stays with the data_pipelines module; this file is just the user-facing
entry point.

Subcommands:
  fetch   --identifier --start --end [--frequency daily] [--json]
  health  [--identifier <id>] [--domain <name>] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Register all domains so DomainRegistry.resolve() works for any prefix.
import data_pipelines.domains.us_equities  # noqa: F401
import data_pipelines.domains.nse_equities  # noqa: F401

from data_pipelines import fetch_with_meta
from data_pipelines.cache import (
    list_cached_identifiers,
    processed_db_path,
    read_processed,
)
from data_pipelines.domain import DomainRegistry
from data_pipelines.errors import (
    AllProvidersFailed,
    MissingAPIKey,
    ProviderError,
    UnknownDomain,
)


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        df, meta = fetch_with_meta(
            args.identifier,
            start=args.start,
            end=args.end,
            frequency=args.frequency,
        )
    except (UnknownDomain, MissingAPIKey, ProviderError, AllProvidersFailed) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    db_path = processed_db_path(Path("data"))
    summary = {
        "identifier": meta.identifier,
        "domain": meta.domain,
        "rows": int(len(df)),
        "range": meta.range,
        "cache_was_cold": meta.cache_was_cold,
        "gaps_filled": meta.gaps_filled,
        "providers_failed": meta.providers_failed,
        "cache_path": str(db_path),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{summary['identifier']}  rows={summary['rows']}  "
              f"range={summary['range']['start']}..{summary['range']['end']}  "
              f"cache_was_cold={summary['cache_was_cold']}  "
              f"providers_failed={len(summary['providers_failed'])}")
        print(f"cache: {summary['cache_path']}")
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root) if args.data_root else Path("data")
    if args.identifier is not None:
        try:
            domain = DomainRegistry.resolve(args.identifier)
        except UnknownDomain as e:
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            return 2
        df, meta = read_processed(data_root, domain, args.identifier)
        if df is None:
            payload = {
                "identifier": args.identifier,
                "domain": domain.name,
                "cached": False,
            }
        else:
            payload = {
                "identifier": args.identifier,
                "domain": domain.name,
                "cached": True,
                "schema_version": meta["schema_version"],
                "rows": int(len(df)),
                "range": meta["range"],
                "last_fetch_utc": meta["last_fetch_utc"],
                "sources": meta["sources"],
            }
        _emit_health(payload, args.json)
        return 0

    # No identifier — either filter by --domain or report the whole cache.
    domains = (
        [DomainRegistry._by_prefix[p] for p in sorted(DomainRegistry._by_prefix.keys())]
        if not args.domain
        else _resolve_domain_by_name(args.domain)
    )
    # Dedup by name (a domain registers under multiple prefixes).
    seen: set[str] = set()
    domain_objs = []
    for d in domains:
        if d.name in seen:
            continue
        seen.add(d.name)
        domain_objs.append(d)

    overall = {
        "data_root": str(data_root),
        "report_generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "per_domain": {},
        "total_identifiers": 0,
        "total_rows": 0,
        "oldest_last_fetch_utc": None,
        "newest_last_fetch_utc": None,
    }
    olds: list[str] = []
    news: list[str] = []
    for d in domain_objs:
        identifiers = list_cached_identifiers(data_root, d)
        rows_for_domain = 0
        per_id = []
        for ident in identifiers:
            sub_df, sub_meta = read_processed(data_root, d, ident)
            if sub_df is None:
                continue
            n = int(len(sub_df))
            rows_for_domain += n
            per_id.append({
                "identifier": ident,
                "rows": n,
                "range": sub_meta["range"],
                "last_fetch_utc": sub_meta["last_fetch_utc"],
            })
            olds.append(sub_meta["last_fetch_utc"])
            news.append(sub_meta["last_fetch_utc"])
        overall["per_domain"][d.name] = {
            "identifier_count": len(identifiers),
            "rows": rows_for_domain,
            "identifiers": per_id,
        }
        overall["total_identifiers"] += len(identifiers)
        overall["total_rows"] += rows_for_domain
    if olds:
        overall["oldest_last_fetch_utc"] = min(olds)
        overall["newest_last_fetch_utc"] = max(news)
    _emit_health(overall, args.json)
    return 0


def _resolve_domain_by_name(name: str):
    """Look up a Domain by .name (rather than prefix)."""
    seen = set()
    for d in DomainRegistry._by_prefix.values():
        if d.name in seen:
            continue
        seen.add(d.name)
        if d.name == name:
            return [d]
    raise SystemExit(f"unknown domain {name!r}; "
                     f"known: {sorted({d.name for d in DomainRegistry._by_prefix.values()})}")


def _emit_health(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    if "per_domain" in payload:
        # Overall report.
        print(f"data_root: {payload['data_root']}")
        print(f"generated: {payload['report_generated_utc']}")
        print(f"total identifiers: {payload['total_identifiers']}")
        print(f"total rows: {payload['total_rows']}")
        if payload["oldest_last_fetch_utc"] is not None:
            print(f"oldest last_fetch_utc: {payload['oldest_last_fetch_utc']}")
            print(f"newest last_fetch_utc: {payload['newest_last_fetch_utc']}")
        for dname, info in payload["per_domain"].items():
            print(f"  {dname}: {info['identifier_count']} identifiers, {info['rows']} rows")
        return
    # Single-identifier report.
    if not payload.get("cached"):
        print(f"{payload['identifier']} ({payload['domain']}): NOT CACHED")
        return
    print(f"{payload['identifier']} ({payload['domain']})")
    print(f"  rows={payload['rows']} range={payload['range']['start']}..{payload['range']['end']}")
    print(f"  last_fetch_utc={payload['last_fetch_utc']}")
    print(f"  sources:")
    for src in payload["sources"]:
        print(f"    - provider={src.get('provider')}  "
              f"adjustment_quality={src.get('adjustment_quality', '?')}  "
              f"covers={src.get('covers', {})}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.data_pipelines.skill_runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="single-identifier fetch via data_pipelines.fetch")
    f.add_argument("--identifier", required=True)
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--frequency", default="daily")
    f.add_argument("--json", action="store_true")
    f.set_defaults(fn=_cmd_fetch)

    h = sub.add_parser("health", help="cache coverage / freshness report")
    h.add_argument("--identifier")
    h.add_argument("--domain")
    h.add_argument("--data-root")
    h.add_argument("--json", action="store_true")
    h.set_defaults(fn=_cmd_health)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
