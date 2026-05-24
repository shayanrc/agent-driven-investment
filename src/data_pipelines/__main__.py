"""data_pipelines CLI.

Invocation: ``python -m data_pipelines <subcommand> [args]`` or via the
project's uv shim: ``uv run python -m data_pipelines ...``.

Subcommands:
  fetch <identifier> --start --end          single-identifier fetch
  seed --domain <name> [--universe sp500]   bulk-seed a universe
  reprocess [--identifier | --domain | --all]
                                             re-derive processed from raw
  list-cached [--domain <name>]              show on-disk cached identifiers
  purge --domain <name> --identifier <id>    remove cached identifier
  health [--domain <name>]                   provider reachability + key check
  list-domains                               show registered domains and chains
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from data_pipelines.cache import (
    list_cached_identifiers,
    merge_cache,
    purge_identifier,
    read_processed,
    write_processed_atomic,
)
from data_pipelines.domain import DomainRegistry
from data_pipelines.dispatch import fetch_with_meta
from data_pipelines.raw_store import list_raw

# Importing this triggers the us_equities registration side effect.
import data_pipelines.domains.us_equities  # noqa: F401
from data_pipelines.domains.us_equities import get_domain as get_us_equities_domain
from data_pipelines.domains.us_equities.universe import load_universe


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def _cmd_fetch(args) -> int:
    df, meta = fetch_with_meta(
        identifier=args.identifier,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        data_root=Path(args.data_root),
    )
    print(json.dumps(meta.to_dict(), indent=2))
    print(f"rows: {len(df)}")
    if args.head:
        print(df.head(args.head).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

def _cmd_seed(args) -> int:
    if args.domain != "us_equities":
        print(f"seed: unsupported domain {args.domain!r} in v1 (us_equities only)",
              file=sys.stderr)
        return 2

    universe = load_universe(args.universe)
    print(f"seeding {len(universe)} identifiers from universe={args.universe} ...")

    ok, failed = 0, []
    for ident in universe:
        try:
            _, meta = fetch_with_meta(
                identifier=ident,
                start=_parse_date(args.start),
                end=_parse_date(args.end),
                data_root=Path(args.data_root),
            )
            ok += 1
            print(f"  ✓ {ident:18s} rows={meta.row_count} "
                  f"cold={meta.cache_was_cold}")
        except Exception as e:
            failed.append({"identifier": ident, "error": str(e)})
            print(f"  ✗ {ident:18s} {e}", file=sys.stderr)

    print(f"\ndone: {ok}/{len(universe)} succeeded, {len(failed)} failed")
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# reprocess (D8 audit path: re-derive processed/ from raw/ without any API)
# ---------------------------------------------------------------------------

def _cmd_reprocess(args) -> int:
    data_root = Path(args.data_root)
    if args.domain != "us_equities":
        print(f"reprocess: only us_equities supported in v1", file=sys.stderr)
        return 2
    domain = _resolve_domain_by_name(args.domain)
    if domain is None:
        print(f"reprocess: domain {args.domain!r} is not registered", file=sys.stderr)
        return 2

    if args.identifier:
        identifiers = [args.identifier]
    else:
        identifiers = _cached_identifiers(data_root, domain.name)
    if not identifiers:
        print("reprocess: no cached identifiers found")
        return 0

    for ident in identifiers:
        exchange, ticker = domain.parse_identifier(ident)
        # Walk raw across all providers for this identifier.
        raws: list[tuple[str, Path]] = []
        for provider in ("stooq", "tiingo", "yfinance"):
            for p in list_raw(data_root, provider, domain.name, exchange, ticker):
                raws.append((provider, p))
        if not raws:
            print(f"  - {ident}: no raw files; skipping")
            continue

        # Sort by raw filename (= timestamp) for deterministic merge order.
        raws.sort(key=lambda pp: (pp[1].name, pp[0]))
        cached_df, cached_meta = None, None
        for provider, raw_path in raws:
            adapter = domain.adapters[provider]
            try:
                src_df = adapter.parse(raw_path)
                norm_df = domain.schema.normalize(
                    src_df,
                    source_column_map=getattr(adapter, "source_column_map", None),
                    provider=provider, identifier=ident,
                )
                domain.schema.validate(norm_df, provider=provider, identifier=ident)
            except Exception as e:
                print(f"  ! {ident} [{provider}/{raw_path.name}]: {e}", file=sys.stderr)
                continue
            new_source = {
                "provider": provider,
                "raw_file": raw_path.name,
                "covers": {
                    "start": norm_df["date"].iloc[0].date().isoformat(),
                    "end": norm_df["date"].iloc[-1].date().isoformat(),
                },
                **getattr(adapter, "extra_meta", {}),
            }
            cached_df, cached_meta = merge_cache(
                cached_df, norm_df, cached_meta, new_source, domain,
            )

        if cached_df is not None:
            write_processed_atomic(data_root, domain, ident, cached_df, cached_meta)
            print(f"  ✓ {ident}: rows={len(cached_df)} sources={len(cached_meta['sources'])}")
    return 0


# ---------------------------------------------------------------------------
# list-cached
# ---------------------------------------------------------------------------

def _cmd_list_cached(args) -> int:
    data_root = Path(args.data_root)
    domains = [args.domain] if args.domain else [d.name for d in DomainRegistry.registered_domains()]
    out = []
    for dname in domains:
        for ident in _cached_identifiers(data_root, dname):
            try:
                domain = DomainRegistry.resolve(ident)
            except Exception:
                continue
            df, meta = read_processed(data_root, domain, ident)
            if df is None:
                continue
            out.append({
                "identifier": ident,
                "domain": dname,
                "row_count": int(len(df)),
                "range": meta.get("range"),
                "providers": [s.get("provider") for s in meta.get("sources", [])],
            })
    print(json.dumps(out, indent=2))
    return 0


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------

def _cmd_purge(args) -> int:
    if not args.yes:
        print("purge: pass --yes to confirm deletion", file=sys.stderr)
        return 2
    data_root = Path(args.data_root)
    domain = DomainRegistry.resolve(args.identifier)
    existed = purge_identifier(data_root, domain, args.identifier)
    if existed:
        print(f"purged {args.identifier} from {domain.name} cache")
    else:
        print("nothing to purge")
    return 0


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

def _cmd_health(args) -> int:
    if args.domain:
        domain = _resolve_domain_by_name(args.domain)
        if domain is None:
            print(f"health: domain {args.domain!r} is not registered",
                  file=sys.stderr)
            return 2
        domains = [domain]
    else:
        domains = list(DomainRegistry.registered_domains())

    out = []
    for d in domains:
        adapters = getattr(d, "adapters", {})
        out.append({
            "domain": d.name,
            "adapters": {name: a.health_check() for name, a in adapters.items()},
        })
    print(json.dumps(out, indent=2))
    return 0


def _resolve_domain_by_name(name: str):
    for d in DomainRegistry.registered_domains():
        if d.name == name:
            return d
    return None


# ---------------------------------------------------------------------------
# list-domains
# ---------------------------------------------------------------------------

def _cmd_list_domains(args) -> int:
    out = []
    for d in DomainRegistry.registered_domains():
        adapters = list(getattr(d, "adapters", {}).keys())
        out.append({
            "name": d.name,
            "prefixes": list(d.identifier_prefixes),
            "adapters": adapters,
            "schema_columns": [c.name for c in d.schema.columns],
        })
    print(json.dumps(out, indent=2))
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cached_identifiers(data_root: Path, domain_name: str) -> list[str]:
    """Identifiers with a meta row in the SQLite cache for `domain_name`."""
    domain = _resolve_domain_by_name(domain_name)
    if domain is None:
        return []
    return list_cached_identifiers(data_root, domain)


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="data_pipelines")
    p.add_argument("--data-root", default="data",
                   help="Cache root (default: ./data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Fetch one identifier")
    f.add_argument("identifier")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--head", type=int, default=0,
                   help="Print first N rows of the result")
    f.set_defaults(func=_cmd_fetch)

    s = sub.add_parser("seed", help="Bulk-seed a universe")
    s.add_argument("--domain", default="us_equities")
    s.add_argument("--universe", default="sp500")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.set_defaults(func=_cmd_seed)

    r = sub.add_parser("reprocess", help="Re-derive processed from raw (no API)")
    r.add_argument("--domain", default="us_equities")
    grp = r.add_mutually_exclusive_group()
    grp.add_argument("--identifier")
    grp.add_argument("--all", action="store_true")
    r.set_defaults(func=_cmd_reprocess)

    lc = sub.add_parser("list-cached", help="List cached identifiers")
    lc.add_argument("--domain")
    lc.set_defaults(func=_cmd_list_cached)

    pg = sub.add_parser("purge", help="Remove a cached identifier")
    pg.add_argument("--identifier", required=True)
    pg.add_argument("--yes", action="store_true")
    pg.set_defaults(func=_cmd_purge)

    h = sub.add_parser("health", help="Provider liveness check")
    h.add_argument("--domain")
    h.set_defaults(func=_cmd_health)

    ld = sub.add_parser("list-domains", help="Show registered domains")
    ld.set_defaults(func=_cmd_list_domains)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
