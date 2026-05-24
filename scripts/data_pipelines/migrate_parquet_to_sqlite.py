"""One-shot migration: read existing parquet+meta processed/ cache and write
the same data into the new SQLite cache (data/processed.db).

Usage:
    uv run python scripts/data_pipelines/migrate_parquet_to_sqlite.py \
        --data-root ./data [--delete-after]

Idempotent (rerunning over the SQLite-only state is a no-op). The
--delete-after flag removes the parquet tree after a successful migration —
omit it if you want to keep parquet around for one verification cycle.

Run before any new fetch / seed against the SQLite cache. New fetches against
the current pipeline write to processed.db, so leaving the old parquet tree
in place is harmless but consumes disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure src/ on path when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import data_pipelines  # noqa: F401  — env load + side-effect domain registration via import below
import data_pipelines.domains.us_equities  # noqa: F401  — registers the domain
from data_pipelines.cache import write_processed_atomic
from data_pipelines.domain import DomainRegistry


def _resolve_domain_by_name(name: str):
    for d in DomainRegistry.registered_domains():
        if d.name == name:
            return d
    return None


def walk_parquet_cache(data_root: Path):
    """Yield (domain_obj, identifier, df, meta) tuples for every parquet
    cached entry under data_root/processed/.
    """
    base = data_root / "processed"
    if not base.is_dir():
        return
    for domain_dir in sorted(base.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = _resolve_domain_by_name(domain_dir.name)
        if domain is None:
            print(f"  ! unknown domain {domain_dir.name!r}; skipping",
                  file=sys.stderr)
            continue
        for exchange_dir in sorted(domain_dir.iterdir()):
            if not exchange_dir.is_dir():
                continue
            for ticker_dir in sorted(exchange_dir.iterdir()):
                if not ticker_dir.is_dir():
                    continue
                parquet = ticker_dir / "daily.parquet"
                meta = ticker_dir / "_meta.json"
                if not (parquet.is_file() and meta.is_file()):
                    continue
                df = pd.read_parquet(parquet)
                meta_obj = json.loads(meta.read_text())
                identifier = f"{exchange_dir.name}:{ticker_dir.name}"
                yield domain, identifier, df, meta_obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--delete-after", action="store_true",
                        help="rm -rf data/processed/<domain>/<exch>/ trees "
                             "after successful migration")
    args = parser.parse_args()
    data_root = Path(args.data_root)

    print(f"Migration: parquet ({data_root}/processed/) → SQLite "
          f"({data_root}/processed.db)")
    t0 = time.time()

    migrated = 0
    failed: list[str] = []
    domains_touched: set[str] = set()

    for domain, identifier, df, meta in walk_parquet_cache(data_root):
        try:
            write_processed_atomic(data_root, domain, identifier, df, meta)
            domains_touched.add(domain.name)
            migrated += 1
            if migrated % 50 == 0:
                print(f"  ... {migrated} migrated ({time.time()-t0:.1f}s elapsed)")
        except Exception as e:
            print(f"  ! {identifier}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            failed.append(identifier)

    print(f"\nDone: {migrated} migrated, {len(failed)} failed, "
          f"{time.time()-t0:.1f}s")
    if failed:
        print("Failures:", failed[:20], file=sys.stderr)
        return 1

    if args.delete_after:
        for dname in domains_touched:
            tree = data_root / "processed" / dname
            if tree.is_dir():
                shutil.rmtree(tree)
                print(f"  removed {tree}")
    else:
        print("Note: parquet tree left in place. Re-run with --delete-after "
              "to free disk once you've verified the SQLite cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
