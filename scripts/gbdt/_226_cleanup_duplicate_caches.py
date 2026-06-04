"""Bug #226 P4 — one-off prune of duplicate universe-feature-cache entries.

Background
----------
The bug-#226 P2 forensics found 3 historical russell1000 universe-cache
entries with identical ``(n_rows, n_cols) = (6_059_743, 279)`` shapes,
written 2 days apart, that all should have collapsed to one shared entry.
The drift was between-process: an auto-fetch or scheduled refresh between
sweep cells shifted the panel's tuple set / right-edge date, drifted
``panel_signature``, and minted a fresh universe-cache key on every
sibling cell. P3 fixes the input drift (``--snapshot-end`` pins the
snapshot for the lifetime of a sweep). This script reclaims the disk
those duplicate cells already burned.

Logic
-----
1. Walk ``<data_root>/gbdt_feature_cache/``; pair each ``<key>.parquet``
   with its ``<key>.key.json`` sidecar.
2. Group entries by ``(n_rows, n_cols)`` — same shape ⇒ same universe
   family (the 279-column matrix is the gbdt v2 universe schema; the
   row-count is the panel's ``len(panel)`` which is universe-specific).
3. Inside each group, keep the newest parquet by mtime + its sidecar;
   delete the older duplicates.
4. **Skip the 9 test fixtures** (rows=1440, cols=6, ~0.1 MB each) —
   leave them alone (the brief explicitly excludes them).
5. Print a summary table: per-group, how many files kept vs deleted,
   bytes freed.

Idempotent — re-running after a clean pass deletes nothing. Safe to
re-run.

Run result (2026-06-04)
-----------------------
Reclaimed 20.43 GB (20917.9 MB) across 3 duplicate universe families:

  group                             total  kept  deleted    freed
  fixtures (rows=1440, cols=6)          9     9        0       —
  rows=  645189, cols=279               2     1        1   679.9 MB
  rows= 3638785, cols=279               3     1        2  7602.5 MB
  rows= 6059743, cols=279               3     1        2 12635.5 MB

17 universe-cache entries → 12 entries (5 duplicates pruned; 9 fixtures
untouched). The (6_059_743, 279) row contains the bug-#226 smoking-gun
trio (russell1000-shaped, written 2026-05-30, 05-31, 06-02); after
cleanup only the newest survives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Fixture cohort (per the P4 brief): rows=1440, cols=6, ~0.1 MB. Leave alone.
FIXTURE_ROWS = 1440
FIXTURE_COLS = 6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_root",
        nargs="?",
        default="/mnt/122CEE982CEE765F/cache_data/gbdt_feature_cache",
        help="Path to gbdt_feature_cache/ directory "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted; don't touch the filesystem.",
    )
    args = parser.parse_args(argv)

    root = Path(args.cache_root)
    if not root.is_dir():
        print(f"[226-cleanup] not a directory: {root}", file=sys.stderr)
        return 2

    # Walk sidecars; pair with parquets; bucket by (n_rows, n_cols).
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    skipped_no_parquet = 0
    for sidecar_path in sorted(root.glob("*.key.json")):
        key = sidecar_path.name[: -len(".key.json")]
        parquet_path = root / f"{key}.parquet"
        if not parquet_path.exists():
            skipped_no_parquet += 1
            continue
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[226-cleanup] skip {key[:12]} (sidecar unreadable: {exc})",
                  file=sys.stderr)
            continue
        n_rows = int(sidecar.get("n_rows", -1))
        n_cols = int(sidecar.get("n_cols", -1))
        size = parquet_path.stat().st_size
        mtime = parquet_path.stat().st_mtime
        groups[(n_rows, n_cols)].append({
            "key": key,
            "sidecar_path": sidecar_path,
            "parquet_path": parquet_path,
            "size_bytes": size,
            "mtime": mtime,
        })

    # Decide kept vs deleted per group.
    summary_rows: list[tuple[str, int, int, int, int]] = []
    total_freed = 0
    for (n_rows, n_cols), entries in sorted(groups.items()):
        # Skip the fixture cohort.
        if (n_rows, n_cols) == (FIXTURE_ROWS, FIXTURE_COLS):
            label = f"fixtures (rows={n_rows}, cols={n_cols})"
            summary_rows.append((label, len(entries), len(entries), 0, 0))
            continue

        # Sort newest-first.
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        keep = entries[:1]
        delete = entries[1:]

        label = f"rows={n_rows:>8}, cols={n_cols:>3}"
        freed_bytes = 0
        for e in delete:
            freed_bytes += e["size_bytes"]
            if args.dry_run:
                print(f"[226-cleanup] DRY-RUN would delete {e['key'][:12]} "
                      f"({e['size_bytes']/1024/1024:.1f} MB, "
                      f"mtime={e['mtime']:.0f})")
            else:
                try:
                    e["parquet_path"].unlink()
                    e["sidecar_path"].unlink()
                    print(f"[226-cleanup] deleted {e['key'][:12]} "
                          f"({e['size_bytes']/1024/1024:.1f} MB)")
                except OSError as exc:
                    print(f"[226-cleanup] FAILED to delete {e['key'][:12]}: "
                          f"{exc}", file=sys.stderr)
                    freed_bytes -= e["size_bytes"]
        summary_rows.append(
            (label, len(entries), len(keep), len(delete), freed_bytes),
        )
        total_freed += freed_bytes

    # Summary table.
    print()
    print(f"{'group':<32} {'total':>6} {'kept':>5} {'deleted':>8} {'freed':>12}")
    print("-" * 70)
    for label, total, kept, deleted, freed in summary_rows:
        freed_str = f"{freed/1024/1024:.1f} MB" if freed else "—"
        print(f"{label:<32} {total:>6} {kept:>5} {deleted:>8} {freed_str:>12}")
    print("-" * 70)
    print(f"{'TOTAL freed':<32} {' ':>6} {' ':>5} {' ':>8} "
          f"{total_freed/1024/1024:.1f} MB "
          f"({total_freed/1024/1024/1024:.2f} GB)")
    if skipped_no_parquet:
        print(f"\nSkipped {skipped_no_parquet} sidecar(s) with no matching "
              f"parquet (orphans — unrelated to this cleanup).")
    print()
    print("DONE." if not args.dry_run else "DRY-RUN — no files modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
