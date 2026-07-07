"""Byte-identity guard for gbdt runner refactors.

Proves a refactor didn't change behavior: run one pinned-snapshot experiment
before the refactor, run it again after, and require the canonicalized
artifacts to match hash-for-hash.

Usage (from the repo root):

    # 1. on the pre-refactor commit
    uv run python -m scripts.gbdt.refactor_byte_guard run --label base

    # 2. after the refactor
    uv run python -m scripts.gbdt.refactor_byte_guard run --label cand

    # 3. gate
    uv run python -m scripts.gbdt.refactor_byte_guard compare base cand

``run`` copies the source spec (default: the v1 pilot cell) to a temp spec
named ``_refactor_guard.yaml`` — the runner derives the artifact dir from the
spec filename stem, so both runs land at
``results/gbdt/experiments/_refactor_guard/`` (gitignored) — pins
``--snapshot-end`` so cache growth between runs can't leak in, and caps the
FS+HP loop at ``--max-iterations`` (default 2) to keep a guard run in minutes.

Two comparison tiers:

- STRICT (mismatch fails the guard): ``metrics.json`` and ``iterations.jsonl``
  after scrubbing volatile keys (wall times, cache mtimes, code commit,
  staleness-vs-today fields), plus raw hashes of ``predictions/*.csv``,
  ``hp.yaml``, ``features.yaml``, ``report.md``.
- INFO (mismatch warns only): model binaries and ``calibration.pkl`` — binary
  container formats may embed non-semantic bytes; the prediction CSVs are the
  behavioral truth they feed.

Canonicalized JSON is also written next to each manifest so a strict mismatch
can be localized with plain ``diff``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
GUARD_DIR = REPO / "runs" / "gbdt" / "refactor_guard"
DEFAULT_SPEC = REPO / "configs" / "gbdt" / "experiments" / "nifty50_up_10pct_20d_pilot.yaml"
ARTIFACT_DIR = REPO / "results" / "gbdt" / "experiments" / "_refactor_guard"

# Keys whose values vary run-to-run without a behavior change. Scrubbed
# recursively by name wherever they appear (plus any key starting with
# "wall_time" — the runner emits several timing variants).
VOLATILE_KEYS = {
    "wall_seconds",
    "preflight",              # cache path/size/mtime, code commit/dirty, host detail
    "cache_age_days_by_ticker",
    "stale_tickers",
    "n_tickers_stale",
    "started_at",
    "finished_at",
    "timestamp",
    "matrix_hit",             # feature-matrix cache cold/warm — state, not behavior
}

# Artifact subtrees that are logs/plots, not behavior.
SKIP_DIRS = {"figs", "loop", "scout", "diagnose"}
SKIP_PREFIXES = ("_feature_matrix_cache",)

INFO_ONLY_SUFFIXES = (".ubj", ".cbm", ".pkl")


def _volatile(key) -> bool:
    return key in VOLATILE_KEYS or (
        isinstance(key, str) and key.startswith("wall_time"))


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if not _volatile(k)}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(path: Path) -> tuple[bytes, bool]:
    """(bytes to hash, was_canonicalized) for one artifact file."""
    if path.name == "metrics.json":
        data = _scrub(json.loads(path.read_text()))
        return json.dumps(data, sort_keys=True, indent=1).encode(), True
    if path.name == "iterations.jsonl":
        lines = [
            json.dumps(_scrub(json.loads(ln)), sort_keys=True)
            for ln in path.read_text().splitlines() if ln.strip()
        ]
        return ("\n".join(lines)).encode(), True
    return path.read_bytes(), False


def _build_manifest(artifact_dir: Path, canon_dir: Path) -> dict:
    strict: dict[str, str] = {}
    info: dict[str, str] = {}
    canon_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(artifact_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(artifact_dir)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if rel.name.startswith(SKIP_PREFIXES):
            continue
        data, canonicalized = _canonical_bytes(p)
        digest = _sha(data)
        if canonicalized:
            out = canon_dir / rel.name
            out.write_bytes(data)
        if p.suffix in INFO_ONLY_SUFFIXES:
            info[str(rel)] = digest
        else:
            strict[str(rel)] = digest
    return {"strict": strict, "info": info}


def cmd_run(args) -> int:
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(Path(args.spec).read_text())
    spec.setdefault("backend", {}).setdefault("fs_hp_loop", {})[
        "max_iterations"] = args.max_iterations
    temp_spec = GUARD_DIR / "_refactor_guard.yaml"
    temp_spec.write_text(yaml.safe_dump(spec, sort_keys=False))

    cmd = [
        sys.executable, "-m", "gbdt", "experiment", str(temp_spec),
        "--snapshot-end", args.snapshot_end, "--overwrite",
    ]
    print(f"[guard] running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        print(f"[guard] runner FAILED (exit {res.returncode})", file=sys.stderr)
        return res.returncode

    manifest = _build_manifest(ARTIFACT_DIR, GUARD_DIR / f"{args.label}.canon")
    out = GUARD_DIR / f"{args.label}.manifest.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"[guard] manifest: {out} "
          f"({len(manifest['strict'])} strict, {len(manifest['info'])} info)")
    return 0


def cmd_compare(args) -> int:
    a = json.loads((GUARD_DIR / f"{args.base}.manifest.json").read_text())
    b = json.loads((GUARD_DIR / f"{args.cand}.manifest.json").read_text())
    failed = False
    for tier, fatal in (("strict", True), ("info", False)):
        keys = sorted(set(a[tier]) | set(b[tier]))
        for k in keys:
            ha, hb = a[tier].get(k), b[tier].get(k)
            if ha == hb:
                continue
            state = "missing-in-base" if ha is None else (
                "missing-in-cand" if hb is None else "hash-mismatch")
            tag = "FAIL" if fatal else "warn"
            print(f"[guard] {tag}: {tier}/{k}: {state}")
            if fatal:
                failed = True
    if failed:
        print(f"[guard] BYTE-IDENTITY FAILED — diff the canon dirs:\n"
              f"  diff -r {GUARD_DIR / (args.base + '.canon')} "
              f"{GUARD_DIR / (args.cand + '.canon')}")
        return 1
    print("[guard] byte-identity holds")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="refactor_byte_guard")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run the guard experiment and emit a manifest")
    r.add_argument("--label", required=True, help="Manifest label (e.g. base, cand)")
    r.add_argument("--spec", default=str(DEFAULT_SPEC))
    r.add_argument("--snapshot-end", default="2026-06-30")
    r.add_argument("--max-iterations", type=int, default=2)
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare", help="Compare two manifests")
    c.add_argument("base")
    c.add_argument("cand")
    c.set_defaults(func=cmd_compare)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
