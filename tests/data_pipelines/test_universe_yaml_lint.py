"""Universe YAML lint — validates every
``configs/data_pipelines/domains/*/universe_*.yaml`` against the schema
documented in ``docs/data_pipelines/universe_yaml_spec.md``, and
cross-checks the ``universes::`` block in ``configs/gbdt/default.yaml``.

The schema is intentionally restrictive: extra top-level keys are rejected
so dead/decorative fields can't sneak back in (this lint exists in part to
prevent reintroduction of ``ticker_prefix`` and friends).

Spec: ``docs/data_pipelines/universe_yaml_spec.md`` is the single source of
truth — if you change this lint, update that doc in the same PR.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

# Allowed top-level keys in a universe YAML, per spec section 3.
_ALLOWED_UNIVERSE_KEYS: frozenset[str] = frozenset({
    "universe", "listed_at", "indices", "tickers",
})

# Allowed keys in a gbdt universes::<name> registry block, per spec section 5.
_ALLOWED_REGISTRY_KEYS: frozenset[str] = frozenset({
    "source", "index_ticker", "annualization_factor",
})

# Domain → set of acceptable ticker prefixes for that domain's constituents.
# Per spec section 3 "Ticker-prefix conventions".
_DOMAIN_TICKER_PREFIXES: dict[str, frozenset[str]] = {
    "nse_equities": frozenset({"NSE:"}),
    "us_equities": frozenset({"NASDAQ:", "NYSE:"}),
}

# Snake-case name pattern: lowercase + digits + underscores; no leading
# digit; no leading/trailing/double underscores.
_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

_UNIVERSES_DIR = REPO_ROOT / "configs" / "data_pipelines" / "domains"
_GBDT_DEFAULT = REPO_ROOT / "configs" / "gbdt" / "default.yaml"


def _discover_universe_yamls() -> list[Path]:
    return sorted(_UNIVERSES_DIR.glob("*/universe_*.yaml"))


_UNIVERSE_FILES = _discover_universe_yamls()


def test_discovery_found_files():
    """Guardrail: if the lint discovers zero files we'd otherwise pass
    vacuously. Fail loudly if the directory layout drifts."""
    assert len(_UNIVERSE_FILES) > 0, (
        f"no universe YAMLs found under {_UNIVERSES_DIR}; lint discovery "
        f"is broken or the directory layout has moved"
    )


# ---------------------------------------------------------------------------
# Per-file lint (parametrized over every discovered universe_*.yaml).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _UNIVERSE_FILES, ids=lambda p: p.name)
def test_universe_yaml_lints(path: Path) -> None:
    """One universe YAML, all schema rules at once."""
    raw = path.read_text()
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        pytest.fail(f"{path.name}: YAML parse error — {e}")

    assert isinstance(doc, dict), f"{path.name}: top-level must be a mapping"

    # --- 1. Allowed-keys check (rejects extras).
    extra = set(doc) - _ALLOWED_UNIVERSE_KEYS
    assert not extra, (
        f"{path.name}: unexpected top-level key(s) {sorted(extra)}; "
        f"allowed = {sorted(_ALLOWED_UNIVERSE_KEYS)} "
        f"(see docs/data_pipelines/universe_yaml_spec.md § 3)"
    )

    # --- 2. Required keys present.
    missing = _ALLOWED_UNIVERSE_KEYS - set(doc)
    assert not missing, (
        f"{path.name}: missing required top-level key(s) {sorted(missing)}"
    )

    # --- 3. universe: type + filename correspondence.
    universe_name = doc["universe"]
    assert isinstance(universe_name, str), (
        f"{path.name}: 'universe' must be a string, got {type(universe_name).__name__}"
    )
    assert _NAME_RE.match(universe_name), (
        f"{path.name}: 'universe' value {universe_name!r} is not snake_case "
        f"(pattern: {_NAME_RE.pattern})"
    )
    expected_stem = f"universe_{universe_name}"
    assert path.stem == expected_stem, (
        f"{path.name}: filename stem {path.stem!r} must match "
        f"'universe_{{universe}}' (= {expected_stem!r})"
    )

    # --- 4. listed_at: ISO date.
    listed_at = doc["listed_at"]
    assert isinstance(listed_at, datetime.date), (
        f"{path.name}: 'listed_at' must be a YAML date (YYYY-MM-DD), "
        f"got {type(listed_at).__name__} = {listed_at!r}"
    )

    # --- 5. indices: non-empty list of strings.
    indices = doc["indices"]
    assert isinstance(indices, list) and indices, (
        f"{path.name}: 'indices' must be a non-empty list, got {indices!r}"
    )
    for idx in indices:
        assert isinstance(idx, str) and idx, (
            f"{path.name}: every entry in 'indices' must be a non-empty "
            f"string; got {idx!r}"
        )

    # --- 6. tickers: non-empty list of strings, all fully prefixed for the
    #         file's domain.
    tickers = doc["tickers"]
    assert isinstance(tickers, list) and tickers, (
        f"{path.name}: 'tickers' must be a non-empty list, got {tickers!r}"
    )
    domain = path.parent.name
    allowed_prefixes = _DOMAIN_TICKER_PREFIXES.get(domain)
    assert allowed_prefixes is not None, (
        f"{path.name}: unknown domain directory {domain!r}; extend "
        f"_DOMAIN_TICKER_PREFIXES in the lint if you added a new domain"
    )
    bad: list[str] = []
    for t in tickers:
        if not (isinstance(t, str) and t):
            bad.append(repr(t))
            continue
        if not any(t.startswith(p) for p in allowed_prefixes):
            bad.append(t)
    assert not bad, (
        f"{path.name}: {len(bad)} ticker(s) lack the expected prefix "
        f"({sorted(allowed_prefixes)}) — first 5: {bad[:5]} "
        f"(see spec § 3 'Ticker-prefix conventions')"
    )


# ---------------------------------------------------------------------------
# gbdt registry lint (configs/gbdt/default.yaml :: universes).
# ---------------------------------------------------------------------------


def _load_gbdt_universes() -> dict[str, dict]:
    """Returns the universes:: block from the gbdt default config (or an
    empty dict if the file or block is missing — the per-entry tests will
    surface that by parameterizing over no entries, caught by the
    discovery guardrail below).
    """
    if not _GBDT_DEFAULT.exists():
        return {}
    doc = yaml.safe_load(_GBDT_DEFAULT.read_text()) or {}
    return doc.get("universes") or {}


_GBDT_UNIVERSES = _load_gbdt_universes()


def test_gbdt_universes_block_nonempty():
    """The gbdt registry should always carry at least one universe (nifty50
    is the worked example shipped on day 1)."""
    assert _GBDT_UNIVERSES, (
        f"no 'universes:' block found in {_GBDT_DEFAULT}; the lint can't "
        f"validate registry entries"
    )


@pytest.mark.parametrize("name", sorted(_GBDT_UNIVERSES), ids=lambda n: n)
def test_gbdt_universe_registry_entry(name: str) -> None:
    """Each gbdt registry block must match the schema in spec § 5 and
    point at an on-disk universe YAML whose 'universe' field matches the
    registry key."""
    block = _GBDT_UNIVERSES[name]
    assert isinstance(block, dict), (
        f"universes::{name}: must be a mapping, got {type(block).__name__}"
    )

    # --- 1. Allowed-keys check (rejects extras such as ticker_prefix).
    extra = set(block) - _ALLOWED_REGISTRY_KEYS
    assert not extra, (
        f"universes::{name}: unexpected key(s) {sorted(extra)}; allowed = "
        f"{sorted(_ALLOWED_REGISTRY_KEYS)} "
        f"(see docs/data_pipelines/universe_yaml_spec.md § 5)"
    )

    # --- 2. Required keys present.
    missing = _ALLOWED_REGISTRY_KEYS - set(block)
    assert not missing, (
        f"universes::{name}: missing required key(s) {sorted(missing)}"
    )

    # --- 3. Field types.
    source = block["source"]
    assert isinstance(source, str) and source, (
        f"universes::{name}: 'source' must be a non-empty string"
    )
    index_ticker = block["index_ticker"]
    assert isinstance(index_ticker, str) and index_ticker, (
        f"universes::{name}: 'index_ticker' must be a non-empty string"
    )
    ann = block["annualization_factor"]
    assert isinstance(ann, int) and not isinstance(ann, bool) and ann > 0, (
        f"universes::{name}: 'annualization_factor' must be a positive int, "
        f"got {ann!r}"
    )

    # --- 4. source: path exists and is a universe YAML inside
    #         configs/data_pipelines/domains/<domain>/.
    source_path = REPO_ROOT / source
    assert source_path.exists(), (
        f"universes::{name}: source path {source!r} does not exist "
        f"(resolved to {source_path})"
    )
    expected_root = _UNIVERSES_DIR.resolve()
    assert expected_root in source_path.resolve().parents, (
        f"universes::{name}: source must live under "
        f"configs/data_pipelines/domains/<domain>/; got {source!r}"
    )

    # --- 5. Cross-check: the YAML's own 'universe' field matches the
    #         registry key (catches typos where the YAML and the registry
    #         drift apart).
    yaml_doc = yaml.safe_load(source_path.read_text()) or {}
    yaml_universe = yaml_doc.get("universe")
    assert yaml_universe == name, (
        f"universes::{name}: source YAML's 'universe' field is "
        f"{yaml_universe!r}, expected {name!r} (registry/YAML out of sync)"
    )
