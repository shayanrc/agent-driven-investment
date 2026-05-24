"""Stage 1 tests: Domain ABC + DomainRegistry."""

from __future__ import annotations

import pytest

from data_pipelines.domain import Domain, DomainRegistry
from data_pipelines.errors import UnknownDomain
from data_pipelines.schema import ColumnSpec, Schema


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Wipe the process-global registry around every test."""
    DomainRegistry._reset()
    yield
    DomainRegistry._reset()


def _make_domain(name: str, prefixes: tuple[str, ...]) -> Domain:
    schema = Schema(columns=(ColumnSpec("date", "datetime64[ns]"),
                             ColumnSpec("value", "float64")))

    class _NullCalendar:
        def trading_days(self, start, end): return []

    class _D(Domain):
        @property
        def name(self): return name
        @property
        def identifier_prefixes(self): return prefixes
        @property
        def schema(self): return schema
        @property
        def calendar(self): return _NullCalendar()
        def parse_identifier(self, identifier):
            prefix, sym = identifier.split(":", 1)
            return prefix, sym
        def chain_for_gap(self, identifier, gap_size_trading_days, has_cache):
            return []

    return _D()


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        Domain()  # type: ignore[abstract]


def test_register_and_resolve_single_prefix():
    d = _make_domain("us_equities", ("NYSE",))
    DomainRegistry.register(d)
    assert DomainRegistry.resolve("NYSE:AAPL") is d


def test_register_multiple_prefixes_one_domain():
    d = _make_domain("us_equities", ("NYSE", "NASDAQ", "INDEX"))
    DomainRegistry.register(d)
    assert DomainRegistry.resolve("NYSE:AAPL") is d
    assert DomainRegistry.resolve("NASDAQ:MSFT") is d
    assert DomainRegistry.resolve("INDEX:^SPX") is d


def test_unknown_prefix_raises():
    DomainRegistry.register(_make_domain("us_equities", ("NYSE",)))
    with pytest.raises(UnknownDomain) as exc:
        DomainRegistry.resolve("FRED:DGS10")
    assert "FRED:DGS10" in str(exc.value)
    assert "NYSE" in str(exc.value)


def test_identifier_without_prefix_raises():
    DomainRegistry.register(_make_domain("us_equities", ("NYSE",)))
    with pytest.raises(UnknownDomain):
        DomainRegistry.resolve("AAPL")


def test_duplicate_prefix_different_domain_rejected():
    DomainRegistry.register(_make_domain("us_equities", ("NYSE",)))
    with pytest.raises(ValueError, match="duplicate domain prefix"):
        DomainRegistry.register(_make_domain("other", ("NYSE",)))


def test_re_register_same_domain_instance_idempotent():
    d = _make_domain("us_equities", ("NYSE",))
    DomainRegistry.register(d)
    DomainRegistry.register(d)  # same instance — no raise
    assert DomainRegistry.resolve("NYSE:AAPL") is d


def test_registered_prefixes_and_domains():
    d1 = _make_domain("us_equities", ("NYSE", "NASDAQ"))
    d2 = _make_domain("fred_macro", ("FRED",))
    DomainRegistry.register(d1)
    DomainRegistry.register(d2)
    assert DomainRegistry.registered_prefixes() == ["FRED", "NASDAQ", "NYSE"]
    assert set(DomainRegistry.registered_domains()) == {d1, d2}


# Identifier safety — these values are interpolated into cache.py DDL via
# f-strings (sqlite3 can't parameterize identifiers). Reject at registration.

@pytest.mark.parametrize("bad_name", [
    "us; DROP TABLE x",  # injection payload
    "us-equities",        # hyphen
    "us equities",        # space
    "1us_equities",       # leading digit
    "",                    # empty
    "a" * 65,              # too long
    "us`equities",        # backtick
    "us'equities",        # quote
])
def test_register_rejects_unsafe_domain_name(bad_name):
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        DomainRegistry.register(_make_domain(bad_name, ("NYSE",)))


def test_register_rejects_unsafe_schema_column_name():
    schema = Schema(columns=(ColumnSpec("date", "datetime64[ns]"),
                             ColumnSpec("value); DROP TABLE x; --", "float64")))

    class _NullCalendar:
        def trading_days(self, start, end): return []

    class _D(Domain):
        @property
        def name(self): return "us_equities"
        @property
        def identifier_prefixes(self): return ("NYSE",)
        @property
        def schema(self): return schema
        @property
        def calendar(self): return _NullCalendar()
        def parse_identifier(self, identifier):
            return identifier.split(":", 1)
        def chain_for_gap(self, identifier, gap_size_trading_days, has_cache):
            return []

    with pytest.raises(ValueError, match="invalid SQL identifier.*column"):
        DomainRegistry.register(_D())


def test_register_accepts_valid_names():
    # Sanity: the validator should pass for normal-looking identifiers.
    DomainRegistry.register(_make_domain("us_equities", ("NYSE",)))
    DomainRegistry.register(_make_domain("fred_macro_v2", ("FRED",)))
