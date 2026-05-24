"""Stage 1 tests: Adapter ABC contract."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data_pipelines.adapter import Adapter


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        Adapter()  # type: ignore[abstract]


def test_subclass_must_implement_fetch_and_parse():
    class HalfAdapter(Adapter):
        name = "half"
        def fetch(self, identifier, start=None, end=None, *, data_root):
            return Path("/tmp/x")
        # missing parse

    with pytest.raises(TypeError):
        HalfAdapter()  # type: ignore[abstract]


def test_concrete_adapter_health_check_default_true(tmp_path: Path):
    class FakeAdapter(Adapter):
        name = "fake"

        def fetch(self, identifier, start=None, end=None, *, data_root):
            p = data_root / "raw.csv"
            p.write_text("a,b\n1,2\n")
            return p

        def parse(self, raw_path):
            return pd.read_csv(raw_path)

    a = FakeAdapter()
    assert a.name == "fake"
    assert a.health_check() is True
    p = a.fetch("X:Y", start=date(2020, 1, 1), end=date(2020, 12, 31),
                data_root=tmp_path)
    df = a.parse(p)
    assert list(df.columns) == ["a", "b"]


def test_health_check_overridable():
    class DownAdapter(Adapter):
        name = "down"
        def fetch(self, identifier, start=None, end=None, *, data_root):
            raise NotImplementedError
        def parse(self, raw_path):
            raise NotImplementedError
        def health_check(self):
            return False

    assert DownAdapter().health_check() is False
