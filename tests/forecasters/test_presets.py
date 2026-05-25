"""Tests for forecasters.presets — load, validate, list."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from forecasters.errors import PresetSchemaError, UnknownPresetError
from forecasters.presets import (
    CURRENT_SCHEMA_VERSION,
    list_presets,
    load_preset,
    preset_content_hash,
    resolve_preset_path,
    validate_preset,
)


def _good_preset_dict() -> dict:
    """Minimal valid preset (matches the v24-default schema, tiny knobs)."""
    return {
        "name": "tiny",
        "backend": "analog_mc",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "hyperparameters": {"n_eff": 50, "block_length": 10},
        "fitted_on": {
            "identifier": "NASDAQ100",
            "start": "1986-01-02",
            "end": "2024-12-31",
            "data_hash": "sha256:deadbeef",
            "n_observations": 9000,
        },
        "fitted_at": "2026-05-24T18:30:00Z",
        "validation_metrics": {"crps_mean": 0.05},
    }


def _write_preset(root: Path, name: str, body: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


# ----------------------------------------------------------------------------
# Canonical preset (v24-default) — shipped with the repo
# ----------------------------------------------------------------------------


def test_canonical_v24_default_loads_from_repo_root(tmp_path: Path) -> None:
    """The shipped canonical preset must load cleanly against the repo."""
    # This test relies on the project layout — run pytest from the repo root.
    preset = load_preset(
        "v24-default",
        canonical_root=Path("configs/forecasters/presets"),
        user_root=tmp_path / "results",  # isolate user root for the test
    )
    assert preset["name"] == "v24-default"
    assert preset["backend"] == "analog_mc"
    assert preset["fitted_on"]["identifier"] == "NASDAQ100"
    assert preset["fitted_on"]["data_hash"].startswith("sha256:")
    assert "__source_path__" in preset
    assert "__content_hash__" in preset
    assert preset["__content_hash__"].startswith("sha256:")


# ----------------------------------------------------------------------------
# resolve_preset_path
# ----------------------------------------------------------------------------


def test_resolve_canonical_then_user(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    user = tmp_path / "user"
    _write_preset(canonical, "shared", _good_preset_dict() | {"name": "shared"})
    _write_preset(user, "user-only", _good_preset_dict() | {"name": "user-only"})

    p1 = resolve_preset_path("shared", canonical_root=canonical, user_root=user)
    assert p1.parent == canonical, "canonical wins when the name appears in both roots"

    p2 = resolve_preset_path("user-only", canonical_root=canonical, user_root=user)
    assert p2.parent == user


def test_resolve_user_wins_over_missing_canonical(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    user = tmp_path / "user"
    _write_preset(canonical, "other", _good_preset_dict() | {"name": "other"})
    _write_preset(user, "u1", _good_preset_dict() | {"name": "u1"})

    p = resolve_preset_path("u1", canonical_root=canonical, user_root=user)
    assert p.parent == user


def test_unknown_preset_raises_with_searched_paths(tmp_path: Path) -> None:
    with pytest.raises(UnknownPresetError) as ei:
        resolve_preset_path(
            "missing",
            canonical_root=tmp_path / "c",
            user_root=tmp_path / "u",
        )
    assert "missing.yaml" in str(ei.value)
    assert ei.value.preset_name == "missing"
    assert len(ei.value.searched) == 2


# ----------------------------------------------------------------------------
# validate_preset — schema rules
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", [
    "name", "backend", "schema_version", "hyperparameters",
    "fitted_on", "fitted_at", "validation_metrics",
])
def test_missing_top_level_key_raises(missing_key: str, tmp_path: Path) -> None:
    body = _good_preset_dict()
    del body[missing_key]
    with pytest.raises(PresetSchemaError) as ei:
        validate_preset(body)
    assert missing_key in ei.value.violated


def test_wrong_schema_version_raises() -> None:
    body = _good_preset_dict()
    body["schema_version"] = 999
    with pytest.raises(PresetSchemaError) as ei:
        validate_preset(body)
    assert "schema_version" in ei.value.violated


def test_name_must_match_filename(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    body = _good_preset_dict()
    body["name"] = "tiny"  # but we write to wrong-name.yaml
    path = _write_preset(canonical, "wrong-name", body)
    with pytest.raises(PresetSchemaError) as ei:
        load_preset("wrong-name", canonical_root=canonical, user_root=tmp_path / "u")
    assert "does not match filename stem" in ei.value.violated
    assert path.exists()  # sanity


def test_bad_data_hash_raises() -> None:
    body = _good_preset_dict()
    body["fitted_on"]["data_hash"] = "md5:abc"
    with pytest.raises(PresetSchemaError) as ei:
        validate_preset(body)
    assert "data_hash" in ei.value.violated


def test_fitted_on_missing_subkey_raises() -> None:
    body = _good_preset_dict()
    del body["fitted_on"]["n_observations"]
    with pytest.raises(PresetSchemaError) as ei:
        validate_preset(body)
    assert "fitted_on" in ei.value.violated
    assert "n_observations" in ei.value.violated


def test_naive_fitted_at_rejected() -> None:
    body = _good_preset_dict()
    body["fitted_at"] = "2026-05-24T18:30:00"  # no tz
    with pytest.raises(PresetSchemaError) as ei:
        validate_preset(body)
    assert "fitted_at" in ei.value.violated


def test_non_utc_fitted_at_rejected() -> None:
    body = _good_preset_dict()
    body["fitted_at"] = "2026-05-24T18:30:00+05:30"
    with pytest.raises(PresetSchemaError) as ei:
        validate_preset(body)
    assert "fitted_at" in ei.value.violated


def test_z_suffix_and_plus_zero_both_accepted() -> None:
    for ts in ("2026-05-24T18:30:00Z", "2026-05-24T18:30:00+00:00"):
        body = _good_preset_dict()
        body["fitted_at"] = ts
        validate_preset(body)  # should not raise


def test_empty_hyperparameters_rejected() -> None:
    body = _good_preset_dict()
    body["hyperparameters"] = {}
    with pytest.raises(PresetSchemaError) as ei:
        validate_preset(body)
    assert "hyperparameters" in ei.value.violated


# ----------------------------------------------------------------------------
# load_preset — round-trip with content hash
# ----------------------------------------------------------------------------


def test_load_preset_round_trip(tmp_path: Path) -> None:
    canonical = tmp_path / "c"
    body = _good_preset_dict()
    path = _write_preset(canonical, "tiny", body)
    loaded = load_preset("tiny", canonical_root=canonical, user_root=tmp_path / "u")
    assert loaded["name"] == "tiny"
    # Loader-supplied metadata.
    assert loaded["__source_path__"] == str(path.resolve())
    assert loaded["__content_hash__"] == preset_content_hash(path)
    # Editing the file changes the content hash.
    body2 = copy.deepcopy(body)
    body2["validation_metrics"]["crps_mean"] = 0.06
    path.write_text(yaml.safe_dump(body2, sort_keys=False))
    loaded2 = load_preset("tiny", canonical_root=canonical, user_root=tmp_path / "u")
    assert loaded2["__content_hash__"] != loaded["__content_hash__"]


def test_top_level_yaml_must_be_mapping(tmp_path: Path) -> None:
    canonical = tmp_path / "c"
    canonical.mkdir()
    (canonical / "weird.yaml").write_text("- a\n- b\n")
    with pytest.raises(PresetSchemaError) as ei:
        load_preset("weird", canonical_root=canonical, user_root=tmp_path / "u")
    assert "not a mapping" in ei.value.violated


# ----------------------------------------------------------------------------
# list_presets
# ----------------------------------------------------------------------------


def test_list_presets_marks_sources(tmp_path: Path) -> None:
    canonical = tmp_path / "c"
    user = tmp_path / "u"
    _write_preset(canonical, "alpha", _good_preset_dict() | {"name": "alpha"})
    _write_preset(user, "beta", _good_preset_dict() | {"name": "beta"})

    rows = list_presets(canonical_root=canonical, user_root=user)
    by_name = {r["name"]: r for r in rows}
    assert by_name["alpha"]["source"] == "canonical"
    assert by_name["beta"]["source"] == "user-tuned"
    assert all(r["backend"] == "analog_mc" for r in rows)
    assert all(r["error"] is None for r in rows)


def test_list_presets_canonical_shadows_user(tmp_path: Path) -> None:
    canonical = tmp_path / "c"
    user = tmp_path / "u"
    _write_preset(canonical, "shared", _good_preset_dict() | {"name": "shared"})
    body_u = _good_preset_dict() | {"name": "shared"}
    body_u["validation_metrics"]["crps_mean"] = 0.99
    _write_preset(user, "shared", body_u)

    rows = list_presets(canonical_root=canonical, user_root=user)
    matching = [r for r in rows if r["name"] == "shared"]
    assert len(matching) == 1
    assert matching[0]["source"] == "canonical"
    assert matching[0]["crps_mean"] != 0.99  # canonical wins


def test_list_presets_filters_by_backend(tmp_path: Path) -> None:
    canonical = tmp_path / "c"
    user = tmp_path / "u"
    _write_preset(canonical, "a", _good_preset_dict() | {"name": "a", "backend": "analog_mc"})
    _write_preset(canonical, "b", _good_preset_dict() | {"name": "b", "backend": "future_arima"})
    rows = list_presets(canonical_root=canonical, user_root=user, backend="analog_mc")
    assert [r["name"] for r in rows] == ["a"]


def test_list_presets_surfaces_bad_files(tmp_path: Path) -> None:
    canonical = tmp_path / "c"
    canonical.mkdir()
    (canonical / "broken.yaml").write_text("name: broken\nbackend: analog_mc\n")  # missing keys
    rows = list_presets(canonical_root=canonical, user_root=tmp_path / "u")
    assert len(rows) == 1
    assert rows[0]["name"] == "broken"
    assert rows[0]["error"] is not None  # surfaced, not silently dropped


def test_list_presets_handles_empty_roots(tmp_path: Path) -> None:
    rows = list_presets(canonical_root=tmp_path / "c", user_root=tmp_path / "u")
    assert rows == []
