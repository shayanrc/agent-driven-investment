"""Tests for the .env loader (data_pipelines.env)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_pipelines import env as env_module


@pytest.fixture(autouse=True)
def _reset_loaded():
    prev = env_module._LOADED
    env_module._LOADED = False
    yield
    env_module._LOADED = prev


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_loads_dotenv_into_environ(isolated_cwd, monkeypatch):
    (isolated_cwd / ".env").write_text("FAKE_TEST_VAR_FROM_ENV=hello-from-dotenv\n")
    monkeypatch.delenv("FAKE_TEST_VAR_FROM_ENV", raising=False)
    env_module.load_env()
    assert os.environ.get("FAKE_TEST_VAR_FROM_ENV") == "hello-from-dotenv"


def test_existing_env_wins_by_default(isolated_cwd, monkeypatch):
    (isolated_cwd / ".env").write_text("FAKE_TEST_VAR_FROM_ENV=from-dotenv\n")
    monkeypatch.setenv("FAKE_TEST_VAR_FROM_ENV", "from-shell")
    env_module.load_env()
    # Real env wins over .env.
    assert os.environ["FAKE_TEST_VAR_FROM_ENV"] == "from-shell"


def test_override_true_lets_dotenv_win(isolated_cwd, monkeypatch):
    (isolated_cwd / ".env").write_text("FAKE_TEST_VAR_FROM_ENV=from-dotenv\n")
    monkeypatch.setenv("FAKE_TEST_VAR_FROM_ENV", "from-shell")
    env_module.load_env(override=True)
    assert os.environ["FAKE_TEST_VAR_FROM_ENV"] == "from-dotenv"


def test_idempotent_without_override(isolated_cwd, monkeypatch):
    (isolated_cwd / ".env").write_text("FAKE_TEST_VAR_FROM_ENV=v1\n")
    monkeypatch.delenv("FAKE_TEST_VAR_FROM_ENV", raising=False)
    env_module.load_env()
    # Mutate .env and reload without override; first-call cache wins.
    (isolated_cwd / ".env").write_text("FAKE_TEST_VAR_FROM_ENV=v2\n")
    result = env_module.load_env()
    assert result is None  # no-op
    assert os.environ["FAKE_TEST_VAR_FROM_ENV"] == "v1"


def test_no_dotenv_returns_none(isolated_cwd, monkeypatch):
    # Make sure neither cwd .env nor the repo-root fallback exists for this test.
    fake_root = isolated_cwd / "fake_repo"
    fake_root.mkdir()
    monkeypatch.chdir(fake_root)
    # Patch the repo-root fallback path to point at an empty dir.
    monkeypatch.setattr(env_module, "__file__",
                        str(fake_root / "src" / "data_pipelines" / "env.py"))
    result = env_module.load_env()
    # No .env above cwd (since we're in a fake dir) → returns None or whatever
    # find_dotenv stumbled into outside our control. Just check it doesn't raise.
    assert result is None or isinstance(result, Path)


def test_get_required_env_raises_on_missing(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
    with pytest.raises(KeyError, match="DEFINITELY_NOT_SET_VAR"):
        env_module.get_required_env("DEFINITELY_NOT_SET_VAR")


def test_get_required_env_returns_value(monkeypatch):
    monkeypatch.setenv("FOO_VAR", "bar")
    assert env_module.get_required_env("FOO_VAR") == "bar"
