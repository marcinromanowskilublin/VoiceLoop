from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_settings_from_local_env_file(monkeypatch):
    """Keep pytest deterministic even when listener/.env exists locally."""

    from voiceloop.settings import Settings, get_settings

    get_settings.cache_clear()
    patched_config = dict(Settings.model_config)
    patched_config["env_file"] = None
    monkeypatch.setattr(Settings, "model_config", patched_config)
    yield
    get_settings.cache_clear()
