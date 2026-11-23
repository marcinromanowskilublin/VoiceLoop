"""Offline regression checks for the opt-in localhost minimalizuj studio."""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


studio = load_module("minimalizuj_studio", "minimalizuj_studio.py")
capture = load_module("minimalizuj_capture_test", "voice_capture_server.py")


def test_difference_explains_error_without_attributing_cause():
    d = studio.difference("pół pit", "pulpit")
    assert d["kind"] == "stt_difference"
    assert d["changes"]
    assert "przyczyny akustycznej" in d["explanation"]


def test_store_persists_alias_and_can_revert(tmp_path):
    db = studio.CalibrationStudio(tmp_path / "samples.sqlite3")
    response = db.save({
        "expected": "Minimalizuj", "heard": "minimalizoo",
        "corrected": "minimalizuj nowe", "expected_action": "minimize_all_windows",
    }, tmp_path)
    assert response["alias_state"] == "proposed"
    assert response["detected_action"] == "unresolved"
    assert db.change_alias(response["id"], True)["scope"] == "calibration_studio_only"
    assert "minimalizuj nowe" in db.snapshot()["aliases"]
    assert db.change_alias(response["id"], False)["alias_state"] == "reverted"
    reopened = studio.CalibrationStudio(tmp_path / "samples.sqlite3")
    assert "minimalizuj nowe" not in reopened.snapshot()["aliases"]


def test_store_rejects_unregistered_action(tmp_path):
    db = studio.CalibrationStudio(tmp_path / "samples.sqlite3")
    with pytest.raises(ValueError, match="Nieobsługiwana akcja"):
        db.save({"expected": "Pulpit", "heard": "Pulpit",
                 "corrected": "Pulpit", "expected_action": "delete_files"}, tmp_path)


def test_local_http_panel_record_and_cross_origin_boundary(tmp_path):
    config = capture.CaptureConfig(
        slug="calibration-phrases", title="VoiceLoop",
        default_port=8792,
        phrases=(capture.CapturePhrase("minimalizuj", "Minimalizuj"),),
        judge=lambda _metadata, _size: (True, "ok"),
    )
    server = capture.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        capture.make_capture_handler(config, tmp_path / "calibration-phrases"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        html = httpx.get(url + "/minimalizuj", timeout=3)
        health = httpx.get(url + "/api/health", timeout=3)
        record = httpx.post(url + "/api/minimalizuj", timeout=3,
                            json={"expected":"Minimalizuj","heard":"minimalizoo",
                                  "corrected":"Minimalizuj"})
        entries = httpx.get(url + "/api/minimalizuj", timeout=3)
        bad_host = httpx.get(url + "/minimalizuj", timeout=3,
                             headers={"Host": "evil.example"})
        bad_origin = httpx.post(url + "/api/minimalizuj", timeout=3,
                                headers={"Origin":"http://evil.example"}, json={})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert html.status_code == 200 and "Spacja" in html.text
    assert health.json()["ok"] is True
    assert record.status_code == 200
    assert record.json()["analysis"]["kind"] == "stt_difference"
    assert len(entries.json()["samples"]) == 1
    assert bad_host.status_code == 400
    assert bad_origin.status_code == 403


def test_explicit_execution_bridge_never_runs_without_confirmation(monkeypatch):
    calls = []

    def fake_core(path, *, payload=None):
        calls.append((path, payload))
        if path.endswith("/commands"):
            return {"request_id": "test-1", "status": "queued"}
        return {"status": "succeeded"}

    monkeypatch.setattr(studio, "core_request", fake_core)
    with pytest.raises(ValueError, match="jawnego potwierdzenia"):
        studio.execute_minimize("")
    assert calls == []
    accepted = studio.execute_minimize("MINIMALIZUJ")
    assert accepted["status"] == "queued"
    assert calls[0][1]["command_id"] == "minimize_all_windows"
    assert calls[0][1]["allow_cloud"] is False
    assert studio.command_status("test-1")["status"] == "succeeded"
    with pytest.raises(ValueError, match="request_id"):
        studio.command_status("../../other")
