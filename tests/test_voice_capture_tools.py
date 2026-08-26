import importlib.util
import sys
import threading
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "voice_capture_server.py"
SPEC = importlib.util.spec_from_file_location("voiceloop_voice_capture_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)


def _config():
    return capture.CaptureConfig(
        slug="test-capture",
        title="Test capture",
        default_port=8799,
        phrases=(capture.CapturePhrase("test-fraza", "testowa fraza"),),
        judge=lambda _metadata, _size: (True, "ok"),
    )


def test_capture_store_is_portable_and_path_confined(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VOICELOOP_DATA_DIR", raising=False)
    data_dir = capture.capture_data_dir("test-capture", tmp_path)
    store = capture.CaptureStore(data_dir)
    metadata = {
        "id": "test-fraza",
        "text": "testowa fraza",
        "family": "test",
        "label": "test",
        "ext": "webm",
        "duration_ms": 900,
    }

    take = store.record(_config(), metadata, b"x" * 2048)

    assert data_dir == tmp_path.resolve() / "test-capture"
    assert store.recording_path(take["file"]).parent == data_dir
    assert store.snapshot(_config())["phrases"][0]["files"] == [take["file"]]
    with pytest.raises(ValueError):
        store.recording_path("../outside.webm")


def test_recording_metadata_round_trip_hides_unicode_from_url() -> None:
    metadata = {
        "id": "polska-fraza",
        "text": "zażółć gęślą jaźń",
        "ext": "webm",
    }

    encoded = capture.encode_recording_metadata(metadata)

    assert capture.decode_recording_metadata(encoded) == metadata
    assert "zażółć" not in encoded
    with pytest.raises(ValueError):
        capture.decode_recording_metadata("not-json")


def test_capture_http_server_binds_recording_contract(tmp_path) -> None:
    config = _config()
    handler = capture.make_capture_handler(
        config,
        capture.capture_data_dir(config.slug, tmp_path),
    )
    server = capture.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    metadata = capture.encode_recording_metadata(
        {
            "id": "test-fraza",
            "text": "testowa fraza",
            "family": "test",
            "label": "test",
            "ext": "webm",
            "duration_ms": 900,
        }
    )

    try:
        health = httpx.get(f"{base_url}/api/health", timeout=3)
        response = httpx.post(
            f"{base_url}/api/record",
            headers={"X-VoiceLoop-Metadata": metadata},
            content=b"x" * 2048,
            timeout=3,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert health.json() == {
        "ok": True,
        "tool": "test-capture",
        "loopback_only": True,
    }
    assert response.status_code == 200
    assert response.json()["saved"] is True
    assert response.request.url.query == b""


def test_capture_tool_wrappers_have_no_user_specific_paths() -> None:
    wrappers = [
        ROOT / "scripts" / "holding-commands" / "server.py",
        ROOT / "scripts" / "calibration-phrases" / "server.py",
    ]

    for wrapper in wrappers:
        source = wrapper.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in source
        assert "run_capture_server" in source
