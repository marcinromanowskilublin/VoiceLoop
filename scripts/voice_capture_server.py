"""Shared loopback-only server for local VoiceLoop voice-sample tools."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

HOST = "127.0.0.1"
MAX_BODY_BYTES = 25 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
SAFE_RECORDING_NAME = re.compile(
    r"^[a-z0-9][a-z0-9_.-]*\.(?:webm|wav|ogg)$",
    re.IGNORECASE,
)
SUPPORTED_EXTENSIONS = {"webm", "wav", "ogg"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = Path(__file__).resolve().with_name("voice_capture.html")

QualityJudge = Callable[[dict[str, Any], int], tuple[bool, str]]


@dataclass(frozen=True)
class CapturePhrase:
    phrase_id: str
    text: str
    family: str = "default"
    label: str = "próbka"

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.phrase_id,
            "text": self.text,
            "family": self.family,
            "label": self.label,
        }


@dataclass(frozen=True)
class CaptureConfig:
    slug: str
    title: str
    default_port: int
    phrases: tuple[CapturePhrase, ...]
    judge: QualityJudge

    def __post_init__(self) -> None:
        if not SAFE_ID.fullmatch(self.slug):
            raise ValueError("Nazwa narzędzia musi być bezpiecznym slugiem.")
        if not self.phrases:
            raise ValueError("Narzędzie musi zawierać co najmniej jedną frazę.")
        phrase_ids = [phrase.phrase_id for phrase in self.phrases]
        if len(phrase_ids) != len(set(phrase_ids)):
            raise ValueError("Identyfikatory fraz nie mogą się powtarzać.")
        if any(not SAFE_ID.fullmatch(phrase_id) for phrase_id in phrase_ids):
            raise ValueError("Każda fraza musi mieć bezpieczny identyfikator.")


def capture_data_dir(slug: str, root: Path | None = None) -> Path:
    configured_root = root
    if configured_root is None:
        env_root = os.environ.get("VOICELOOP_DATA_DIR", "").strip()
        configured_root = Path(env_root).expanduser() if env_root else PROJECT_ROOT / "data"
    return configured_root.expanduser().resolve() / slug


def encode_recording_metadata(metadata: dict[str, Any]) -> str:
    raw = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_recording_metadata(value: str) -> dict[str, Any]:
    if not value or len(value) > MAX_METADATA_BYTES:
        raise ValueError("Brak lub zbyt duże metadane nagrania.")
    padding = "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode(value + padding)
        metadata = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Nieprawidłowe metadane nagrania.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Metadane nagrania muszą być obiektem JSON.")
    return metadata


class CaptureStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.state_path = self.data_dir / "session.json"

    def load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"takes": [], "items": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"takes": [], "items": {}}
        takes = payload.get("takes")
        items = payload.get("items")
        return {
            "takes": takes if isinstance(takes, list) else [],
            "items": items if isinstance(items, dict) else {},
        }

    def save(self, state: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "takes": list(state.get("takes", []))[-500:],
            "items": state.get("items", {}),
        }
        temp_path = self.state_path.with_name(
            f".{self.state_path.name}.{os.getpid()}.tmp"
        )
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temp_path.replace(self.state_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def snapshot(self, config: CaptureConfig) -> dict[str, Any]:
        state = self.load()
        items = state["items"]
        phrases: list[dict[str, Any]] = []
        known_ids: set[str] = set()
        for phrase in config.phrases:
            entry = items.get(phrase.phrase_id, {})
            phrases.append(
                {
                    **phrase.as_dict(),
                    "done": bool(entry.get("done")) if isinstance(entry, dict) else False,
                    "files": (
                        list(entry.get("files", []))
                        if isinstance(entry, dict)
                        and isinstance(entry.get("files"), list)
                        else []
                    ),
                }
            )
            known_ids.add(phrase.phrase_id)
        for phrase_id, entry in items.items():
            if (
                phrase_id in known_ids
                or not SAFE_ID.fullmatch(str(phrase_id))
                or not isinstance(entry, dict)
            ):
                continue
            phrases.append(
                {
                    "id": str(phrase_id),
                    "text": str(entry.get("text") or phrase_id),
                    "family": str(entry.get("family") or "generated"),
                    "label": str(entry.get("label") or "własna fraza"),
                    "done": bool(entry.get("done")),
                    "files": list(entry.get("files", [])),
                }
            )
        return {
            "ok": True,
            "title": config.title,
            "tool": config.slug,
            "save_dir": str(self.data_dir),
            "phrases": phrases,
            "takes": list(reversed(state["takes"]))[:100],
        }

    def set_done(self, phrase_id: str, done: bool) -> None:
        if not SAFE_ID.fullmatch(phrase_id):
            raise ValueError("Nieprawidłowy identyfikator frazy.")
        state = self.load()
        entry = state["items"].setdefault(phrase_id, {})
        entry["done"] = bool(done)
        self.save(state)

    def record(
        self,
        config: CaptureConfig,
        metadata: dict[str, Any],
        body: bytes,
    ) -> dict[str, Any]:
        phrase_id = str(metadata.get("id") or "").strip()
        text = str(metadata.get("text") or "").strip()
        family = str(metadata.get("family") or "default").strip()[:80]
        label = str(metadata.get("label") or "próbka").strip()[:120]
        extension = str(metadata.get("ext") or "webm").strip().lower()
        if not SAFE_ID.fullmatch(phrase_id):
            raise ValueError("Nieprawidłowy identyfikator frazy.")
        if not text or len(text) > 1000:
            raise ValueError("Fraza jest pusta albo zbyt długa.")
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError("Nieobsługiwany format nagrania.")
        if len(body) < 64:
            raise ValueError("Nagranie jest puste albo zbyt krótkie.")
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("Nagranie przekracza limit 25 MB.")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"{phrase_id}_{stamp}.{extension}"
        destination = self.data_dir / name
        temp_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temp_path.write_bytes(body)
            temp_path.replace(destination)
        finally:
            temp_path.unlink(missing_ok=True)
        if destination.stat().st_size != len(body):
            destination.unlink(missing_ok=True)
            raise OSError("Rozmiar zapisanego nagrania nie zgadza się z wejściem.")

        ok_take, reason = config.judge(metadata, len(body))
        take = {
            "id": phrase_id,
            "text": text,
            "family": family,
            "label": label,
            "file": name,
            "bytes": len(body),
            "duration_ms": _safe_int(metadata.get("duration_ms")),
            "peak": _safe_float(metadata.get("peak")),
            "rms": _safe_float(metadata.get("rms")),
            "ok": ok_take,
            "reason": reason,
            "at": datetime.now(UTC).isoformat(),
        }
        state = self.load()
        state["takes"].append(take)
        entry = state["items"].setdefault(phrase_id, {})
        entry.update(
            {
                "text": text,
                "family": family,
                "label": label,
                "done": bool(entry.get("done")),
            }
        )
        files = entry.setdefault("files", [])
        if name not in files:
            files.insert(0, name)
        self.save(state)
        return take

    def recording_path(self, name: str) -> Path:
        if not SAFE_RECORDING_NAME.fullmatch(name):
            raise ValueError("Nieprawidłowa nazwa nagrania.")
        path = (self.data_dir / name).resolve()
        if path.parent != self.data_dir or not path.is_file():
            raise FileNotFoundError(name)
        return path


def make_capture_handler(
    config: CaptureConfig,
    data_dir: Path,
) -> type[BaseHTTPRequestHandler]:
    store = CaptureStore(data_dir)

    class CaptureHandler(BaseHTTPRequestHandler):
        server_version = "VoiceLoopCapture/1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            print(f"[{config.slug}] {self.address_string()} {format % args}", flush=True)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path in {"/", "/index.html"}:
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._json(
                    200,
                    {
                        "ok": True,
                        "tool": config.slug,
                        "loopback_only": True,
                    },
                )
                return
            if path == "/api/state":
                self._json(200, store.snapshot(config))
                return
            if path == "/api/draw":
                phrase = random.choice(config.phrases)
                self._json(200, {**store.snapshot(config), "phrase": phrase.as_dict()})
                return
            if path.startswith("/files/"):
                try:
                    recording = store.recording_path(path.removeprefix("/files/"))
                except ValueError:
                    self._json(400, {"ok": False, "error": "Nieprawidłowa nazwa."})
                    return
                except FileNotFoundError:
                    self._json(404, {"ok": False, "error": "Brak nagrania."})
                    return
                content_type = {
                    ".wav": "audio/wav",
                    ".ogg": "audio/ogg",
                }.get(recording.suffix.casefold(), "audio/webm")
                self._send(200, recording.read_bytes(), content_type)
                return
            self._json(404, {"ok": False, "error": "Nie znaleziono."})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                self._json(400, {"ok": False, "error": "Błędny Content-Length."})
                return
            if length < 0 or length > MAX_BODY_BYTES:
                self._json(413, {"ok": False, "error": "Przekroczony limit danych."})
                return
            body = self.rfile.read(length) if length else b""
            try:
                if path == "/api/record":
                    metadata = decode_recording_metadata(
                        self.headers.get("X-VoiceLoop-Metadata", "")
                    )
                    take = store.record(config, metadata, body)
                    self._json(
                        200,
                        {
                            **store.snapshot(config),
                            "saved": True,
                            "take": take,
                        },
                    )
                    return
                if path == "/api/check":
                    payload = json.loads(body.decode("utf-8") or "{}")
                    store.set_done(
                        str(payload.get("id") or ""),
                        bool(payload.get("done")),
                    )
                    self._json(200, store.snapshot(config))
                    return
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"ok": False, "error": str(exc)})
                return
            self._json(404, {"ok": False, "error": "Nie znaleziono."})

    return CaptureHandler


def run_capture_server(
    config: CaptureConfig,
    argv: list[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=config.title)
    parser.add_argument("--port", type=int, default=config.default_port)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("Port musi należeć do zakresu 1024–65535.")
    data_dir = capture_data_dir(config.slug, args.data_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        (HOST, args.port),
        make_capture_handler(config, data_dir),
    )
    print(f"{config.title}: http://{HOST}:{args.port}/", flush=True)
    print(f"zapis lokalny: {data_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
