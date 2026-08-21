from __future__ import annotations

import array
import asyncio
import base64
import json
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from pydantic import SecretStr

from .settings import Settings

LOGGER = logging.getLogger("voiceloop.hume_emotion")


class HumeEmotionError(RuntimeError):
    """Raised when Hume emotion analysis cannot complete."""


@dataclass(frozen=True)
class EmotionScore:
    name: str
    score: float


@dataclass(frozen=True)
class HumeEmotionWindow:
    begin_seconds: float | None
    end_seconds: float | None
    emotions: tuple[EmotionScore, ...]


@dataclass(frozen=True)
class Linear16Audio:
    sample_rate: int
    channels: int
    pcm: bytes
    duration_seconds: float


class HumeEmotionClient:
    """Hume expression/prosody layer.

    VoiceLoop keeps Deepgram responsible for Polish transcription, punctuation,
    and diarization. This client only attaches text emotion labels extracted
    from the matching audio chunk.
    """

    def __init__(self, settings: Settings) -> None:
        self.enabled = bool(settings.hume_emotion_analysis_enabled)
        self.api_key = settings.hume_api_key
        self.endpoint = settings.hume_emotion_endpoint.strip()
        self.timeout_seconds = max(3.0, settings.hume_emotion_timeout_seconds)
        self.top_n = max(1, min(settings.hume_emotion_top_n, 8))
        self.min_score = max(0.0, min(settings.hume_emotion_min_score, 1.0))
        self._last_status = "skonfigurowany" if self.enabled else "wyłączony"

    def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączony"
        if not self._api_key_value():
            return False, "brak HUME_API_KEY"
        return True, self._last_status

    async def analyze_file(self, path: Path) -> list[HumeEmotionWindow]:
        if not self.enabled:
            return []
        if not self._api_key_value():
            self._last_status = "brak HUME_API_KEY"
            return []
        if not path.is_file():
            raise HumeEmotionError(f"Brak pliku audio: {path}")

        try:
            if self._uses_legacy_stream_models():
                payload = await asyncio.to_thread(self._payload_for_file, path)
                response = await asyncio.wait_for(
                    self._send_legacy_payload(payload),
                    timeout=self.timeout_seconds,
                )
                windows = self._extract_windows(response)
            else:
                audio = await asyncio.to_thread(self._linear16_audio_for_file, path)
                windows = await asyncio.wait_for(
                    self._stream_evi_audio(audio),
                    timeout=self.timeout_seconds,
                )
        except TimeoutError as exc:
            self._last_status = "timeout"
            raise HumeEmotionError("Hume emotion analysis przekroczył limit czasu.") from exc
        except Exception as exc:
            self._last_status = "błąd"
            raise HumeEmotionError(str(exc)) from exc

        self._last_status = f"ok · {len(windows)} okien emocji"
        return windows

    def emotions_for_interval(
        self,
        windows: list[HumeEmotionWindow],
        *,
        begin_seconds: float,
        end_seconds: float,
    ) -> list[dict[str, object]]:
        if not windows:
            return []
        weighted: dict[str, float] = {}
        total_weight = 0.0
        for window in windows:
            weight = self._overlap_weight(
                window.begin_seconds,
                window.end_seconds,
                begin_seconds,
                end_seconds,
            )
            if weight <= 0.0:
                continue
            total_weight += weight
            for emotion in window.emotions:
                weighted[emotion.name] = weighted.get(emotion.name, 0.0) + (
                    emotion.score * weight
                )

        if total_weight <= 0.0:
            for emotion in windows[0].emotions:
                weighted[emotion.name] = max(weighted.get(emotion.name, 0.0), emotion.score)
            total_weight = 1.0

        ranked = sorted(
            (
                {"name": name, "score": score / total_weight}
                for name, score in weighted.items()
            ),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        return [
            item
            for item in ranked[: self.top_n]
            if float(item["score"]) >= self.min_score
        ]

    def _api_key_value(self) -> str:
        if self.api_key is None:
            return ""
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value().strip()
        return str(self.api_key).strip()

    def _payload_for_file(self, path: Path) -> dict[str, Any]:
        audio = path.read_bytes()
        return {
            "data": base64.b64encode(audio).decode("ascii"),
            "models": {
                "prosody": {},
            },
        }

    async def _send_legacy_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        uri = self._endpoint_uri()
        async with websockets.connect(
            uri,
            max_size=32 * 1024 * 1024,
            additional_headers=self._connection_headers(),
        ) as websocket:
            await websocket.send(json.dumps(payload, separators=(",", ":")))
            raw = await websocket.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HumeEmotionError("Hume zwrócił niepoprawny JSON.") from exc
        if not isinstance(parsed, dict):
            raise HumeEmotionError("Hume zwrócił nieoczekiwany format odpowiedzi.")
        if "error" in parsed:
            raise HumeEmotionError(str(parsed["error"]))
        return parsed

    def _endpoint_uri(self) -> str:
        parts = urlsplit(self.endpoint)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.pop("api_key", None)
        query.pop("access_token", None)
        if not self._uses_legacy_stream_models():
            query.setdefault("verbose_transcription", "true")
        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(query),
                parts.fragment,
            )
        )

    def _connection_headers(self) -> dict[str, str]:
        return {"X-Hume-Api-Key": self._api_key_value()}

    def _uses_legacy_stream_models(self) -> bool:
        return "/stream/models" in urlsplit(self.endpoint).path

    def _linear16_audio_for_file(self, path: Path) -> Linear16Audio:
        if path.suffix.casefold() != ".wav":
            raise HumeEmotionError(
                "Hume EVI dostaje teraz tylko lokalne WAV/linear16. "
                f"Pominięto format {path.suffix or '(bez rozszerzenia)'}."
            )
        try:
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frame_count = handle.getnframes()
                frames = handle.readframes(frame_count)
        except wave.Error as exc:
            raise HumeEmotionError(f"Niepoprawny plik WAV dla Hume: {path}") from exc

        if channels <= 0 or sample_rate <= 0:
            raise HumeEmotionError("Niepoprawne parametry WAV dla Hume.")
        if sample_width != 2:
            raise HumeEmotionError(
                "Hume EVI wymaga linear16 PCM, czyli 16-bit signed little-endian."
            )

        duration_seconds = frame_count / sample_rate if sample_rate else 0.0
        pcm = frames
        output_channels = channels
        if channels > 1:
            pcm = self._mix_pcm16_to_mono(frames, channels)
            output_channels = 1
        return Linear16Audio(
            sample_rate=sample_rate,
            channels=output_channels,
            pcm=pcm,
            duration_seconds=duration_seconds,
        )

    @staticmethod
    def _mix_pcm16_to_mono(frames: bytes, channels: int) -> bytes:
        samples = array.array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        mono = array.array("h")
        usable = len(samples) - (len(samples) % channels)
        for index in range(0, usable, channels):
            total = 0
            for channel_index in range(channels):
                total += samples[index + channel_index]
            average = round(total / channels)
            mono.append(max(-32768, min(32767, int(average))))
        if sys.byteorder != "little":
            mono.byteswap()
        return mono.tobytes()

    async def _stream_evi_audio(self, audio: Linear16Audio) -> list[HumeEmotionWindow]:
        uri = self._endpoint_uri()
        windows: list[HumeEmotionWindow] = []
        async with websockets.connect(
            uri,
            max_size=32 * 1024 * 1024,
            additional_headers=self._connection_headers(),
        ) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "session_settings",
                        "audio": {
                            "encoding": "linear16",
                            "sample_rate": audio.sample_rate,
                            "channels": audio.channels,
                        },
                    },
                    separators=(",", ":"),
                )
            )
            for chunk in self._pcm_chunks(audio.pcm, audio.sample_rate, audio.channels):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "audio_input",
                            "data": base64.b64encode(chunk).decode("ascii"),
                        },
                        separators=(",", ":"),
                    )
                )
                await asyncio.sleep(0)

            silence = b"\x00\x00" * int(audio.sample_rate * audio.channels * 0.5)
            if silence:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "audio_input",
                            "data": base64.b64encode(silence).decode("ascii"),
                        },
                        separators=(",", ":"),
                    )
                )

            deadline = asyncio.get_running_loop().time() + min(
                max(3.0, self.timeout_seconds / 2),
                10.0,
            )
            while asyncio.get_running_loop().time() < deadline:
                remaining = max(0.1, deadline - asyncio.get_running_loop().time())
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                except TimeoutError:
                    break
                event = self._json_event(raw)
                if not event:
                    continue
                if event.get("type") == "error":
                    code = str(event.get("code") or "").strip()
                    message = str(event.get("message") or event.get("error") or event)
                    raise HumeEmotionError(
                        f"Hume EVI zwrócił błąd{f' {code}' if code else ''}: {message}"
                    )
                if event.get("type") != "user_message":
                    continue
                extracted = self._extract_windows(event)
                if extracted:
                    windows.extend(extracted)
                    if not event.get("interim"):
                        break

        return self._dedupe_windows(windows)

    @staticmethod
    def _pcm_chunks(pcm: bytes, sample_rate: int, channels: int) -> list[bytes]:
        bytes_per_sample = 2
        frame_width = max(1, channels) * bytes_per_sample
        frames_per_chunk = max(1, sample_rate // 10)
        chunk_bytes = frames_per_chunk * frame_width
        return [
            pcm[index : index + chunk_bytes]
            for index in range(0, len(pcm), chunk_bytes)
            if pcm[index : index + chunk_bytes]
        ]

    @staticmethod
    def _json_event(raw: str | bytes) -> dict[str, Any] | None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return event if isinstance(event, dict) else None

    def _extract_windows(self, response: dict[str, Any]) -> list[HumeEmotionWindow]:
        raw_windows: list[HumeEmotionWindow] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                scores = self._prosody_scores(node)
                if scores:
                    parsed_scores = self._parse_score_map(scores)
                    if parsed_scores:
                        begin, end = self._parse_time(node)
                        raw_windows.append(
                            HumeEmotionWindow(
                                begin_seconds=begin,
                                end_seconds=end,
                                emotions=tuple(parsed_scores),
                            )
                        )
                emotions = node.get("emotions")
                if isinstance(emotions, list):
                    parsed = self._parse_emotions(emotions)
                    if parsed:
                        begin, end = self._parse_time(node)
                        raw_windows.append(
                            HumeEmotionWindow(
                                begin_seconds=begin,
                                end_seconds=end,
                                emotions=tuple(parsed),
                            )
                        )
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(response)
        raw_windows.sort(
            key=lambda item: (
                float("inf") if item.begin_seconds is None else item.begin_seconds,
                float("inf") if item.end_seconds is None else item.end_seconds,
            )
        )
        return self._dedupe_windows(raw_windows)

    @staticmethod
    def _prosody_scores(node: dict[str, Any]) -> dict[str, Any] | None:
        models = node.get("models")
        if isinstance(models, dict):
            prosody = models.get("prosody")
            if isinstance(prosody, dict):
                scores = prosody.get("scores")
                if isinstance(scores, dict):
                    return scores
        prosody = node.get("prosody")
        if isinstance(prosody, dict):
            scores = prosody.get("scores")
            if isinstance(scores, dict):
                return scores
        return None

    def _parse_score_map(self, scores: dict[str, Any]) -> list[EmotionScore]:
        parsed: list[EmotionScore] = []
        for name, raw_score in scores.items():
            label = str(name).strip()
            if not label:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            parsed.append(EmotionScore(name=label, score=max(0.0, min(score, 1.0))))
        parsed.sort(key=lambda item: item.score, reverse=True)
        return parsed[: self.top_n]

    def _parse_emotions(self, emotions: list[Any]) -> list[EmotionScore]:
        parsed: list[EmotionScore] = []
        for item in emotions:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("label") or "").strip()
            if not name:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            parsed.append(EmotionScore(name=name, score=max(0.0, min(score, 1.0))))
        parsed.sort(key=lambda item: item.score, reverse=True)
        return parsed[: self.top_n]

    @staticmethod
    def _dedupe_windows(windows: list[HumeEmotionWindow]) -> list[HumeEmotionWindow]:
        deduped: list[HumeEmotionWindow] = []
        seen: set[tuple[float | None, float | None, tuple[tuple[str, float], ...]]] = set()
        for window in windows:
            key = (
                window.begin_seconds,
                window.end_seconds,
                tuple((emotion.name, round(emotion.score, 6)) for emotion in window.emotions),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(window)
        return deduped

    @staticmethod
    def _parse_time(node: dict[str, Any]) -> tuple[float | None, float | None]:
        time_value = node.get("time")
        begin: Any = None
        end: Any = None
        if isinstance(time_value, dict):
            begin = time_value.get("begin", time_value.get("start"))
            end = time_value.get("end")
        begin = node.get("begin", node.get("start", begin))
        end = node.get("end", end)
        try:
            parsed_begin = float(begin) if begin is not None else None
        except (TypeError, ValueError):
            parsed_begin = None
        try:
            parsed_end = float(end) if end is not None else None
        except (TypeError, ValueError):
            parsed_end = None
        return parsed_begin, parsed_end

    @staticmethod
    def _overlap_weight(
        window_begin: float | None,
        window_end: float | None,
        target_begin: float,
        target_end: float,
    ) -> float:
        if window_begin is None or window_end is None:
            return 1.0
        begin = max(window_begin, target_begin)
        end = min(window_end, target_end)
        return max(0.0, end - begin)
