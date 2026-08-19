from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

import sounddevice as sd
import websockets

from .events import EventBus
from .models import (
    TranscriptEnvelopeV1,
    TranscriptWordV1,
    normalize_transcript_text,
)
from .settings import Settings

FinalCallback = Callable[..., Awaitable[None]]
InterimCallback = Callable[..., Awaitable[None]]
PriorityFinalPredicate = Callable[[str], bool]
AudioSink = Callable[[bytes], None]
LOGGER = logging.getLogger("voiceloop.deepgram")


class DeepgramListener:
    def __init__(
        self,
        *,
        settings: Settings,
        events: EventBus,
        on_final: FinalCallback,
        on_interim: InterimCallback | None = None,
    ) -> None:
        self.settings = settings
        self.events = events
        self.on_final = on_final
        self.on_interim = on_interim
        self.running = False
        self.connected = False
        self.last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._callback_tasks: set[asyncio.Task[None]] = set()
        self._final_queue: asyncio.Queue[
            TranscriptEnvelopeV1
        ] = asyncio.Queue(maxsize=16)
        self._interim_queue: asyncio.Queue[
            tuple[str, tuple[int, ...]]
        ] = asyncio.Queue(maxsize=16)
        self._final_worker_task: asyncio.Task[None] | None = None
        self._interim_worker_task: asyncio.Task[None] | None = None
        self._priority_final_predicate: PriorityFinalPredicate | None = None
        self._audio_sink: AudioSink | None = None
        self._on_final_accepts_transcript = self._callback_accepts_keyword(
            on_final,
            "transcript",
        )
        self._one_shot = False
        self._conversation_mode = False
        self._one_shot_prefix = ""
        self._one_shot_timeout_seconds = 30.0
        self._connected_event = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        self.connection_count = 0

    def set_priority_final_predicate(
        self,
        predicate: PriorityFinalPredicate,
    ) -> None:
        self._priority_final_predicate = predicate

    def set_audio_sink(self, sink: AudioSink | None) -> None:
        self._audio_sink = sink

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if (self.running and self._one_shot) or (
                not self.running and self._task is not None and not self._task.done()
            ):
                await self._stop_locked()
            self._one_shot = False
            self._conversation_mode = False
            self._one_shot_prefix = ""
            await self._start_locked()

    async def start_conversation(self) -> None:
        async with self._lifecycle_lock:
            if self.running and self._conversation_mode and self._task is not None:
                return
            if self.running or (self._task is not None and not self._task.done()):
                await self._stop_locked()
            self._one_shot = False
            self._conversation_mode = True
            self._one_shot_prefix = ""
            await self._start_locked()

    async def start_once(
        self,
        *,
        prefix: str = "",
        timeout_seconds: float = 30.0,
    ) -> None:
        async with self._lifecycle_lock:
            if self.running or (self._task is not None and not self._task.done()):
                await self._stop_locked()
            self._one_shot = True
            self._conversation_mode = False
            self._one_shot_prefix = prefix.strip()
            if self._one_shot_prefix:
                self._one_shot_prefix += " "
            self._one_shot_timeout_seconds = max(5.0, min(timeout_seconds, 120.0))
            await self._start_locked()

    async def _start_locked(self) -> None:
        if self.running:
            return
        if not self.settings.deepgram_api_key:
            raise RuntimeError("Brak DEEPGRAM_API_KEY w listener/.env")
        if not self.settings.deepgram_api_key.get_secret_value().strip():
            raise RuntimeError("DEEPGRAM_API_KEY jest pusty")
        self.running = True
        self.last_error = None
        self._connected_event.clear()
        self._task = asyncio.create_task(self._run_forever(), name="deepgram-listener")
        await self.events.publish("listening.started")

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        current_task = asyncio.current_task()
        self.running = False
        self._one_shot = False
        self._conversation_mode = False
        self._one_shot_prefix = ""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.connected = False
        self._connected_event.clear()
        callback_tasks = [
            task
            for task in tuple(self._callback_tasks)
            if task is not current_task and not task.done()
        ]
        for task in callback_tasks:
            task.cancel()
        if callback_tasks:
            await asyncio.gather(*callback_tasks, return_exceptions=True)
        self._callback_tasks = {
            task for task in self._callback_tasks if task is current_task and not task.done()
        }
        self._clear_queue(self._final_queue)
        self._clear_queue(self._interim_queue)
        await self.events.publish("listening.stopped")

    async def wait_until_connected(self, *, timeout_seconds: float = 5.0) -> None:
        if self.connected:
            return
        try:
            await asyncio.wait_for(
                self._connected_event.wait(),
                timeout=max(0.1, timeout_seconds),
            )
        except TimeoutError as exc:
            raise RuntimeError(
                self.last_error or "Deepgram nie połączył się w wymaganym czasie."
            ) from exc

    def health(self) -> tuple[bool, str]:
        if self.connected:
            mode = (
                ", conversation"
                if self._conversation_mode
                else (", one-shot" if self._one_shot else "")
            )
            diarization = ", diarization" if self.settings.deepgram_diarization_enabled else ""
            return (
                True,
                f"connected ({self.settings.deepgram_model}, "
                f"{self.settings.deepgram_language}{mode}{diarization})",
            )
        if self.running:
            mode = (
                " (conversation)"
                if self._conversation_mode
                else (" (one-shot)" if self._one_shot else "")
            )
            return False, self.last_error or f"connecting{mode}"
        return False, self.last_error or "stopped"

    def _url(self) -> str:
        parameters = {
            "model": self.settings.deepgram_model,
            "language": self.settings.deepgram_language,
            "encoding": "linear16",
            "sample_rate": str(self.settings.sample_rate),
            "channels": "1",
            "smart_format": "true",
            "punctuate": "true",
            "interim_results": "true",
            "endpointing": str(
                max(100, min(int(self.settings.deepgram_endpointing_ms), 5000))
            ),
            "utterance_end_ms": str(
                max(500, min(int(self.settings.deepgram_utterance_end_ms), 10000))
            ),
            "vad_events": "true",
        }
        if self.settings.deepgram_diarization_enabled:
            parameters["diarize_model"] = (
                self.settings.deepgram_diarization_model.strip() or "latest"
            )
        return f"wss://api.deepgram.com/v1/listen?{urlencode(parameters)}"

    async def _run_forever(self) -> None:
        delays = (1, 2, 4, 8, 16, 30)
        attempt = 0
        while self.running:
            try:
                if self._one_shot:
                    await asyncio.wait_for(
                        self._run_connection(),
                        timeout=self._one_shot_timeout_seconds,
                    )
                else:
                    await self._run_connection()
                attempt = 0
            except TimeoutError:
                self.connected = False
                self._connected_event.clear()
                self.running = False
                self._one_shot = False
                self._one_shot_prefix = ""
                self.last_error = "Upłynął czas oczekiwania na wypowiedź."
                await self.events.publish(
                    "listening.timeout",
                    {"error": self.last_error},
                )
                await self.events.publish(
                    "listening.stopped",
                    {"reason": "one_shot_timeout"},
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self._connected_event.clear()
                self.last_error = str(exc)
                await self.events.publish("listening.error", {"error": self.last_error})
                if self._one_shot:
                    self.running = False
                    self._one_shot = False
                    self._one_shot_prefix = ""
                    await self.events.publish(
                        "listening.stopped",
                        {"reason": "one_shot_error"},
                    )
                    break
                if not self.running:
                    break
                delay = delays[min(attempt, len(delays) - 1)]
                attempt += 1
                await asyncio.sleep(delay)

    async def _run_connection(self) -> None:
        key = self.settings.deepgram_api_key
        if key is None:
            raise RuntimeError("Brak klucza Deepgram")
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        loop = asyncio.get_running_loop()
        finals: list[str] = []
        confidences: list[float] = []
        final_words: list[TranscriptWordV1] = []
        final_speaker_ids: set[int] = set()
        last_sent = ""

        def enqueue_audio(data: bytes) -> None:
            if audio_queue.full():
                return
            audio_queue.put_nowait(data)

        def microphone_callback(indata, frames, time_info, status) -> None:
            if status:
                loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self.events.publish("audio.warning", {"status": str(status)}),
                )
            audio = bytes(indata)
            if self._audio_sink is not None:
                try:
                    self._audio_sink(audio)
                except Exception:
                    LOGGER.exception("Deepgram microphone audio sink failed")
            loop.call_soon_threadsafe(enqueue_audio, audio)

        async def flush(*, speech_final: bool) -> None:
            nonlocal last_sent
            transcript = " ".join(finals).strip()
            word_confidences = [
                word.confidence
                for word in final_words
                if word.confidence is not None
            ]
            confidence_values = word_confidences or confidences
            confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            )
            confidence_min = min(confidence_values) if confidence_values else None
            word_speakers = {
                word.speaker_id
                for word in final_words
                if word.speaker_id is not None
            }
            speaker_ids = tuple(sorted(final_speaker_ids | word_speakers))
            words = tuple(final_words)
            finals.clear()
            confidences.clear()
            final_words.clear()
            final_speaker_ids.clear()
            if not transcript or transcript == last_sent:
                return
            last_sent = transcript
            text = f"{self._one_shot_prefix}{transcript}".strip()
            starts = [word.start_seconds for word in words]
            ends = [word.end_seconds for word in words]
            envelope = TranscriptEnvelopeV1(
                raw_text=text,
                normalized_text=normalize_transcript_text(text),
                language=self.settings.deepgram_language,
                confidence_mean=confidence,
                confidence_min=confidence_min,
                words=words,
                started_at_seconds=min(starts) if starts else None,
                ended_at_seconds=max(ends) if ends else None,
                speaker_ids=speaker_ids,
                is_final=True,
                speech_final=speech_final,
                model=self.settings.deepgram_model,
            )
            if self._one_shot:
                self.running = False
                self._one_shot = False
                self._one_shot_prefix = ""
                await websocket.close()
                await self.events.publish(
                    "listening.stopped",
                    {"reason": "one_shot_completed"},
                )
            await self.events.publish(
                "transcript.final",
                {
                    "text": text,
                    "confidence": confidence,
                    "speaker_ids": list(speaker_ids),
                    "transcript": envelope.model_dump(mode="json"),
                },
            )
            self._dispatch_final(
                text,
                confidence=confidence,
                speaker_ids=speaker_ids,
                transcript=envelope,
            )

        async with websockets.connect(
            self._url(),
            subprotocols=["token", key.get_secret_value()],
            open_timeout=10,
            close_timeout=3,
            max_size=4 * 1024 * 1024,
        ) as websocket:
            self.connected = True
            self.connection_count += 1
            self._connected_event.set()
            self.last_error = None
            await self.events.publish(
                "listening.connected",
                {
                    "model": self.settings.deepgram_model,
                    "language": self.settings.deepgram_language,
                    "connection_count": self.connection_count,
                },
            )

            async def sender() -> None:
                stream = sd.RawInputStream(
                    samplerate=self.settings.sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=self.settings.sample_rate // 10,
                    device=self.settings.microphone_device,
                    callback=microphone_callback,
                )
                with stream:
                    while self.running:
                        chunk = await audio_queue.get()
                        await websocket.send(chunk)

            async def receiver() -> None:
                async for raw in websocket:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    message_type = message.get("type")
                    if message_type == "SpeechStarted":
                        await self.events.publish(
                            "transcript.speech_started",
                            {"mode": "one_shot" if self._one_shot else "continuous"},
                        )
                    elif message_type == "Results":
                        alternatives = (message.get("channel") or {}).get("alternatives") or [{}]
                        alternative = alternatives[0]
                        transcript = (alternative.get("transcript") or "").strip()
                        speaker_ids = self._extract_speaker_ids(alternative)
                        if message.get("is_final"):
                            if transcript:
                                finals.append(transcript)
                                final_speaker_ids.update(speaker_ids)
                                final_words.extend(
                                    self._extract_transcript_words(alternative)
                                )
                                raw_confidence = alternative.get("confidence")
                                if isinstance(raw_confidence, (int, float)):
                                    confidences.append(float(raw_confidence))
                            if message.get("speech_final"):
                                await flush(speech_final=True)
                        elif transcript:
                            combined = " ".join([*finals, transcript]).strip()
                            combined_speaker_ids = tuple(
                                sorted(final_speaker_ids | set(speaker_ids))
                            )
                            await self.events.publish(
                                "transcript.interim",
                                {
                                    "text": combined,
                                    "speaker_ids": list(combined_speaker_ids),
                                },
                            )
                            if self.on_interim is not None and combined:
                                self._dispatch_interim(
                                    combined,
                                    speaker_ids=combined_speaker_ids,
                                )
                    elif message_type == "UtteranceEnd":
                        await flush(speech_final=False)

            async def keepalive() -> None:
                while self.running:
                    await asyncio.sleep(8)
                    await websocket.send(json.dumps({"type": "KeepAlive"}))

            tasks = [
                asyncio.create_task(sender()),
                asyncio.create_task(receiver()),
                asyncio.create_task(keepalive()),
            ]
            try:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    exception = task.exception()
                    if exception:
                        raise exception
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.connected = False
                self._connected_event.clear()

    def _dispatch_final(
        self,
        text: str,
        *,
        confidence: float | None = None,
        speaker_ids: tuple[int, ...] = (),
        transcript: TranscriptEnvelopeV1 | None = None,
    ) -> asyncio.Task[None]:
        envelope = transcript or TranscriptEnvelopeV1.from_text(
            text,
            language=self.settings.deepgram_language,
            confidence=confidence,
            speaker_ids=speaker_ids,
            model=self.settings.deepgram_model,
        )
        if (
            self._priority_final_predicate is not None
            and self._priority_final_predicate(text)
        ):
            task = asyncio.create_task(
                self._invoke_final_callback(envelope),
                name="deepgram-priority-final-callback",
            )
            self._callback_tasks.add(task)
            task.add_done_callback(self._finish_callback)
            return task
        self._put_latest_bounded(
            self._final_queue,
            envelope,
            kind="final",
        )
        task = self._final_worker_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(
            self._drain_final_callbacks(),
            name="deepgram-final-callback-worker",
        )
        self._final_worker_task = task
        self._callback_tasks.add(task)
        task.add_done_callback(self._finish_callback)
        return task

    def _dispatch_interim(
        self,
        text: str,
        *,
        speaker_ids: tuple[int, ...] = (),
    ) -> asyncio.Task[None]:
        assert self.on_interim is not None
        self._put_latest_bounded(
            self._interim_queue,
            (text, speaker_ids),
            kind="interim",
        )
        task = self._interim_worker_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(
            self._drain_interim_callbacks(),
            name="deepgram-interim-callback-worker",
        )
        self._interim_worker_task = task
        self._callback_tasks.add(task)
        task.add_done_callback(self._finish_callback)
        return task

    async def _drain_final_callbacks(self) -> None:
        while not self._final_queue.empty():
            transcript = self._final_queue.get_nowait()
            await self._invoke_final_callback(transcript)

    async def _invoke_final_callback(
        self,
        transcript: TranscriptEnvelopeV1,
    ) -> None:
        try:
            kwargs = {
                "confidence": transcript.confidence_mean,
                "speaker_ids": transcript.speaker_ids,
            }
            if self._on_final_accepts_transcript:
                kwargs["transcript"] = transcript
            await self.on_final(transcript.normalized_text, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Deepgram final callback failed")

    async def _drain_interim_callbacks(self) -> None:
        assert self.on_interim is not None
        while not self._interim_queue.empty():
            text, speaker_ids = self._interim_queue.get_nowait()
            try:
                await self.on_interim(text, speaker_ids=speaker_ids)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Deepgram interim callback failed")

    @staticmethod
    def _put_latest_bounded(
        queue: asyncio.Queue,
        item: object,
        *,
        kind: str,
    ) -> None:
        if queue.full():
            queue.get_nowait()
            LOGGER.warning("Dropping oldest queued Deepgram %s callback", kind)
        queue.put_nowait(item)

    @staticmethod
    def _clear_queue(queue: asyncio.Queue) -> None:
        while not queue.empty():
            queue.get_nowait()

    @staticmethod
    def _extract_speaker_ids(alternative: dict) -> tuple[int, ...]:
        speaker_ids: set[int] = set()
        for word in alternative.get("words") or ():
            speaker = word.get("speaker") if isinstance(word, dict) else None
            if isinstance(speaker, int) and not isinstance(speaker, bool):
                speaker_ids.add(speaker)
        return tuple(sorted(speaker_ids))

    @staticmethod
    def _extract_transcript_words(
        alternative: dict,
    ) -> tuple[TranscriptWordV1, ...]:
        result: list[TranscriptWordV1] = []
        for item in alternative.get("words") or ():
            if not isinstance(item, dict):
                continue
            raw_word = str(
                item.get("word")
                or item.get("punctuated_word")
                or ""
            ).strip()
            if not raw_word:
                continue
            start = item.get("start")
            end = item.get("end")
            if not isinstance(start, int | float) or isinstance(start, bool):
                continue
            if not isinstance(end, int | float) or isinstance(end, bool):
                continue
            confidence = item.get("confidence")
            speaker = item.get("speaker")
            result.append(
                TranscriptWordV1(
                    word=raw_word,
                    punctuated_word=(
                        str(item.get("punctuated_word")).strip()
                        if item.get("punctuated_word")
                        else None
                    ),
                    start_seconds=float(start),
                    end_seconds=float(end),
                    confidence=(
                        float(confidence)
                        if isinstance(confidence, int | float)
                        and not isinstance(confidence, bool)
                        else None
                    ),
                    speaker_id=(
                        int(speaker)
                        if isinstance(speaker, int)
                        and not isinstance(speaker, bool)
                        and speaker >= 0
                        else None
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def _callback_accepts_keyword(callback: Callable, keyword: str) -> bool:
        try:
            parameters = inspect.signature(callback).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or (
                parameter.name == keyword
                and parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            )
            for parameter in parameters
        )

    def _finish_callback(self, task: asyncio.Task[None]) -> None:
        self._callback_tasks.discard(task)
        if task is self._final_worker_task:
            self._final_worker_task = None
        if task is self._interim_worker_task:
            self._interim_worker_task = None
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            LOGGER.error(
                "Deepgram final callback failed",
                exc_info=(type(exception), exception, exception.__traceback__),
            )
