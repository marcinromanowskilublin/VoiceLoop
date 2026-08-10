from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

import sounddevice as sd
import websockets

from .events import EventBus
from .settings import Settings

TextCallback = Callable[[str], Awaitable[None]]


class DeepgramListener:
    def __init__(
        self,
        *,
        settings: Settings,
        events: EventBus,
        on_final: TextCallback,
    ) -> None:
        self.settings = settings
        self.events = events
        self.on_final = on_final
        self.running = False
        self.connected = False
        self.last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._one_shot = False
        self._one_shot_prefix = ""
        self._one_shot_timeout_seconds = 30.0

    async def start(self) -> None:
        self._one_shot = False
        self._one_shot_prefix = ""
        await self._start()

    async def start_once(
        self,
        *,
        prefix: str = "",
        timeout_seconds: float = 30.0,
    ) -> None:
        if self.running:
            await self.stop()
        self._one_shot = True
        self._one_shot_prefix = prefix.strip()
        if self._one_shot_prefix:
            self._one_shot_prefix += " "
        self._one_shot_timeout_seconds = max(5.0, min(timeout_seconds, 120.0))
        await self._start()

    async def _start(self) -> None:
        if self.running:
            return
        if not self.settings.deepgram_api_key:
            raise RuntimeError("Brak DEEPGRAM_API_KEY w listener/.env")
        if not self.settings.deepgram_api_key.get_secret_value().strip():
            raise RuntimeError("DEEPGRAM_API_KEY jest pusty")
        self.running = True
        self.last_error = None
        self._task = asyncio.create_task(self._run_forever(), name="deepgram-listener")
        await self.events.publish("listening.started")

    async def stop(self) -> None:
        self.running = False
        self._one_shot = False
        self._one_shot_prefix = ""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.connected = False
        await self.events.publish("listening.stopped")

    def health(self) -> tuple[bool, str]:
        if self.connected:
            mode = ", one-shot" if self._one_shot else ""
            return (
                True,
                f"connected ({self.settings.deepgram_model}, "
                f"{self.settings.deepgram_language}{mode})",
            )
        if self.running:
            mode = " (one-shot)" if self._one_shot else ""
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
            "endpointing": "300",
            "utterance_end_ms": "1200",
            "vad_events": "true",
        }
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
            loop.call_soon_threadsafe(enqueue_audio, bytes(indata))

        async def flush() -> None:
            nonlocal last_sent
            transcript = " ".join(finals).strip()
            finals.clear()
            if not transcript or transcript == last_sent:
                return
            last_sent = transcript
            text = f"{self._one_shot_prefix}{transcript}".strip()
            if self._one_shot:
                self.running = False
                self._one_shot = False
                self._one_shot_prefix = ""
                await websocket.close()
                await self.events.publish(
                    "listening.stopped",
                    {"reason": "one_shot_completed"},
                )
            await self.events.publish("transcript.final", {"text": text})
            await self.on_final(text)

        async with websockets.connect(
            self._url(),
            subprotocols=["token", key.get_secret_value()],
            open_timeout=10,
            close_timeout=3,
            max_size=4 * 1024 * 1024,
        ) as websocket:
            self.connected = True
            self.last_error = None
            await self.events.publish(
                "listening.connected",
                {
                    "model": self.settings.deepgram_model,
                    "language": self.settings.deepgram_language,
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
                    if message_type == "Results":
                        alternatives = (message.get("channel") or {}).get("alternatives") or [{}]
                        transcript = (alternatives[0].get("transcript") or "").strip()
                        if message.get("is_final"):
                            if transcript:
                                finals.append(transcript)
                            if message.get("speech_final"):
                                await flush()
                        elif transcript:
                            combined = " ".join([*finals, transcript]).strip()
                            await self.events.publish(
                                "transcript.interim",
                                {"text": combined},
                            )
                    elif message_type == "UtteranceEnd":
                        await flush()

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
