from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

from .assistant import AssistantService
from .conversation_telemetry import ConversationTelemetry
from .deepgram import DeepgramListener
from .events import EventBus
from .models import CommandRequest, CommandSource, CommandStatus, TranscriptEnvelopeV1
from .router import normalize_text
from .tts import WindowsTTS

LOGGER = logging.getLogger("voiceloop.voice_conversation")

# Soft barge-in while listening / via API: cut speech and keep the session.
INTERRUPT_PHRASES = {
    "stop",
    "przerwij",
    "cisza",
    "zamilcz",
    "stoj",
    "dosc",
    "wystarczy",
    "nie mow",
    "zamilknij",
    "pauza",
    "pause",
    "poczekaj",
    "chwila",
}
PAUSE_CONFIRMATIONS = {
    "tak",
    "potwierdz",
    "potwierdzam",
    "tak potwierdz",
    "tak potwierdzam",
    "zgadza sie",
}
PAUSE_CANCELLATIONS = {
    "nie",
    "anuluj",
    "anuluj przerwe",
    "nie wstrzymuj",
    "rezygnuje",
    *INTERRUPT_PHRASES,
}
RESUME_PHRASES = {
    "wznow",
    "wznow dzialanie",
    "wlacz sie",
    "wracaj",
    "koniec przerwy",
    "anuluj przerwe",
    "kontynuuj dzialanie",
}
DIRECT_ADDRESS_PREFIXES = (
    "asystent",
    "asystencie",
    "assistant",
    "venice",
    "venive",
    "wenice",
    "voiceloop",
)
ECHO_LOW_INFORMATION_TOKENS = {
    "a",
    "ale",
    "co",
    "czego",
    "czy",
    "do",
    "gdzie",
    "i",
    "jak",
    "jest",
    "kiedy",
    "kto",
    "mi",
    "na",
    "no",
    "o",
    "od",
    "po",
    "sa",
    "sie",
    "ta",
    "te",
    "ten",
    "to",
    "w",
    "we",
    "z",
    "ze",
}
MAX_PAUSE_SECONDS = 24 * 60 * 60
PAUSE_PATTERN = re.compile(
    r"\b(?:przerwij|wstrzymaj|zatrzymaj|zawies)\s+"
    r"(?:dzialanie|prace|asystenta)\s+na\s+"
    r"(?P<value>\d+|[a-z]+(?:\s+[a-z]+)?)\s*"
    r"(?P<unit>sekund(?:e|a|y)?|sek|s|minut(?:e|a|y)?|min|"
    r"godzin(?:e|a|y)?|godz|h)\b"
)
POLISH_NUMBERS = {
    "jeden": 1,
    "jedna": 1,
    "jedno": 1,
    "dwa": 2,
    "dwie": 2,
    "trzy": 3,
    "cztery": 4,
    "piec": 5,
    "szesc": 6,
    "siedem": 7,
    "osiem": 8,
    "dziewiec": 9,
    "dziesiec": 10,
    "jedenascie": 11,
    "dwanascie": 12,
    "trzynascie": 13,
    "czternascie": 14,
    "pietnascie": 15,
    "szesnascie": 16,
    "siedemnascie": 17,
    "osiemnascie": 18,
    "dziewietnascie": 19,
}
POLISH_TENS = {
    "dwadziescia": 20,
    "trzydziesci": 30,
    "czterdziesci": 40,
    "piecdziesiat": 50,
    "szescdziesiat": 60,
    "siedemdziesiat": 70,
    "osiemdziesiat": 80,
    "dziewiecdziesiat": 90,
}


@dataclass(frozen=True, slots=True)
class PauseRequest:
    value: int
    unit: str
    seconds: int

    @property
    def display(self) -> str:
        forms = {
            "second": ("sekundę", "sekundy", "sekund"),
            "minute": ("minutę", "minuty", "minut"),
            "hour": ("godzinę", "godziny", "godzin"),
        }[self.unit]
        if self.value == 1:
            label = forms[0]
        elif self.value % 10 in {2, 3, 4} and self.value % 100 not in {12, 13, 14}:
            label = forms[1]
        else:
            label = forms[2]
        return f"{self.value} {label}"


@dataclass(frozen=True, slots=True)
class EchoDecision:
    score: float
    is_echo: bool
    new_tokens: tuple[str, ...] = ()


class ConversationState(StrEnum):
    IDLE = "idle"
    SPEAKING = "speaking"
    LISTENING_ONCE = "listening_once"
    PROCESSING = "processing"
    PAUSED = "paused"


class VoiceConversationCoordinator:
    """Half-duplex owner of greeting → listen → reply → listen again."""

    def __init__(
        self,
        *,
        assistant: AssistantService,
        deepgram: DeepgramListener,
        tts: WindowsTTS,
        events: EventBus,
        greeting: str,
        cooldown_ms: int = 350,
        barge_in_after_ms: int = 3000,
        direct_address_after_seconds: float = 30.0,
        ignore_multi_speaker: bool = True,
        allow_cloud: bool = False,
        telemetry: ConversationTelemetry | None = None,
        stream_reuse_enabled: bool = False,
        hybrid_barge_in_enabled: bool = False,
        hybrid_barge_in_grace_ms: int = 700,
        barge_in_stability_ms: int = 300,
        barge_in_profile: str = "auto",
    ) -> None:
        self.assistant = assistant
        self.deepgram = deepgram
        self.tts = tts
        self.events = events
        self.greeting = (
            greeting
            or (
                "Cześć. Możemy porozmawiać albo możesz od razu wydać polecenie. "
                "Możesz też zapytać, co potrafię. W czym mogę pomóc?"
            )
        ).strip()
        self.cooldown_seconds = max(0.0, cooldown_ms / 1000.0)
        self.stream_reuse_enabled = bool(stream_reuse_enabled)
        self.hybrid_barge_in_enabled = bool(hybrid_barge_in_enabled)
        effective_grace_ms = (
            hybrid_barge_in_grace_ms
            if self.stream_reuse_enabled and self.hybrid_barge_in_enabled
            else barge_in_after_ms
        )
        self.barge_in_after_seconds = max(0.0, effective_grace_ms / 1000.0)
        self.barge_in_stability_seconds = max(0.05, barge_in_stability_ms / 1000.0)
        profile = (barge_in_profile or "auto").strip().lower()
        self.barge_in_profile = (
            profile if profile in {"auto", "headphones", "speakers"} else "auto"
        )
        self.direct_address_after_seconds = max(
            0.0,
            direct_address_after_seconds,
        )
        self.ignore_multi_speaker = ignore_multi_speaker
        self.allow_cloud = allow_cloud
        self.telemetry = telemetry
        self.state = ConversationState.IDLE
        self._lock = asyncio.Lock()
        self._pause_lock = asyncio.Lock()
        self._blocked = False
        self._transcribe_only = False
        self._turn_active = False
        self._accept_final = False
        self._turn_id = 0
        self._interrupt_requested = False
        self._barge_in_armed = False
        self._pending_barge_in: tuple[
            str,
            float | None,
            TranscriptEnvelopeV1 | None,
        ] | None = None
        self._current_spoken_text = ""
        self._recent_spoken_text = ""
        self._recent_spoken_until = 0.0
        self._address_free_until = 0.0
        self._refresh_address_window = False
        self._barge_arm_task: asyncio.Task[None] | None = None
        self._pending_pause: PauseRequest | None = None
        self._pause_until: float | None = None
        self._pause_task: asyncio.Task[None] | None = None
        self._last_tts_ms = 0
        self._last_cooldown_ms = 0
        self._pending_speech_started_perf: float | None = None
        self._pending_first_interim_perf: float | None = None
        self._pending_barge_detected_perf: float | None = None
        self._pending_tts_stopped_perf: float | None = None
        self._trace_sessions: dict[int, str] = {}
        self._barge_candidate_text = ""
        self._barge_candidate_since = 0.0
        self._tts_started_monotonic = 0.0
        self._last_connection_count = int(
            getattr(self.deepgram, "connection_count", 0)
        )

    def health(self) -> tuple[bool, str]:
        if self._transcribe_only:
            return True, "transcribe_only"
        if self._blocked:
            return False, f"blocked ({self.state.value})"
        detail = self.state.value
        if self.state is ConversationState.PAUSED and self._pause_until is not None:
            remaining = max(0, round(self._pause_until - time.monotonic()))
            detail = f"paused ({remaining}s remaining)"
        elif self._pending_pause is not None:
            detail = f"{detail}+pause_confirmation"
        elif (
            self.state is ConversationState.LISTENING_ONCE
            and time.monotonic() > self._address_free_until
        ):
            detail = f"{detail}+address_required"
        if self._barge_in_armed:
            detail = f"{detail}+barge_in"
        return True, detail

    def should_handle_control_transcript(self, text: str) -> bool:
        if self._transcribe_only:
            return False
        return (
            self.state is ConversationState.PAUSED
            or self._pending_pause is not None
            or self._parse_pause_request(text) is not None
            or self._is_interrupt_phrase(text)
        )

    def should_route_to_assistant(self) -> bool:
        return not self._transcribe_only

    def accepts_speaking_transcript(self) -> bool:
        return (
            self.state is ConversationState.SPEAKING
            and (self._barge_in_armed or self.stream_reuse_enabled)
        )

    async def handle_speech_started(self) -> None:
        if self._transcribe_only or self._blocked:
            return
        if self.state not in {
            ConversationState.LISTENING_ONCE,
            ConversationState.SPEAKING,
            ConversationState.PAUSED,
        }:
            return
        self._pending_speech_started_perf = time.perf_counter()
        self._pending_first_interim_perf = None

    async def start_transcribe_only(self) -> dict[str, str]:
        await self._cancel_barge_arm()
        await self._cancel_pause_task()
        self._transcribe_only = True
        self._blocked = True
        self._interrupt_requested = True
        self._barge_in_armed = False
        self._pending_barge_in = None
        self._pending_pause = None
        self._pause_until = None
        self._accept_final = False
        self._turn_active = False
        self._turn_id += 1
        await self.tts.stop()
        await self.assistant.interrupt(end_conversation=True)
        self.state = ConversationState.IDLE
        await self.deepgram.start()
        await self.events.publish(
            "conversation.transcribe_only",
            {"enabled": True, "status": "transcribe_only"},
        )
        return {"status": "transcribe_only", "state": self.state.value}

    def end_transcribe_only(self) -> None:
        self._transcribe_only = False

    async def start_conversation(self) -> dict[str, str]:
        async with self._lock:
            if self._blocked:
                return {"status": "blocked", "state": self.state.value}
            if self.state != ConversationState.IDLE or self._turn_active:
                return {"status": "busy", "state": self.state.value}
            self._turn_active = True
            self._accept_final = False
            self._interrupt_requested = False
            self._turn_id += 1
            turn_id = self._turn_id

        try:
            if self.stream_reuse_enabled:
                await self._ensure_conversation_stream()
            else:
                await self.deepgram.stop()
            new_session = not self.assistant.conversation_active
            prompt = self.greeting if new_session else "Słucham."
            if new_session:
                await self.assistant.begin_conversation_session(self.greeting)
            await self.events.publish(
                "conversation.greeting",
                {"text": prompt, "new_session": new_session},
            )
            barge = await self._speak_with_barge_in(prompt, turn_id=turn_id)
            if barge is not None:
                text, confidence, transcript = barge
                self._interrupt_requested = False
                await self._run_user_turn(
                    text,
                    confidence=confidence,
                    transcript=transcript,
                    turn_id=self._turn_id,
                )
                return {"status": self.state.value, "state": self.state.value}
            if self._interrupted(turn_id):
                return await self._resume_listening(continued=True, turn_id=turn_id)
            await self._wait_cooldown()
            if self._interrupted(turn_id):
                return await self._resume_listening(continued=True, turn_id=turn_id)
            return await self._resume_listening(continued=False, turn_id=turn_id)
        except Exception:
            LOGGER.exception("Conversation start failed")
            self._accept_final = False
            self._turn_active = False
            self.state = ConversationState.IDLE
            raise

    async def handle_interim(
        self,
        text: str,
        *,
        speaker_ids: tuple[int, ...] = (),
    ) -> None:
        """Cut TTS as soon as user speech is heard after the barge-in grace period."""
        if self._transcribe_only:
            return
        cleaned = (text or "").strip()
        if len(cleaned) < 3:
            return
        if self._pending_first_interim_perf is None:
            self._pending_first_interim_perf = time.perf_counter()
        if self.state is not ConversationState.SPEAKING or (
            not self._barge_in_armed and not self.stream_reuse_enabled
        ):
            return
        explicit_interrupt = self._is_interrupt_phrase(cleaned)
        echo = self._echo_decision(cleaned)
        if not explicit_interrupt and echo.is_echo:
            await self.events.publish(
                "conversation.echo_ignored",
                {
                    "phase": "interim",
                    "text": cleaned[:120],
                    "speaker_ids": list(speaker_ids),
                    "echo_score": round(echo.score, 3),
                },
            )
            return
        if len(normalize_text(cleaned).split()) < 2 and not explicit_interrupt:
            return
        if not explicit_interrupt and not self.hybrid_barge_in_enabled:
            await self.events.publish(
                "conversation.barge_in_candidate",
                {
                    "text": cleaned[:120],
                    "speaker_ids": list(speaker_ids),
                },
            )
            return
        if not explicit_interrupt:
            normalized = self._normalized_phrase(cleaned)
            now = time.monotonic()
            if (
                self._barge_candidate_text
                and (
                    normalized.startswith(self._barge_candidate_text)
                    or self._barge_candidate_text.startswith(normalized)
                )
            ):
                stable_for = now - self._barge_candidate_since
            else:
                self._barge_candidate_text = normalized
                self._barge_candidate_since = now
                stable_for = 0.0
            await self.events.publish(
                "conversation.barge_in_candidate",
                {
                    "text": cleaned[:120],
                    "speaker_ids": list(speaker_ids),
                    "echo_score": round(echo.score, 3),
                    "new_tokens": list(echo.new_tokens),
                    "stable_ms": round(stable_for * 1000),
                },
            )
            if (
                len(echo.new_tokens) < 2
                or stable_for < self.barge_in_stability_seconds
                or (
                    not self._barge_in_armed
                    and now - self._tts_started_monotonic
                    < self._effective_barge_in_delay_seconds(echo)
                )
            ):
                return
        await self.events.publish(
            "conversation.interrupted",
            {
                "reason": "barge_in_interim",
                "text": cleaned[:120],
                "speaker_ids": list(speaker_ids),
            },
        )
        self._interrupt_requested = True
        self._barge_candidate_text = ""
        self._pending_barge_detected_perf = time.perf_counter()
        await self.tts.stop()
        self._pending_tts_stopped_perf = time.perf_counter()

    async def handle_transcript(
        self,
        text: str,
        *,
        confidence: float | None = None,
        speaker_ids: tuple[int, ...] = (),
        transcript: TranscriptEnvelopeV1 | None = None,
    ) -> None:
        if self._transcribe_only:
            return
        cleaned = (text or "").strip()
        if not cleaned:
            return

        if self.state is ConversationState.PAUSED:
            self._accept_final = False
            await self._handle_paused_transcript(cleaned)
            return
        if self._is_interrupt_phrase(cleaned):
            await self.interrupt_speech()
            return

        # Barge-in final during her speech: stash and cut; speak() returns to caller.
        if self.state is ConversationState.SPEAKING and (
            self._barge_in_armed or self.stream_reuse_enabled
        ):
            echo = self._echo_decision(cleaned)
            if not self._is_interrupt_phrase(cleaned) and echo.is_echo:
                await self.events.publish(
                    "conversation.echo_ignored",
                    {
                        "phase": "final",
                        "text": cleaned[:120],
                        "speaker_ids": list(speaker_ids),
                        "echo_score": round(echo.score, 3),
                    },
                )
                if not self.stream_reuse_enabled:
                    await self._rearm_barge_in_listener(reason="tts_echo_ignored")
                return
            self._pending_barge_in = (cleaned, confidence, transcript)
            self._accept_final = False
            if not self._barge_in_armed and self.stream_reuse_enabled:
                await self.events.publish(
                    "conversation.barge_in_candidate",
                    {
                        "text": cleaned[:120],
                        "speaker_ids": list(speaker_ids),
                        "reason": "early_final_buffered",
                        "echo_score": round(echo.score, 3),
                    },
                )
                return
            self._interrupt_requested = True
            self._pending_barge_detected_perf = time.perf_counter()
            await self.tts.stop()
            self._pending_tts_stopped_perf = time.perf_counter()
            await self.events.publish(
                "conversation.interrupted",
                {
                    "reason": "barge_in_final",
                    "text": cleaned[:120],
                    "speaker_ids": list(speaker_ids),
                },
            )
            return

        if not self._accept_final:
            if self._parse_pause_request(cleaned) is not None and not self._blocked:
                if not self.assistant.conversation_active:
                    await self.assistant.begin_conversation_session("")
                self._turn_active = True
                self._interrupt_requested = False
                self._turn_id += 1
                turn_id = self._turn_id
                await self._run_user_turn(
                    cleaned,
                    confidence=confidence,
                    transcript=transcript,
                    turn_id=turn_id,
                )
                return
            LOGGER.info("Ignoring transcript outside conversation turn: %s", cleaned[:80])
            return

        control_phrase = (
            self._pending_pause is not None
            or self._parse_pause_request(cleaned) is not None
            or self._is_interrupt_phrase(cleaned)
        )
        if (
            not control_phrase
            and not self._is_direct_address(cleaned)
            and self._is_likely_tts_echo(cleaned)
        ):
            self._accept_final = False
            self._turn_id += 1
            turn_id = self._turn_id
            await self.events.publish(
                "conversation.echo_ignored",
                {
                    "phase": "post_tts_final",
                    "speaker_ids": list(speaker_ids),
                },
            )
            if not self.stream_reuse_enabled:
                await self.deepgram.stop()
            await self._resume_listening(continued=True, turn_id=turn_id)
            return
        if (
            time.monotonic() > self._address_free_until
            and not self._is_direct_address(cleaned)
            and not control_phrase
        ):
            self._accept_final = False
            self._turn_id += 1
            turn_id = self._turn_id
            await self.events.publish(
                "conversation.ambient_ignored",
                {
                    "text": cleaned[:120],
                    "reason": "direct_address_required_after_idle",
                },
            )
            if not self.stream_reuse_enabled:
                await self.deepgram.stop()
            await self._resume_listening(continued=True, turn_id=turn_id)
            return

        unique_speakers = set(speaker_ids)
        if (
            self.ignore_multi_speaker
            and len(unique_speakers) > 1
            and not self._is_direct_address(cleaned)
            and self._parse_pause_request(cleaned) is None
            and not self._is_interrupt_phrase(cleaned)
        ):
            self._accept_final = False
            self._turn_id += 1
            turn_id = self._turn_id
            await self.events.publish(
                "conversation.multi_speaker_ignored",
                {
                    "text": cleaned[:120],
                    "speaker_ids": sorted(unique_speakers),
                    "reason": "direct_address_required",
                },
            )
            if not self.stream_reuse_enabled:
                await self.deepgram.stop()
            await self._resume_listening(continued=True, turn_id=turn_id)
            return

        self._accept_final = False
        self._interrupt_requested = False
        self._turn_id += 1
        turn_id = self._turn_id
        await self._run_user_turn(
            cleaned,
            confidence=confidence,
            transcript=transcript,
            turn_id=turn_id,
        )

    async def interrupt_speech(self) -> dict[str, str]:
        """Soft barge-in: cut TTS/model work, keep session, listen immediately."""
        if self._transcribe_only:
            await self.tts.stop()
            await self.assistant.interrupt(end_conversation=True)
            return {"status": "transcribe_only", "state": self.state.value}
        if self.state is ConversationState.PAUSED:
            return {"status": "paused", "state": self.state.value}
        await self._cancel_barge_arm()
        self._interrupt_requested = True
        self._barge_in_armed = False
        self._pending_barge_in = None
        self._turn_id += 1
        turn_id = self._turn_id
        self._accept_final = False
        if not self.stream_reuse_enabled:
            await self.deepgram.stop()
        await self.tts.stop()
        await self.assistant.interrupt(end_conversation=False)
        if not self.assistant.conversation_active:
            await self.assistant.begin_conversation_session(self.greeting)
        await self.events.publish(
            "conversation.interrupted",
            {"reason": "api_stop", "status": "listening_once"},
        )
        self._refresh_address_window = True
        return await self._resume_listening(continued=True, turn_id=turn_id)

    async def hard_stop(self) -> dict[str, str]:
        if self._transcribe_only:
            await self.tts.stop()
            await self.assistant.interrupt(end_conversation=True)
            return {"status": "transcribe_only", "state": self.state.value}
        await self._cancel_barge_arm()
        await self._cancel_pause_task()
        self._blocked = True
        self._interrupt_requested = True
        self._barge_in_armed = False
        self._pending_barge_in = None
        self._pending_pause = None
        self._pause_until = None
        self._accept_final = False
        self._turn_active = False
        self._turn_id += 1
        await self.deepgram.stop()
        await self.tts.stop()
        await self.assistant.interrupt(end_conversation=True)
        self.state = ConversationState.IDLE
        await self.events.publish("conversation.stopped", {"status": "stopped"})
        return {"status": "stopped", "state": self.state.value}

    def clear_block(self) -> None:
        self._blocked = False
        self._interrupt_requested = False

    async def resume_from_pause(self, *, source: str = "api") -> dict[str, str]:
        if self.state is not ConversationState.PAUSED:
            return {"status": "not_paused", "state": self.state.value}
        return await self._leave_pause(source=source)

    async def close(self) -> None:
        await self._cancel_barge_arm()
        await self._cancel_pause_task()

    async def handle_listen_timeout(self, *, reason: str = "one_shot_timeout") -> None:
        """Re-arm the mic when Deepgram one-shot ends without a final transcript.

        Without this, conversation stays in LISTENING_ONCE with _accept_final=True
        while Deepgram is already stopped — permanent silence.
        """
        if self._blocked:
            return
        if self.state is ConversationState.SPEAKING:
            if self._barge_in_armed:
                LOGGER.warning(
                    "Barge-in listen ended during speaking (%s); re-arming",
                    reason,
                )
                await self._rearm_barge_in_listener(
                    reason=f"{reason}_during_tts",
                )
            else:
                LOGGER.info("Ignoring listen timeout during speaking (%s)", reason)
            return
        if self.state is ConversationState.PAUSED:
            if self._pause_until is not None and time.monotonic() >= self._pause_until:
                await self._leave_pause(source="timer")
                return
            LOGGER.info("Paused-listener timeout (%s); re-arming resume command", reason)
            if not self.stream_reuse_enabled:
                await self.deepgram.stop()
            await self._arm_paused_listening()
            return
        if self.state is not ConversationState.LISTENING_ONCE or not self._turn_active:
            return
        LOGGER.warning("Conversation listen timed out (%s); re-arming mic", reason)
        self._turn_id += 1
        turn_id = self._turn_id
        await self.events.publish(
            "conversation.listen_timeout",
            {"reason": reason, "state": self.state.value},
        )
        await self._resume_listening(continued=True, turn_id=turn_id)

    async def _run_user_turn(
        self,
        cleaned: str,
        *,
        confidence: float | None,
        transcript: TranscriptEnvelopeV1 | None,
        turn_id: int,
    ) -> None:
        session_id = self.assistant.conversation_session_id or "voice-session"
        now_perf = time.perf_counter()
        started_perf = self._pending_speech_started_perf
        if started_perf is None and transcript is not None:
            if (
                transcript.started_at_seconds is not None
                and transcript.ended_at_seconds is not None
            ):
                duration = max(
                    0.0,
                    transcript.ended_at_seconds - transcript.started_at_seconds,
                )
                started_perf = now_perf - duration
        started_perf = started_perf or now_perf
        initial_phases = {"speech_started": 0}
        if self._pending_first_interim_perf is not None:
            initial_phases["first_interim"] = max(
                0,
                round((self._pending_first_interim_perf - started_perf) * 1000),
            )
        if self._pending_barge_detected_perf is not None:
            initial_phases["barge_detected"] = max(
                0,
                round((self._pending_barge_detected_perf - started_perf) * 1000),
            )
        if self._pending_tts_stopped_perf is not None:
            initial_phases["tts_stopped"] = max(
                0,
                round((self._pending_tts_stopped_perf - started_perf) * 1000),
            )
        if self.telemetry is not None:
            self._trace_sessions[turn_id] = session_id
            await self.telemetry.begin(
                session_id=session_id,
                turn_id=turn_id,
                started_perf=started_perf,
                initial_phases_ms=initial_phases,
                metadata={
                    "transcript_confidence": confidence,
                    "segment_id": transcript.segment_id if transcript is not None else None,
                    "conversation_mode": (
                        "hybrid"
                        if self.stream_reuse_enabled and self.hybrid_barge_in_enabled
                        else "legacy"
                    ),
                    "barge_in_profile": self.barge_in_profile,
                },
            )
            await self.telemetry.mark(
                session_id=session_id,
                turn_id=turn_id,
                phase="speech_end",
            )
        self._pending_speech_started_perf = None
        self._pending_first_interim_perf = None
        self._pending_barge_detected_perf = None
        self._pending_tts_stopped_perf = None
        self.state = ConversationState.PROCESSING
        try:
            if not self.stream_reuse_enabled:
                await self.deepgram.stop()
            if self._pending_pause is not None:
                await self._handle_pause_confirmation(cleaned, turn_id=turn_id)
                return

            pause_request = self._parse_pause_request(cleaned)
            if pause_request is not None:
                if pause_request.seconds <= 0:
                    await self._speak_control_prompt(
                        "Czas przerwy musi być większy od zera.",
                        turn_id=turn_id,
                    )
                    return
                if pause_request.seconds > MAX_PAUSE_SECONDS:
                    await self._speak_control_prompt(
                        "Maksymalna przerwa to 24 godziny. Podaj krótszy czas.",
                        turn_id=turn_id,
                    )
                    return
                self._pending_pause = pause_request
                question = (
                    "Czy na pewno wstrzymać działanie na "
                    f"{pause_request.display}?"
                )
                await self.events.publish(
                    "conversation.pause_confirmation_requested",
                    {
                        "value": pause_request.value,
                        "unit": pause_request.unit,
                        "seconds": pause_request.seconds,
                    },
                )
                await self._speak_control_prompt(question, turn_id=turn_id)
                return

            if self._is_interrupt_phrase(cleaned):
                await self.events.publish(
                    "conversation.interrupted",
                    {"reason": "spoken_stop", "text": cleaned[:120]},
                )
                await self.assistant.interrupt(end_conversation=False)
                await self.tts.stop()
                self._refresh_address_window = True
                await self._resume_listening(continued=True, turn_id=turn_id)
                return

            request = (
                CommandRequest.from_transcript(
                    transcript,
                    allow_cloud=self.allow_cloud,
                    managed_voice_turn=True,
                    interaction_session_id=self.assistant.conversation_session_id,
                    conversation_turn_id=turn_id,
                )
                if transcript is not None
                else CommandRequest(
                    source=CommandSource.DEEPGRAM,
                    text=cleaned,
                    allow_cloud=self.allow_cloud,
                    transcript_confidence=confidence,
                    managed_voice_turn=True,
                    interaction_session_id=self.assistant.conversation_session_id,
                    conversation_turn_id=turn_id,
                )
            )
            if self.telemetry is not None:
                await self.telemetry.mark(
                    session_id=session_id,
                    turn_id=turn_id,
                    phase="request_created",
                    request_id=request.request_id,
                )
            accepted = await self.assistant.handle(request)
            if self._interrupted(turn_id):
                await self._resume_listening(continued=True, turn_id=turn_id)
                return

            spoken = None
            managed_execution = bool(
                accepted.plan is not None
                and accepted.plan.steps
                and not accepted.plan.confirmation_required
                and not accepted.plan.requires_clarification
            )
            if managed_execution:
                completed = await self.assistant.executor.wait_for_completion(
                    accepted.request_id
                )
                if self._interrupted(turn_id):
                    await self._resume_listening(continued=True, turn_id=turn_id)
                    return
                result_messages = (
                    [
                        result.message.strip()
                        for result in completed.results
                        if result.message.strip()
                    ]
                    if completed is not None
                    else []
                )
                if result_messages:
                    spoken = " ".join(dict.fromkeys(result_messages))[:2000]
                elif completed is not None and completed.status is CommandStatus.FAILED:
                    detail = (completed.error or "").strip()
                    spoken = (
                        f"Nie udało się wykonać polecenia. {detail}".strip()[:2000]
                    )
                elif completed is not None and completed.status is CommandStatus.CANCELLED:
                    spoken = ""
                else:
                    spoken = (
                        accepted.plan.response_text if accepted.plan is not None else ""
                    ) or "Gotowe."
                await self.assistant.remember_managed_turn_result(
                    request_id=accepted.request_id,
                    user_text=cleaned,
                    assistant_text=spoken,
                )
            if accepted.plan is not None:
                spoken = spoken or (
                    accepted.plan.clarification_question or accepted.plan.response_text
                )
            should_speak = bool(
                spoken
                and accepted.plan is not None
                and (
                    managed_execution
                    or
                    accepted.plan.intent == "conversation"
                    or accepted.plan.intent == "conversation_end"
                    or not accepted.plan.speak_result
                    or accepted.plan.confirmation_required
                    or accepted.plan.requires_clarification
                )
            )
            if should_speak and spoken:
                await self.events.publish(
                    "conversation.reply",
                    {
                        "request_id": accepted.request_id,
                        "text": spoken[:500],
                    },
                )
                barge = await self._speak_with_barge_in(spoken, turn_id=turn_id)
                if barge is not None:
                    text, barge_confidence, barge_transcript = barge
                    # New user turn after barge-in.
                    self._turn_id += 1
                    self._interrupt_requested = False
                    await self._run_user_turn(
                        text,
                        confidence=barge_confidence,
                        transcript=barge_transcript,
                        turn_id=self._turn_id,
                    )
                    return

            if self._interrupted(turn_id):
                await self._resume_listening(continued=True, turn_id=turn_id)
                return

            ended = (
                accepted.plan is not None
                and accepted.plan.intent == "conversation_end"
            ) or not self.assistant.conversation_active
            if ended or self._blocked:
                self._turn_active = False
                self._accept_final = False
                self.state = ConversationState.IDLE
                await self.events.publish(
                    "conversation.turn_complete",
                    {"status": "idle", "continue": False},
                )
                return

            await self._wait_cooldown()
            if self._interrupted(turn_id) or self._blocked:
                if self._blocked:
                    self._turn_active = False
                    self._accept_final = False
                    self.state = ConversationState.IDLE
                    return
                await self._resume_listening(continued=True, turn_id=turn_id)
                return
            await self._resume_listening(continued=True, turn_id=turn_id)
        except asyncio.CancelledError:
            if turn_id == self._turn_id:
                self._turn_active = False
                self._accept_final = False
                self.state = ConversationState.IDLE
            raise
        except Exception as exc:
            LOGGER.exception("Conversation turn failed; re-arming microphone")
            await self.events.publish(
                "conversation.turn_error",
                {"error": str(exc)[:500], "continue": not self._blocked},
            )
            if self._blocked or turn_id != self._turn_id:
                self._turn_active = False
                self._accept_final = False
                self.state = ConversationState.IDLE
                return
            try:
                await self._resume_listening(continued=True, turn_id=turn_id)
            except Exception:
                LOGGER.exception("Could not recover conversation listening after turn error")
                self._turn_active = False
                self._accept_final = False
                self.state = ConversationState.IDLE

    async def _speak_with_barge_in(
        self,
        text: str,
        *,
        turn_id: int,
    ) -> tuple[str, float | None, TranscriptEnvelopeV1 | None] | None:
        self.state = ConversationState.SPEAKING
        self._barge_in_armed = False
        self._pending_barge_in = None
        self._barge_candidate_text = ""
        self._barge_candidate_since = 0.0
        self._current_spoken_text = text
        if self.stream_reuse_enabled:
            await self._ensure_conversation_stream()
        self._accept_final = self.stream_reuse_enabled
        await self._cancel_barge_arm()
        barge_listener_started = False
        self._tts_started_monotonic = time.monotonic()

        async def arm_barge_in() -> None:
            nonlocal barge_listener_started
            if self.barge_in_after_seconds > 0:
                await asyncio.sleep(self.barge_in_after_seconds)
            if self._interrupted(turn_id) or self.state is not ConversationState.SPEAKING:
                return
            try:
                if self.stream_reuse_enabled:
                    await self._ensure_conversation_stream()
                else:
                    # Mark before awaiting: start_once may create its task and then be
                    # cancelled while publishing the start event.
                    barge_listener_started = True
                    await self.deepgram.start_once(timeout_seconds=45.0)
                await self._wait_for_deepgram_ready()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if barge_listener_started:
                    await self.deepgram.stop()
                    barge_listener_started = False
                LOGGER.warning("Could not arm conversation barge-in: %s", exc)
                await self.events.publish(
                    "conversation.barge_in_unavailable",
                    {"error": str(exc)[:300]},
                )
                return
            if self._interrupted(turn_id) or self.state is not ConversationState.SPEAKING:
                return
            self._barge_in_armed = True
            self._accept_final = True
            await self.events.publish(
                "conversation.barge_in_armed",
                {"after_ms": int(self.barge_in_after_seconds * 1000)},
            )
            if self._pending_barge_in is not None:
                self._interrupt_requested = True
                self._pending_barge_detected_perf = time.perf_counter()
                await self.tts.stop()
                self._pending_tts_stopped_perf = time.perf_counter()

        self._barge_arm_task = asyncio.create_task(
            arm_barge_in(),
            name="conversation-barge-in-arm",
        )
        started = time.perf_counter()
        try:
            await self._trace_mark(turn_id, "tts_started")
            await self._trace_mark(
                turn_id,
                "tts_first_audio",
                metadata={"first_audio_estimated": True},
            )
            await self.tts.speak(text)
        finally:
            self._last_tts_ms = round((time.perf_counter() - started) * 1000)
            await self._trace_mark(turn_id, "tts_completed")
            await self._cancel_barge_arm()
            armed = self._barge_in_armed
            self._barge_in_armed = False
            if not self.stream_reuse_enabled and (armed or barge_listener_started):
                # Stop one-shot listen opened for barge-in; next turn reopens mic.
                await self.deepgram.stop()
            self._accept_final = self.stream_reuse_enabled
            interrupted = (
                self._interrupt_requested or self._pending_barge_in is not None
            )
            await self.events.publish(
                "conversation.tts_completed",
                {
                    "completed": not interrupted,
                    "interrupted": interrupted,
                    "tts_ms": self._last_tts_ms,
                    "text_length": len(text),
                },
            )
            self._refresh_address_window = not interrupted
            self._recent_spoken_text = text
            self._recent_spoken_until = time.monotonic() + 2.5
            self._current_spoken_text = ""

        pending = self._pending_barge_in
        self._pending_barge_in = None
        if pending is not None:
            return pending
        return None

    async def _rearm_barge_in_listener(self, *, reason: str) -> None:
        if self.stream_reuse_enabled:
            await self._ensure_conversation_stream()
            self._accept_final = True
            await self.events.publish(
                "conversation.barge_in_rearmed",
                {"reason": reason, "reused": True},
            )
            return
        turn_id = self._turn_id
        self._accept_final = False
        await self.deepgram.stop()
        if (
            self._blocked
            or self._interrupt_requested
            or turn_id != self._turn_id
            or self.state is not ConversationState.SPEAKING
            or not self._barge_in_armed
        ):
            return
        try:
            await self.deepgram.start_once(timeout_seconds=45.0)
            await self._wait_for_deepgram_ready()
        except Exception as exc:
            LOGGER.warning("Could not re-arm barge-in (%s): %s", reason, exc)
            await self.deepgram.stop()
            self._barge_in_armed = False
            self._accept_final = False
            await self.events.publish(
                "conversation.barge_in_unavailable",
                {"error": str(exc)[:300], "reason": f"{reason}_rearm_failed"},
            )
            return
        if (
            turn_id == self._turn_id
            and self.state is ConversationState.SPEAKING
            and not self._interrupt_requested
        ):
            self._accept_final = True
            await self.events.publish(
                "conversation.barge_in_rearmed",
                {"reason": reason},
            )

    async def _speak_control_prompt(self, text: str, *, turn_id: int) -> None:
        await self.events.publish(
            "conversation.control_prompt",
            {"text": text, "turn_id": turn_id},
        )
        barge = await self._speak_with_barge_in(text, turn_id=turn_id)
        if barge is not None:
            reply, confidence, transcript = barge
            self._turn_id += 1
            self._interrupt_requested = False
            await self._run_user_turn(
                reply,
                confidence=confidence,
                transcript=transcript,
                turn_id=self._turn_id,
            )
            return
        if self._interrupted(turn_id):
            await self._resume_listening(continued=True, turn_id=turn_id)
            return
        await self._wait_cooldown()
        await self._resume_listening(continued=True, turn_id=turn_id)

    async def _handle_pause_confirmation(self, text: str, *, turn_id: int) -> None:
        normalized = self._normalized_phrase(text)
        request = self._pending_pause
        if request is None:
            await self._resume_listening(continued=True, turn_id=turn_id)
            return
        if normalized in PAUSE_CONFIRMATIONS:
            self._pending_pause = None
            await self._enter_pause(request, turn_id=turn_id)
            return
        if normalized in PAUSE_CANCELLATIONS or self._is_resume_phrase(normalized):
            self._pending_pause = None
            await self.events.publish(
                "conversation.pause_cancelled",
                {"reason": "voice_confirmation_declined"},
            )
            await self._speak_control_prompt(
                "Nie wstrzymuję działania.",
                turn_id=turn_id,
            )
            return
        await self._speak_control_prompt(
            "Powiedz „potwierdź”, aby rozpocząć przerwę, albo „anuluj”.",
            turn_id=turn_id,
        )

    async def _enter_pause(self, request: PauseRequest, *, turn_id: int) -> None:
        await self.assistant.interrupt(end_conversation=False)
        self._accept_final = False
        self._interrupt_requested = False
        self.state = ConversationState.SPEAKING
        await self.events.publish(
            "conversation.pause_confirmed",
            {
                "value": request.value,
                "unit": request.unit,
                "seconds": request.seconds,
            },
        )
        await self._trace_mark(turn_id, "pause_confirmed")
        barge = await self._speak_with_barge_in(
            f"Wstrzymuję działanie na {request.display}.",
            turn_id=turn_id,
        )
        if self._blocked or turn_id != self._turn_id:
            return
        if barge is not None and self._is_resume_phrase(barge[0]):
            await self.events.publish(
                "conversation.pause_cancelled",
                {"reason": "resume_during_acknowledgement"},
            )
            await self._resume_listening(continued=True, turn_id=turn_id)
            return

        self._pause_until = time.monotonic() + request.seconds
        self.state = ConversationState.PAUSED
        self._turn_active = True
        self._accept_final = True
        self._pause_task = asyncio.create_task(
            self._pause_countdown(self._pause_until, request.seconds),
            name="conversation-pause-timer",
        )
        await self.events.publish(
            "conversation.paused",
            {
                "seconds": request.seconds,
                "resume_phrases": sorted(RESUME_PHRASES),
            },
        )
        await self._trace_mark(
            turn_id,
            "pause_started",
            metadata={"pause_seconds": request.seconds},
        )
        await self._wait_cooldown()
        await self._arm_paused_listening()

    async def _pause_countdown(self, pause_until: float, seconds: int) -> None:
        try:
            await asyncio.sleep(seconds)
            if (
                self.state is ConversationState.PAUSED
                and self._pause_until == pause_until
            ):
                await self._leave_pause(source="timer")
        except asyncio.CancelledError:
            raise

    async def _arm_paused_listening(self) -> None:
        if self.state is not ConversationState.PAUSED or self._pause_until is None:
            return
        remaining = max(1.0, self._pause_until - time.monotonic())
        self._accept_final = True
        self._turn_active = True
        if self.stream_reuse_enabled:
            await self._ensure_conversation_stream()
        else:
            await self.deepgram.start_once(timeout_seconds=min(60.0, remaining + 1.0))
            await self._wait_for_deepgram_ready()
        await self._trace_mark(self._turn_id, "pause_listener_ready")
        await self.events.publish(
            "conversation.paused_listening",
            {"purpose": "resume_only", "remaining_seconds": round(remaining)},
        )

    async def _handle_paused_transcript(self, text: str) -> None:
        if self._is_resume_phrase(text):
            await self._leave_pause(source="voice")
            return
        await self.events.publish(
            "conversation.paused_transcript_ignored",
            {"text": text[:120], "reason": "resume_phrase_required"},
        )
        if not self.stream_reuse_enabled:
            await self.deepgram.stop()
            await self._arm_paused_listening()

    async def _leave_pause(self, *, source: str) -> dict[str, str]:
        async with self._pause_lock:
            if self.state is not ConversationState.PAUSED:
                return {"status": "not_paused", "state": self.state.value}
            await self._cancel_pause_task()
            self._pause_until = None
            self._accept_final = False
            self._turn_id += 1
            turn_id = self._turn_id
            if not self.stream_reuse_enabled:
                await self.deepgram.stop()
            self.state = ConversationState.SPEAKING
            await self.events.publish(
                "conversation.resumed",
                {"source": source},
            )
            prompt = (
                "Przerwa minęła. Wznawiam działanie."
                if source == "timer"
                else "Wznawiam działanie."
            )
            try:
                barge = await self._speak_with_barge_in(prompt, turn_id=turn_id)
            except Exception:
                LOGGER.exception("Could not speak pause-resume confirmation")
                barge = None
            self._refresh_address_window = True
            if barge is not None:
                text, confidence, transcript = barge
                self._turn_id += 1
                await self._run_user_turn(
                    text,
                    confidence=confidence,
                    transcript=transcript,
                    turn_id=self._turn_id,
                )
                return {"status": self.state.value, "state": self.state.value}
            if self._blocked:
                self._turn_active = False
                self.state = ConversationState.IDLE
                return {"status": "idle", "state": self.state.value}
            await self._wait_cooldown()
            return await self._resume_listening(continued=True, turn_id=turn_id)

    async def _cancel_pause_task(self) -> None:
        task = self._pause_task
        self._pause_task = None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _cancel_barge_arm(self) -> None:
        task = self._barge_arm_task
        self._barge_arm_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _interrupted(self, turn_id: int) -> bool:
        return self._interrupt_requested or turn_id != self._turn_id or self._blocked

    @staticmethod
    def _parse_pause_request(text: str) -> PauseRequest | None:
        match = PAUSE_PATTERN.search(normalize_text(text))
        if match is None:
            return None
        value = VoiceConversationCoordinator._parse_polish_number(
            match.group("value")
        )
        if value is None:
            return None
        raw_unit = match.group("unit")
        if raw_unit.startswith(("s", "sek")):
            unit = "second"
            multiplier = 1
        elif raw_unit.startswith(("m", "min")):
            unit = "minute"
            multiplier = 60
        else:
            unit = "hour"
            multiplier = 60 * 60
        return PauseRequest(value=value, unit=unit, seconds=value * multiplier)

    @staticmethod
    def _parse_polish_number(value: str) -> int | None:
        cleaned = " ".join((value or "").split())
        if cleaned.isdigit():
            return int(cleaned)
        parts = cleaned.split()
        if len(parts) == 1:
            return POLISH_NUMBERS.get(parts[0], POLISH_TENS.get(parts[0]))
        if (
            len(parts) == 2
            and parts[0] in POLISH_TENS
            and 1 <= POLISH_NUMBERS.get(parts[1], 0) <= 9
        ):
            return POLISH_TENS[parts[0]] + POLISH_NUMBERS[parts[1]]
        return None

    def _is_likely_tts_echo(self, text: str) -> bool:
        return self._echo_decision(text).is_echo

    def _echo_decision(self, text: str) -> EchoDecision:
        heard = self._normalized_phrase(text)
        spoken_source = self._current_spoken_text
        if not spoken_source and time.monotonic() <= self._recent_spoken_until:
            spoken_source = self._recent_spoken_text
        spoken = self._normalized_phrase(spoken_source)
        if not heard or not spoken:
            return EchoDecision(score=0.0, is_echo=False)
        heard_tokens = heard.split()
        spoken_tokens = spoken.split()
        significant_tokens = [
            token
            for token in heard_tokens
            if len(token) >= 4 and token not in ECHO_LOW_INFORMATION_TOKENS
        ]
        new_tokens = tuple(
            token for token in significant_tokens if token not in spoken_tokens
        )
        longest = SequenceMatcher(
            None,
            heard_tokens,
            spoken_tokens,
            autojunk=False,
        ).find_longest_match()
        token_score = longest.size / max(1, len(heard_tokens))
        significant_score = (
            sum(token in spoken_tokens for token in significant_tokens)
            / len(significant_tokens)
            if significant_tokens
            else 0.0
        )
        character_match = SequenceMatcher(
            None,
            heard,
            spoken,
            autojunk=False,
        ).find_longest_match()
        character_score = character_match.size / max(1, len(heard))
        substring_score = 1.0 if heard in spoken and len(heard) >= 6 else 0.0
        score = max(substring_score, token_score, significant_score, character_score)
        threshold = {
            "headphones": 0.82,
            "speakers": 0.62,
            "auto": 0.66,
        }[self.barge_in_profile]
        is_echo = score >= threshold and len(new_tokens) < 2
        return EchoDecision(
            score=max(0.0, min(score, 1.0)),
            is_echo=is_echo,
            new_tokens=new_tokens,
        )

    def _effective_barge_in_delay_seconds(self, echo: EchoDecision) -> float:
        if not self.hybrid_barge_in_enabled or len(echo.new_tokens) < 2:
            return self.barge_in_after_seconds
        if echo.score >= 0.5:
            return self.barge_in_after_seconds
        clean_delay = {
            "headphones": 0.1,
            "speakers": 0.28,
            "auto": 0.18,
        }[self.barge_in_profile]
        if len(echo.new_tokens) >= 4:
            clean_delay *= 0.75
        return min(self.barge_in_after_seconds, clean_delay)

    @staticmethod
    def _is_direct_address(text: str) -> bool:
        normalized = VoiceConversationCoordinator._normalized_phrase(text)
        return any(
            normalized == prefix or normalized.startswith(f"{prefix} ")
            for prefix in DIRECT_ADDRESS_PREFIXES
        )

    @staticmethod
    def _is_resume_phrase(text: str) -> bool:
        normalized = VoiceConversationCoordinator._normalized_phrase(text)
        for prefix in DIRECT_ADDRESS_PREFIXES:
            if normalized.startswith(f"{prefix} "):
                normalized = normalized[len(prefix) :].strip()
                break
        return normalized in RESUME_PHRASES

    @staticmethod
    def _normalized_phrase(text: str) -> str:
        normalized = " ".join(normalize_text(text).split())
        return normalized.strip(" .,!?:;\"'…„”")

    async def _wait_cooldown(self) -> None:
        started = time.perf_counter()
        if self.cooldown_seconds and not self.stream_reuse_enabled:
            await asyncio.sleep(self.cooldown_seconds)
        self._last_cooldown_ms = round((time.perf_counter() - started) * 1000)

    async def _wait_for_deepgram_ready(self) -> None:
        wait_until_connected = getattr(self.deepgram, "wait_until_connected", None)
        if wait_until_connected is not None:
            await wait_until_connected(timeout_seconds=5.0)

    async def _ensure_conversation_stream(self) -> None:
        starter = getattr(self.deepgram, "start_conversation", None)
        if starter is not None:
            await starter()
        else:
            await self.deepgram.start()
        await self._wait_for_deepgram_ready()

    async def _trace_mark(
        self,
        turn_id: int,
        phase: str,
        *,
        request_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.telemetry is None:
            return
        session_id = self._trace_sessions.get(turn_id)
        if session_id is None:
            return
        await self.telemetry.mark(
            session_id=session_id,
            turn_id=turn_id,
            phase=phase,
            request_id=request_id,
            metadata=metadata,
        )

    async def _trace_finish(
        self,
        turn_id: int,
        status: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.telemetry is None:
            return
        session_id = self._trace_sessions.pop(turn_id, None)
        if session_id is None:
            return
        await self.telemetry.finish(
            session_id=session_id,
            turn_id=turn_id,
            status=status,
            metadata=metadata,
        )

    @staticmethod
    def _is_interrupt_phrase(text: str) -> bool:
        normalized = VoiceConversationCoordinator._normalized_phrase(text)
        return normalized in INTERRUPT_PHRASES

    async def _resume_listening(self, *, continued: bool, turn_id: int) -> dict[str, str]:
        if self._blocked or turn_id != self._turn_id:
            self._turn_active = False
            self._accept_final = False
            self.state = ConversationState.IDLE
            return {"status": "idle", "state": self.state.value}
        self.state = ConversationState.LISTENING_ONCE
        self._accept_final = True
        self._turn_active = True
        self._interrupt_requested = False
        self._barge_in_armed = False
        if self._refresh_address_window:
            self._address_free_until = (
                time.monotonic() + self.direct_address_after_seconds
            )
            self._refresh_address_window = False
        # Longer window for natural pauses; timeout handler re-arms if needed.
        reconnect_started = time.perf_counter()
        if self.stream_reuse_enabled:
            await self._ensure_conversation_stream()
        else:
            await self.deepgram.start_once(timeout_seconds=60.0)
            await self._wait_for_deepgram_ready()
        reconnect_ms = round((time.perf_counter() - reconnect_started) * 1000)
        connection_count = int(
            getattr(self.deepgram, "connection_count", self._last_connection_count)
        )
        connection_delta = max(0, connection_count - self._last_connection_count)
        reconnect_count = max(
            0,
            connection_delta - (1 if self._last_connection_count == 0 else 0),
        )
        self._last_connection_count = connection_count
        await self._trace_mark(
            turn_id,
            "listening_ready",
            metadata={
                "deepgram_reconnect_ms": reconnect_ms,
                "deepgram_reconnect_count": reconnect_count,
            },
        )
        timing = {
            "tts_ms": self._last_tts_ms,
            "cooldown_ms": self._last_cooldown_ms,
            "deepgram_reconnect_ms": reconnect_ms,
            "deepgram_reconnect_count": reconnect_count,
            "continued": continued,
        }
        LOGGER.info(
            "Conversation latency tts_ms=%d cooldown_ms=%d "
            "deepgram_reconnect_ms=%d continued=%s",
            timing["tts_ms"],
            timing["cooldown_ms"],
            timing["deepgram_reconnect_ms"],
            continued,
        )
        await self.events.publish("conversation.timing", timing)
        self._last_tts_ms = 0
        self._last_cooldown_ms = 0
        await self.events.publish(
            "conversation.listening",
            {"mode": "once", "continued": continued},
        )
        await self.events.publish(
            "conversation.turn_complete",
            {"status": "listening_once", "continue": True},
        )
        await self._trace_finish(turn_id, "listening_once")
        return {"status": "listening_once", "state": self.state.value}
