"""Transkrypcja przez Deepgram — model i język brane z ustawień VoiceLoopa."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from voiceloop.settings import Settings

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float
    speaker: int | None


@dataclass(frozen=True)
class Transcription:
    text: str
    words: list[Word]
    sentences: list[str]
    utterances: list[dict[str, Any]]
    model: str
    language: str
    raw: dict[str, Any]


def deepgram_params(settings: Settings) -> dict[str, str]:
    params = {
        "model": settings.deepgram_model,
        "language": settings.deepgram_language,
        "smart_format": "true",
        "punctuate": "true",
        "paragraphs": "true",
        "utterances": "true",
    }
    if settings.deepgram_diarization_enabled:
        params["diarize"] = "true"
    return params


async def transcribe_audio(
    *,
    settings: Settings,
    payload: bytes,
    content_type: str,
) -> Transcription:
    if not settings.deepgram_api_key:
        raise TranscriptionError(
            "Brak DEEPGRAM_API_KEY w listener/.env — transkrypcja niemożliwa."
        )
    key = settings.deepgram_api_key.get_secret_value().strip()
    if not key:
        raise TranscriptionError("DEEPGRAM_API_KEY jest puste.")
    if not payload:
        raise TranscriptionError("Nagranie jest puste.")

    headers = {
        "Authorization": f"Token {key}",
        "Content-Type": content_type or "audio/webm",
    }
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                DEEPGRAM_URL,
                params=deepgram_params(settings),
                headers=headers,
                content=payload,
            )
            response.raise_for_status()
            raw = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise TranscriptionError(
            f"Deepgram odrzucił żądanie ({exc.response.status_code}): {detail}"
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise TranscriptionError(f"Deepgram niedostępny: {exc}") from exc

    return _parse(raw, settings)


def _parse(raw: dict[str, Any], settings: Settings) -> Transcription:
    results = raw.get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        raise TranscriptionError("Deepgram nie zwrócił kanałów wyniku.")
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        raise TranscriptionError("Deepgram nie zwrócił alternatyw transkrypcji.")
    alternative = alternatives[0]

    words: list[Word] = []
    for item in alternative.get("words") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("punctuated_word") or item.get("word") or "").strip()
        if not text:
            continue
        speaker = item.get("speaker")
        words.append(
            Word(
                text=text,
                start=float(item.get("start") or 0.0),
                end=float(item.get("end") or 0.0),
                confidence=float(item.get("confidence") or 0.0),
                speaker=int(speaker) if isinstance(speaker, (int, float)) else None,
            )
        )

    sentences: list[str] = []
    paragraphs = alternative.get("paragraphs") or {}
    for paragraph in paragraphs.get("paragraphs") or []:
        for sentence in paragraph.get("sentences") or []:
            value = str(sentence.get("text") or "").strip()
            if value:
                sentences.append(value)

    utterances = [
        item for item in (results.get("utterances") or []) if isinstance(item, dict)
    ]

    return Transcription(
        text=str(alternative.get("transcript") or "").strip(),
        words=words,
        sentences=sentences,
        utterances=utterances,
        model=settings.deepgram_model,
        language=settings.deepgram_language,
        raw=raw,
    )


def transcription_payload(transcription: Transcription) -> dict[str, Any]:
    return {
        "text": transcription.text,
        "model": transcription.model,
        "language": transcription.language,
        "words": [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "confidence": word.confidence,
                "speaker": word.speaker,
            }
            for word in transcription.words
        ],
        "sentences": transcription.sentences,
        "utterances": transcription.utterances,
        "raw": transcription.raw,
    }
