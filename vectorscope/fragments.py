"""Hierarchia fragmentów: słowo → fraza → zdanie → wypowiedź.

Każdy fragment jest osobnym punktem z własnym identyfikatorem, czasem
i wskaźnikiem na rodzica. Poziomy celowo nie są named vectors — named vector to
osobna przestrzeń znaczeniowa, a poziom segmentacji to tylko ziarnistość tego
samego znaczenia. Mieszanie tych dwóch rzeczy dałoby pięć kopii jednego wektora.

Embedding widzi wyłącznie tekst, więc dwa fragmenty o identycznej treści mają
identyczny wektor niezależnie od tego, jak brzmiały.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

LEVEL_WORD = "word"
LEVEL_PHRASE = "phrase"
LEVEL_SENTENCE = "sentence"
LEVEL_UTTERANCE = "utterance"
LEVELS = (LEVEL_WORD, LEVEL_PHRASE, LEVEL_SENTENCE, LEVEL_UTTERANCE)

PHRASE_PAUSE_MS = 350
PHRASE_MAX_WORDS = 6
PHRASE_MIN_WORDS = 2
SENTENCE_ENDINGS = (".", "!", "?", "…")
PHRASE_BREAKS = (",", ";", ":", "—", "–")

SEGMENTATION_VERSION = "vectorscope-segmentation-v1"
SEGMENTATION_RULE = (
    "utterance = Deepgram utterances[] (albo całość, gdy brak); "
    "sentence = podział na [.!?…]; "
    f"phrase = pauza > {PHRASE_PAUSE_MS} ms albo [,;:—–] albo {PHRASE_MAX_WORDS} słów, "
    f"min {PHRASE_MIN_WORDS} słowa; "
    "word = token Deepgrama"
)

STRIP_EDGES = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Fragment:
    id: str
    level: str
    text: str
    start_ms: int | None
    end_ms: int | None
    parent_id: str | None
    recording_id: str
    recording_label: str
    speaker: int | None
    order: int
    word_count: int

    def to_payload(self, index: int, *, parent_text: str | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["index"] = index
        payload["parent_text"] = parent_text
        return payload


def normalise_key(text: str) -> str:
    return STRIP_EDGES.sub("", text).casefold().strip()


def _display(text: str) -> str:
    return STRIP_EDGES.sub("", text).strip()


def _ms(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(round(float(value) * 1000))
    return None


def normalise_transcript(transcript: dict[str, Any]) -> dict[str, Any]:
    """Sprowadza transkrypt do jednego kształtu niezależnie od jego pochodzenia.

    Na dysku mogą leżeć dwa formaty: znormalizowany (klucze `text`, `words`,
    `utterances` na wierzchu) oraz surowa odpowiedź Deepgrama (`results`).
    Panel ma czytać nagrania nagrane wcześniej, więc rozumie oba.
    """

    if transcript.get("words") or transcript.get("text"):
        return transcript

    results = transcript.get("results") or {}
    channels = results.get("channels") or []
    alternative: dict[str, Any] = {}
    if channels and isinstance(channels[0], dict):
        alternatives = channels[0].get("alternatives") or []
        if alternatives and isinstance(alternatives[0], dict):
            alternative = alternatives[0]

    words = [
        {
            "text": str(item.get("punctuated_word") or item.get("word") or "").strip(),
            "start": item.get("start"),
            "end": item.get("end"),
            "confidence": item.get("confidence"),
            "speaker": item.get("speaker"),
        }
        for item in (alternative.get("words") or [])
        if isinstance(item, dict)
    ]

    sentences: list[str] = []
    paragraphs = (alternative.get("paragraphs") or {}).get("paragraphs") or []
    for paragraph in paragraphs:
        for sentence in (paragraph or {}).get("sentences") or []:
            value = str((sentence or {}).get("text") or "").strip()
            if value:
                sentences.append(value)

    return {
        "text": str(alternative.get("transcript") or "").strip(),
        "words": [item for item in words if item["text"]],
        "sentences": sentences,
        "utterances": results.get("utterances") or [],
    }


def build_fragments(
    *,
    transcript: dict[str, Any],
    recording_id: str,
    recording_label: str,
) -> list[Fragment]:
    """Zwraca wszystkie poziomy naraz, z poprawnymi wskaźnikami na rodzica."""

    transcript = normalise_transcript(transcript)
    words = _collect_words(transcript)
    if not words:
        return _text_only_fragments(transcript, recording_id, recording_label)

    utterance_ranges = _utterance_ranges(transcript, words)
    fragments: list[Fragment] = []
    counters = {level: 0 for level in LEVELS}

    def next_id(level: str) -> str:
        counters[level] += 1
        prefix = {
            LEVEL_WORD: "w",
            LEVEL_PHRASE: "p",
            LEVEL_SENTENCE: "s",
            LEVEL_UTTERANCE: "u",
        }[level]
        return f"{recording_id}:{prefix}{counters[level]}"

    order = 0
    for utterance_words, utterance_speaker in utterance_ranges:
        if not utterance_words:
            continue
        utterance_id = next_id(LEVEL_UTTERANCE)
        utterance_text = _display(" ".join(item["text"] for item in utterance_words))
        if utterance_text and HAS_LETTER.search(utterance_text):
            fragments.append(
                Fragment(
                    id=utterance_id,
                    level=LEVEL_UTTERANCE,
                    text=utterance_text,
                    start_ms=utterance_words[0]["start_ms"],
                    end_ms=utterance_words[-1]["end_ms"],
                    parent_id=None,
                    recording_id=recording_id,
                    recording_label=recording_label,
                    speaker=utterance_speaker,
                    order=order,
                    word_count=len(utterance_words),
                )
            )
            order += 1

        for sentence_words in _split_sentences(utterance_words):
            sentence_id = next_id(LEVEL_SENTENCE)
            sentence_text = _display(" ".join(item["text"] for item in sentence_words))
            if sentence_text and HAS_LETTER.search(sentence_text):
                fragments.append(
                    Fragment(
                        id=sentence_id,
                        level=LEVEL_SENTENCE,
                        text=sentence_text,
                        start_ms=sentence_words[0]["start_ms"],
                        end_ms=sentence_words[-1]["end_ms"],
                        parent_id=utterance_id,
                        recording_id=recording_id,
                        recording_label=recording_label,
                        speaker=sentence_words[0]["speaker"],
                        order=order,
                        word_count=len(sentence_words),
                    )
                )
                order += 1

            for phrase_words in _split_phrases(sentence_words):
                phrase_parent = sentence_id
                phrase_id: str | None = None
                if len(phrase_words) >= PHRASE_MIN_WORDS:
                    phrase_text = _display(" ".join(item["text"] for item in phrase_words))
                    if phrase_text and HAS_LETTER.search(phrase_text):
                        phrase_id = next_id(LEVEL_PHRASE)
                        fragments.append(
                            Fragment(
                                id=phrase_id,
                                level=LEVEL_PHRASE,
                                text=phrase_text,
                                start_ms=phrase_words[0]["start_ms"],
                                end_ms=phrase_words[-1]["end_ms"],
                                parent_id=phrase_parent,
                                recording_id=recording_id,
                                recording_label=recording_label,
                                speaker=phrase_words[0]["speaker"],
                                order=order,
                                word_count=len(phrase_words),
                            )
                        )
                        order += 1

                for item in phrase_words:
                    word_text = _display(item["text"])
                    if not word_text or not HAS_LETTER.search(word_text):
                        continue
                    fragments.append(
                        Fragment(
                            id=next_id(LEVEL_WORD),
                            level=LEVEL_WORD,
                            text=word_text,
                            start_ms=item["start_ms"],
                            end_ms=item["end_ms"],
                            parent_id=phrase_id or phrase_parent,
                            recording_id=recording_id,
                            recording_label=recording_label,
                            speaker=item["speaker"],
                            order=order,
                            word_count=1,
                        )
                    )
                    order += 1

    return fragments


def _collect_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for item in transcript.get("words") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        speaker = item.get("speaker")
        collected.append(
            {
                "text": text,
                "start_ms": _ms(item.get("start")),
                "end_ms": _ms(item.get("end")),
                "speaker": int(speaker) if isinstance(speaker, (int, float)) else None,
            }
        )
    return collected


def _utterance_ranges(
    transcript: dict[str, Any],
    words: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], int | None]]:
    utterances = [
        item for item in (transcript.get("utterances") or []) if isinstance(item, dict)
    ]
    if not utterances:
        speakers = {item["speaker"] for item in words if item["speaker"] is not None}
        speaker = speakers.pop() if len(speakers) == 1 else None
        return [(words, speaker)]

    grouped: list[tuple[list[dict[str, Any]], int | None]] = []
    consumed = 0
    for utterance in utterances:
        start = _ms(utterance.get("start"))
        end = _ms(utterance.get("end"))
        speaker = utterance.get("speaker")
        speaker_id = int(speaker) if isinstance(speaker, (int, float)) else None
        if start is None or end is None:
            continue
        bucket: list[dict[str, Any]] = []
        for item in words[consumed:]:
            centre = _centre(item)
            if centre is None or centre < start:
                continue
            if centre > end:
                break
            bucket.append(item)
        consumed += len(bucket)
        if bucket:
            grouped.append((bucket, speaker_id))

    if not grouped:
        return [(words, None)]
    assigned = sum(len(bucket) for bucket, _ in grouped)
    if assigned < len(words):
        grouped.append((words[assigned:], None))
    return grouped


def _centre(item: dict[str, Any]) -> int | None:
    start, end = item["start_ms"], item["end_ms"]
    if start is None and end is None:
        return None
    if start is None:
        return end
    if end is None:
        return start
    return (start + end) // 2


def _split_sentences(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in words:
        current.append(item)
        if item["text"].rstrip().endswith(SENTENCE_ENDINGS):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _split_phrases(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_end: int | None = None
    for item in words:
        start = item["start_ms"]
        long_pause = (
            start is not None
            and previous_end is not None
            and (start - previous_end) > PHRASE_PAUSE_MS
        )
        if current and (long_pause or len(current) >= PHRASE_MAX_WORDS):
            groups.append(current)
            current = []
        current.append(item)
        previous_end = item["end_ms"] if item["end_ms"] is not None else previous_end
        if item["text"].rstrip().endswith(PHRASE_BREAKS + SENTENCE_ENDINGS):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _text_only_fragments(
    transcript: dict[str, Any],
    recording_id: str,
    recording_label: str,
) -> list[Fragment]:
    """Awaryjna ścieżka, gdy Deepgram nie zwrócił czasów słów."""

    sentences = [
        str(value).strip()
        for value in (transcript.get("sentences") or [])
        if str(value or "").strip()
    ]
    if not sentences:
        text = str(transcript.get("text") or "").strip()
        sentences = [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]
    if not sentences:
        return []

    fragments: list[Fragment] = []
    utterance_id = f"{recording_id}:u1"
    fragments.append(
        Fragment(
            id=utterance_id,
            level=LEVEL_UTTERANCE,
            text=" ".join(sentences),
            start_ms=None,
            end_ms=None,
            parent_id=None,
            recording_id=recording_id,
            recording_label=recording_label,
            speaker=None,
            order=0,
            word_count=sum(len(item.split()) for item in sentences),
        )
    )
    order = 1
    for sentence_index, sentence in enumerate(sentences, start=1):
        sentence_id = f"{recording_id}:s{sentence_index}"
        fragments.append(
            Fragment(
                id=sentence_id,
                level=LEVEL_SENTENCE,
                text=sentence,
                start_ms=None,
                end_ms=None,
                parent_id=utterance_id,
                recording_id=recording_id,
                recording_label=recording_label,
                speaker=None,
                order=order,
                word_count=len(sentence.split()),
            )
        )
        order += 1
        for word_index, token in enumerate(sentence.split(), start=1):
            word_text = _display(token)
            if not word_text or not HAS_LETTER.search(word_text):
                continue
            fragments.append(
                Fragment(
                    id=f"{recording_id}:s{sentence_index}w{word_index}",
                    level=LEVEL_WORD,
                    text=word_text,
                    start_ms=None,
                    end_ms=None,
                    parent_id=sentence_id,
                    recording_id=recording_id,
                    recording_label=recording_label,
                    speaker=None,
                    order=order,
                    word_count=1,
                )
            )
            order += 1
    return fragments


def select_levels(fragments: list[Fragment], levels: list[str]) -> list[Fragment]:
    wanted = {level for level in levels if level in LEVELS}
    return [fragment for fragment in fragments if fragment.level in wanted]


def merge_identical(fragments: list[Fragment]) -> tuple[list[Fragment], list[list[str]]]:
    """Scala fragmenty o identycznym tekście w obrębie nagrania i poziomu.

    Ich wektory są i tak identyczne, bo model widzi wyłącznie tekst — scalenie
    nic nie ukrywa, tylko zdejmuje z grafu zduplikowane węzły o cosinusie 1.00.
    """

    grouped: dict[tuple[str, str, str], Fragment] = {}
    members: dict[tuple[str, str, str], list[str]] = {}
    for fragment in fragments:
        key = (fragment.recording_id, fragment.level, normalise_key(fragment.text))
        if key not in grouped:
            grouped[key] = fragment
            members[key] = [fragment.id]
        else:
            members[key].append(fragment.id)
    ordered_keys = list(grouped.keys())
    return (
        [grouped[key] for key in ordered_keys],
        [members[key] for key in ordered_keys],
    )
