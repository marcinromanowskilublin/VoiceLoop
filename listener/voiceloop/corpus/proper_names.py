from __future__ import annotations

import re
from difflib import SequenceMatcher

from .schema import (
    ProperNameEntryV1,
    ProperNameLexiconV1,
    VoiceEvalPredictionV1,
    VoiceGoldAnnotationV1,
)


def build_proper_name_lexicon(
    annotations: list[VoiceGoldAnnotationV1],
    predictions: list[VoiceEvalPredictionV1],
    *,
    existing: ProperNameLexiconV1 | None = None,
) -> ProperNameLexiconV1:
    prediction_by_id = {item.sample_id: item for item in predictions}
    existing_by_name = {
        entry.canonical.casefold(): entry for entry in (existing.entries if existing else ())
    }
    evidence: dict[str, set[str]] = {}
    errors: dict[str, set[str]] = {}
    display_names: dict[str, str] = {}
    for annotation in annotations:
        prediction = prediction_by_id.get(annotation.sample_id)
        transcript = prediction.transcript_text if prediction is not None else ""
        for proper_name in annotation.proper_names:
            canonical = proper_name.strip()
            if not canonical:
                continue
            key = canonical.casefold()
            display_names.setdefault(key, canonical)
            evidence.setdefault(key, set()).add(annotation.sample_id)
            if canonical.casefold() not in transcript.casefold():
                candidate = _closest_phrase(canonical, transcript)
                if candidate and candidate.casefold() != key:
                    errors.setdefault(key, set()).add(candidate)

    entries: list[ProperNameEntryV1] = []
    all_names = set(existing_by_name) | set(evidence)
    for key in sorted(all_names):
        current = existing_by_name.get(key)
        entries.append(
            ProperNameEntryV1(
                canonical=(current.canonical if current else display_names[key]),
                aliases=(current.aliases if current else ()),
                common_stt_errors=tuple(
                    sorted(
                        set(current.common_stt_errors if current else ())
                        | errors.get(key, set())
                    )
                ),
                pronunciation_hint=(current.pronunciation_hint if current else ""),
                category=(current.category if current else "project_or_tool"),
                evidence_sample_ids=tuple(
                    sorted(
                        set(current.evidence_sample_ids if current else ())
                        | evidence.get(key, set())
                    )
                ),
                approved=(current.approved if current else False),
            )
        )
    return ProperNameLexiconV1(entries=tuple(entries))


def apply_proper_name_lexicon(
    text: str,
    lexicon: ProperNameLexiconV1,
) -> tuple[str, tuple[dict[str, str], ...]]:
    corrected = text
    changes: list[dict[str, str]] = []
    replacements: list[tuple[str, str]] = []
    for entry in lexicon.entries:
        if not entry.approved:
            continue
        for variant in (*entry.aliases, *entry.common_stt_errors):
            value = variant.strip()
            if value and value.casefold() != entry.canonical.casefold():
                replacements.append((value, entry.canonical))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    for source, target in replacements:
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        updated, count = pattern.subn(target, corrected)
        if count:
            changes.append({"before": source, "after": target, "count": str(count)})
            corrected = updated
    return corrected, tuple(changes)


def _closest_phrase(canonical: str, transcript: str) -> str:
    target_words = canonical.split()
    words = transcript.split()
    if not target_words or not words:
        return ""
    widths = {
        max(1, len(target_words) - 1),
        len(target_words),
        len(target_words) + 1,
    }
    best_phrase = ""
    best_score = 0.0
    for width in widths:
        for index in range(0, max(0, len(words) - width + 1)):
            phrase = " ".join(words[index : index + width])
            score = SequenceMatcher(None, canonical.casefold(), phrase.casefold()).ratio()
            if score > best_score:
                best_score = score
                best_phrase = phrase
    return best_phrase if best_score >= 0.45 else ""
