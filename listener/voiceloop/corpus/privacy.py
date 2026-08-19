from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import (
    QuarantineRecord,
    SpeakerStatus,
    UtteranceRecord,
)

_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,63}(?![\w-])",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|"
    r"(?:api[_-]?key|token|secret)\s*[=:]\s*[^\s,;]{8,})",
    re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?48[\s.-]?)?(?:\d[\s.-]?){9}(?!\d)")
_PESEL_PATTERN = re.compile(r"(?<!\d)(\d{11})(?!\d)")
_URL_PATTERN = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
_MEDICAL_THIRD_PARTY_PATTERN = re.compile(
    r"\b(?:pacjent\w*|pesel\w*|diagnoz\w*|recept\w*|dawk\w*|"
    r"wizyta\s+pacjent\w*|histori[ai]\s+chorob\w*)\b",
    re.IGNORECASE,
)
_MEDICAL_SELF_PATTERN = re.compile(
    r"\b(?:mam|choruję\s+na|zdiagnozowano\s+u\s+mnie|moja\s+diagnoza)\s+"
    r"(?:cukrzyc\w*|depresj\w*|nadciśnieni\w*|nowotw\w*|astm\w*|"
    r"chorob\w*|adhd\w*|autyzm\w*)\b",
    re.IGNORECASE,
)
_MEDICAL_GENERAL_PATTERN = re.compile(
    r"\b(?:jestem\s+w\s+ciąż\w*|leczę\s+się|"
    r"(?:przyjmuję|zażywam|stosuję|biorę)\s+"
    r"(?!(?:udział|pod\s+uwagę)\b)\w+|moje\s+lek\w*|"
    r"wynik\w*\s+badań|mam\s+recept\w*)\b",
    re.IGNORECASE,
)
_PSYCHOLOGICAL_INFERENCE_PATTERN = re.compile(
    r"\b(?:ma\s+(?:depresj|adhd|autyzm)|jest\s+(?:narcyz|psychopat)|"
    r"zaburzeni\w+\s+osobowości)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PrivacyGateResult:
    clean: UtteranceRecord | None
    quarantine: QuarantineRecord | None


def apply_privacy_gate(record: UtteranceRecord) -> PrivacyGateResult:
    redacted_text, flags = redact_text(record.text)
    reason = _quarantine_reason(record, flags, redacted_text)
    if reason:
        return PrivacyGateResult(
            clean=None,
            quarantine=QuarantineRecord(
                utterance_id=record.utterance_id,
                source_id=record.source_id,
                origin=record.origin,
                session_id=record.session_id,
                captured_at=record.captured_at,
                word_count=record.word_count,
                char_count=record.char_count,
                text_sha256=record.text_sha256,
                speaker_status=record.speaker_status,
                pii_flags=tuple(flags),
                reason=reason,
            ),
        )
    clean = record.model_copy(
        update={
            "text": redacted_text,
            "word_count": len(redacted_text.split()),
            "char_count": len(redacted_text),
            "pii_flags": tuple(flags),
            "redacted": bool(flags),
            "quarantine_reason": None,
        }
    )
    return PrivacyGateResult(clean=clean, quarantine=None)


def redact_text(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    value = text
    value = _replace(value, _SECRET_PATTERN, "[SECRET]", "secret", flags)
    value = _replace_valid_pesel(value, flags)
    value = _replace(value, _EMAIL_PATTERN, "[EMAIL]", "email", flags)
    value = _replace(value, _PHONE_PATTERN, "[PHONE]", "phone", flags)
    value = _replace(value, _URL_PATTERN, "[URL]", "url", flags)
    if _MEDICAL_THIRD_PARTY_PATTERN.search(value):
        flags.append("medical_or_third_party")
    if _MEDICAL_SELF_PATTERN.search(value) or _MEDICAL_GENERAL_PATTERN.search(value):
        flags.append("medical_sensitive")
    if _PSYCHOLOGICAL_INFERENCE_PATTERN.search(value):
        flags.append("psychological_inference")
    return " ".join(value.split()), _unique(flags)


def sensitive_reason(text: str) -> str | None:
    _, flags = redact_text(text)
    if "medical_or_third_party" in flags:
        return "medical_or_third_party"
    if "medical_sensitive" in flags:
        return "medical_sensitive"
    if "psychological_inference" in flags:
        return "psychological_inference"
    if "secret" in flags or "pesel" in flags:
        return "high_risk_identifier"
    return None


def _quarantine_reason(
    record: UtteranceRecord,
    flags: list[str],
    redacted_text: str,
) -> str | None:
    if record.speaker_status is not SpeakerStatus.SELF:
        return f"speaker_{record.speaker_status.value}"
    if not redacted_text:
        return "empty_after_redaction"
    if "medical_or_third_party" in flags:
        return "medical_or_third_party"
    if "medical_sensitive" in flags:
        return "medical_sensitive"
    if "psychological_inference" in flags:
        return "psychological_inference"
    if "secret" in flags or "pesel" in flags:
        return "high_risk_identifier"
    return None


def _replace(
    text: str,
    pattern: re.Pattern[str],
    replacement: str,
    flag: str,
    flags: list[str],
) -> str:
    if pattern.search(text):
        flags.append(flag)
        return pattern.sub(replacement, text)
    return text


def _replace_valid_pesel(text: str, flags: list[str]) -> str:
    found = False

    def replacement(match: re.Match[str]) -> str:
        nonlocal found
        value = match.group(1)
        if not _valid_pesel(value):
            return value
        found = True
        return "[PESEL]"

    result = _PESEL_PATTERN.sub(replacement, text)
    if found:
        flags.append("pesel")
    return result


def _valid_pesel(value: str) -> bool:
    if len(value) != 11 or not value.isdigit():
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    checksum = (
        10
        - sum(
            int(char) * weight
            for char, weight in zip(value, weights, strict=False)
        )
    ) % 10
    return checksum == int(value[-1])


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
