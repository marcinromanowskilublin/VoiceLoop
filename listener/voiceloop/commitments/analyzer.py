from __future__ import annotations

from collections.abc import Iterable

from .detector import detect_commitment_items
from .schema import CommitmentAnalysisResult, TranscriptChunk
from .scoring import score_commitment


def analyze_commitments(
    chunks: Iterable[TranscriptChunk],
    *,
    user_speakers: set[str] | None = None,
) -> CommitmentAnalysisResult:
    detected = detect_commitment_items(chunks, user_speakers=user_speakers)
    scored = tuple(score_commitment(item) for item in detected)
    return CommitmentAnalysisResult(items=scored)
