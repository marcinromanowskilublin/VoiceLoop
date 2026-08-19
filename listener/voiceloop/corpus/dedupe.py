from __future__ import annotations

import hashlib
from collections import defaultdict
from difflib import SequenceMatcher

from ..router import normalize_text
from .schema import CorpusSplit, UtteranceRecord


def deduplicate(
    records: list[UtteranceRecord],
    *,
    near_duplicate_threshold: float = 0.88,
) -> list[UtteranceRecord]:
    canonical_by_hash: dict[str, str] = {}
    shingle_index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    short_token_index: dict[str, set[int]] = defaultdict(set)
    canonical_shingles: list[set[tuple[str, ...]]] = []
    canonical_normalized: list[str] = []
    canonical_tokens: list[set[str]] = []
    canonical_ids: list[str] = []
    result: list[UtteranceRecord] = []

    for record in records:
        exact_id = canonical_by_hash.get(record.text_sha256)
        if exact_id:
            result.append(
                record.model_copy(
                    update={
                        "is_near_duplicate": True,
                        "duplicate_of": exact_id,
                        "split": CorpusSplit.UNUSED,
                    }
                )
            )
            continue

        normalized = normalize_text(record.text)
        tokens = set(normalized.split())
        short_candidate_indexes: set[int] = set()
        if len(normalized.split()) <= 5:
            for token in tokens:
                short_candidate_indexes.update(short_token_index.get(token, set()))
        duplicate_id = _best_short_duplicate(
            normalized,
            tokens,
            short_candidate_indexes,
            canonical_normalized,
            canonical_tokens,
            canonical_ids,
        )
        if duplicate_id:
            result.append(
                record.model_copy(
                    update={
                        "is_near_duplicate": True,
                        "duplicate_of": duplicate_id,
                        "split": CorpusSplit.UNUSED,
                    }
                )
            )
            continue

        shingles = _word_shingles(record.text)
        candidate_indexes: set[int] = set()
        for shingle in shingles:
            candidate_indexes.update(shingle_index.get(shingle, set()))
        duplicate_id = _best_duplicate(
            shingles,
            candidate_indexes,
            canonical_shingles,
            canonical_ids,
            threshold=near_duplicate_threshold,
        )
        if duplicate_id:
            result.append(
                record.model_copy(
                    update={
                        "is_near_duplicate": True,
                        "duplicate_of": duplicate_id,
                        "split": CorpusSplit.UNUSED,
                    }
                )
            )
            continue

        canonical_index = len(canonical_ids)
        canonical_ids.append(record.utterance_id)
        canonical_shingles.append(shingles)
        canonical_normalized.append(normalized)
        canonical_tokens.append(tokens)
        canonical_by_hash[record.text_sha256] = record.utterance_id
        for shingle in shingles:
            shingle_index[shingle].add(canonical_index)
        if len(normalized.split()) <= 5:
            for token in tokens:
                short_token_index[token].add(canonical_index)
        result.append(record)
    return result


def assign_splits(
    records: list[UtteranceRecord],
    *,
    holdout_percent: int = 20,
) -> list[UtteranceRecord]:
    safe_percent = max(1, min(holdout_percent, 49))
    result: list[UtteranceRecord] = []
    for record in records:
        if record.is_near_duplicate or record.quarantine_reason:
            split = CorpusSplit.UNUSED
        else:
            bucket = int(
                hashlib.sha256(record.session_id.encode("utf-8")).hexdigest()[:8],
                16,
            ) % 100
            split = (
                CorpusSplit.HOLDOUT
                if bucket < safe_percent
                else CorpusSplit.TRAIN
            )
        result.append(record.model_copy(update={"split": split}))
    return result


def _word_shingles(text: str, *, width: int = 5) -> set[tuple[str, ...]]:
    words = normalize_text(text).split()
    if len(words) < width:
        return set()
    return {
        tuple(words[index : index + width])
        for index in range(len(words) - width + 1)
    }


def _best_duplicate(
    shingles: set[tuple[str, ...]],
    candidate_indexes: set[int],
    canonical_shingles: list[set[tuple[str, ...]]],
    canonical_ids: list[str],
    *,
    threshold: float,
) -> str | None:
    if not shingles:
        return None
    best_score = 0.0
    best_id: str | None = None
    for index in candidate_indexes:
        other = canonical_shingles[index]
        union_size = len(shingles | other)
        if not union_size:
            continue
        score = len(shingles & other) / union_size
        if score >= threshold and score > best_score:
            best_score = score
            best_id = canonical_ids[index]
    return best_id


def _best_short_duplicate(
    normalized: str,
    tokens: set[str],
    candidate_indexes: set[int],
    canonical_normalized: list[str],
    canonical_tokens: list[set[str]],
    canonical_ids: list[str],
) -> str | None:
    if not normalized or not tokens:
        return None
    best_score = 0.0
    best_id: str | None = None
    for index in candidate_indexes:
        other_tokens = canonical_tokens[index]
        token_union = tokens | other_tokens
        token_similarity = len(tokens & other_tokens) / len(token_union)
        text_similarity = SequenceMatcher(
            None,
            normalized,
            canonical_normalized[index],
        ).ratio()
        if (
            token_similarity >= 0.6
            and text_similarity >= 0.82
            and text_similarity > best_score
        ):
            best_score = text_similarity
            best_id = canonical_ids[index]
    return best_id
