from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..memory_vectorization import (
    MEMORY_VECTOR_NAMES,
    MEMORY_VECTOR_WEIGHTS,
    memory_query_weights,
)

RetrievalDomain = Literal["general", "windows", "rpg"]
RetrievalQueryClass = Literal[
    "factual",
    "operational",
    "creative",
    "exploratory",
]
RetrievalRisk = Literal["low", "medium", "high"]
RetrievalStrategy = Literal["exact_only", "fts_then_cascade", "hybrid_cascade"]

POLICY_VERSION = "dynamic-memory-coupling-v1"

_CREATIVE = re.compile(
    r"\b(?:wymysl|stworz|zaproponuj|motyw|fabul|narrac|inspirac|wariant|"
    r"alternatyw|tworcz|kreatywn)\w*",
)
_OPERATIONAL = re.compile(
    r"\b(?:otworz|kliknij|uruchom|zamknij|przenies|usun|zmien|wybierz|"
    r"wklej|zapisz|folder|plik|okn|aplikac)\w*",
)
_FACTUAL = re.compile(
    r"\b(?:kto|kiedy|gdzie|co ustal|decyzj|postanow|fakt|dokladn|"
    r"jaki byl|jaka byla|przypomnij)\w*",
)
_EXACT_IDENTIFIER = re.compile(
    r"(?:[a-z]:\\|[/\\][\w.-]+|[\w.-]+\.(?:py|js|ts|tsx|json|md|txt|exe)\b|"
    r"\b[a-f0-9]{7,40}\b|\b\d{2,}\b)",
)


class RetrievalPolicyV1(BaseModel):
    """Versioned, deterministic policy for one retrieval subquery.

    The policy can vary ranking and cost. It can never disable canonical SQL
    verification or authorize an action.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_version: Literal["dynamic-memory-coupling-v1"] = POLICY_VERSION
    policy_id: str = Field(min_length=16, max_length=16)
    domain: RetrievalDomain
    query_class: RetrievalQueryClass
    risk: RetrievalRisk
    strategy: RetrievalStrategy
    fts_weight: float = Field(ge=0.0, le=1.0)
    vector_weight: float = Field(ge=0.0, le=1.0)
    axis_weights: dict[str, float]
    preferred_axes: tuple[str, ...]
    qmax: int = Field(ge=1, le=len(MEMORY_VECTOR_NAMES))
    wave_size: int = Field(ge=1, le=2)
    max_waves: int = Field(ge=1, le=3)
    min_evidence: int = Field(ge=1, le=10)
    min_axis_coverage: int = Field(ge=1, le=len(MEMORY_VECTOR_NAMES))
    candidate_multiplier: int = Field(ge=1, le=5)
    always_vector_scout: bool = False
    require_time_bound: bool = True
    require_canonical_sql: Literal[True] = True
    serving_eligible: Literal[False] = False
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_policy(self) -> RetrievalPolicyV1:
        if abs((self.fts_weight + self.vector_weight) - 1.0) > 1e-9:
            raise ValueError("fts_weight and vector_weight must sum to 1")
        if set(self.axis_weights) != set(MEMORY_VECTOR_NAMES):
            raise ValueError("axis_weights must contain every named memory axis")
        if abs(sum(self.axis_weights.values()) - 1.0) > 1e-9:
            raise ValueError("axis_weights must sum to 1")
        if any(axis not in MEMORY_VECTOR_NAMES for axis in self.preferred_axes):
            raise ValueError("preferred_axes contains an unknown memory axis")
        if self.min_axis_coverage > self.qmax:
            raise ValueError("min_axis_coverage cannot exceed qmax")
        return self

    @property
    def selected_axes(self) -> tuple[str, ...]:
        ordered = [
            *self.preferred_axes,
            *sorted(
                MEMORY_VECTOR_NAMES,
                key=lambda axis: (
                    self.axis_weights[axis],
                    -MEMORY_VECTOR_NAMES.index(axis),
                ),
                reverse=True,
            ),
        ]
        return tuple(dict.fromkeys(ordered))[: self.qmax]

    def vector_expansion_needed(
        self,
        *,
        exact_evidence: int,
        accepted_evidence: int = 0,
        covered_axes: int = 0,
    ) -> bool:
        if self.strategy == "exact_only":
            return False
        if self.always_vector_scout and accepted_evidence < self.min_evidence:
            return True
        if accepted_evidence == 0 and exact_evidence >= self.min_evidence:
            return False
        return (
            exact_evidence + accepted_evidence < self.min_evidence
            or covered_axes < self.min_axis_coverage
        )


def resolve_retrieval_policy(
    query: str,
    *,
    domain_hint: RetrievalDomain = "general",
    risk: RetrievalRisk = "low",
    time_bounded: bool = False,
) -> RetrievalPolicyV1:
    """Resolve a policy from trusted scope plus deterministic query signals."""

    normalized = _normalize(query)
    query_class = _query_class(normalized)
    exact_identifier = bool(_EXACT_IDENTIFIER.search(normalized))
    reasons = [
        f"domain:{domain_hint}",
        f"class:{query_class}",
        f"risk:{risk}",
        "time:bounded" if time_bounded else "time:unbounded",
    ]
    if exact_identifier:
        reasons.append("signal:exact_identifier")

    fts_weight, vector_weight = 0.65, 0.35
    qmax, wave_size, max_waves = 3, 2, 2
    min_evidence, min_axis_coverage = 2, 1
    candidate_multiplier = 3
    always_vector_scout = False
    require_time_bound = True
    strategy: RetrievalStrategy = "fts_then_cascade"

    if domain_hint == "windows":
        fts_weight, vector_weight = 0.82, 0.18
        qmax, wave_size, max_waves = 2, 2, 1
        min_evidence, min_axis_coverage = 1, 1
        candidate_multiplier = 2
        if query_class == "exploratory":
            fts_weight, vector_weight = 0.55, 0.45
            qmax, max_waves, min_evidence = 3, 2, 2
        if query_class == "operational" and exact_identifier:
            strategy = "exact_only"
            reasons.append("guard:windows_exact_first")
    elif domain_hint == "rpg":
        qmax, wave_size, max_waves = 4, 2, 2
        min_evidence, min_axis_coverage = 2, 1
        candidate_multiplier = 4
        if query_class == "creative":
            fts_weight, vector_weight = 0.25, 0.75
            qmax, max_waves = 5, 3
            min_evidence, min_axis_coverage = 3, 2
            always_vector_scout = True
            strategy = "hybrid_cascade"
            reasons.append("mode:associative_not_canonical")
        elif query_class == "factual":
            fts_weight, vector_weight = 0.72, 0.28
            reasons.append("guard:rpg_canon_first")
    elif query_class in {"creative", "exploratory"}:
        fts_weight, vector_weight = 0.45, 0.55
        qmax, max_waves = 4, 2
        always_vector_scout = query_class == "creative"
        strategy = "hybrid_cascade"

    if risk == "high":
        fts_weight = min(0.9, fts_weight + 0.15)
        vector_weight = 1.0 - fts_weight
        min_evidence = max(min_evidence, 2)
        min_axis_coverage = min(qmax, max(min_axis_coverage, 2))
        reasons.append("guard:high_risk_abstention")

    if exact_identifier and query_class != "creative":
        fts_weight = min(0.9, fts_weight + 0.1)
        vector_weight = 1.0 - fts_weight

    base_axis_weights = memory_query_weights(
        query,
        adaptive=True,
        base_weights=_domain_axis_weights(domain_hint, query_class),
    )
    preferred_axes = _preferred_axes(
        query,
        domain=domain_hint,
        query_class=query_class,
        weights=base_axis_weights,
    )
    if not time_bounded:
        reasons.append("guard:shadow_until_scoped")

    payload = {
        "version": POLICY_VERSION,
        "domain": domain_hint,
        "query_class": query_class,
        "risk": risk,
        "strategy": strategy,
        "fts_weight": round(fts_weight, 6),
        "vector_weight": round(vector_weight, 6),
        "axis_weights": base_axis_weights,
        "preferred_axes": preferred_axes,
        "qmax": qmax,
        "wave_size": wave_size,
        "max_waves": max_waves,
        "min_evidence": min_evidence,
        "min_axis_coverage": min_axis_coverage,
        "candidate_multiplier": candidate_multiplier,
        "always_vector_scout": always_vector_scout,
        "require_time_bound": require_time_bound,
        "reasons": reasons,
    }
    policy_id = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return RetrievalPolicyV1(
        policy_id=policy_id,
        domain=domain_hint,
        query_class=query_class,
        risk=risk,
        strategy=strategy,
        fts_weight=fts_weight,
        vector_weight=vector_weight,
        axis_weights=base_axis_weights,
        preferred_axes=preferred_axes,
        qmax=qmax,
        wave_size=wave_size,
        max_waves=max_waves,
        min_evidence=min_evidence,
        min_axis_coverage=min_axis_coverage,
        candidate_multiplier=candidate_multiplier,
        always_vector_scout=always_vector_scout,
        require_time_bound=require_time_bound,
        reason_codes=tuple(reasons),
    )


def _domain_axis_weights(
    domain: RetrievalDomain,
    query_class: RetrievalQueryClass,
) -> dict[str, float]:
    if domain == "windows":
        return {
            "semantic": 0.50,
            "topic": 0.20,
            "intent": 0.20,
            "decision": 0.07,
            "person_context": 0.03,
        }
    if domain == "rpg" and query_class == "creative":
        return {
            "semantic": 0.35,
            "topic": 0.25,
            "intent": 0.10,
            "decision": 0.08,
            "person_context": 0.22,
        }
    return dict(MEMORY_VECTOR_WEIGHTS)


def _preferred_axes(
    query: str,
    *,
    domain: RetrievalDomain,
    query_class: RetrievalQueryClass,
    weights: dict[str, float],
) -> tuple[str, ...]:
    normalized = _normalize(query)
    explicit: list[str] = []
    markers = (
        ("decision", ("ustal", "decyz", "postanow", "wybor")),
        ("intent", ("dlaczego", "po co", "cel", "zamiar")),
        ("person_context", ("kto", "komu", "kogo", "z kim", "postac")),
        ("topic", ("temat", "o czym", "projekt", "motyw")),
    )
    for axis, values in markers:
        if any(value in normalized for value in values):
            explicit.append(axis)
    domain_defaults: tuple[str, ...]
    if domain == "windows":
        domain_defaults = ("semantic", "intent", "topic")
    elif domain == "rpg" and query_class == "creative":
        domain_defaults = ("semantic", "topic", "person_context", "intent")
    elif query_class == "factual":
        domain_defaults = ("semantic", "decision", "person_context", "topic")
    else:
        domain_defaults = ("semantic", "topic", "intent")
    weighted = tuple(
        sorted(
            MEMORY_VECTOR_NAMES,
            key=lambda axis: weights[axis],
            reverse=True,
        )
    )
    return tuple(dict.fromkeys((*explicit, *domain_defaults, *weighted)))


def _query_class(normalized: str) -> RetrievalQueryClass:
    if _CREATIVE.search(normalized):
        return "creative"
    if _OPERATIONAL.search(normalized):
        return "operational"
    if _FACTUAL.search(normalized):
        return "factual"
    return "exploratory"


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("ł", "l"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))
