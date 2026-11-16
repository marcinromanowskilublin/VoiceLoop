"""Typed, non-executable context contracts and deterministic assembly."""

from .adaptive_retrieval import (
    AdaptiveHybridRetriever,
    AdaptiveRetrievalRun,
    AdaptiveRetrievalTraceV1,
)
from .assembler import ContextAssembler, ContextBudgets
from .entities import (
    ContextEntityKind,
    ContextEntityResolver,
    ResolvedEntity,
    stable_entity_id,
)
from .episodes import activity_bucket, group_events_into_episodes
from .policy import RetrievalPolicyV1, resolve_retrieval_policy
from .retrieval import (
    TimeFirstQueryPlan,
    TimeFirstRetriever,
    TimePrecision,
    TimeSource,
    build_context_search_text,
    build_time_first_query,
    normalize_context_text,
)
from .schema import (
    ContextEpisodeV1,
    ContextEventV1,
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
    ForegroundConfidence,
    content_hash_for,
    stable_context_id,
    stable_episode_id,
)
from .shadow_statements import (
    AtomicClaimProposal,
    TemporalRoleV1,
    TemporalStatementShadowV1,
    analyze_temporal_statement_shadow,
    build_temporal_shadow_context,
    comparative_shadow_question,
    shadow_relevance_gate,
)

__all__ = [
    "AdaptiveHybridRetriever",
    "AdaptiveRetrievalRun",
    "AdaptiveRetrievalTraceV1",
    "AtomicClaimProposal",
    "ContextAssembler",
    "ContextBudgets",
    "ContextEntityKind",
    "ContextEntityResolver",
    "ContextEventV1",
    "ContextEpisodeV1",
    "ContextItemV1",
    "ContextPackV1",
    "ContextScope",
    "ContextTrust",
    "ForegroundConfidence",
    "ResolvedEntity",
    "RetrievalPolicyV1",
    "TemporalRoleV1",
    "TemporalStatementShadowV1",
    "TimeFirstQueryPlan",
    "TimeFirstRetriever",
    "TimePrecision",
    "TimeSource",
    "activity_bucket",
    "analyze_temporal_statement_shadow",
    "build_context_search_text",
    "build_temporal_shadow_context",
    "build_time_first_query",
    "comparative_shadow_question",
    "content_hash_for",
    "normalize_context_text",
    "resolve_retrieval_policy",
    "shadow_relevance_gate",
    "stable_context_id",
    "stable_episode_id",
    "stable_entity_id",
    "group_events_into_episodes",
]
