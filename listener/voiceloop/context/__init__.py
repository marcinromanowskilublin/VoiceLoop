"""Typed, non-executable context contracts and deterministic assembly."""

from .assembler import ContextAssembler, ContextBudgets
from .entities import (
    ContextEntityKind,
    ContextEntityResolver,
    ResolvedEntity,
    stable_entity_id,
)
from .episodes import activity_bucket, group_events_into_episodes
from .retrieval import TimeFirstQueryPlan, TimeFirstRetriever, build_time_first_query
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

__all__ = [
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
    "TimeFirstQueryPlan",
    "TimeFirstRetriever",
    "activity_bucket",
    "build_time_first_query",
    "content_hash_for",
    "stable_context_id",
    "stable_episode_id",
    "stable_entity_id",
    "group_events_into_episodes",
]
