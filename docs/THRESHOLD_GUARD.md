# Threshold Guard And Runtime Gates

This document records the runtime threshold contracts that must not be replaced
by guessed cosine cutoffs.

## Vector Memory Retrieval

`VECTOR_MEMORY_MIN_SCORE=0.0` is deliberate. The memory retriever fuses named
vector rankings with weighted RRF, so one absolute cosine gate across all memory
axes would mean a different thing in each space.

The retriever still limits results by ranking and context size. It does not
return the whole collection. The LLM receives retrieval evidence such as
`rank_fusion`, `best_cosine`, matched spaces, source and timestamp.

Do not introduce `thresholds.yaml` or per-axis values such as `0.75` without a
new measurement that proves where the gate acts.

## Screenpipe Deduplication

`ACTIVITY_DUPLICATE_MIN_SCORE=0.97` applies only after digesting activity, when
the candidate semantic document and stored semantic documents are in the same
document-document space.

Screenpipe deduplication has two separate stages:

1. Before digest: exact `content_hash`, no embedding call.
2. After digest: semantic duplicate check using the exact vector that would be
   stored.

An unavailable Qdrant store must fail closed during ingestion. It must not be
treated as "duplicate absent", because that writes copies into memory.

## User Retrieval Fallback

User-facing retrieval is different from ingestion. If Qdrant is unavailable
during assistant context lookup or the `recall` action, VoiceLoop may fall back
to the local SQLite vector store. This preserves local functionality without
weakening ingestion deduplication.

## Routing V2 Gates

Routing V2 intentionally disables the capability-index prefilter in the service
path by calling capability search with `min_score=-1.0`. Resolution then applies
the real gates:

- `STT_MIN_ACTION_CONFIDENCE=0.75` gates uncertain Deepgram commands.
- `ROUTING_V2_EXECUTE_MIN_SCORE=0.50` gates the combined resolver score.
- `ROUTING_V2_EXECUTE_MIN_MARGIN=0.10` gates the top-2 margin.
- A single candidate without a comparator asks for clarification.
- Insufficient vector coverage asks for clarification or rejects the candidate.

These gates are resolver contracts, not raw cosine thresholds.

## Threshold Guard Scope

`threshold_guard.py` is a read-only measurement worker. It samples Qdrant with
`scroll` and `query_points`, records verdicts for health, and must not write
points or replace runtime memory.

The guard can measure vector-memory geometry and the Screenpipe duplicate gate.
It intentionally reports some settings as out of scope:

- `capability_match_min_score` is a normalized rank-fusion threshold in a
  separate capability corpus.
- `behavior_digest_min_confidence` is a model-output confidence threshold, not
  an embedding geometry threshold.

Out-of-scope thresholds should be named as `unmeasured`, not silently ignored.

## Retention And Prune

Vector memories can carry `expires_at`, but automatic pruning remains off by
default with `VECTOR_MEMORY_PRUNE_ENABLED=false`. Keep it that way unless a
dry-run has shown exactly how many Qdrant and SQLite records would be removed.

`QdrantVectorStore.prune_expired()` is dry-run by default. Deleting expired
records requires an explicit runtime flag and should be treated as an operational
cleanup step, not as part of threshold calibration.

## RRF Variants

There are currently two RRF normalizations:

- Memory retrieval normalizes by vector spaces that actually contributed.
- Capability matching normalizes against the fixed capability-vector set.

Do not extract a shared helper until tests characterize both variants. The goal
is not identical formulas everywhere; the goal is explicit semantics in each
ranking path.
