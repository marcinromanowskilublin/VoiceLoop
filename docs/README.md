# VoiceLoop documentation

Public documentation for VoiceLoop 0.3.0.

The source of truth is the current code under `listener/voiceloop/`. Documents
below are grouped by role so historical handoffs and experiments do not look
like current runtime behavior.

## Start here

- [Repository README](../README.md) — product overview, live demo and quick start.
- [Current architecture](ARCHITECTURE_CURRENT.md) — active, experimental and
  planned components mapped to code.
- [Architecture invariants](ARCHITECTURE_INVARIANTS.md) — safety boundaries
  enforced by tests.

## Context, memory and evaluation

- [Context Timeline V1](CONTEXT_TIMELINE_V1.md) — time-first local context,
  explicit adapters, SQL-verified semantic candidates, CLI operations and
  rollout limits. Includes legacy-index compatibility requirements.
- [Memory-axis navigation design](VECTOR_NAVIGATION_DESIGN.md) — agreed direction
  for a larger set of specialized axes, evidence requirements and stopping
  rules; separates future design from current code and human labeling duties.
- [Threshold Guard](THRESHOLD_GUARD.md) — measured vector thresholds and
  fail-closed behavior.
- [Safe user corpus](SAFE_USER_CORPUS.md) — local corpus, privacy gates,
  holdouts and controlled remote evaluation.

## Experimental contracts

These contracts exist in code but do not grant execution authority.

- [Commitment Layer](COMMITMENT_LAYER.md) — requests, promises, boundaries and
  review states.
- [EvidenceItem V1](EVIDENCE_V1.md) — provenance and trust classes.
- [SituationState V1](SITUATION_STATE_V1.md) — read-only event ledger.
- [StateProposal V1](STATE_PROPOSAL_V1.md) — locally governed state proposals.

## Visual reference

- [System architecture](img/voiceloop-system-architecture.svg)
- [Memory and context architecture](img/voiceloop-memory-architecture.svg)
- [Safe execution model](img/voiceloop-safe-execution.svg)
- [Diagnostics panel](img/voiceloop-panel.png)

## Historical material

Historical plans, handoffs and point-in-time audits are not part of the current
architecture contract. They live under `archive/` only when retaining them is
useful for project archaeology.
