# VoiceLoop documentation

Public documentation for VoiceLoop 0.3.0.

The source of truth is the current code under `listener/voiceloop/`. Documents
below are grouped by role so historical handoffs and experiments do not look
like current runtime behavior.

## Start here

- [Repository README](../README.md) — context-layer overview, truthful system
  diagrams, current status and quick start.
- [Current architecture](ARCHITECTURE_CURRENT.md) — active, experimental and
  planned components mapped to code.
- [Architecture invariants](ARCHITECTURE_INVARIANTS.md) — safety boundaries
  enforced by tests.

## Context, memory and evaluation

- [Context Timeline V1](CONTEXT_TIMELINE_V1.md) — time-first local context,
  explicit adapters, SQL-verified semantic candidates, CLI operations and
  rollout limits. Includes legacy-index compatibility requirements.
- [Cascade 2+2+1 and security boundary](KASKADA_BEZPIECZENSTWO.md) — shadow
  measurer on the five existing axes, the HTTP, file, search and retention
  boundaries, and the decisions only a person can make.
- [Dual-track cascade prototype](DUAL_TRACK_CASCADE_PROTOTYPE.md) — isolated
  episode collection, explicit idempotent indexer, parallel RRF5/2+2+1,
  deterministic arbiter, content-free traces and read-only demonstrator.
- [Manual human contribution](DO_WYKONANIA_PRZEZ_CZLOWIEKA.md) — decisions,
  private labeling and operator steps that cannot be completed by code.
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

- [Local context dual-track flow](img/voiceloop-dual-track-flow.svg) — local
  adapters, canonical SQL time, parallel retrieval and shadow evidence.
- [Arbiter outcomes](img/voiceloop-arbiter-outcomes.svg) — deterministic
  agreement, conflict and uncertain behavior.
- [Time certainty boundary](img/voiceloop-time-certainty.svg) — bounded,
  approximate and missing time remain distinct states.
- [Context workspace — user view](img/voiceloop-context-workspace.svg)
- [Live context workspace](img/voiceloop-panel.png)
- [System architecture](img/voiceloop-system-architecture.svg)
- [Memory and context architecture](img/voiceloop-memory-architecture.svg)
- [Safe execution model](img/voiceloop-safe-execution.svg)

## Optional integration directories

The core lives in `listener/voiceloop/`. The directories below are tracked
because the optional stack depends on them, but none of them is required to run
the core, and none of them can bypass the executor or local policy.

- [`panel/`](../panel/) — the local control panel served by the FastAPI core.
- [`vectorscope/`](../vectorscope/README.md) — read-only Qdrant inspection lab
  and threshold measurements; it never writes to memory.
- [`voiceattack/`](../voiceattack/INSTRUKCJA.md) — VoiceAttack profiles that
  forward configured commands to the core as the `voiceattack` source.
- [`uivision/`](../uivision/) — UI.Vision macros used by the confirmed
  `run_uivision_macro` action.
- [`n8n/`](../n8n/) — exact-phrase webhook workflow for the optional n8n
  fallback (`n8n_enabled=false` by default).

## Historical material

Historical plans, handoffs and point-in-time audits are not part of the current
architecture contract. They live under [`archive/`](archive/README.md) only when
retaining them is useful for project archaeology.
