# VoiceLoop public context evidence v1

- Source revision: `372b642`
- Data: public synthetic only
- Network: not used
- SQLite events/episodes: 5
- Qdrant points: 5
- Idempotent second index: true
- Promotion allowed: `false`

## Retrieval comparison

| Track | Recall@3 | MRR | Negative abstention | Result provenance | Smoke P50 ms | Smoke P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| FTS only | 1.000 | 1.000 | 1.000 | 1.000 | 1.511 | 1.877 |
| RRF5 | 0.750 | 0.500 | 1.000 | 1.000 | 9.728 | 10.883 |
| Cascade 2+2+1 | 1.000 | 0.833 | 1.000 | 1.000 | 10.176 | 14.271 |

## Observable outcomes

| Scenario | Verdict | Baseline | Cascade | Safety behavior |
|---|---|---:|---:|---|
| agreement | agreement | 3 | 3 | record agreement; no serving switch |
| conflict | conflict | 1 | 1 | record conflict for human review |
| abstention | uncertain | 0 | 0 | return no bounded evidence |
| unbounded | uncertain | 0 | 0 | reject unbounded dual-track measurement |

## Action boundary

- Proposed action: `remember`
- Risk raised locally: `low` -> `medium`
- Confirmation required: `true`
- Executor status: `awaiting_confirmation`
- Handler calls: `0`
- Handler executed: `false`

## Failure analysis

- **fts_only:** no misses in this synthetic fixture
- **rrf5:** topic:miss
- **cascade_2_2_1:** no misses in this synthetic fixture

## What this does not prove

- Synthetic data is not evidence of private-data quality.
- Lexical hash embeddings make the run deterministic; they are not a model benchmark.
- Qdrant runs in the real qdrant-client local engine, not a network server.
- Latency is runner-specific and must not be generalized to target Windows hardware.
- Agreement means the same top canonical episode version; the remaining ranked lists may differ.
- The arbiter remains shadow-only and cannot authorize an action.
