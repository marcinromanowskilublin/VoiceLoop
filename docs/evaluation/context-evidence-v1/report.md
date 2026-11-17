# VoiceLoop public context evidence v1

- Source revision: `372b642`
- Data: public synthetic only
- Network: not used
- SQLite events/episodes: 5
- Qdrant points: 5
- Idempotent second index: true
- Index coupling consistent: true
- Promotion allowed: `false`

## Retrieval comparison

| Track | Recall@3 | MRR | Negative abstention | Result provenance | Smoke P50 ms | Smoke P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| FTS only | 1.000 | 1.000 | 1.000 | 1.000 | 1.729 | 1.899 |
| RRF5 | 0.750 | 0.500 | 1.000 | 1.000 | 12.837 | 13.439 |
| Cascade 2+2+1 | 1.000 | 0.833 | 1.000 | 1.000 | 11.461 | 28.733 |
| Adaptive hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 12.687 | 14.637 |

## Dynamic policy profiles

| Profile | Strategy | FTS weight | Vector weight | Max axes (qmax) | Serving eligible |
|---|---|---:|---:|---:|---|
| Windows exact operation | `exact_only` | 0.900 | 0.100 | 2 | `false` |
| Windows architectural exploration | `fts_then_cascade` | 0.550 | 0.450 | 3 | `false` |
| RPG canonical fact | `fts_then_cascade` | 0.720 | 0.280 | 4 | `false` |
| RPG creative association | `hybrid_cascade` | 0.250 | 0.750 | 5 | `false` |

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
- **adaptive_hybrid:** no misses in this synthetic fixture

## What this does not prove

- Synthetic data is not evidence of private-data quality.
- Lexical hash embeddings make the run deterministic; they are not a model benchmark.
- Qdrant runs in the real qdrant-client local engine, not a network server.
- Latency is runner-specific and must not be generalized to target Windows hardware.
- Agreement means the same top canonical episode version; the remaining ranked lists may differ.
- The arbiter remains shadow-only and cannot authorize an action.
