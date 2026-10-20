# VoiceLoop public context evidence v1

- Source revision: `deb77fe`
- Data: public synthetic only
- Network: not used
- SQLite events/episodes: 5
- Qdrant points: 5
- Idempotent second index: true
- Promotion allowed: `false`

## Retrieval comparison

| Track | Recall@3 | MRR | Abstention | Provenance | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| FTS only | 1.000 | 0.800 | 1.000 | 0.800 | 2.240 | 2.433 |
| RRF5 | 0.750 | 0.400 | 0.800 | 0.800 | 10.101 | 11.770 |
| Cascade 2+2+1 | 1.000 | 0.667 | 1.000 | 0.800 | 12.157 | 14.747 |

## Observable outcomes

| Scenario | Verdict | Baseline | Cascade | Safety behavior |
|---|---|---:|---:|---|
| agreement | agreement | 3 | 3 | record agreement; no serving switch |
| conflict | conflict | 1 | 1 | record conflict for human review |
| abstention | uncertain | 0 | 0 | return no bounded evidence |
| unbounded | uncertain | 0 | 0 | reject unbounded dual-track measurement |

## Action boundary

- Proposed action: `open_shell_item`
- Risk raised locally: `low` -> `medium`
- Confirmation required: `true`
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
- The arbiter remains shadow-only and cannot authorize an action.
