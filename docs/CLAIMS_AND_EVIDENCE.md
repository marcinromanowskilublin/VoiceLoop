# Claims and evidence

**Purpose:** keep public statements aligned with code, configuration and
reproducible evidence.  
**Authority order:** current code and tests → generated evidence → current
architecture documentation → historical material.

VoiceLoop distinguishes five evidence levels:

1. **Implemented** — the code path exists.
2. **Regression-tested** — automated tests cover specified behavior.
3. **Reproduced publicly** — a committed synthetic scenario executes the real
   integration contract without private data.
4. **Measured privately** — representative private examples have been reviewed
   outside the repository.
5. **Validated in deployment** — target users, hardware and workflow have been
   evaluated under a defined protocol.

Higher levels are never inferred from lower ones.

## Claim register

| Claim | Status | Evidence | Permitted wording |
|---|---|---|---|
| SQLite is canonical for timeline content, time and source identity. | Implemented and regression-tested. | `listener/voiceloop/memory.py`, `tests/test_context_foundation.py`, `tests/test_context_retrieval_integrity.py`. | “SQLite is the canonical context store.” |
| Qdrant is a derived candidate index. | Implemented and regression-tested. | `listener/voiceloop/qdrant_memory.py`, `listener/voiceloop/context/vectorization.py`, `tests/test_context_index_coupling.py`. | “Vectors rank candidates; SQL verifies current records.” |
| Stale vector writes cannot silently become the applied generation. | Implemented and regression-tested. | Desired/applied generations, claim tokens, CAS and delete fences in `memory.py` and `vectorization.py`. | “The index projection is generation-fenced and retryable.” |
| Retrieved context cannot grant action authority. | Implemented and regression-tested for specified paths. | INV-04, `tests/test_architecture_invariants.py`, `tests/test_model_router.py`. | “Retrieved evidence does not authorize execution.” |
| Local policy can raise risk and require confirmation. | Implemented, regression-tested and publicly reproduced. | `actions.py`, `executor.py`, `tests/test_executor.py`, public evidence pack. | “Local policy, not model text, determines required confirmation.” |
| Dual-track and adaptive retrieval run without changing serving context. | Implemented and regression-tested in shadow mode. | `context/dual_track.py`, `context/adaptive_retrieval.py`, associated tests. | “Candidate retrieval strategies are measured in shadow.” |
| The public evidence command reproduces SQLite/Qdrant integration offline. | Publicly reproduced. | `scripts/run_context_evidence_demo.py`, CI artifact and committed report. | “A no-network synthetic scenario reproduces the integration contract.” |
| Adaptive retrieval is better than the baseline. | Not established generally. | One small lexical fixture reports results and preserves an RRF5 miss. | “Adaptive retrieval is a candidate under evaluation.” |
| Context Timeline works well on private history. | Not established. | A private reviewed gold set is still required. | “Private-data quality remains an evaluation gate.” |
| VoiceLoop is clinically safe or effective. | Not established. | No clinical validation or outcome study. | “The repository contains engineering controls relevant to safety evaluation.” |
| VoiceLoop supports arbitrary host environments with the same behavior. | Design direction only. | Adapter contracts and demonstrations are incomplete. | “The architecture is designed for explicit adapters; portability remains to be demonstrated.” |

## Public evidence package

Run from `listener/`:

```powershell
.\.venv\Scripts\python.exe ..\scripts\run_context_evidence_demo.py `
  --output ..\artifacts\context-evidence-demo
```

Outputs:

- `report.json` — machine-readable facts and limitations;
- `report.md` — evaluation card and failure analysis;
- `report.html` — standalone visual walkthrough.

CI runs the same scenario and uploads `context-evidence-<sha>`. The committed
reference is [VoiceLoop public context evidence v1](evaluation/context-evidence-v1/report.md).

The fixture is intentionally small and synthetic. Its deterministic lexical
embeddings are useful for reproducibility, not for asserting model quality,
private-data performance or target-Windows latency.

## Wording rules

Use:

- “implemented” when a code path exists;
- “regression-tested” only for the behavior a named test covers;
- “reproduced” when the exact command and artifact are available;
- “measured” only with dataset, split, sample count, revision and configuration;
- “shadow” for paths that cannot select serving context;
- “proposed” or “design direction” for unimplemented behavior.

Avoid:

- “safe” without naming the bounded property;
- “production-ready” based on CI alone;
- “accurate” without a metric, denominator and dataset;
- “clinically validated” or “medical” without the applicable evidence;
- “real-time” without a latency definition and measurement;
- “private” when a configured provider may receive explicitly permitted data;
- “supports” when only a schema, mock or diagram exists.

## Release checklist

Before a public release or portfolio update:

1. Pin the source revision.
2. Run the documented Ruff and pytest scope.
3. Run dependency audit against the locked requirements.
4. Regenerate the evidence package.
5. Review every numerical claim against `report.json`.
6. Keep known misses and failed checks visible.
7. Separate synthetic, private and target-machine evidence.
8. Verify local links, image descriptions and status labels.
9. Confirm that planned paths are not described as serving.
10. Record remaining unknowns rather than replacing them with estimates.

## Related contracts

- [Current architecture](ARCHITECTURE_CURRENT.md)
- [Architecture invariants](ARCHITECTURE_INVARIANTS.md)
- [Safety evaluation and claim boundaries](SAFETY_EVALUATION.md)
- [Dynamic Memory Coupling V1](DYNAMIC_MEMORY_COUPLING_V1.md)
- [Context Timeline V1](CONTEXT_TIMELINE_V1.md)
- [Context Evidence Pack V1](CONTEXT_EVIDENCE_RELEASE_V1.md)
- [Portfolio case study](PORTFOLIO_CASE_STUDY.md)
