# Context Evidence Pack V1 — release notes

**Status:** release candidate; no Git tag is created by this document.

## Included evidence

- deterministic public fixture with five timeline events and episodes;
- real SQLite writes, FTS reads and source provenance;
- real `qdrant-client` local engine with five named-vector points;
- idempotent second indexing pass;
- FTS-only, RRF5 and cascade 2+2+1 comparison;
- agreement, conflict, bounded abstention and unbounded rejection scenarios;
- local action-policy example that raises risk and requires confirmation without
  executing a handler;
- JSON, Markdown and standalone HTML reports.

## Reproduction

```powershell
cd listener
.\.venv\Scripts\python.exe ..\scripts\run_context_evidence_demo.py `
  --output ..\artifacts\context-evidence-demo
```

CI runs the same command on Ubuntu after the Windows quality job and uploads
`context-evidence-<sha>` for 14 days. The generated Markdown is also added to
the GitHub Actions job summary.

## Interpretation

The pack proves that the public integration contract runs end to end without
network access. It does not prove private-data quality, clinical validity,
target-Windows performance or production readiness. `promotion_allowed`
remains `false`.

## Known result

The committed fixture exposes an RRF5 miss for the topic example. This is kept
as an explicit failure rather than removed from the dataset. On this tiny
lexical fixture, FTS is the fastest and strongest baseline.

## Human release step

After merge, a person must choose the repository version, review the CI
artifact, create the Git tag and GitHub Release, and attach or link the
evidence artifact. Automated publication is intentionally absent.

