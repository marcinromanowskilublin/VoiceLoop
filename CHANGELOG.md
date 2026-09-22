# Changelog

All notable public changes to VoiceLoop are documented here.

## [Unreleased]

### Added

- Explicit document ingest, Windows project projection, SQL memory migration,
  deictic live-screen items, commitment review rows and shadow retrieval
  comparison. Runtime integrations remain opt-in and off by default.
- Timeline adapter retention keeps explicit user memories without a TTL;
  document and project records preserve source timestamps rather than scan time.
- Document events store a digest plus a reference and re-validate configured
  roots when the full text is read on demand.
- Local CLI commands for document ingest, memory migration, timeline pruning
  and the private retrieval comparison report.

### Fixed

- Time-first recall keeps an empty answer within its requested time range
  instead of falling through to unbounded legacy recall.
- Semantic timeline candidates are checked against canonical SQLite episodes
  and source-event versions, including identity, deletion, expiry and time.
  Returned content comes from SQL; ranking no longer becomes `confidence`.
- Invalid, timezone-free and out-of-range Screenpipe timestamps are discarded.

### Changed

- Explicit episode indexing records source-event hashes when SQLite is supplied.
  Legacy points without canonical references and source versions are excluded
  from time-first semantic recall; no automatic migration is performed.
- README and architecture docs distinguish planner/default recall from opt-in
  time-first retrieval, document adapter write effects and indexing limits,
  and record the larger memory-axis navigation design as future work.

### Verification

- Code checkpoint `bf26abe`: local full suite `916 passed, 1 skipped`; Ruff passed.
  This does not establish live audio, provider availability or private recall quality.

## [0.3.0] - 2026-09-20

### Added

- Context Timeline V1 with canonical SQLite events and episodes, FTS5 search,
  paginated Screenpipe history, meeting adapters and explicit Win32 foreground
  observations.
- Typed `ContextPackV1`, deterministic context budgets and session-scoped
  conversation history.
- Reviewed person/project entity registry with stable identifiers and candidate
  approval gates.
- Selective episode vectors, time-first retrieval, semantic scout, lazy reserve
  axes and retrieval evaluation metrics.
- Focus-bound active Notepad read/write actions with explicit replacement
  confirmation and read-back verification.

### Changed

- Ranked context is assembled explicitly instead of repeatedly taking the tail
  of concatenated memory lists.
- Context retention coordinates SQLite, FTS5 and Qdrant fail-closed.
- Public README and architecture diagrams now separate active, opt-in,
  context-only and shadow behavior.
- Architecture and invariant documentation now reflects 44 registered actions.

### Safety and rollout

- Context Timeline recall remains disabled by default.
- Automatic timeline ingest, foreground polling and automatic entity approval
  remain disabled.
- Routing V2 remains shadow by default.
- Retrieved context still cannot create an action, lower risk, replace required
  confirmation or claim execution success.

### Verification

- Windows CI: Ruff and the full pytest suite.
- Live Windows automation remains a separate opt-in/manual verification.

## [0.2.0]

- Previous public baseline before the Notepad lane and Context Timeline work.
