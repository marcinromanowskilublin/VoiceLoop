# Changelog

All notable public changes to VoiceLoop are documented here.

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
