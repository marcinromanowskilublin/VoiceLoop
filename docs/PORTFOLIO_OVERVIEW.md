# VoiceLoop — technical portfolio at a glance

<img src="img/portfolio-hero.svg" alt="VoiceLoop project hero." width="100%">

## The design challenge

Build a locally anchored AI context layer that can distinguish **what was observed**, **what is being requested**, and **what is actually permitted to happen**.

<img src="img/portfolio-system.svg" alt="Source-to-evidence architecture with a separate action authority boundary." width="100%">

## Three engineering contributions

### 1. A time- and provenance-aware memory core
SQLite records the authoritative event, timestamp and source. Qdrant provides derived candidate ranking. An outbox and monotonic generations keep index updates observable. Time-first recall and adaptive vector strategies remain gated or shadow-only where specified.

### 2. An incremental conversation graph
<img src="img/portfolio-conversation.svg" alt="Concept graph with two diarized speakers and preserved nodes." width="100%">

Operations are **add / enrich / link / revise**, applied to an existing graph; IDs, positions and history persist. A deterministic heuristic can propose nodes, but a semantic-quality evaluation on real conversations is not established. Tests: [graph regression suite](../tests/test_conversation_graph.py).

### 3. Proof-oriented evaluation
<img src="img/portfolio-evidence.svg" alt="Five-record synthetic Recall-at-3 comparison; FTS 1.0, RRF5 0.75, cascade 1.0, adaptive 1.0." width="100%">

The reported results are from a **synthetic five-record fixture**, with no private data and deterministic lexical hash embeddings. RRF5's topic miss remains visible. The evidence pack also checks a local action stops at confirmation without running its handler. [Full reproducible results](evaluation/context-evidence-v1/report.md).

## The security principle

> A model proposes. A local policy validates. An executor reports success.

Retrieved data never becomes executable intent by itself. This is an engineering architecture with bounded regression tests, **not** a claim of clinical safety, real-world reliability or full deployment validation.

## Read the implementation

[Source code](../listener/voiceloop/) · [Current architecture](ARCHITECTURE_CURRENT.md) · [Case study](PORTFOLIO_CASE_STUDY.md) · [Evidence and claims](CLAIMS_AND_EVIDENCE.md) · [Repository home](../README.md)

<sub>Portfolio status reviewed 2026-11-24. No release version change.</sub>
