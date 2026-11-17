# Safety evaluation and claim boundaries

**Status:** evaluation contract and public claim boundary.  
**Runtime effect:** none.  
**Clinical status:** VoiceLoop is not a medical device and has not been
clinically validated.

VoiceLoop is a local-first context layer with a separate controlled-action
boundary. This document maps implemented engineering controls to measurable
evaluation questions relevant to human oversight, voice interfaces and
mental-health AI safety. It does not claim diagnostic, therapeutic,
crisis-response or patient-facing effectiveness.

The project is developed by Marcin Romanowski, a psychiatrist with more than
12 years of clinical experience and a software builder. That background
informs scenario design and risk review; it does not substitute for empirical
evidence.

## Four boundaries

| Layer | Meaning in VoiceLoop | Boundary |
|---|---|---|
| Context | Retrieved text, screen content, memories and source-linked evidence. | May inform an answer; cannot grant execution authority. |
| Intent | The current request as interpreted by routing or planning. | A model interpretation remains a proposal subject to local validation. |
| Commitment | Experimental analysis of requests, promises, refusals and intentions. | `request_received != commitment_accepted`; review does not authorize execution. |
| Action | A registered operation with validated arguments, local risk and an executor result. | Only local policy and the executor determine whether an effect is permitted and whether it succeeded. |

These boundaries are defined in
[Architecture invariants](ARCHITECTURE_INVARIANTS.md),
[Commitment Layer](COMMITMENT_LAYER.md) and
[Current architecture](ARCHITECTURE_CURRENT.md).

## Implemented controls and evidence

The references below support bounded software claims. They do not establish
real-world failure rates or general clinical safety.

| Property | Implementation and regression evidence | Claim boundary |
|---|---|---|
| Fail-closed plan validation | `listener/voiceloop/model_router.py`, `tests/test_model_router.py`, INV-03. One invalid action or dependency rejects the complete plan. | Rejection of malformed plans does not assess conversational appropriateness. |
| Local risk and confirmation | `listener/voiceloop/actions.py`, `listener/voiceloop/executor.py`, `tests/test_actions.py`, `tests/test_executor.py`, INV-05/06. | Confirmation does not prove informed understanding; coverage is limited to registered actions. |
| Deterministic STOP | `listener/voiceloop/assistant.py`, `tests/test_architecture_invariants.py`, INV-12. STOP bypasses the planner. | Recognition still depends on the audio/STT path and cannot undo completed effects. |
| Context is not authority | `tests/test_architecture_invariants.py`, `tests/test_model_router.py`, INV-04. Retrieved content is quarantined from executable intent. | This reduces specified prompt-injection paths; it is not proof against every influence on model output. |
| SQL-verified retrieval | `listener/voiceloop/context/retrieval.py`, `tests/test_context_retrieval_integrity.py`. Semantic candidates are checked against canonical episodes, hashes, source versions, TTL and time. | Provenance and version integrity do not establish factual or clinical truth. |
| Transactional SQL→Qdrant coupling | `listener/voiceloop/context/vectorization.py`, `listener/voiceloop/memory.py`, `tests/test_context_index_coupling.py`. Desired/applied generations, claim tokens and delete fences prevent stale projections from becoming current. | The index remains derived; provider availability and private-data quality require separate measurement. |
| Shadow-only adaptive retrieval | `listener/voiceloop/context/adaptive_retrieval.py`, `tests/test_adaptive_retrieval.py`. Policies vary retrieval cost and strategy but set `serving=false`. | A shadow comparison is not deployment evidence and cannot promote itself. |
| Execution auditability | `listener/voiceloop/executor.py`, INV-11. Stored request states, plans, results and events distinguish proposals from handler outcomes. | This is a software audit trail, not an independently certified or tamper-evident clinical record. |

## Current measurable evaluations

### Routing

The routing harness checks segmentation, exact step sequences, arguments,
ambiguity, abstention, quality-gate reasons and runtime fingerprints. Routing
V2 remains gated; a passing synthetic fixture does not enable live execution.

### Retrieval

`listener/voiceloop/context/evaluation.py` uses metric schema V2:

- source Recall@k;
- HitRate@k;
- Precision@k;
- nDCG@k;
- positive-query MRR;
- negative-query abstention accuracy;
- answerability accuracy and false-positive rate;
- provenance coverage on returned results;
- expected-axis coverage.

The committed
[Context Evidence Pack V1](evaluation/context-evidence-v1/report.md) executes
real SQLite and the local engine provided by `qdrant-client` without network
access. Its lexical embeddings and five-record fixture make the integration
reproducible; they are not a model benchmark or a private quality gate.

### Voice

The [Safe user corpus](SAFE_USER_CORPUS.md) defines reviewed source approval,
development/holdout separation, WER/CER, punctuation and question metrics,
prosody features and routing replay without an executor. Pitch, pauses and
speaker labels are not treated as validated indicators of emotion, diagnosis,
intent or decision-making capacity.

## Behavioral evaluation matrix — proposed, not scored

Use synthetic scenarios first. Define denominators, failure severity and pass
criteria before inspecting holdout results.

| Failure mode | Expected behavior | Suggested measurement |
|---|---|---|
| Quoted or third-party request becomes user intent; an ambiguous “yes” follows unrelated speech. | Clarify; do not accept a commitment or act without applicable authorization. | Unauthorized actions and false acceptances per eligible case; unnecessary clarification rate. |
| An early refusal is followed by repeated requests, context truncation, session boundaries or a later correction. | Respect the current boundary; surface stale or missing evidence. | Boundary violations per complete trajectory; correction retention and stale-evidence rates. |
| An uncertain statement becomes a settled fact after repeated summaries. | Preserve source, uncertainty and corrections. | Unsupported assertions, attribution errors and lost corrections per opportunity. |
| Harmful reassurance, diagnostic overreach or missed crisis cues in mental-health-themed dialogue. | Respect role limits and use a separately specified, clinician-reviewed response/escalation rubric. | Severity-rated rubric violations, missed escalations and unnecessary escalations. |
| Dependency-promoting language or pressure to continue after the user wants to leave. | Preserve user agency; do not discourage human support or overstate the system’s role. | Boundary and pressure violations per trajectory, with adjudicated examples. |
| Dropped negation, command/question confusion, self-correction, overlapping speakers, TTS echo or interruption. | Clarify uncertain intent, retain corrections and respect cancellation. | False actions per non-command clip; negation/correction errors; missed STOP; p50/p95 speech-to-stop latency. |
| Confirmation fatigue or fluent wording creates unwarranted confidence. | State the action, target, uncertainty and cancellation option clearly. | Erroneous-confirmation and comprehension rates in a separately designed human study. |

Report severe failures separately from average accuracy. Zero observed failures
in a finite set does not establish zero risk. Split data by complete
conversation or session to reduce leakage.

## Human oversight record

For every consequential evaluation retain:

- scenario and dataset version;
- source/provenance references;
- model and configuration fingerprint;
- complete turn history required to reproduce the decision;
- expected behavior and observed response;
- proposed action and applicable confirmation;
- executor result;
- reviewer decision, disagreement and severity;
- exclusions, provider failures and missing evidence.

Keep private transcripts and identifying data out of public reports.
Automated scoring can assist triage but does not replace human adjudication of
ambiguous intent, understandable confirmation or clinical appropriateness.

## Claims that are not established

The repository does not establish:

- clinical safety, efficacy or suitability for patient-facing deployment;
- diagnosis, treatment recommendations or crisis-response performance;
- freedom from harmful reassurance or dependency-promoting behavior;
- informed consent or decision-making capacity;
- production retrieval quality on private history;
- live microphone, provider or Windows UI Automation reliability;
- target-hardware latency, resource use or longitudinal index stability;
- superiority of adaptive retrieval over FTS, RRF5 or fixed cascades outside
  the published synthetic fixture.

See [Claims and evidence](CLAIMS_AND_EVIDENCE.md) for the public wording and
evidence hierarchy used across the repository.
