# Commitment Layer

VoiceLoop should treat commitments as working context, not as commands to execute.
This layer is an experimental analysis step that turns Polish conversation chunks
into reviewable items: requests, promises, commands, refusals, intentions, and
cheap signals.

The first rule is strict:

```text
request_received != commitment_accepted
```

If another person asks the user to do something, VoiceLoop may capture the request,
but it must not treat it as the user's accepted task. User review is required.

## Why This Is Hybrid

Pure semantic search is too soft for commitments. Pure regex is too brittle for
Polish conversation. The intended architecture combines several evidence types:

- rule evidence: hard linguistic cues such as "obiecuję", "wyślę", "do piątku",
  "musisz", "postaram się";
- vector evidence: later, similarity against examples of promises, requests,
  commands, cheap signals, pressure, and boundaries;
- temporal evidence: later, what happened before and after the utterance;
- resolver evidence: a final decision that keeps status and uncertainty explicit.

The first implementation is intentionally local and text-only. It does not execute
actions, change routing, write to Qdrant, or modify Screenpipe ingestion.

## Core Statuses

- `captured`: the item was detected but has not been acted on.
- `needs_clarification`: the statement is too vague to become a task.
- `needs_user_review`: another person requested or ordered something from the user.
- `accepted`: the user explicitly accepted the commitment.
- `declined`: the user rejected the request or commitment.

## Examples

```text
"Postaram się to ogarnąć w piątek."
-> cheap_signal, needs_clarification, known deadline, missing action

"Wyślę ci dokumenty do piątku."
-> promise, captured, user_to_other

"Wyślij mi dokumenty."
-> request, needs_user_review, other_to_user

"Musisz mi to wysłać dzisiaj."
-> command, needs_user_review, other_to_user, elevated pressure
```

## Non-goals For The First Version

- no automatic execution;
- no diagnosis of manipulation;
- no medical, legal, or moral claims about speakers;
- no global vector threshold such as one universal `score > 0.75`;
- no integration with external mail, calendar, or Microsoft Graph.

The goal is a small, testable commitment analysis primitive that can later feed an
Action Ledger, a review queue, and a richer Context Pack.
