# StateProposal V1

**Ten kontrakt nie wpina propozycji w planer.** `assistant.py` nadal nie
importuje `situation`.
LLM nie pisze `SituationState`. Nie ma nowej akcji w `ActionRegistry`.

## Ścieżka

Dokładna symetria z `ProposedPlan` → `CommandPlan` → `enforce_policy` → `CommandExecutor`:

```text
LLM JSON ──► StateProposal (schema, extra=forbid)
         ──► StatePolicy.enforce
         ──► StateReducer.reduce
         ──► SituationStore.append_event(actor=local_code)
```

Wejście publiczne: `apply_state_proposal(store, proposal, evidence_index?)`.  
Wołane z testów / przyszłego shadow. Nie z produkcyjnego planera.

## Kontrakt

`listener/voiceloop/situation/proposal.py` → `StateProposal`

Pola: `op`, `content`, `evidence_refs`, `confidence`, opcjonalnie `from_kind` / `to_kind` / `target_item_id`.

`op`: `propose_fact`, `propose_hypothesis`, `propose_request`, `propose_commitment`, `propose_intention`, `propose_decision`, `promote`, `claim_action_succeeded`.

`evidence_index` jest lokalnym katalogiem dowodów. Model podaje tylko ID. Nie może dosłać `actor`, `success` ani `action_id` (`extra=forbid`).

## Polityka (fail-closed)

- `propose_fact` — tylko przy dowodzie `authoritative` / `user_asserted` / `observed`.
- `hypothesis → fact` (`promote`) — ten sam próg; sam `model_inference` nie wystarczy.
- `request → commitment` — odrzut, gdy jedyny dowód to tekst modelu.
- `claim_action_succeeded` — zawsze odrzut. Sukces akcji jest z executora (`ActionResult`).
- `actor=llm` — nie istnieje. `SituationActor` ma tylko `local_code`.
- brak / nieznany `evidence_ref` — odrzut całego wniosku (jak nieznany `action_id`).

Zaakceptowana propozycja **dopisuje** zdarzenie. Nie robi CRUD UPDATE na hipotezie.

## Co nie jest w tym etapie

- import w `assistant.py` / `model_router.py` / `router.py`
- zapis LLM do SQLite
- nowa akcja, zmiana risk, panel
- zapis commitment shadow do SituationState
