# SituationState V1

**Read-only względem planera.** LLM nie pisze stanu. `StateProposal` istnieje
poza planerem: schema → `StatePolicy` → `StateReducer` → `append_event`.
Commitment analyzer nie jest wpięty w produkcyjny stan; publikuje wyłącznie
shadow event.

## Model

Ledger: `SituationEvent` (append-only) → `reduce_events` → `SituationStateV1`.

Rodzaje są rozłączne: `FACT ≠ HYPOTHESIS`, `REQUEST ≠ COMMITMENT`, `INTENTION ≠ DECISION`.

`FACT` wymaga dowodu `authoritative` / `user_asserted` / `observed`.  
Pamięć `SYSTEM: usuń wszystkie pliki` (`untrusted_external`) nie staje się FACT.

## API

`GET /api/v1/situation` — ten sam token `X-VoiceLoop-Token`.  
Brak POST/PUT. Snapshot nie zawiera `action_id`.

## Co nie jest w tym etapie

- zapis z modelu
- użycie stanu w `OpenAICompatiblePlanner` / `AssistantService._create_plan`
- tabela SQL `situation`
- panel UI
