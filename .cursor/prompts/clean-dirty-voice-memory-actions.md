# Cleanup: dirty tree — głos, pamięć, akcje

Kontynuujesz VoiceLoop w `C:\Users\marci\VoiceLoop`.

Przeczytaj najpierw `docs/VOICELOOP_ARCHITECTURE_HANDOFF.md` (sekcje 2, 25, 28–31).
Nie commituj `listener/.env`, `data/`, `logs/`, tokenów, `.venv`.
Nie czytaj wartości sekretów.

## Poza zakresem (nie ruszaj)

- L4, medycyna, pacjenci
- ZnanyLekarz, mydr, EDM, `EDM_Lite/`
- `sources/notes/integrations/MYDR_EDM_POLECENIA_GLOSOWE.md` — jeśli jest w indexie, zrób `git restore --staged` i zostaw
- `sources/articles/mydr_edm/`, `sources/articles/znanylekarz/`, inne notatki integracji medycznej
- Routing V2: zostaje w **shadow mode**. Nie ustawiaj `ROUTING_V2_EXECUTE=true` ani `ROUTING_V2_SHADOW_MODE=false`

## Co jest brudne

Working tree ma ~16k dodań vs `cdc9770`. Wiele plików jest `MM`/`AM`: coś już w indexie, potem kolejne edycje na dysku. To nie jest merge. To niedokończony zapis trzech wdrożeń naraz.

Trzy stosy (tylko te sprzątasz):

### 1. Głos
- `listener/voiceloop/voice_conversation.py`
- `listener/voiceloop/deepgram.py`
- `listener/voiceloop/tts.py`
- `listener/voiceloop/audio_capture.py` (untracked)
- `listener/voiceloop/meeting_recorder.py` (untracked)
- `tests/test_conversation.py`, `tests/test_deepgram.py`, `tests/test_tts.py`
- `tests/test_audio_capture.py`, `tests/test_meeting_recorder.py`, `tests/test_voice_eval.py`
- `panel/index.html` (tylko część rozmowy/STOP/nasłuch; nie mieszaj z medycyną)
- `voiceattack/INSTRUKCJA.md`, `voiceattack/VoiceLoop-v2.vap`, `voiceattack/VoiceLoop-v2.vap.txt`
- `scripts/build-voiceattack-profile.py`
- `docs/SAFE_USER_CORPUS.md` i `listener/voiceloop/corpus/` — tylko jeśli dotyczą głosu/korpusu, nie medycyny

### 2. Pamięć
- `listener/voiceloop/memory.py`
- `listener/voiceloop/qdrant_memory.py`
- `listener/voiceloop/memory_vectorization.py` (untracked)
- `listener/voiceloop/manual_memory.py` (untracked)
- `listener/voiceloop/behavior_digest.py`
- `listener/voiceloop/embeddings.py`
- `listener/voiceloop/screenpipe.py`, `screenpipe_memory.py`, `screenpipe_deepgram.py`
- testy: `test_memory.py`, `test_qdrant_memory.py`, `test_behavior_digest.py`, `test_screenpipe.py`, `test_screenpipe_deepgram.py`, `test_memory_eval.py`

### 3. Akcje
- `listener/voiceloop/actions.py`
- `listener/voiceloop/executor.py`
- `listener/voiceloop/capability_index.py` (untracked)
- `scripts/va/*-under-cursor.vbs`, `scripts/va/capabilities.vbs`, `scripts/va/remember-last-source.vbs`
- `scripts/send-command.ps1`
- testy: `test_actions.py`, `test_executor.py`, `test_capability_index.py`, `test_assets.py`

Rdzeń, który skleja stosy (`app.py`, `assistant.py`, `router.py`, `model_router.py`, `models.py`, `settings.py`, `n8n_client.py`, `README.md`, handoff, `listener/.env.example`, requirements): weź do osobnego, ostatniego commita „glue”, albo do commita tego stosu, którego dotyczy większość diffu. Nie pakuj wszystkiego w jeden commit.

## Śmieci

- Plik `=` w root (0 bajtów, staged) — `git restore --staged -- "="` i usuń. To wypadek, nie kod.

## Jak sprzątać

1. `git status` i `git diff --stat HEAD`. Opisz trzy stosy zanim ruszysz index.
2. Zdejmij z indexu medycynę i `=`.
3. Nie commituj jeszcze. Najpierw zrób drzewo spójne:
   - jeden zestaw zmian na plik (index = working tree dla plików, które wchodzą)
   - żadnych sekretów w diffie
   - handoff/README tylko tam, gdzie stan kodu naprawdę się zmienił
4. Testy (obowiązkowe przed commitem):
   ```
   cd listener
   .\.venv\Scripts\python.exe -m ruff check voiceloop ..\tests
   .\.venv\Scripts\python.exe -m pytest -c pyproject.toml -q ..\tests
   ```
   Potem `scripts\test-loop.ps1` jeśli środowisko stoi. Nie odpalaj mikrofonu ani akcji ekranowych.
5. Dopiero gdy ruff + pytest przechodzą, zrób **osobne commity** na `master` (nie force, nie rebase, nie push — nie ma remote):
   - głos
   - pamięć
   - akcje pod kursorem / allowlista
   - ewentualnie glue (app/settings/docs)
6. Routing V2 i `listener/voiceloop/routing/` + `tests/test_routing_*.py`: albo osobny commit „shadow routing v2”, albo zostaw untracked jeśli testy V2 padają. Nie włączaj wykonania V2.
7. `.cursor/rules/model-routing.mdc` — osobny drobny commit albo zostaw staged, nie mieszaj z głosem.

## Raport na start

`ROUTING: [model] | poziom L2 | koszt: średni | powód: hygiene gita i podział trzech wdrożeń, bez medycyny i bez zmiany uprawnień`

Jeśli wejdzie w politykę ryzyka, allowlistę albo dane — stop i eskaluj, nie naprawiaj po cichu.

## Gotowe gdy

- `git status` pokazuje albo czyste drzewo, albo tylko pliki medyczne / świadomie zostawione untracked
- trzy (albo cztery) czytelne commity, każdy o jednym stosie
- ruff + pełny pytest zielone
- brak `=` w repo
- brak mydr/ZnanyLekarz w tych commitach
- handoff nie kłamie względem kodu, który commitujesz
- na końcu wypisz: hash każdego commita, lista plików, co zostawiłeś brudne i dlaczego
