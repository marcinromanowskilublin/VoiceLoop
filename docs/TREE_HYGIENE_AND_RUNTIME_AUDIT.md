# Higiena drzewa i audyt działania

Data audytu: 2026-08-19
Repozytorium: `C:\Users\marci\VoiceLoop`

## Cel

Utrzymać bezpieczny, przewidywalny stan roboczy bez utraty lokalnych zmian:

- brak przypadkowo staged plików,
- brak śmieci i artefaktów w indexie,
- potwierdzenie, że program przechodzi lint i pełne testy.

## Zakres wykonanych działań

1. Usunięcie ewidentnych śmieci z repo:
   - plik `=` (0 B) został wcześniej zdjęty ze stage i usunięty.
2. Ochrona obszaru medycznego:
   - `sources/notes/integrations/MYDR_EDM_POLECENIA_GLOSOWE.md` został wcześniej zdjęty ze stage
     i pozostaje `untracked`.
3. Higiena indexu:
   - wykonano `git restore --staged .`,
   - wynik: `staged=0` (brak plików gotowych do przypadkowego commita).
4. Audyt działania aplikacji:
   - lint:
     - `listener\.venv\Scripts\python.exe -m ruff check voiceloop ..\tests ..\scripts\voice_capture_server.py ..\scripts\holding-commands\server.py ..\scripts\calibration-phrases\server.py`
   - testy:
     - `listener\.venv\Scripts\python.exe -m pytest -c pyproject.toml -q`

## Wyniki audytu

- Ruff: **PASS**
- Pytest: **PASS** według aktualnego przebiegu CI/lokalnego `pytest`
- Health API: **PASS** (`GET /api/v1/health`, HTTP 200)

Stan drzewa po audycie:

- `staged=0`
- `modified=46`
- `untracked=78`

To oznacza, że zmiany robocze nadal są obecne (celowo), ale index jest czysty i bezpieczny.

## Wdrożenie runtime (po audycie)

Po przejściu testów wykonano kontrolowany restart listenera na porcie `8765`, aby
działający proces załadował aktualny katalog akcji.

Komenda restartu:

- `listener\.venv\Scripts\python.exe -m uvicorn voiceloop.app:app --host 127.0.0.1 --port 8765`
  (z wcześniejszym zatrzymaniem procesu nasłuchującego na porcie `8765`)

Potwierdzenie po restarcie (`/api/v1/health`):

- `routing_v2=canary; quality gate: zaliczona; calibration: report_only`
- `capability_embeddings=voiceloop_capabilities_v2: 30 możliwości; katalog 57172dbd6cf51bfb`

## Wnioski operacyjne

1. Program jest sprawny na aktualnym stanie kodu (lint + pełny test suite przechodzi).
2. Drzewo jest "czyste logicznie" do dalszego porządkowania:
   - brak staged zmian,
   - mniejsze ryzyko przypadkowego commita.
3. Kolejny krok powinien być już wyłącznie merytoryczny:
   - podział na commity tematyczne (`głos`, `pamięć`, `akcje`, opcjonalnie `routing/glue`),
   - bez mieszania obszarów medycznych.

## Minimalna procedura przed każdym commitem

1. `git status --short`
2. dodać tylko wybrany stos zmian (`git add <zakres>`)
3. `ruff` + `pytest`
4. commit jednego tematu

## Czego nie robić

- nie commitować `listener/.env`, `data/`, `logs/`, sekretów,
- nie mieszać zmian medycznych z rdzeniem asystenta,
- nie robić "wszystko naraz" w jednym commicie.
