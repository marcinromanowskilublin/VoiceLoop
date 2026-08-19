# VoiceLoop — handoff dla Mikołaja

Ten dokument jest krótką instrukcją przekazania projektu po dużym wdrożeniu
rozmowy głosowej, telemetrii i warstwy wiedzy.

## 1) Co zostało dowiezione

- Jedna, zarządzana ścieżka rozmowy głosowej z kontrolą barge-in i pauzy.
- Telemetria tury rozmowy (`conversation.trace`) oraz endpointy:
  - `GET /api/v1/conversation/traces`
  - `GET /api/v1/conversation/quality`
- Rozszerzony routing i kontekst rozmowy (historia, pamięć, źródła web).
- Orkiestracja wiedzy i źródeł (`knowledge_tools`, `ToolObservation`).
- Ujednolicony katalog możliwości oraz lepsze odpowiedzi typu
  "czy umiesz..." / "jak powiedzieć...".
- Aktualizacje panelu (`panel/index.html`) dla metryk i źródeł.
- Rozszerzony zestaw testów regresyjnych.

## 2) Stan spójności na moment przekazania

- `530 passed` w pełnym `pytest` (lokalne uruchomienie na tym drzewie).
- `compileall` dla `listener/voiceloop` bez błędów.
- `git diff --check` bez błędów whitespace.

## 3) Jak uruchomić lokalnie

1. Przygotuj plik `listener/.env` na bazie `listener/.env.example`.
2. Upewnij się, że działają usługi zależne:
   - LM Studio (`127.0.0.1:1234`)
   - Qdrant (`127.0.0.1:6333`)
   - Screenpipe (`127.0.0.1:3030`) — opcjonalnie, ale zalecane
3. Uruchom listener:

```powershell
$env:PYTHONPATH='listener'
listener/.venv/Scripts/python.exe -m uvicorn voiceloop.app:app --host 127.0.0.1 --port 8765
```

4. Sprawdź health:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/v1/health"
```

## 4) Checklista odbioru (15 min)

1. Otwórz panel `http://127.0.0.1:8765/`.
2. Zrób jedną turę rozmowy i potwierdź, że pojawia się `conversation.trace`.
3. Zadaj pytanie "aktualne" (np. o wersję) i sprawdź sekcję źródeł.
4. Przetestuj przerwanie (`stop`/`pauza`) podczas TTS.
5. Uruchom pełne testy:

```powershell
$env:PYTHONPATH='listener'
listener/.venv/Scripts/python.exe -m pytest -q
```

## 5) Pliki, które Mikołaj powinien przeczytać najpierw

- `README.md`
- `docs/VOICELOOP_ARCHITECTURE_HANDOFF.md`
- `listener/voiceloop/voice_conversation.py`
- `listener/voiceloop/assistant.py`
- `listener/voiceloop/conversation_telemetry.py`
- `listener/voiceloop/knowledge_tools.py`

## 6) Uwaga operacyjna

- Nie commitować `listener/.env` ani sekretów.
- Jeżeli provider web ma limit (np. 402), fallback jest skonfigurowany i
  raportowany w health.
- Projekt jest gotowy do przekazania jako snapshot kodu + testy.
