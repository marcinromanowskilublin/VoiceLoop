# Handoff: tor Notatnika (etap naprawczy)

## Zakres
Naprawiono odczyt/zapis aktywnego Notatnika bez myszy, rozróżnienie zaznaczenia i schowka, głosowe potwierdzenie/anulowanie jednej oczekującej operacji, weryfikację odczytem po zapisie oraz wąski Router V1. Nie ma redagowania treści przez model.

Nowe akcje: `read_active_notepad`, `write_active_notepad`.
Flaga sesji: `NOTEPAD_VOICE_LANE=true` (V1, allowlista tych dwóch akcji, bez wykonywania V2).

## Uruchomienie
W `listener/.env` dodaj tylko:

```
NOTEPAD_VOICE_LANE=true
```

Nie zmieniaj polityki prywatności. Nie uruchamiaj `start-all`.

```
cd listener
.venv\Scripts\python -m uvicorn voiceloop.app:app --host 127.0.0.1 --port 8765
```

Panel: `http://127.0.0.1:8765/`. Sesja głosowa z panelu. Przygotuj własny plik `VoiceLoop_demo_notatka.txt` z fikcyjną treścią i zostaw go jako jedyne okno Notatnika.

Polecenia: `odczytaj notatkę` · `wpisz w notatniku: …` · `potwierdzam` / `anuluj zadanie` · `stop`.

## Testy
- Baseline przed zmianą: 845 passed.
- Po zmianie: **860 passed**, ruff i invariants zielone.
- Live UIA na własnym Notatniku testowym: **zaliczone ręcznie** (to nie jest test STT/mikrofonu).
- Domyślny pytest nie uruchamia Notatnika. Live test wymaga jawnego:
  `VOICELOOP_RUN_WINDOWS_LIVE_TESTS=1`.
- Lokalnie nie działały porty 8765, 1234, 6333, 3030. `ROUTING_V2_EXECUTE=true` nadal rozjeżdża się z bramką V2; tor demo jej nie obchodzi.

## Ograniczenia
- Tylko Notatnik, jedna karta/kontrolka. Inny fokus albo kilka edytorów = odmowa.
- Dyktowana treść jest dosłowna. Tekst dokumentu nie uruchamia akcji.
- STOP/anulowanie nie przerywa już trwającego wywołania Windows i nie cofa zapisu.
- STT Deepgram i TTS nie były sprawdzane Twoim mikrofonem.

## Pięć prób z Twoim udziałem
1. Jedna otwarta notatka testowa. Powiedz: `odczytaj notatkę`. Usłysz treść, bez myszy.
2. `wpisz w notatniku: VoiceLoop demo jeden` → pytanie o zgodę → `potwierdzam`. Tekst w Notatniku ma być dokładnie ten, asystent ma potwierdzić po odczycie.
3. To samo wpisanie, potem `anuluj zadanie`. Notatnik bez zmiany.
4. Przygotuj propozycję, przełącz okno na inną aplikację, powiedz `potwierdzam`. Ma odmówić.
5. Przy `NOTEPAD_VOICE_LANE=true` powiedz `otwórz przeglądarkę` (odmowa zakresu), potem `stop` w trakcie pytania o zgodę.
