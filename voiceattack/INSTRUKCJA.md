# VoiceAttack — konfiguracja (5–10 minut)

VoiceAttack to w tej pętli **refleks**: krótkie, sztywne komendy działające zawsze,
nawet gdy nasłuch Deepgram jest wyłączony. Wysyła je do lokalnego rdzenia VoiceLoop,
a STOP omija n8n.

> **Stan licencji.** VoiceAttack jest zarejestrowany (pełna wersja). Import profilu
> i tworzenie nowych profili są dostępne. Klucz licencji trzymamy lokalnie w
> `listener/.env` jako `VOICEATTACK_REGISTRATION_KEY` oraz w konfiguracji VoiceAttack.

## Krok 1 — pełna wersja: import gotowego profilu

1. Uruchom VoiceAttack.
2. Kliknij **More Actions → Import Profile**.
3. Wybierz `C:\Users\marci\VoiceLoop\voiceattack\VoiceLoop-profil.vap`.
4. Jeśli profil "VoiceLoop" pojawił się na liście i ma 6 komend — gotowe, przejdź do kroku 3.

## Krok 2 — wersja próbna: konfiguracja ręczna

1. VoiceAttack → **Edit Profile** dla istniejącego profilu domyślnego.
2. Dla każdej pozycji z tabeli: **New Command**:
   - w polu **When I say** wpisz frazę,
   - kliknij **Other** → **Windows** → **Run an application**,
   - w **Application** wskaż plik `.vbs` z tabeli,
   - zatwierdź (OK → Done).

| When I say | Application (w `C:\Users\marci\VoiceLoop\scripts\va\`) |
|---|---|
| `voice test` | `voice-test.vbs` |
| `open calendar` | `open-calendar.vbs` |
| `open browser` | `open-browser.vbs` |
| `open chat` | `open-chat.vbs` |
| `take a note;new note` | `note.vbs` |
| `stop now;abort` | `stop.vbs` |

(Średnik w "When I say" = alternatywne frazy tej samej komendy.)

## Krok 3 — mikrofon i test

1. Ustawienia VoiceAttack (ikona klucza) → zakładka **Audio** → wybierz swój mikrofon.
2. Upewnij się, że VA "słucha" (ikona mikrofonu nie jest przekreślona).
3. Powiedz: **"voice test"** → VoiceLoop odpowie głosem, że pętla działa.
4. Powiedz: **"open calendar"** → otworzy się kalendarz Windows.

Jeśli rozpoznawanie działa słabo, sprawdź wybrany polski silnik mowy oraz trening
głosu w ustawieniach Windows.

## Jak to działa pod spodem

```
"open calendar" → VoiceAttack → scripts\va\open-calendar.vbs (cicho, bez okna)
    → scripts\send-command.ps1
    → POST http://127.0.0.1:8765/api/v1/commands
    → n8n/LM Studio → bezpieczny executor
```

Nowa komenda głosowa wymaga frazy w VoiceAttack i dozwolonego `action_id` w
lokalnym rejestrze VoiceLoop. Nie dodawaj dowolnych poleceń shell.
