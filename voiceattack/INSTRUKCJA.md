# VoiceAttack Profile v2

VoiceAttack pełni rolę niezawodnego przycisku głosowego i warstwy awaryjnej.
Rozpoznaje krótkie, z góry zdefiniowane polskie frazy. Swobodną wypowiedź po
komendzie **„Asystent”** rozpoznaje Deepgram, nie wildcard VoiceAttack.

Routing komend jest automatyczny:

- poza trybem rozmowy znane komendy lecą standardowo przez `CommandId`,
- nieznane komendy przełączają się na `listening/once` (Deepgram),
- w trybie rozmowy (aktywny nasłuch) komendy VoiceAttack są mapowane na tekst i
  zawsze idą ścieżką tekstową.

Gotowy profil:

`C:\Users\marci\VoiceLoop\voiceattack\VoiceLoop-v2.vap`

Profil zawiera 24 komendy i korzysta wyłącznie z lokalnego API VoiceLoop na
`127.0.0.1:8765`.

## Jak dokładnie działa „Asystent”

Rozmowa jest dwustopniowa:

1. Powiedz **„Asystent”** albo **„Hej asystent”**.
2. VoiceAttack uruchomi `scripts\va\assistant.vbs`.
3. Skrypt wywoła lokalny endpoint `POST /api/v1/listening/once`.
4. VoiceLoop powie **„Słucham”**.
5. Dopiero teraz wypowiedz pełne polecenie, na przykład:
   **„Podsumuj, czym zajmowałem się rano i przygotuj krótką odpowiedź”**.
6. Deepgram rozpozna jedną polską wypowiedź i automatycznie zamknie nasłuch.
   Jeśli nic nie powiesz, tryb wyłączy się po 30 sekundach.
7. Rdzeń zapisze żądanie, sprawdzi n8n i router deterministyczny, pobierze lokalny
   kontekst z pamięci wektorowej, a dopiero dla wolnego języka użyje Venice.
8. Venice zwróci wyłącznie ustrukturyzowany plan z dozwolonymi `action_id`.
   Gdy Venice jest niedostępny, plan może przejąć lokalny Qwen.
9. Executor sprawdzi ryzyko i ewentualne potwierdzenie. VoiceLoop wypowie
   odpowiedź lub końcowy wynik akcji.

Schemat:

```text
„Asystent”
  → VoiceAttack (pewna, stała fraza)
  → lokalne /listening/once
  → „Słucham”
  → następna wypowiedź
  → Deepgram STT
  → n8n → router deterministyczny → lokalny retrieval
  → Venice primary / Qwen fallback
  → allowlista akcji + potwierdzenie
  → executor → odpowiedź głosowa
```

Dwustopniowy tryb jest celowy. Polski Deepgram znacznie lepiej rozpoznaje dowolne
zdania niż wildcard systemowego silnika VoiceAttack. Stałe polecenia, takie jak
„otwórz kalendarz”, wypowiadaj bez poprzedzania ich słowem „Asystent”.

## Pakiet komend (20) z parafrazami

| Komenda główna | Przykładowe parafrazy | Działanie |
|---|---|---|
| `Asystent` | `Hej asystent`, `Słuchaj asystencie`, `Mam polecenie`, skrót: `Asys` | Otwiera jednorazowy nasłuch dowolnego polecenia. |
| `Zapisz notatkę` | `Nowa notatka`, `Utwórz notatkę`, `Zanotuj to`, skrót: `Notka` | Pyta o treść i zapisuje ją kontrolowaną akcją UI.Vision. |
| `Zapamiętaj` | `Zapamiętaj to`, `Pamiętaj to`, `Zapisz to w pamięci`, skrót: `Pamiętaj` | Pyta o fakt i przygotowuje lokalny zapis wymagający zgody. |
| `Co robiłem ostatnio` | `Co się działo na ekranie`, `Pokaż historię ekranu`, `Podsumuj aktywność`, skrót: `Aktywność` | Czyta lokalne podsumowanie aktywności Screenpipe. |
| `Opisz aktywne okno` | `Co mam otwarte`, `Jakie okno jest aktywne`, `Na jakim oknie jestem`, skrót: `Aktywne okno` | Lokalnie odczytuje aktywne okno i program. |
| `Sprawdź pole tekstowe` | `Gdzie teraz piszę`, `Czy to pasek adresu`, `Gdzie trafi tekst` | Sprawdza, czy wpisywanie trafi do właściwego pola, a nie np. paska adresu. |
| `Zminimalizuj okno` | `Schowaj okno`, `Ukryj okno`, `Zwiń aktywne okno`, skrót: `Zwiń` | Minimalizuje aktualnie aktywne okno. |
| `Zminimalizuj wszystkie` | `Pokaż pulpit`, `Minimalizuj wszystko`, `Zwiń wszystkie okna`, skrót: `Pulpit` | Minimalizuje wszystkie okna i pokazuje pulpit. |
| `Kopiuj zaznaczony tekst` | `Skopiuj zaznaczony tekst`, `Kopiuj zaznaczony fragment`, skrót: `Kopiuj zaznaczenie` | Kopiuje aktualnie zaznaczony tekst do schowka. |
| `Kopiuj numer pod kursorem` | `Skopiuj numer pod kursorem`, `Kopiuj liczbę pod kursorem`, `Skopiuj telefon pod kursorem`, skrót: `Kopiuj numer` | Wykrywa i kopiuje numer pod kursorem. |
| `Kopiuj całe zdanie pod kursorem` | `Skopiuj całe zdanie pod kursorem`, `Kopiuj tekst pod kursorem`, `Skopiuj zdanie pod myszką`, skrót: `Kopiuj zdanie` | Wykrywa i kopiuje całe zdanie pod kursorem. |
| `Włącz nasłuch` | `Zacznij nasłuch`, `Start nasłuchu`, `Słuchaj ciągle`, skrót: `Nasłuch on` | Uruchamia ciągły nasłuch Deepgram. |
| `Wyłącz nasłuch` | `Zatrzymaj nasłuch`, `Stop nasłuchu`, `Przestań słuchać`, skrót: `Nasłuch off` | Wyłącza tylko Deepgram. |
| `Status Voice Loop` | `Czy działasz`, `Jaki jest status`, `Podaj status systemu`, skrót: `Status` | Czyta stan rdzenia, modelu i nasłuchu. |
| `Potwierdź` | `Tak potwierdzam`, `Wykonaj to`, `Zatwierdź`, skrót: `Potwierdź` | Potwierdza najnowszą akcję oczekującą na zgodę. |
| `Anuluj zadanie` | `Nie rób tego`, `Przerwij to`, `Odrzuć polecenie`, skrót: `Anuluj` | Odrzuca najnowszą akcję oczekującą na zgodę. |
| `Test pętli` | `Test głosu`, `Test VoiceLoop`, `Sprawdź głos`, skrót: `Test` | Sprawdza lokalną pętlę API i TTS. |
| `Otwórz kalendarz` | `Uruchom kalendarz`, `Pokaż kalendarz`, `Włącz kalendarz`, skrót: `Kalendarz` | Uruchamia dozwoloną akcję kalendarza. |
| `Otwórz przeglądarkę` | `Uruchom przeglądarkę`, `Otwórz internet`, `Odpal przeglądarkę`, skrót: `Przeglądarka` | Uruchamia dozwoloną akcję przeglądarki. |
| `Otwórz czat` | `Open chat`, `Otwórz ChatGPT`, `Nowy chat GPT`, skrót: `ChatGPT` | Uruchamia dozwoloną akcję czatu. |
| `Otwórz GPT` | `Open GPT`, `Otwórz chat GPT`, `Uruchom GPT` | Otwiera osobny czat GPT. |
| `Otwórz Gemini` | `Open Gemini`, `Czat Gemini`, `Uruchom Gemini` | Otwiera osobny czat Gemini. |
| `Wyszukaj w internecie` | `Sprawdź w necie`, `Szukaj online` | Uruchamia szybkie wyszukiwanie internetowe (potem doprecyzuj zapytanie). |
| `Stop teraz` | `Przerwij wszystko`, `Awaryjnie stop`, `Panic stop`, skrót: `Stop` | Natychmiast zatrzymuje nasłuch, TTS, kolejkę i bieżącą akcję. |

### Ręczne uruchomienie skryptów minimalizacji

- `Uruchom: C:\Users\marci\VoiceLoop\scripts\va\minimize-window.vbs`
- `Uruchom: C:\Users\marci\VoiceLoop\scripts\va\minimize-all.vbs`

## Notatka i pamięć — różnica

### „Zapisz notatkę”

VoiceLoop odpowiada „Co zapisać w notatce?”, słucha jednej wypowiedzi, a następnie
tworzy notatkę. Treść nie musi przechodzić przez Venice, ponieważ lokalny router
rozpoznaje prefiks i buduje akcję `create_note`.

### „Zapamiętaj”

VoiceLoop odpowiada „Co mam zapamiętać?”, słucha faktu i tworzy akcję `remember`.
Nic nie zostanie zapisane od razu. Asystent poprosi o decyzję:

- **„Potwierdź”** — zapis do lokalnej pamięci;
- **„Anuluj zadanie”** — odrzucenie.

Zgoda wygasa po 5 minutach. To zabezpiecza pamięć przed przypadkowym zapisem
źle rozpoznanego zdania oraz przed wykonaniem starego polecenia po restarcie.

## Tryb jednorazowy a ciągły

- **„Asystent”**, **„Zapisz notatkę”** i **„Zapamiętaj”** otwierają mikrofon
  Deepgram tylko na jedną wypowiedź.
- **„Włącz nasłuch”** pozostawia Deepgram aktywny i każde zakończone zdanie
  przesyła jako nowe polecenie.
- **„Wyłącz nasłuch”** zatrzymuje tylko Deepgram.
- **„Stop teraz”** jest panic buttonem i zatrzymuje także akcje oraz TTS.

## Instalacja profilu

1. Uruchom cały stos przez `scripts\start-all.ps1`.
2. Otwórz VoiceAttack.
3. Wybierz **More Actions → Import Profile**.
4. Wskaż `C:\Users\marci\VoiceLoop\voiceattack\VoiceLoop-v2.vap`.
5. Wybierz na liście profil **VoiceLoop v2**.
6. Upewnij się, że VoiceAttack słucha i ma ustawiony właściwy mikrofon.
7. Powiedz **„Test pętli”**, a następnie **„Asystent”**.

Stary profil `VoiceLoop` może pozostać jako archiwum, ale aktywny powinien być
tylko `VoiceLoop v2`.

## Test przyjęcia

1. **„Test pętli”** → „Pętla VoiceLoop działa”.
2. **„Status Voice Loop”** → krótki komunikat o modelu i Deepgramie.
3. **„Asystent”** → „Słucham” → powiedz „Jaki model jest teraz główny?”.
4. **„Opisz aktywne okno”** → asystent czyta rzeczywisty tytuł okna.
5. **„Zapamiętaj”** → podaj fakt → **„Potwierdź”** → „Zapamiętano”.
6. **„Stop teraz”** → aktywne zadania i głos zostają przerwane.

## Bezpieczeństwo

- Skrypty `.vbs` nie wykonują treści wypowiedzi jako PowerShell ani CMD.
- Wszystkie żądania wymagają lokalnego tokenu `X-VoiceLoop-Token`.
- STOP idzie bezpośrednio do rdzenia i omija n8n oraz modele.
- Screenpipe, aktywne okno i Qdrant są przetwarzane lokalnie.
- Do Venice trafia tekst wolnego polecenia i wybrany kontekst, nie dowolny plik.
- Model może wskazać tylko akcje z allowlisty.
- Akcje wysokiego ryzyka oraz zapis pamięci wymagają potwierdzenia.

Profil można odtworzyć po zmianie ścieżki projektu:

```powershell
.\listener\.venv\Scripts\python.exe .\scripts\build-voiceattack-profile.py
```
