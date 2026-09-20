# VoiceLoop — dokumentacja programu

Wersja: 0.2.0  
Platforma: Windows (lokalne uruchomienie)  
Repozytorium: `C:\Users\marci\VoiceLoop`

## 1. Czym jest VoiceLoop

VoiceLoop to lokalny asystent Windows sterowany po polsku (głos + tekst), który:

- przyjmuje polecenia z panelu, mikrofonu, VoiceAttack i lokalnego API,
- rozdziela rozmowę od zadań operacyjnych,
- wykonuje tylko dozwolone akcje z lokalnej allowlisty,
- ma pamięć lokalną i kontekst aktywności użytkownika,
- wymusza potwierdzenia dla operacji ryzykownych.

Kluczowa zasada: model AI nie wykonuje dowolnych komend systemowych; zwraca tylko
typowany plan (`action_id` + argumenty), który jest lokalnie walidowany.

## 2. Budowa systemu

## 2.1 Warstwa wejścia

- `panel/index.html` — panel WWW (`http://127.0.0.1:8765`)
- Deepgram live (PL) — transkrypcja mowy
- VoiceAttack — stałe komendy i wake-word
- lokalne REST API — integracje i automatyzacje

## 2.2 Warstwa rdzenia

- `listener/voiceloop/app.py` — FastAPI, endpointy, health
- `listener/voiceloop/assistant.py` — główny orkiestrator decyzji
- `listener/voiceloop/router.py` — deterministyczny fast-path
- `listener/voiceloop/model_router.py` — routing do modeli
- `listener/voiceloop/actions.py` — katalog i polityka akcji
- `listener/voiceloop/executor.py` — kolejka wykonawcza (single-flight)
- `listener/voiceloop/models.py` — kontrakty danych i planów

## 2.3 Warstwa AI i pamięci

- Gemini / Venice / lokalny Qwen (LM Studio) — planowanie i rozmowa
- Screenpipe — lokalny kontekst aktywności
- Qdrant — pamięć wektorowa (named vectors)
- SQLite — stan operacyjny, historia i fallback
- corpus/routing V2 — ewaluacja jakości i quality gate

## 2.4 Warstwa wykonawcza

- Windows API / UI Automation
- UI.Vision (makra z allowlisty)
- VoiceAttack wrappers (`scripts/va/*.vbs`)
- TTS: Azure Speech SDK → Azure REST → Windows TTS (fallback)

## 3. Jak działa program (end-to-end)

1. Użytkownik wydaje polecenie (głos/tekst/API).  
2. Polecenie trafia do `CommandRequest`, jest deduplikowane i logowane.  
3. Priorytetowo sprawdzany jest STOP oraz komendy ochronne.  
4. Routing:
   - najpierw reguły deterministyczne,
   - potem model + kontekst pamięci, jeśli potrzeba.
5. Powstaje typowany plan (`CommandPlan`) z krokami.  
6. Plan przechodzi walidację argumentów i politykę ryzyka.  
7. Kroki wykonują się sekwencyjnie przez executor.  
8. Wynik wraca do panelu/API/SSE i opcjonalnie TTS.

## 4. Bezpieczeństwo i niezawodność

Najważniejsze mechanizmy:

- usługi związane z kontrolą systemu działają na loopback,
- brak wykonywania dowolnego shell z odpowiedzi modelu,
- lokalna allowlista akcji i walidowane `args_schema`,
- potwierdzenia dla operacji średniego/wysokiego ryzyka,
- deduplikacja requestów i natychmiastowy STOP,
- quality gate dla Routing V2 przed wykonaniem live.

Granice:

- projekt jest local-first, nie jest gotowy do publicznej ekspozycji portów,
- sekretów (`.env`, tokeny, dane runtime) nie wolno versionować.

## 5. Możliwości wdrożenia

## 5.1 Wdrożenie lokalne (zalecane)

Scenariusz: jedna stacja robocza użytkownika.

- pełna funkcjonalność głosu, pamięci i akcji,
- najmniejsze ryzyko wycieku danych,
- najprostszy model utrzymania i diagnostyki.

## 5.2 Wdrożenie etapowe (stabilizacja)

Scenariusz: zmiany wdrażane warstwowo.

1. `shadow/observe` (brak live execution nowej logiki),
2. canary na bezpiecznych akcjach LOW,
3. pełne wykonanie dopiero po PASS quality gate.

## 5.3 Wdrożenie zespołowe (wewnętrzne)

Scenariusz: kilka stacji, wspólne standardy.

- ten sam kod i checklisty quality gate,
- lokalne sekrety per stacja (`listener/.env`),
- centralnie współdzielona dokumentacja i test plan.

## 6. Możliwości rozwijania

## 6.1 Rozwój funkcjonalny

- rozszerzanie katalogu akcji (`ActionSpec`) o nowe bezpieczne operacje,
- lepsze rozumienie parafraz i skrótów mowy PL,
- rozwój trybu rozmowy i zarządzania sesją.

## 6.2 Rozwój pamięci i kontekstu

- lepsze rankowanie kontekstu i scoring trafień,
- rozszerzenie jakości digestów ze Screenpipe,
- lepsze reguły retencji i profilowania pamięci użytkownika.

## 6.3 Rozwój jakości i testów

- większy korpus eval (routing, głos, argumenty),
- rozszerzanie testów regresji dla akcji i bezpieczeństwa,
- ciągłe monitorowanie quality gate/fingerprint runtime.

## 6.4 Rozwój operacyjny

- dokładniejsze healthchecki komponentów,
- standard runbooków incydentowych,
- automatyzacja restartu/diagnozy lokalnych usług.

## 7. Rekomendowany porządek dalszych prac

1. Utrzymać higienę drzewa (`staged=0`, tematyczne commity).  
2. Domknąć paczki zmian: `głos`, `pamięć`, `akcje`, `routing`.  
3. Każdą paczkę przepuszczać przez `ruff + pytest`.  
4. Trzymać dokumentację zgodną z realnym stanem runtime (`/api/v1/health`).

## 8. Gdzie czytać dalej

- architektura i handoff: `docs/VOICELOOP_ARCHITECTURE_HANDOFF.md`
- korpus i ewaluacja: `docs/SAFE_USER_CORPUS.md`
- higiena drzewa i ostatni audyt: `docs/TREE_HYGIENE_AND_RUNTIME_AUDIT.md`
