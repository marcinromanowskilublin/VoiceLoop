# Bezpieczny korpus użytkownika

Pipeline działa offline i zapisuje artefakty wyłącznie w ignorowanym przez Git
`data/corpus`. Surowe transkrypcje pozostają w lokalizacjach źródłowych.

## Uruchomienie

Z katalogu `listener`:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus inventory
.\.venv\Scripts\python.exe -m voiceloop.corpus run
.\.venv\Scripts\python.exe -m voiceloop.corpus evaluate-routing
.\.venv\Scripts\python.exe -m voiceloop.corpus evaluate-routing-v2
```

Komenda `inventory` zapisuje tylko ścieżki, rozmiary, liczby słów, daty i hashe.
`run` tworzy zredagowany zbiór pochodny, metadane kwarantanny bez tekstu,
podział sesjami, zestaw ewaluacyjny, agregat stylu i kolejkę kandydatów pamięci.
Benchmark routingu korzysta z ręcznego holdoutu nieobecnego w
`routing_examples`; raport zawiera jawne `quality_gate_passed` i przyczyny
niezaliczenia. `evaluate-routing` zwraca kod `3`, jeśli bramka nie przejdzie,
więc nieudany eksperyment nie włącza się sam do runtime.

`evaluate-routing-v2` odtwarza aktualny ręczny holdout i sprawdza dodatkowo
segmentację, dokładną sekwencję kroków, argumenty, resolver, margines top-2 oraz
bezpieczną abstencję. Przypadki negatywne obejmują polecenie bez separatora,
niejednoznaczny spójnik, wielu mówców i próbę zamknięcia okna po nazwie.
Raport jest zapisywany do
`data/corpus/eval/routing-v2-metrics.json`. Routing V2 działa domyślnie w
`shadow mode`: publikuje porównanie V1/V2, ale nie przekazuje planu V2 do
executora.

Wykonanie V2 jest możliwe dopiero po spełnieniu wszystkich warunków:

1. raport ma `quality_gate_passed=true`;
2. `catalog_coverage=1.0` i liczba akcji obejmuje aktualny katalog;
3. `catalog_hash` odpowiada bieżącemu indeksowi, a raport ma pełny schemat V2;
4. `runtime_fingerprint` odpowiada aktualnej wersji implementacji, progom,
   limitowi top-k, modelowi i wymiarowi embeddingów, kolekcji Qdrant oraz
   metryce cosine;
5. lokalnie ustawiono `ROUTING_V2_SHADOW_MODE=false`;
6. lokalnie ustawiono `ROUTING_V2_EXECUTE=true`.

Brak raportu, niepełne pokrycie, zmiana katalogu albo konfiguracji runtime
wymuszają powrót do `shadow mode`. Nierozstrzygnięte lub niedostępne V2 w trybie
wykonawczym kończy się pytaniem doprecyzowującym, bez przejścia do legacy/LLM.
Polecenie złożone nadal jest blokowane przed starym fast pathem, więc nie może
wykonać tylko pierwszego rozpoznanego fragmentu.

## Mówca i kwarantanna

Wiadomości `role=user` z głównych transkryptów Cursora są oznaczane jako własne.
Audio bez jawnej decyzji pozostaje w kwarantannie. Numery mówców z diarizacji nie
są traktowane jako weryfikacja tożsamości.

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus source-status
.\.venv\Scripts\python.exe -m voiceloop.corpus approve-audio-source SOURCE_ID `
  --confirm SOURCE_ID
.\.venv\Scripts\python.exe -m voiceloop.corpus revoke-audio-source SOURCE_ID
```

Decyzja jest związana z hashem pliku. Zmiana źródła powoduje nowy `source_id`
i ponownie wymaga zatwierdzenia.

## Profil stylu

Profil zawiera tylko agregaty, bez cytatów i embeddingów wypowiedzi. Powstaje na
zbiorze treningowym, a osobny raport holdout musi przejść bramkę jakości. Profil
jest domyślnie wyłączony. Do użycia runtime wymagane są równocześnie:

1. `enabled=true` w lokalnym `profile-v1.json`;
2. `passes_quality_gate=true` w `holdout-report-v1.json`;
3. `CORPUS_STYLE_PROFILE_ENABLED=true`.

Instrukcja stylu jest przekazywana wyłącznie do LM Studio na adresie loopback.

## Kandydaci pamięci

Pipeline nigdy nie zapisuje kandydatów do tabeli `memories`. Przegląd lokalny:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus list-candidates
.\.venv\Scripts\python.exe -m voiceloop.corpus show-candidate CANDIDATE_ID
.\.venv\Scripts\python.exe -m voiceloop.corpus approve CANDIDATE_ID `
  --confirm CANDIDATE_ID --content-sha256 HASH_POKAZANEJ_TREŚCI
.\.venv\Scripts\python.exe -m voiceloop.corpus reject CANDIDATE_ID
```

Chronione tokenem API udostępnia odpowiedniki:

- `GET/POST /api/v1/memory-candidates`
- `POST /api/v1/memory-candidates/{id}/approve`
- `POST /api/v1/memory-candidates/{id}/reject`

Dla `approve` wymagane jest ciało `{"content_sha256":"..."}` odpowiadające
treści pokazanej podczas przeglądu. Identyfikator nie może zostać ponownie użyty
dla innej treści.

Dane medyczne osób trzecich, sekrety, identyfikatory wysokiego ryzyka i wnioski
psychologiczne są blokowane przed kolejką.

## Gwarancje lokalności

- Pipeline nie ma opcji `allow_cloud`.
- Ewaluacja odrzuca endpointy embeddings i Qdrant inne niż loopback.
- Gemini, Venice i provider `cloud` nie otrzymują historii, pamięci, ekranu,
  obrazów ani profilu stylu, niezależnie od tego, czy są modelem primary.
- Korpus nie jest zapisywany do `voiceloop_capabilities_v1`.
- Trening i holdout są rozdzielane całymi sesjami; duplikaty są wyłączane przed
  podziałem.

## Zamrożony zestaw głosowy V1

Zestaw głosowy jest kolejnym etapem tego samego pipeline'u korpusu. Nie tworzy
drugiej pamięci ani drugiej bazy wektorowej. Źródłem audio jest lokalne API
Screenpipe oraz, gdy API nie indeksuje nagrań bez transkrypcji, bezpośredni
odczyt metadanych plików wyłącznie z `~/.screenpipe/data`. Manifest nie zawiera
tekstu ani bajtów nagrania.

Kolejność:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus inventory-voice-eval `
  --start 2026-08-01 --end 2026-09-01

# Dopiero po sprawdzeniu urządzeń i potwierdzeniu, że to własny kanał użytkownika:
.\.venv\Scripts\python.exe -m voiceloop.corpus build-voice-candidates `
  --confirm SELF_AUDIO_ONLY

.\.venv\Scripts\python.exe -m voiceloop.corpus select-voice-eval `
  --target 120 --development-count 30
```

`build-voice-candidates`:

- dopuszcza tylko pliki zamknięte w `~/.screenpipe/data`;
- wybiera kanał wejściowy;
- dekoduje przez `ffmpeg` do mono PCM 16 kHz;
- wykrywa fragmenty mowy na podstawie energii i ciszy;
- zapisuje tylko wybrane lokalne klipy w `data/corpus/eval/voice-v1/audio`;
- zachowuje hash oryginału, hash klipu, urządzenie, kierunek i absolutny czas.

Po ręcznym uzupełnieniu szablonu
`data/corpus/eval/voice-v1/annotation-template-v1.jsonl` zatwierdzone rekordy
należy zapisać jako `annotations-v1.jsonl`.

Wygodniejszy lokalny formularz jest generowany równolegle jako
`data/corpus/eval/voice-v1/review-v1.html`. Odtwarza względne pliki audio,
trzyma wersję roboczą w `localStorage` i pobiera gotowy `annotations-v1.jsonl`;
nie wysyła formularza do serwera.
Są też osobne formularze `review-development-v1.html` i
`review-holdout-v1.html`, eksportujące odpowiednio
`annotations-development-v1.jsonl` oraz `annotations-holdout-v1.jsonl`.

Następnie trzeba uruchomić:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus validate-voice-eval
```

Walidator wymaga 120 unikalnych próbek, splitu 30/90, kompletnych adnotacji,
braku przecieku grup duplikatów oraz pokrycia pytań, intonacji, poleceń,
samokorekt, anulowań, barge-in, nazw własnych i trudniejszego audio.
Każda adnotacja wymaga też `speaker_role="self"` i
`speaker_confirmed=true`; samo urządzenie mikrofonowe nie dowodzi tożsamości
mówcy.
Raport ostrzega również, gdy ponad połowa próbek pochodzi z jednego dnia; takie
ostrzeżenie nie jest ukrywane przez sam fakt uzyskania 120 plików.

### Replay Deepgram

Wysłanie zamrożonego audio do Deepgram nigdy nie jest domyślne:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus transcribe-voice-eval `
  --split development --allow-remote --confirm DEEPGRAM_AUDIO_UPLOAD

# Holdout dopiero po zamrożeniu konfiguracji:
.\.venv\Scripts\python.exe -m voiceloop.corpus transcribe-voice-eval `
  --split holdout --allow-remote --confirm DEEPGRAM_AUDIO_UPLOAD `
  --holdout-confirm HOLDOUT_FINAL_EVALUATION
```

Odpowiedź zachowuje tekst, confidence, słowa, timestampy i speaker IDs.
Cache jest związany z hashem audio, modelem, językiem i parametrami. Ponowne
uruchomienie tej samej konfiguracji czyta wynik lokalnie.

### Tekst, prozodia i semantyka

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus evaluate-voice-eval `
  --split development

# Holdout uruchamia się dopiero raz po zamrożeniu konfiguracji:
.\.venv\Scripts\python.exe -m voiceloop.corpus evaluate-voice-eval `
  --split holdout --confirm HOLDOUT_FINAL_EVALUATION
```

Ewaluacja:

- mierzy WER, CER, F1 interpunkcji i trafność znaku zapytania;
- lokalnie oblicza F0, końcowy ruch tonu, energię, pauzy, ciszę i tempo;
- uruchamia Routing V2 bez executora i zapisuje top-k, margines oraz pełny plan;
- porównuje osobno wynik tekstowy, prozodyczny, semantyczny i połączony;
- zapisuje fingerprint konfiguracji oraz wynik do osobnego katalogu `runs`.

Wektory wypowiedzi są chwilowe. Nie trafiają do `voiceloop_memory`; są używane
wyłącznie do porównania z kontrolowanym katalogiem akcji.

### Słownik, niezawodność i dziennik

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus build-proper-names
.\.venv\Scripts\python.exe -m voiceloop.corpus report-actions
.\.venv\Scripts\python.exe -m voiceloop.corpus extract-project-journal
.\.venv\Scripts\python.exe -m voiceloop.corpus list-journal-candidates
```

Słownik zbiera nazwy i błędy STT, ale korekty działają tylko dla wpisów z
`approved=true`. Raport działań czyta SQLite w trybie read-only i domyślnie nie
publikuje tekstów poleceń. Dziennik przyjmuje wyłącznie jawne wypowiedzi
użytkownika, odrzuca pytania jako trwałe decyzje i wymaga osobnego polecenia
`approve-journal ... --confirm ...`.
