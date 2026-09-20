# Context Timeline V1

Context Timeline jest lokalną, niewykonywalną warstwą dowodową VoiceLoop.
Czas jest osią główną; nazwy aplikacji, okien i stan foreground są metadanymi,
a embeddingi dotyczą znaczenia epizodów.

## Status

- Schematy `ContextEventV1`, `ContextEpisodeV1`, `ContextItemV1` i
  `ContextPackV1` są zaimplementowane.
- SQLite przechowuje `context_events` i `context_episodes`; obie warstwy mają
  lokalny indeks FTS5.
- `ScreenpipeClient.text_activity_between()` obsługuje zakres czasu,
  zapytanie tekstowe i paginację.
- `ScreenpipeTimelineIngestor` jest jawnie wywoływanym adapterem. Nie ma pętli
  background i nie uruchamia się podczas startu aplikacji.
- `CONTEXT_TIMELINE_RECALL_ENABLED=false` pozostaje domyślne. Włączenie zmienia
  tylko akcję `recall` z jawnym zakresem czasu; nie zmienia routingu ani polityki
  wykonania.

## Przepływ

```text
Screenpipe / inne lokalne źródła
  -> context_events (czas, źródło, okno, tekst, provenance)
  -> context_episodes (10-minutowy deterministyczny fallback)
  -> SQLite FTS5
  -> TimeFirstRetriever
  -> ContextPackV1
  -> memories: list[str] jako kompatybilny widok dla planera
```

`ContextPackV1` jest typowany, ale `TurnContext.memories` pozostaje listą
stringów. Zachowuje to invariant, że pamięć jest danymi kontekstowymi, a nie
źródłem `action_id`.

## Foreground

Screenpipe nie dostarcza obecnie wiarygodnego `is_foreground` ani HWND.
Zdarzenia Screenpipe zapisują więc `foreground_confidence=unknown`.
Znany foreground będzie wymagał osobnego samplera Win32/UIA. Kod odrzuca
ustawienie `is_foreground`, jeżeli confidence nadal jest `unknown`.

## Encje

`ContextEntityResolver` wykorzystuje wyłącznie zatwierdzone wpisy istniejącego
`ProperNameLexiconV1`. Stabilne `entity_id` nie zawiera widocznej nazwy.
Niepewne nazwy i ekstrakcje modelu nie są automatycznie scalane ani awansowane
do kanonicznej osoby.

## Retencja

`prune_expired_context_records()` jest dry-run domyślnie. Jawne
`dry_run=False` oznacza tombstone oraz usunięcie projekcji FTS; rekord
kanoniczny pozostaje do audytu. Automatyczne kasowanie i kaskada Qdrant nie są
jeszcze uruchomione.

## Ograniczenia V1

- Brak automatycznego ingestu.
- Brak embeddingów epizodów i osi rezerwowych w nowym retrieverze.
- Brak samplera foreground.
- Brak automatycznej akceptacji encji osób.
- `TimeFirstRetriever` używa FTS i źródłowego Screenpipe; semantic scout zostaje
  kolejnym, mierzalnym etapem.
