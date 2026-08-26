# Lokalne narzędzia zbierania próbek głosu

Te dwa małe panele pomagają nagrywać wyłącznie własny głos do ręcznej oceny i
lokalnego korpusu VoiceLoop:

- `holding-commands` — stały zestaw krótkich, często używanych poleceń;
- `calibration-phrases` — zróżnicowane polskie frazy: pytania, korekty,
  anulowanie, nazwy własne i zadania wieloetapowe.

## Uruchomienie

Z katalogu głównego projektu:

```powershell
python .\scripts\holding-commands\server.py
python .\scripts\calibration-phrases\server.py
```

Panele są dostępne odpowiednio na:

```text
http://127.0.0.1:8791/
http://127.0.0.1:8792/
```

Port i katalog danych można zmienić bez edycji kodu:

```powershell
python .\scripts\holding-commands\server.py --port 8891 --data-root D:\VoiceLoopData
```

Można również ustawić `VOICELOOP_DATA_DIR`. Każde narzędzie tworzy w tym
katalogu własny podkatalog.

## Granice prywatności

- Serwery nasłuchują wyłącznie na `127.0.0.1`.
- Nagrania i `session.json` trafiają do `data/`, które jest ignorowane przez Git.
- Metadane frazy są przesyłane w nagłówku, nie w adresie URL ani logu żądania.
- Narzędzie nie wysyła audio do Deepgram, Hume ani innej usługi chmurowej.
- Automatyczna ocena sprawdza tylko podstawowe parametry pliku. Nie zastępuje
  odsłuchu, ręcznej adnotacji ani świadomego zatwierdzenia mówcy.

Jednorazowy `scripts/staging-voice-eval/prepare.py` pozostaje narzędziem lokalnym
i nie jest częścią portfolio ani produkcyjnego pipeline’u korpusu.
