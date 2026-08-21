"""Local recorder for a stable holding set of VoiceLoop commands."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from voice_capture_server import (  # noqa: E402
    CaptureConfig,
    CapturePhrase,
    run_capture_server,
)

PHRASES = (
    CapturePhrase("otworz-chrome", "otwórz Chrome", "otwieranie", "aplikacje"),
    CapturePhrase("otworz-gemini", "otwórz Gemini", "otwieranie", "aplikacje"),
    CapturePhrase("otworz-youtube", "otwórz YouTube", "otwieranie", "aplikacje"),
    CapturePhrase("otworz-nowa-karte", "otwórz nową kartę", "okna", "przeglądarka"),
    CapturePhrase("zamknij-okno", "zamknij okno", "okna", "sterowanie oknem"),
    CapturePhrase("zamknij-karte", "zamknij kartę", "okna", "przeglądarka"),
    CapturePhrase(
        "zminimalizuj-okno",
        "zminimalizuj aktywne okno",
        "okna",
        "sterowanie oknem",
    ),
    CapturePhrase("pokaz-pulpit", "pokaż pulpit", "okna", "sterowanie oknem"),
    CapturePhrase("odswiez-strone", "odśwież stronę", "okna", "przeglądarka"),
    CapturePhrase("przelacz-karte", "przełącz kartę", "okna", "przeglądarka"),
    CapturePhrase(
        "kopiuj-zaznaczenie",
        "skopiuj zaznaczony tekst",
        "kursor",
        "tekst i schowek",
    ),
    CapturePhrase(
        "kopiuj-email",
        "skopiuj e-mail pod kursorem",
        "kursor",
        "tekst i schowek",
    ),
    CapturePhrase(
        "zaznacz-zdanie",
        "zaznacz zdanie pod kursorem",
        "kursor",
        "tekst i schowek",
    ),
    CapturePhrase("zapisz-notatke", "zapisz notatkę", "pamięć", "notatki"),
    CapturePhrase("zapamietaj", "zapamiętaj to", "pamięć", "pamięć prywatna"),
    CapturePhrase("wlacz-nasluch", "włącz nasłuch", "sterowanie", "nasłuch"),
    CapturePhrase("wylacz-nasluch", "wyłącz nasłuch", "sterowanie", "nasłuch"),
    CapturePhrase("potwierdz", "potwierdź", "sterowanie", "zgoda"),
    CapturePhrase("anuluj", "anuluj", "sterowanie", "anulowanie"),
    CapturePhrase("stop", "stop", "sterowanie", "przerwanie"),
)


def judge_holding(metadata: dict[str, object], size: int) -> tuple[bool, str]:
    try:
        duration_ms = int(metadata.get("duration_ms") or 0)
    except (TypeError, ValueError):
        duration_ms = 0
    if size < 2048:
        return False, "plik jest zbyt mały"
    if duration_ms and duration_ms < 350:
        return False, "nagranie jest zbyt krótkie"
    if duration_ms > 20000:
        return False, "nagranie jest zbyt długie na jedną komendę"
    return True, "próbka wygląda poprawnie"


CONFIG = CaptureConfig(
    slug="holding-commands",
    title="VoiceLoop — stabilny zestaw poleceń",
    default_port=8791,
    phrases=PHRASES,
    judge=judge_holding,
)


if __name__ == "__main__":
    run_capture_server(CONFIG)
