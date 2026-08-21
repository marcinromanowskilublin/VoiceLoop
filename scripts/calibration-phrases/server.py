"""Local recorder for diverse Polish VoiceLoop calibration phrases."""

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
    CapturePhrase("stop", "stop", "sterowanie", "krótka komenda"),
    CapturePhrase("anuluj", "anuluj", "sterowanie", "krótka komenda"),
    CapturePhrase("niewazne", "nieważne", "sterowanie", "krótka komenda"),
    CapturePhrase("potwierdz", "potwierdź", "sterowanie", "krótka komenda"),
    CapturePhrase("asystencie", "asystencie", "budzenie", "słowo wybudzające"),
    CapturePhrase(
        "hej-asystencie",
        "hej asystencie",
        "budzenie",
        "słowo wybudzające",
    ),
    CapturePhrase("voiceloop", "VoiceLoop", "trudne", "nazwa własna"),
    CapturePhrase("deepgram", "Deepgram", "trudne", "nazwa własna"),
    CapturePhrase("screenpipe", "Screenpipe", "trudne", "nazwa własna"),
    CapturePhrase("otworz-chrome", "otwórz Chrome", "otwieranie", "zadanie"),
    CapturePhrase("otworz-gmail", "otwórz Gmail", "otwieranie", "zadanie"),
    CapturePhrase(
        "otworz-dwie-aplikacje",
        "otwórz Chrome i kalendarz",
        "złożone",
        "zadanie wieloetapowe",
    ),
    CapturePhrase(
        "kopiuj-email",
        "czy możesz skopiować e-mail pod kursorem",
        "kursor",
        "pytanie-zadanie",
    ),
    CapturePhrase(
        "zaznacz-akapit",
        "zaznacz akapit pod kursorem",
        "kursor",
        "zadanie",
    ),
    CapturePhrase(
        "poprawka-otworz",
        "znaczy, otwórz jednak Gemini",
        "korekta",
        "samokorekta",
    ),
    CapturePhrase(
        "anulowanie-zdanie",
        "otwórz Chrome, nie, nieważne",
        "anulowanie",
        "zmiana decyzji",
    ),
    CapturePhrase(
        "pytanie-status",
        "czy VoiceLoop działa poprawnie?",
        "rozmowa",
        "pytanie",
    ),
    CapturePhrase(
        "rozmowa-spokojna",
        "nie wykonuj żadnej akcji, tylko potwierdź że mnie słyszysz",
        "rozmowa",
        "wypowiedź neutralna",
    ),
    CapturePhrase(
        "rozmowa-dluga",
        "sprawdzam, czy transkrypcja zachowuje całe zdanie i polskie końcówki",
        "rozmowa",
        "dłuższa wypowiedź",
    ),
    CapturePhrase(
        "compound-copy-note",
        "skopiuj zaznaczenie, a potem otwórz notatkę",
        "złożone",
        "zadanie wieloetapowe",
    ),
)


def judge_calibration(metadata: dict[str, object], size: int) -> tuple[bool, str]:
    def number(name: str) -> float:
        try:
            return float(metadata.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    duration_ms = number("duration_ms")
    peak = number("peak")
    rms = number("rms")
    words = max(1, len(str(metadata.get("text") or "").split()))
    minimum_ms = min(8000, max(300, words * 140))
    if size < 2048:
        return False, "plik jest zbyt mały"
    if duration_ms and duration_ms < minimum_ms:
        return False, f"nagranie jest zbyt krótkie; cel to co najmniej {minimum_ms} ms"
    if duration_ms > 20000:
        return False, "nagranie jest zbyt długie na jedną frazę"
    if peak and peak < 0.03:
        return False, "sygnał jest zbyt cichy"
    if rms and rms < 0.004:
        return False, "średni poziom sygnału jest zbyt niski"
    return True, "próbka nadaje się do ręcznej oceny"


CONFIG = CaptureConfig(
    slug="calibration-phrases",
    title="VoiceLoop — kalibracja polskiego głosu",
    default_port=8792,
    phrases=PHRASES,
    judge=judge_calibration,
)


if __name__ == "__main__":
    run_capture_server(CONFIG)
