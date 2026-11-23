"""Local SQLite voice-command calibration; never performs Windows actions."""
from __future__ import annotations

import difflib
import json
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

STARTER_ALIASES = ("minimalizuj", "pulpit", "pokaż pulpit", "minimalizuj wszystko", "zminimalizuj wszystkie okna")


def normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("ł", "l"))
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def difference(heard: str, corrected: str) -> dict:
    a, b = normalize(heard), normalize(corrected)
    changes = [
        {"operation": tag, "recognized": a[i:j], "correct": b[k:m]}
        for tag, i, j, k, m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
        if tag != "equal"
    ]
    if not heard:
        kind = "no_transcription"
        explanation = "Brak transkrypcji. Sprawdź mikrofon i dostępność STT; nie można wywnioskować przyczyny."
    elif changes:
        kind = "stt_difference"
        explanation = "Rozpoznana treść różni się od zatwierdzonej. Pokazano znaki zmienione, ale nie znamy przyczyny akustycznej."
    else:
        kind = "match"
        explanation = "Frazy są zgodne po normalizacji pisowni."
    return {"kind": kind, "explanation": explanation, "changes": changes}


class CalibrationStudio:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS corrections(
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                expected TEXT NOT NULL, heard TEXT NOT NULL, corrected TEXT NOT NULL,
                expected_action TEXT NOT NULL, detected_action TEXT NOT NULL,
                kind TEXT NOT NULL, explanation TEXT NOT NULL, changes TEXT NOT NULL,
                audio_file TEXT NOT NULL, alias_state TEXT NOT NULL)""")

    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _aliases(self, db) -> set[str]:
        defaults = {normalize(s) for s in STARTER_ALIASES}
        defaults.update(normalize(row[0]) for row in db.execute(
            "SELECT corrected FROM corrections WHERE alias_state='approved'"
        ))
        return defaults

    def snapshot(self) -> dict:
        with self.lock, self._db() as db:
            rows = db.execute("SELECT * FROM corrections ORDER BY timestamp DESC LIMIT 50").fetchall()
            return {"ok": True, "aliases": sorted(self._aliases(db)),
                    "samples": [dict(r) | {"changes": json.loads(r["changes"])} for r in rows]}

    def save(self, data: dict, folder: Path) -> dict:
        expected = str(data.get("expected") or "").strip()
        heard = str(data.get("heard") or "").strip()
        corrected = str(data.get("corrected") or "").strip()
        audio_file = str(data.get("audio_file") or "").strip()
        if not expected or not corrected or any(len(x)>300 for x in (expected,heard,corrected)):
            raise ValueError("Podaj poprawne frazy, maksymalnie 300 znaków każda.")
        if data.get("expected_action", "minimize_all_windows") != "minimize_all_windows":
            raise ValueError("Nieobsługiwana akcja; dozwolone tylko minimize_all_windows.")
        if audio_file and (
            not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.(webm|wav|ogg)", audio_file, re.I)
            or not (folder / audio_file).is_file()
        ):
            raise ValueError("Plik audio nie istnieje w katalogu nagrań.")
        diff = difference(heard,corrected)
        with self.lock, self._db() as db:
            aliases = self._aliases(db)
            detected = "minimize_all_windows" if normalize(heard) in aliases else "unresolved"
            state = "already_known" if normalize(corrected) in aliases else "proposed"
            identifier = uuid4().hex
            db.execute(
                "INSERT INTO corrections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (identifier,datetime.now(timezone.utc).isoformat(),expected,heard,corrected,
                 "minimize_all_windows",detected,diff["kind"],diff["explanation"],
                 json.dumps(diff["changes"],ensure_ascii=False),audio_file,state))
            return {"ok": True, "id": identifier, "analysis": diff, "detected_action": detected, "alias_state": state}

    def change_alias(self, identifier: str, approve: bool) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}",identifier):
            raise ValueError("Nieprawidłowy identyfikator.")
        with self.lock,self._db() as db:
            row=db.execute("SELECT corrected FROM corrections WHERE id=?",(identifier,)).fetchone()
            if row is None:
                raise ValueError("Nie znaleziono poprawki.")
            if approve and normalize(row["corrected"]) in {"stop","anuluj","potwierdz","zamknij"}:
                raise ValueError("Konflikt ze słowem sterującym.")
            state="approved" if approve else "reverted"
            db.execute("UPDATE corrections SET alias_state=? WHERE id=?",(state,identifier))
            return {"ok":True,"id":identifier,"alias_state":state,"scope":"calibration_studio_only"}
