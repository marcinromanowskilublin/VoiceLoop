"""Układ na dysku i warstwa odtwarzalności eksperymentu.

Jedno nagranie to jeden katalog: surowe audio bez konwersji, transkrypt, wektory
i `meta.json`, w którym zapisujemy wszystko, co wpływa na wynik. Bez tego
eksperyment jest ładny, ale nie do powtórzenia — a wtedy nie jest pomiarem.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import VECTORSCOPE_VERSION
from .fragments import SEGMENTATION_RULE, SEGMENTATION_VERSION

MIME_EXTENSIONS = {
    "audio/webm": ".webm",
    "audio/webm;codecs=opus": ".webm",
    "audio/ogg": ".ogg",
    "audio/ogg;codecs=opus": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
}

# Wektory zapisujemy dokładnie tak, jak zwróciło je LM Studio. Normalizacja L2
# dzieje się dopiero przy liczeniu cosinusa, na kopii — żeby w pliku został
# surowy wynik modelu, a nie nasza interpretacja.
VECTOR_STORAGE_POLICY = "surowe z LM Studio; L2 dopiero przy cosinusie"


@dataclass
class RecordingMeta:
    id: str
    experiment_id: str
    created_at: str
    label: str
    mime: str
    audio_file: str
    size_bytes: int
    duration_seconds: float | None = None
    microphone_processing: bool = False
    vectorscope_version: str = VECTORSCOPE_VERSION
    segmentation_rule: str = SEGMENTATION_RULE
    segmentation_version: str = SEGMENTATION_VERSION
    vector_storage_policy: str = VECTOR_STORAGE_POLICY
    transcript_status: str = "pending"
    transcript_error: str | None = None
    transcript_model: str | None = None
    transcript_language: str | None = None
    transcript_hash: str | None = None
    deepgram_params: dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    text_preview: str = ""
    embedding_runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def record_timing(self, stage: str, milliseconds: float) -> None:
        self.timings_ms[stage] = round(float(milliseconds), 2)

    def record_error(self, stage: str, message: str) -> None:
        self.errors.append(
            {
                "stage": stage,
                "message": message[:1000],
                "at": datetime.now(UTC).isoformat(),
            }
        )

    def record_embedding_run(
        self,
        *,
        level: str,
        prefix: str,
        model: str,
        dimension: int,
        fragment_count: int,
        over_context: int,
    ) -> None:
        self.embedding_runs[f"{level}|{prefix}"] = {
            "level": level,
            "prefix": prefix,
            "model": model,
            "dimension": dimension,
            "fragment_count": fragment_count,
            "over_context": over_context,
            "normalized_on_disk": False,
            "at": datetime.now(UTC).isoformat(),
        }


def transcript_hash(payload: dict[str, Any]) -> str:
    """Skrót tego, co realnie determinuje segmentację i wektory."""

    canonical = json.dumps(
        {
            "text": payload.get("text") or "",
            "words": [
                {
                    "text": item.get("text"),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "speaker": item.get("speaker"),
                }
                for item in (payload.get("words") or [])
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RecordingStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, recording_id: str) -> Path:
        safe = "".join(
            character
            for character in recording_id
            if character.isalnum() or character in {"-", "_"}
        )
        if not safe or safe != recording_id:
            raise ValueError("Niepoprawny identyfikator nagrania.")
        return self.root / safe

    @staticmethod
    def new_id() -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{secrets.token_hex(2)}"

    def create(
        self,
        *,
        payload: bytes,
        mime: str,
        label: str,
        duration_seconds: float | None,
        microphone_processing: bool,
        upload_ms: float | None = None,
    ) -> RecordingMeta:
        recording_id = self.new_id()
        directory = self._directory(recording_id)
        directory.mkdir(parents=True, exist_ok=False)

        base_mime = mime.split(";")[0].strip().casefold() or "audio/webm"
        extension = MIME_EXTENSIONS.get(mime.strip().casefold()) or MIME_EXTENSIONS.get(
            base_mime, ".bin"
        )
        audio_name = f"audio{extension}"
        (directory / audio_name).write_bytes(payload)

        meta = RecordingMeta(
            id=recording_id,
            experiment_id=f"vs-{recording_id}",
            created_at=datetime.now(UTC).isoformat(),
            label=label.strip() or recording_id,
            mime=mime,
            audio_file=audio_name,
            size_bytes=len(payload),
            duration_seconds=duration_seconds,
            microphone_processing=microphone_processing,
        )
        if upload_ms is not None:
            meta.record_timing("upload", upload_ms)
        self.write_meta(meta)
        return meta

    def write_meta(self, meta: RecordingMeta) -> None:
        directory = self._directory(meta.id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "meta.json"
        path.write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_meta(self, recording_id: str) -> RecordingMeta:
        path = self._directory(recording_id) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"Nagranie {recording_id} nie istnieje.")
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = RecordingMeta.__dataclass_fields__
        known: dict[str, Any] = {}
        for name in fields:
            if name in raw and raw[name] is not None:
                known[name] = raw[name]
        known["id"] = raw.get("id", recording_id)
        known.setdefault("experiment_id", f"vs-{known['id']}")
        known.setdefault("created_at", datetime.now(UTC).isoformat())
        known.setdefault("label", known["id"])
        known.setdefault("mime", "audio/webm")

        # Nagrania z wcześniejszych wersji panelu używały innych nazw pól.
        # Czytamy je zamiast pokazywać zera obok poprawnego pliku audio.
        if "size_bytes" not in known and isinstance(raw.get("bytes"), int):
            known["size_bytes"] = raw["bytes"]
        known.setdefault("size_bytes", 0)
        if "microphone_processing" not in known and "mic_processing" in raw:
            known["microphone_processing"] = bool(raw["mic_processing"])
        if "transcript_language" not in known and raw.get("language"):
            known["transcript_language"] = str(raw["language"])
        if "text_preview" not in known and raw.get("text"):
            known["text_preview"] = str(raw["text"])[:300]

        if "audio_file" not in known:
            directory = self._directory(known["id"])
            found = next(
                (
                    candidate.name
                    for candidate in sorted(directory.glob("audio.*"))
                    if candidate.is_file()
                ),
                None,
            )
            known["audio_file"] = found or "audio.webm"

        return RecordingMeta(**known)

    def list_recordings(self) -> list[RecordingMeta]:
        items: list[RecordingMeta] = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            try:
                items.append(self.read_meta(directory.name))
            except (FileNotFoundError, ValueError, json.JSONDecodeError, TypeError):
                continue
        return items

    def audio_path(self, recording_id: str) -> Path:
        meta = self.read_meta(recording_id)
        return self._directory(recording_id) / meta.audio_file

    def write_transcript(self, recording_id: str, payload: dict[str, Any]) -> None:
        path = self._directory(recording_id) / "transcript.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_transcript(self, recording_id: str) -> dict[str, Any] | None:
        path = self._directory(recording_id) / "transcript.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_fragments(
        self,
        recording_id: str,
        payload: list[dict[str, Any]],
    ) -> None:
        path = self._directory(recording_id) / "fragments.json"
        path.write_text(
            json.dumps(
                {
                    "segmentation_rule": SEGMENTATION_RULE,
                    "segmentation_version": SEGMENTATION_VERSION,
                    "fragments": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def vectors_path(self, recording_id: str, level: str, prefix: str) -> Path:
        return self._directory(recording_id) / f"vectors-{level}-{prefix}.npz"

    def read_vectors(
        self,
        recording_id: str,
        level: str,
        prefix: str,
        expected_texts: list[str],
    ) -> np.ndarray | None:
        """Cache jest ważny tylko dla dokładnie tych samych tekstów w tej kolejności."""

        path = self.vectors_path(recording_id, level, prefix)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                stored = [str(value) for value in archive["texts"].tolist()]
                vectors = archive["vectors"]
        except (OSError, KeyError, ValueError):
            return None
        if stored != expected_texts:
            return None
        return np.asarray(vectors, dtype=float)

    def write_vectors(
        self,
        recording_id: str,
        level: str,
        prefix: str,
        texts: list[str],
        vectors: np.ndarray,
        *,
        model: str,
    ) -> None:
        path = self.vectors_path(recording_id, level, prefix)
        np.savez_compressed(
            path,
            texts=np.array(texts, dtype=object).astype("U"),
            vectors=np.asarray(vectors, dtype=float),
            model=np.array([model], dtype="U"),
            prefix=np.array([prefix], dtype="U"),
            normalized=np.array([0], dtype=np.int8),
        )

    def delete(self, recording_id: str) -> None:
        directory = self._directory(recording_id)
        if directory.exists():
            shutil.rmtree(directory)
