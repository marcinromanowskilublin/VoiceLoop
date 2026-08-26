"""Vectorscope — aplikacja FastAPI na loopbacku, port 8770.

Panel jest przyrządem lokalnym: nasłuchuje wyłącznie na 127.0.0.1 i nigdy nie
przekazuje klucza Deepgram do przeglądarki. Transkrypcja idzie przez backend.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voiceloop.embeddings import EmbeddingUnavailableError
from voiceloop.qdrant_memory import QdrantVectorStore

from .analysis import (
    MAX_FRAGMENTS,
    MAX_HEATMAP_FRAGMENTS,
    PROJECTIONS,
    AnalysisRequest,
    run_analysis,
)
from .anchors import anchor_pair_payload
from .config import (
    EMBEDDING_CONTEXT_TOKENS,
    PREFIX_DOCUMENT,
    PREFIXES,
    VECTORSCOPE_HOST,
    VECTORSCOPE_PORT,
    build_embedding_client,
    collect_thresholds,
    settings,
    vectorscope_data_dir,
)
from .diagnostics import DiagnosticsRequest, run_diagnostics
from .fragments import LEVELS, SEGMENTATION_RULE, SEGMENTATION_VERSION
from .live import measure_dedup_probe, measure_live_collection
from .prefix_check import run_prefix_check
from .scale import measure_axis_floors, measure_scale, measure_threshold_reachability
from .store import RecordingStore, transcript_hash
from .transcribe import (
    TranscriptionError,
    deepgram_params,
    transcribe_audio,
    transcription_payload,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="Vectorscope", docs_url="/api/docs", redoc_url=None)


def get_store() -> RecordingStore:
    return RecordingStore(vectorscope_data_dir(settings()))


class AnalyzePayload(BaseModel):
    recording_ids: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=lambda: ["word"])
    prefix: str = "search_document"
    neighbours: int = Field(default=4, ge=0, le=30)
    threshold: float | None = Field(default=0.15, ge=-1.0, le=1.0)
    projection: str = "mds"
    include_anchors: bool = True
    reference_texts: list[str] = Field(default_factory=list)
    merge_identical: bool = False


class DiagnosePayload(BaseModel):
    query: str
    limit: int | None = Field(default=None, ge=1, le=30)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    adaptive: bool | None = None


class ScaleFloorPayload(BaseModel):
    prefix: str = Field(default=PREFIX_DOCUMENT)


class LiveCollectionPayload(BaseModel):
    probes: int = Field(default=30, ge=5, le=120)
    depth: int = Field(default=200, ge=20, le=1000)


class DedupProbePayload(BaseModel):
    sample: int = Field(default=60, ge=10, le=300)


@app.get("/")
async def index() -> HTMLResponse:
    """Doklejamy do zasobów znacznik zmiany pliku.

    Same nagłówki `no-store` nie wystarczają: osadzone przeglądarki potrafią
    trzymać stary `app.js` mimo nich, a wtedy panel liczy nowe dane i rysuje je
    starym kodem. Przy narzędziu pomiarowym to błąd nie do wykrycia okiem.
    """

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for asset in ("app.js", "styles.css"):
        path = STATIC_DIR / asset
        if path.exists():
            stamp = int(path.stat().st_mtime)
            html = html.replace(f"/static/{asset}", f"/static/{asset}?v={stamp}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/api/config")
async def api_config() -> dict[str, Any]:
    active = settings()
    return {
        "levels": list(LEVELS),
        "prefixes": list(PREFIXES),
        "projections": list(PROJECTIONS),
        "max_fragments": MAX_FRAGMENTS,
        "max_heatmap_fragments": MAX_HEATMAP_FRAGMENTS,
        "context_tokens": EMBEDDING_CONTEXT_TOKENS,
        "segmentation": {
            "rule": SEGMENTATION_RULE,
            "version": SEGMENTATION_VERSION,
        },
        "deepgram": {
            "model": active.deepgram_model,
            "language": active.deepgram_language,
            "diarization": active.deepgram_diarization_enabled,
            "params": deepgram_params(active),
        },
        "embeddings": {
            "base_url": active.local_embeddings_base_url or active.lm_studio_base_url,
            "configured_model": active.local_embeddings_model,
        },
        "thresholds": [
            {
                "key": item.key,
                "value": item.value,
                "label": item.label,
                "origin": item.origin,
            }
            for item in collect_thresholds(active)
        ],
        "anchors": anchor_pair_payload(),
        "data_dir": str(vectorscope_data_dir(active)),
    }


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    active = settings()
    client = build_embedding_client(active)

    # Uwaga: OpenAICompatibleEmbeddingClient.health() potrafi zwrócić True przy
    # całkowicie wyłączonym LM Studio, bo resolve_model() oddaje nazwę modelu
    # z konfiguracji, nie pytając serwera. Panel diagnostyczny nie może świecić
    # na zielono, kiedy każda wektoryzacja poleci wyjątkiem, więc sprawdzamy to
    # jedyną wiarygodną metodą: realnym żądaniem embeddingu.
    embeddings_ok, embeddings_detail = await client.health()
    if embeddings_ok:
        try:
            probe = await client.embed_texts(["ping"])
        except EmbeddingUnavailableError as exc:
            embeddings_ok = False
            embeddings_detail = f"konfiguracja wygląda dobrze, ale żądanie padło: {exc}"
        else:
            if probe:
                embeddings_detail = (
                    f"{embeddings_detail}, odpowiada realnie ({len(probe[0])}D)"
                )
            else:
                embeddings_ok = False
                embeddings_detail = "serwer odpowiedział, ale nie zwrócił wektora"

    deepgram_key = (
        active.deepgram_api_key.get_secret_value().strip()
        if active.deepgram_api_key
        else ""
    )

    store = QdrantVectorStore(active)
    try:
        qdrant_ok, qdrant_detail = await store.health()
        collection = store.collection_name
    finally:
        await store.close()

    return {
        "embeddings": {
            "ok": embeddings_ok,
            "detail": embeddings_detail,
            "base_url": client.base_url,
        },
        "deepgram": {
            "ok": bool(deepgram_key),
            "detail": (
                f"klucz obecny ({len(deepgram_key)} znaków), model "
                f"{active.deepgram_model}/{active.deepgram_language}"
                if deepgram_key
                else "brak DEEPGRAM_API_KEY w listener/.env"
            ),
        },
        "qdrant": {
            "ok": qdrant_ok,
            "detail": qdrant_detail,
            "collection": collection,
        },
    }


@app.get("/api/recordings")
async def api_recordings() -> dict[str, Any]:
    store = get_store()
    return {
        "items": [
            {
                "id": meta.id,
                "label": meta.label,
                "created_at": meta.created_at,
                "size_bytes": meta.size_bytes,
                "duration_seconds": meta.duration_seconds,
                "mime": meta.mime,
                "microphone_processing": meta.microphone_processing,
                "transcript_status": meta.transcript_status,
                "transcript_error": meta.transcript_error,
                "word_count": meta.word_count,
                "text_preview": meta.text_preview,
            }
            for meta in store.list_recordings()
        ]
    }


@app.post("/api/recordings")
async def api_create_recording(request: Request) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Puste ciało żądania — brak audio.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Nagranie przekracza 200 MB.")

    params = request.query_params
    duration_raw = params.get("duration")
    try:
        duration = float(duration_raw) if duration_raw else None
    except ValueError:
        duration = None

    store = get_store()
    meta = store.create(
        payload=payload,
        mime=request.headers.get("content-type") or "audio/webm",
        label=params.get("label") or "",
        duration_seconds=duration,
        microphone_processing=params.get("processing") == "1",
        upload_ms=(time.perf_counter() - started) * 1000.0,
    )
    return {"id": meta.id, "label": meta.label, "size_bytes": meta.size_bytes}


@app.get("/api/recordings/{recording_id}")
async def api_recording(recording_id: str) -> dict[str, Any]:
    store = get_store()
    try:
        meta = store.read_meta(recording_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    transcript = store.read_transcript(recording_id)
    if transcript is not None:
        transcript = {
            key: value for key, value in transcript.items() if key != "raw"
        }
    return {"meta": meta.__dict__, "transcript": transcript}


@app.get("/api/recordings/{recording_id}/audio")
async def api_recording_audio(recording_id: str) -> FileResponse:
    store = get_store()
    try:
        path = store.audio_path(recording_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plik audio nie istnieje.")
    return FileResponse(path)


@app.delete("/api/recordings/{recording_id}")
async def api_delete_recording(recording_id: str) -> dict[str, bool]:
    store = get_store()
    try:
        store.delete(recording_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True}


@app.post("/api/recordings/{recording_id}/transcribe")
async def api_transcribe(recording_id: str) -> dict[str, Any]:
    active = settings()
    store = get_store()
    try:
        meta = store.read_meta(recording_id)
        audio_path = store.audio_path(recording_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    started = time.perf_counter()
    meta.deepgram_params = deepgram_params(active)
    try:
        transcription = await transcribe_audio(
            settings=active,
            payload=audio_path.read_bytes(),
            content_type=meta.mime,
        )
    except TranscriptionError as exc:
        meta.transcript_status = "error"
        meta.transcript_error = str(exc)
        meta.record_error("transcribe", str(exc))
        meta.record_timing("transcribe", (time.perf_counter() - started) * 1000.0)
        store.write_meta(meta)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    payload = transcription_payload(transcription)
    store.write_transcript(recording_id, payload)
    meta.transcript_status = "ok"
    meta.transcript_error = None
    meta.transcript_model = transcription.model
    meta.transcript_language = transcription.language
    meta.transcript_hash = transcript_hash(payload)
    meta.word_count = len(transcription.words)
    meta.text_preview = transcription.text[:300]
    meta.record_timing("transcribe", (time.perf_counter() - started) * 1000.0)
    store.write_meta(meta)

    return {
        "id": recording_id,
        "text": transcription.text,
        "word_count": len(transcription.words),
        "sentence_count": len(transcription.sentences),
        "model": transcription.model,
        "language": transcription.language,
        "transcript_hash": meta.transcript_hash,
    }


@app.post("/api/analyze")
async def api_analyze(payload: AnalyzePayload) -> JSONResponse:
    if not payload.recording_ids and not payload.reference_texts:
        raise HTTPException(
            status_code=400,
            detail="Wybierz przynajmniej jedno nagranie albo podaj tekst referencyjny.",
        )
    store = get_store()
    try:
        result = await run_analysis(
            AnalysisRequest(
                recording_ids=payload.recording_ids,
                levels=payload.levels,
                prefix=payload.prefix,
                neighbours=payload.neighbours,
                threshold=payload.threshold,
                projection=payload.projection,
                include_anchors=payload.include_anchors,
                reference_texts=payload.reference_texts,
                merge_identical=payload.merge_identical,
            ),
            store,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(result)


@app.post("/api/diagnose")
async def api_diagnose(payload: DiagnosePayload) -> JSONResponse:
    result = await run_diagnostics(
        DiagnosticsRequest(
            query=payload.query,
            limit=payload.limit,
            min_score=payload.min_score,
            adaptive=payload.adaptive,
        )
    )
    return JSONResponse(result)


@app.post("/api/prefix-check")
async def api_prefix_check() -> JSONResponse:
    # Bez `min_score`: ten pomiar dotyczy geometrii przestrzeni, a nie tego,
    # co przy jakimś progu przechodzi. Progi rozstrzyga scale.py i live.py.
    return JSONResponse(await run_prefix_check())


@app.post("/api/scale-floor")
async def api_scale_floor(payload: ScaleFloorPayload) -> JSONResponse:
    result = await measure_scale(prefix=payload.prefix)
    return JSONResponse(result)


@app.post("/api/threshold-reachability")
async def api_threshold_reachability() -> JSONResponse:
    return JSONResponse(await measure_threshold_reachability())


@app.post("/api/live-collection")
async def api_live_collection(payload: LiveCollectionPayload) -> JSONResponse:
    # Tylko odczyt: `scroll` i `query_points`. Panel nigdy nie pisze do pamięci
    # VoiceLoopa, bo przyrząd pomiarowy, który zmienia mierzony obiekt, jest
    # bezużyteczny.
    return JSONResponse(
        await measure_live_collection(probe_count=payload.probes, depth=payload.depth)
    )


@app.post("/api/dedup-probe")
async def api_dedup_probe(payload: DedupProbePayload) -> JSONResponse:
    return JSONResponse(await measure_dedup_probe(sample=payload.sample))


@app.post("/api/axis-floor")
async def api_axis_floor() -> JSONResponse:
    return JSONResponse(await measure_axis_floors())


class NoCacheStaticFiles(StaticFiles):
    """Panel jest narzędziem diagnostycznym, nie stroną produkcyjną.

    Bez tego przeglądarka trzyma stary `app.js` po każdej zmianie kodu i pokazuje
    wnioski wyliczone poprzednią wersją — a to najgorszy możliwy rodzaj błędu
    w przyrządzie pomiarowym.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: ANN001
        return False

    async def get_response(self, path: str, scope):  # noqa: ANN001, ANN201
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "vectorscope.app:app",
        host=VECTORSCOPE_HOST,
        port=VECTORSCOPE_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
