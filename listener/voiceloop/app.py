from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .actions import ActionRegistry
from .assistant import AssistantService
from .behavior_digest import LocalBehaviorDigestClient
from .deepgram import DeepgramListener
from .embeddings import OpenAICompatibleEmbeddingClient
from .events import EventBus
from .executor import CommandExecutor
from .memory import MemoryStore
from .model_router import ModelRouter, ModelUnavailableError, OpenAICompatiblePlanner
from .models import (
    CommandAccepted,
    CommandRequest,
    CommandSource,
    CommandView,
    HealthComponent,
    HealthResponse,
    MemoryCreate,
    MemoryItem,
)
from .n8n_client import N8nClient
from .qdrant_memory import QdrantVectorStore
from .screen import ScreenContextService
from .screenpipe import ScreenpipeClient
from .screenpipe_deepgram import ScreenpipeMeetingTranscriber
from .screenpipe_memory import ScreenpipeVectorMemoryWorker
from .settings import Settings, get_settings
from .tts import WindowsTTS
from .web_search import WebSearchClient

LOGGER = logging.getLogger("voiceloop")

LISTEN_ONCE_MODES = {
    "assistant": ("Słucham.", ""),
    "note": ("Co zapisać w notatce?", "Zapisz notatkę"),
    "remember": ("Co mam zapamiętać?", "Zapamiętaj"),
}


@dataclass
class Services:
    settings: Settings
    token: str
    memory: MemoryStore
    events: EventBus
    screenpipe: ScreenpipeClient
    web_search: WebSearchClient
    screenpipe_transcriber: ScreenpipeMeetingTranscriber
    embeddings: OpenAICompatibleEmbeddingClient
    qdrant: QdrantVectorStore
    behavior_digester: LocalBehaviorDigestClient
    screenpipe_vector_memory: ScreenpipeVectorMemoryWorker
    actions: ActionRegistry
    tts: WindowsTTS
    executor: CommandExecutor
    assistant: AssistantService
    deepgram: DeepgramListener
    local_planner: OpenAICompatiblePlanner
    cloud_planner: OpenAICompatiblePlanner | None
    n8n: N8nClient


def build_services(settings: Settings) -> Services:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    token = settings.ensure_local_token()
    memory = MemoryStore(settings.data_dir / "voiceloop.db")
    events = EventBus()
    tts = WindowsTTS(
        azure_enabled=settings.azure_tts_enabled,
        azure_key=(
            settings.azure_tts_key.get_secret_value().strip()
            if settings.azure_tts_key
            else None
        ),
        azure_region=settings.azure_tts_region,
        azure_voice=settings.azure_tts_voice,
        azure_timeout_seconds=settings.azure_tts_timeout_seconds,
    )
    screenpipe = ScreenpipeClient(settings)
    web_search = WebSearchClient(settings)
    screenpipe_transcriber = ScreenpipeMeetingTranscriber(settings, screenpipe, memory)
    embeddings = OpenAICompatibleEmbeddingClient(
        base_url=(settings.local_embeddings_base_url or settings.lm_studio_base_url),
        api_key=settings.local_embeddings_api_key or settings.lm_studio_api_key,
        model=settings.local_embeddings_model,
        timeout_seconds=settings.local_embeddings_timeout_seconds,
        enabled=settings.local_embeddings_enabled,
    )
    qdrant = QdrantVectorStore(settings)
    behavior_digester = LocalBehaviorDigestClient(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        model=settings.behavior_digest_model or settings.lm_studio_model,
        timeout_seconds=settings.behavior_digest_timeout_seconds,
        enabled=settings.behavior_digest_enabled,
    )
    screenpipe_vector_memory = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=screenpipe,
        memory=memory,
        embeddings=embeddings,
        qdrant=qdrant,
        digester=behavior_digester,
    )
    actions = ActionRegistry(settings, memory, tts, screenpipe, web_search)
    executor = CommandExecutor(
        memory=memory,
        actions=actions,
        events=events,
        queue_limit=settings.command_queue_limit,
    )
    n8n = N8nClient(
        base_url=settings.n8n_base_url,
        webhook_url=settings.n8n_webhook_url,
        token=settings.n8n_token,
        timeout_seconds=settings.n8n_timeout_seconds,
    )
    local_planner = OpenAICompatiblePlanner(
        provider="lm_studio",
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        model=settings.lm_studio_model,
        timeout_seconds=settings.lm_studio_timeout_seconds,
    )
    cloud_planner = None
    if settings.cloud_llm_enabled and settings.cloud_llm_base_url and settings.cloud_llm_model:
        cloud_planner = OpenAICompatiblePlanner(
            provider="cloud",
            base_url=settings.cloud_llm_base_url,
            api_key=settings.cloud_llm_api_key,
            model=settings.cloud_llm_model,
            timeout_seconds=settings.lm_studio_timeout_seconds,
        )
    llm_primary = settings.llm_primary.strip().lower()
    if llm_primary in {"cloud", "venice"} and cloud_planner is not None:
        model_router = ModelRouter(
            local=cloud_planner,
            cloud=local_planner,
            fallback_requires_allow_cloud=False,
        )
    else:
        model_router = ModelRouter(local=local_planner, cloud=cloud_planner)
    screen = ScreenContextService(settings.data_dir / "screens")
    assistant = AssistantService(
        memory=memory,
        events=events,
        executor=executor,
        n8n=n8n,
        model_router=model_router,
        screen=screen,
        tts=tts,
        embeddings=embeddings,
        qdrant=qdrant,
        action_definitions=actions.definitions(),
        dedupe_seconds=settings.command_dedupe_seconds,
        vector_context_limit=settings.vector_memory_context_limit,
    )

    async def on_deepgram_final(text: str) -> None:
        try:
            await assistant.handle(
                CommandRequest(
                    source=CommandSource.DEEPGRAM,
                    text=text,
                    allow_cloud=settings.cloud_llm_enabled,
                )
            )
        except Exception:
            LOGGER.exception("Deepgram command failed")

    deepgram = DeepgramListener(
        settings=settings,
        events=events,
        on_final=on_deepgram_final,
    )
    return Services(
        settings=settings,
        token=token,
        memory=memory,
        events=events,
        screenpipe=screenpipe,
        web_search=web_search,
        screenpipe_transcriber=screenpipe_transcriber,
        embeddings=embeddings,
        qdrant=qdrant,
        behavior_digester=behavior_digester,
        screenpipe_vector_memory=screenpipe_vector_memory,
        actions=actions,
        tts=tts,
        executor=executor,
        assistant=assistant,
        deepgram=deepgram,
        local_planner=local_planner,
        cloud_planner=cloud_planner,
        n8n=n8n,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.voiceloop_log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    services = build_services(settings)
    app.state.services = services
    await services.memory.initialize()
    await services.executor.start()
    await services.screenpipe_transcriber.start()
    await services.screenpipe_vector_memory.start()
    if settings.auto_start_listening:
        try:
            await services.deepgram.start()
        except Exception:
            LOGGER.exception("Could not auto-start Deepgram")
    try:
        yield
    finally:
        await services.screenpipe_vector_memory.stop()
        await services.screenpipe_transcriber.stop()
        await services.deepgram.stop()
        await services.assistant.close()
        await services.executor.close()
        await services.qdrant.close()


app = FastAPI(
    title="VoiceLoop",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


def services_from(request: Request) -> Services:
    return request.app.state.services


def require_token(
    request: Request,
    x_voiceloop_token: Annotated[str | None, Header()] = None,
) -> Services:
    services = services_from(request)
    if not x_voiceloop_token or not secrets.compare_digest(
        x_voiceloop_token,
        services.token,
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return services


@app.get("/", include_in_schema=False)
async def panel(request: Request) -> FileResponse:
    services = services_from(request)
    panel_path = services.settings.panel_dir / "index.html"
    if not panel_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Panel VoiceLoop nie jest dostępny.",
        )
    return FileResponse(panel_path)


@app.get("/api/v1/session", include_in_schema=False)
async def session(request: Request) -> dict[str, str]:
    services = services_from(request)
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Local access only")
    return {
        "token": services.token,
        "llm_primary": services.settings.llm_primary.strip().lower(),
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    services = services_from(request)
    cloud_health = (
        services.cloud_planner.health()
        if services.cloud_planner is not None
        else asyncio.sleep(0, result=(False, "wyłączony"))
    )
    (
        (lm_ok, lm_detail),
        (cloud_ok, cloud_detail),
        (embeddings_ok, embeddings_detail),
        (qdrant_ok, qdrant_detail),
        (digest_ok, digest_detail),
        (n8n_ok, n8n_detail),
        (screenpipe_ok, screenpipe_detail),
        (web_search_ok, web_search_detail),
    ) = await asyncio.gather(
        services.local_planner.health(),
        cloud_health,
        services.embeddings.health(),
        services.qdrant.health(),
        services.behavior_digester.health(),
        services.n8n.health(),
        services.screenpipe.health(),
        services.web_search.health(),
    )
    dg_ok, dg_detail = services.deepgram.health()
    screenpipe_dg_ok, screenpipe_dg_detail = services.screenpipe_transcriber.health()
    memory_worker_ok, memory_worker_detail = services.screenpipe_vector_memory.health()
    ui_page = services.settings.ui_vision_home_path / "ui.vision.html"
    va_path = services.settings.voiceattack_path
    components = {
        "core": HealthComponent(status="ok", detail="local API"),
        "lm_studio": HealthComponent(
            status="ok" if lm_ok else "error",
            detail=lm_detail,
        ),
        "cloud_llm": HealthComponent(
            status="ok" if cloud_ok else ("error" if services.cloud_planner else "stopped"),
            detail=cloud_detail,
        ),
        "local_embeddings": HealthComponent(
            status="ok" if embeddings_ok else "stopped",
            detail=embeddings_detail,
        ),
        "qdrant": HealthComponent(
            status="ok" if qdrant_ok else "stopped",
            detail=qdrant_detail,
        ),
        "behavior_digest": HealthComponent(
            status="ok" if digest_ok else "stopped",
            detail=digest_detail,
        ),
        "n8n": HealthComponent(
            status="ok" if n8n_ok else "error",
            detail=n8n_detail,
        ),
        "deepgram": HealthComponent(
            status="ok" if dg_ok else "stopped",
            detail=dg_detail,
        ),
        "screenpipe": HealthComponent(
            status="ok" if screenpipe_ok else "stopped",
            detail=screenpipe_detail,
        ),
        "web_search": HealthComponent(
            status="ok" if web_search_ok else "stopped",
            detail=web_search_detail,
        ),
        "screenpipe_deepgram": HealthComponent(
            status="ok" if screenpipe_dg_ok else "stopped",
            detail=screenpipe_dg_detail,
        ),
        "screenpipe_vector_memory": HealthComponent(
            status="ok" if memory_worker_ok else "stopped",
            detail=memory_worker_detail,
        ),
        "ui_vision": HealthComponent(
            status="ok" if ui_page.exists() else "setup_required",
            detail=str(ui_page),
        ),
        "voiceattack": HealthComponent(
            status="ok" if va_path.exists() else "error",
            detail=str(va_path),
        ),
    }
    vector_backend_ok = qdrant_ok or not services.settings.qdrant_enabled
    overall = (
        "ok"
        if (lm_ok or cloud_ok) and screenpipe_ok and vector_backend_ok
        else "degraded"
    )
    return HealthResponse(status=overall, version=__version__, components=components)


@app.post("/api/v1/commands", response_model=CommandAccepted)
async def create_command(
    command: CommandRequest,
    services: Annotated[Services, Depends(require_token)],
) -> CommandAccepted:
    try:
        return await services.assistant.handle(command)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@app.get("/api/v1/commands", response_model=list[CommandView])
async def list_commands(
    services: Annotated[Services, Depends(require_token)],
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
) -> list[CommandView]:
    return await services.memory.recent_commands(limit)


@app.get("/api/v1/commands/{request_id}", response_model=CommandView)
async def get_command(
    request_id: str,
    services: Annotated[Services, Depends(require_token)],
) -> CommandView:
    command = await services.memory.get_command(request_id)
    if command is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Command not found")
    return command


@app.post("/api/v1/commands/{request_id}/confirm", response_model=CommandView)
async def confirm_command(
    request_id: str,
    services: Annotated[Services, Depends(require_token)],
) -> CommandView:
    command = await services.executor.confirm(request_id)
    if command is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Command not found")
    return command


@app.post("/api/v1/commands/{request_id}/cancel", response_model=CommandView)
async def cancel_command(
    request_id: str,
    services: Annotated[Services, Depends(require_token)],
) -> CommandView:
    command = await services.executor.cancel(request_id)
    if command is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Command not found")
    return command


@app.post("/api/v1/stop")
async def stop_all(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, str]:
    await services.deepgram.stop()
    await services.executor.stop_all()
    return {"status": "stopped"}


@app.post("/api/v1/listening/start")
async def start_listening(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, str]:
    await services.deepgram.start()
    return {"status": "starting"}


@app.post("/api/v1/listening/once")
async def listen_once(
    services: Annotated[Services, Depends(require_token)],
    mode: Annotated[
        Literal["assistant", "note", "remember"],
        Query(),
    ] = "assistant",
) -> dict[str, str]:
    prompt, prefix = LISTEN_ONCE_MODES[mode]
    await services.tts.speak(prompt)
    await services.deepgram.start_once(prefix=prefix)
    return {"status": "listening_once", "mode": mode}


@app.post("/api/v1/listening/stop")
async def stop_listening(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, str]:
    await services.deepgram.stop()
    return {"status": "stopped"}


@app.get("/api/v1/memories", response_model=list[MemoryItem])
async def list_memories(
    services: Annotated[Services, Depends(require_token)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    kind: str | None = None,
) -> list[MemoryItem]:
    return await services.memory.list_memories(limit=limit, kind=kind)


@app.post("/api/v1/memories", response_model=MemoryItem)
async def create_memory(
    item: MemoryCreate,
    services: Annotated[Services, Depends(require_token)],
) -> MemoryItem:
    return await services.memory.create_memory(item)


@app.delete("/api/v1/memories/{memory_id}")
async def delete_memory(
    memory_id: int,
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, bool]:
    return {"deleted": await services.memory.delete_memory(memory_id)}


@app.get("/api/v1/events", include_in_schema=False)
async def events(request: Request) -> StreamingResponse:
    services = services_from(request)

    async def stream():
        iterator = services.events.subscribe().__aiter__()
        next_event = asyncio.create_task(anext(iterator))
        try:
            while True:
                if await request.is_disconnected():
                    break
                done, _ = await asyncio.wait({next_event}, timeout=15)
                if not done:
                    yield ": keepalive\n\n"
                    continue
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                next_event = asyncio.create_task(anext(iterator))
        finally:
            next_event.cancel()
            await asyncio.gather(next_event, return_exceptions=True)
            await iterator.aclose()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


static_dir = Path(__file__).resolve().parents[2] / "panel"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
