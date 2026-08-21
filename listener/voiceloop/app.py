from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .actions import ActionRegistry
from .assistant import AssistantService
from .behavior_digest import LocalBehaviorDigestClient
from .capability_index import CapabilityIndex, CapabilityIndexError
from .conversation_telemetry import ConversationTelemetry
from .corpus.candidates import (
    CandidateDecisionError,
    MemoryCandidateStore,
)
from .corpus.local_only import LocalOnlyViolation, load_style_profile
from .corpus.privacy import sensitive_reason
from .corpus.schema import (
    CandidateStatus,
    MemoryCandidate,
    MemoryCandidateApprovalRequest,
    MemoryCandidateCreate,
)
from .deepgram import DeepgramListener
from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .events import EventBus
from .executor import CommandExecutor
from .knowledge_tools import KnowledgeToolOrchestrator
from .manual_memory import ManualMemoryService
from .meeting_recorder import MeetingRecorder
from .memory import MemoryStore
from .model_router import ModelRouter, ModelUnavailableError, OpenAICompatiblePlanner
from .models import (
    CapabilityMatchRequest,
    CommandAccepted,
    CommandRequest,
    CommandView,
    ConversationTraceSnapshot,
    HealthComponent,
    HealthResponse,
    MemoryCreate,
    MemoryItem,
    TranscriptEnvelopeV1,
)
from .n8n_client import N8nClient
from .qdrant_memory import QdrantMemoryError, QdrantVectorStore
from .routing.service import RoutingV2Service
from .screen import ScreenContextService
from .screenpipe import ScreenpipeClient
from .screenpipe_deepgram import ScreenpipeMeetingTranscriber
from .screenpipe_memory import ScreenpipeVectorMemoryWorker
from .settings import Settings, get_settings
from .tts import WindowsTTS
from .voice_conversation import VoiceConversationCoordinator
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
    telemetry: ConversationTelemetry
    screenpipe: ScreenpipeClient
    web_search: WebSearchClient
    knowledge_tools: KnowledgeToolOrchestrator
    screenpipe_transcriber: ScreenpipeMeetingTranscriber
    meeting_recorder: MeetingRecorder
    embeddings: OpenAICompatibleEmbeddingClient
    qdrant: QdrantVectorStore
    qdrant_shadow: QdrantVectorStore | None
    manual_memory: ManualMemoryService
    behavior_digester: LocalBehaviorDigestClient
    screenpipe_vector_memory: ScreenpipeVectorMemoryWorker
    actions: ActionRegistry
    capability_index: CapabilityIndex
    routing_v2: RoutingV2Service
    corpus_candidates: MemoryCandidateStore
    tts: WindowsTTS
    executor: CommandExecutor
    assistant: AssistantService
    deepgram: DeepgramListener
    conversation: VoiceConversationCoordinator
    local_planner: OpenAICompatiblePlanner
    cloud_planner: OpenAICompatiblePlanner | None
    gemini_planner: OpenAICompatiblePlanner | None
    n8n: N8nClient


def _build_gemini_planner(settings: Settings) -> OpenAICompatiblePlanner | None:
    key = (
        settings.gemini_api_key.get_secret_value().strip()
        if settings.gemini_api_key
        else ""
    )
    if not key:
        return None
    return OpenAICompatiblePlanner(
        provider="gemini",
        base_url=settings.gemini_base_url,
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
        context_policy=settings.conversation_context_policy,
    )


def build_services(settings: Settings) -> Services:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    token = settings.ensure_local_token()
    memory = MemoryStore(settings.data_dir / "voiceloop.db")
    events = EventBus()
    telemetry = ConversationTelemetry(
        events,
        storage_path=settings.data_dir / "conversation-traces.jsonl",
    )
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
        speaking_rate_percent=settings.tts_rate_percent,
        speaking_pitch_percent=settings.tts_pitch_percent,
    )
    screenpipe = ScreenpipeClient(settings)
    web_search = WebSearchClient(settings)
    knowledge_tools = KnowledgeToolOrchestrator(
        settings=settings,
        web_search=web_search,
        events=events,
        telemetry=telemetry,
    )
    screenpipe_transcriber = ScreenpipeMeetingTranscriber(settings, screenpipe, memory)
    meeting_recorder = MeetingRecorder(
        settings=settings,
        memory=memory,
        events=events,
        screenpipe=screenpipe,
    )
    embeddings = OpenAICompatibleEmbeddingClient(
        base_url=(settings.local_embeddings_base_url or settings.lm_studio_base_url),
        api_key=settings.local_embeddings_api_key or settings.lm_studio_api_key,
        model=settings.local_embeddings_model,
        timeout_seconds=settings.local_embeddings_timeout_seconds,
        enabled=settings.local_embeddings_enabled,
    )
    qdrant = QdrantVectorStore(settings)
    qdrant_shadow = None
    next_memory_collection = (settings.qdrant_memory_next_collection or "").strip()
    if next_memory_collection and next_memory_collection != settings.qdrant_collection:
        qdrant_shadow = QdrantVectorStore(
            settings.model_copy(update={"qdrant_collection": next_memory_collection})
        )
    manual_memory = ManualMemoryService(
        memory=memory,
        embeddings=embeddings,
        qdrant=qdrant,
    )
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
    actions = ActionRegistry(
        settings,
        memory,
        tts,
        screenpipe,
        web_search,
        embeddings=embeddings,
        qdrant=qdrant,
        manual_memory=manual_memory,
    )
    capability_index = CapabilityIndex(
        settings,
        embeddings=embeddings,
        definitions=actions.capability_catalog()["voiceloop_actions"],
    )
    routing_v2 = RoutingV2Service(
        settings,
        capability_index=capability_index,
        definitions=actions.capability_catalog()["voiceloop_actions"],
    )
    corpus_candidates = MemoryCandidateStore(
        settings.corpus_dir / "memory_candidates" / "candidates.db"
    )
    executor = CommandExecutor(
        memory=memory,
        actions=actions,
        events=events,
        queue_limit=settings.command_queue_limit,
        telemetry=telemetry,
    )
    n8n = N8nClient(
        base_url=settings.n8n_base_url,
        webhook_url=settings.n8n_webhook_url,
        token=settings.n8n_token,
        timeout_seconds=settings.n8n_timeout_seconds,
        enabled=settings.n8n_enabled,
    )
    local_planner = OpenAICompatiblePlanner(
        provider="lm_studio",
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        model=settings.lm_studio_model,
        timeout_seconds=settings.lm_studio_timeout_seconds,
        context_policy=settings.conversation_context_policy,
    )
    cloud_planner = None
    if settings.cloud_llm_enabled and settings.cloud_llm_base_url and settings.cloud_llm_model:
        cloud_planner = OpenAICompatiblePlanner(
            provider="cloud",
            base_url=settings.cloud_llm_base_url,
            api_key=settings.cloud_llm_api_key,
            model=settings.cloud_llm_model,
            timeout_seconds=settings.lm_studio_timeout_seconds,
            context_policy=settings.conversation_context_policy,
        )
    gemini_planner = _build_gemini_planner(settings)
    llm_primary = settings.llm_primary.strip().lower()
    if llm_primary == "gemini" and gemini_planner is not None:
        model_router = ModelRouter(
            local=gemini_planner,
            cloud=local_planner,
            fallback_requires_allow_cloud=False,
        )
    elif llm_primary in {"cloud", "venice"} and cloud_planner is not None:
        model_router = ModelRouter(
            local=cloud_planner,
            cloud=local_planner,
            fallback_requires_allow_cloud=False,
        )
    else:
        fallback = gemini_planner or cloud_planner
        model_router = ModelRouter(local=local_planner, cloud=fallback)
    private_style_instruction = None
    try:
        style_profile = load_style_profile(
            settings.corpus_style_profile_path,
            enabled=settings.corpus_style_profile_enabled,
        )
        if style_profile is not None:
            private_style_instruction = style_profile.prompt_instruction()
    except LocalOnlyViolation as exc:
        LOGGER.warning("Corpus style profile disabled: %s", exc)
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
        capability_index=capability_index,
        routing_v2=routing_v2,
        action_definitions=actions.definitions(),
        dedupe_seconds=settings.command_dedupe_seconds,
        vector_context_limit=settings.vector_memory_context_limit,
        vector_query_adaptive_weights=settings.vector_memory_adaptive_query_weights,
        vector_query_weights=settings.vector_memory_weights,
        vector_memory_min_score=settings.vector_memory_min_score,
        vector_memory_rrf_k=settings.vector_memory_rrf_k,
        qdrant_shadow=qdrant_shadow,
        private_style_instruction=private_style_instruction,
        telemetry=telemetry,
        knowledge_tools=knowledge_tools,
    )
    assistant.configure_stt_threshold(settings.stt_min_action_confidence)
    assistant.configure_speaker_gate(
        reject_multi_speaker=settings.conversation_ignore_multi_speaker
    )

    conversation_holder: dict[str, VoiceConversationCoordinator] = {}

    async def on_deepgram_final(
        text: str,
        *,
        confidence: float | None = None,
        speaker_ids: tuple[int, ...] = (),
        transcript: TranscriptEnvelopeV1 | None = None,
    ) -> None:
        try:
            envelope = transcript or TranscriptEnvelopeV1.from_text(
                text,
                language=settings.deepgram_language,
                confidence=confidence,
                speaker_ids=speaker_ids,
                model=settings.deepgram_model,
            )
            if meeting_recorder.active:
                await meeting_recorder.record_live(envelope)
                return
            coordinator = conversation_holder.get("conversation")
            if coordinator is not None and not coordinator.should_route_to_assistant():
                return
            if coordinator is not None and (
                coordinator._accept_final
                or coordinator.should_handle_control_transcript(text)
                or coordinator.accepts_speaking_transcript()
            ):
                await coordinator.handle_transcript(
                    text,
                    confidence=confidence,
                    speaker_ids=speaker_ids,
                    transcript=envelope,
                )
                return
            await assistant.handle(
                CommandRequest.from_transcript(
                    envelope,
                    allow_cloud=(
                        settings.cloud_llm_enabled
                        or settings.llm_primary.strip().lower() in {"gemini", "cloud", "venice"}
                    ),
                    interaction_session_id=(
                        assistant.conversation_session_id
                        if assistant.conversation_active
                        else None
                    ),
                )
            )
        except Exception:
            LOGGER.exception("Deepgram command failed")

    async def on_deepgram_interim(
        text: str,
        *,
        speaker_ids: tuple[int, ...] = (),
    ) -> None:
        try:
            coordinator = conversation_holder.get("conversation")
            if coordinator is not None:
                await coordinator.handle_interim(text, speaker_ids=speaker_ids)
        except Exception:
            LOGGER.exception("Deepgram interim barge-in failed")

    deepgram = DeepgramListener(
        settings=settings,
        events=events,
        on_final=on_deepgram_final,
        on_interim=on_deepgram_interim,
    )
    deepgram.set_audio_sink(meeting_recorder.feed_microphone_audio)
    conversation = VoiceConversationCoordinator(
        assistant=assistant,
        deepgram=deepgram,
        tts=tts,
        events=events,
        greeting=settings.conversation_greeting,
        cooldown_ms=settings.conversation_cooldown_ms,
        barge_in_after_ms=settings.conversation_barge_in_after_ms,
        direct_address_after_seconds=(
            settings.conversation_direct_address_after_seconds
        ),
        ignore_multi_speaker=settings.conversation_ignore_multi_speaker,
        allow_cloud=(
            settings.cloud_llm_enabled
            or settings.llm_primary.strip().lower() in {"gemini", "cloud", "venice"}
        ),
        telemetry=telemetry,
        stream_reuse_enabled=settings.conversation_stream_reuse_enabled,
        hybrid_barge_in_enabled=settings.conversation_hybrid_barge_in_enabled,
        hybrid_barge_in_grace_ms=settings.conversation_hybrid_barge_in_grace_ms,
        barge_in_stability_ms=settings.conversation_barge_in_stability_ms,
        barge_in_profile=settings.conversation_barge_in_profile,
    )
    conversation_holder["conversation"] = conversation
    deepgram.set_priority_final_predicate(
        conversation.should_handle_control_transcript
    )
    return Services(
        settings=settings,
        token=token,
        memory=memory,
        events=events,
        telemetry=telemetry,
        screenpipe=screenpipe,
        web_search=web_search,
        knowledge_tools=knowledge_tools,
        screenpipe_transcriber=screenpipe_transcriber,
        meeting_recorder=meeting_recorder,
        embeddings=embeddings,
        qdrant=qdrant,
        qdrant_shadow=qdrant_shadow,
        manual_memory=manual_memory,
        behavior_digester=behavior_digester,
        screenpipe_vector_memory=screenpipe_vector_memory,
        actions=actions,
        capability_index=capability_index,
        routing_v2=routing_v2,
        corpus_candidates=corpus_candidates,
        tts=tts,
        executor=executor,
        assistant=assistant,
        deepgram=deepgram,
        conversation=conversation,
        local_planner=local_planner,
        cloud_planner=cloud_planner,
        gemini_planner=gemini_planner,
        n8n=n8n,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.voiceloop_log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    services = build_services(settings)
    app.state.services = services
    await services.memory.initialize()
    try:
        indexed_manual_memories = await services.manual_memory.sync()
        if indexed_manual_memories:
            LOGGER.info(
                "Indexed %s manual memories in local Qdrant.",
                indexed_manual_memories,
            )
    except (EmbeddingUnavailableError, QdrantMemoryError) as exc:
        LOGGER.warning("Manual memory vector sync unavailable: %s", exc)
    if settings.corpus_enabled:
        await services.corpus_candidates.initialize()
    try:
        await services.capability_index.start()
    except CapabilityIndexError as exc:
        LOGGER.warning("Capability index unavailable at startup: %s", exc)
    await services.routing_v2.start()
    await services.executor.start()
    await services.screenpipe_transcriber.start()
    await services.screenpipe_vector_memory.start()
    listen_watch_task: asyncio.Task[None] | None = None
    conversation_start_task: asyncio.Task[dict[str, str]] | None = None

    async def _watch_conversation_listen_timeouts() -> None:
        try:
            async for event in services.events.subscribe():
                event_type = event.get("type")
                payload = event.get("payload") or {}
                if event_type == "transcript.speech_started":
                    await services.conversation.handle_speech_started()
                elif event_type == "listening.timeout":
                    await services.conversation.handle_listen_timeout(reason="timeout")
                elif (
                    event_type == "listening.stopped"
                    and payload.get("reason") == "one_shot_error"
                ):
                    await services.conversation.handle_listen_timeout(
                        reason="one_shot_error"
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Conversation listen-timeout watcher failed")

    listen_watch_task = asyncio.create_task(
        _watch_conversation_listen_timeouts(),
        name="conversation-listen-timeout-watch",
    )
    if settings.auto_start_conversation:
        try:
            conversation_start_task = asyncio.create_task(
                services.conversation.start_conversation(),
                name="auto-start-conversation",
            )
        except Exception:
            LOGGER.exception("Could not schedule auto-start conversation")
    elif settings.auto_start_listening:
        try:
            await services.deepgram.start()
        except Exception:
            LOGGER.exception("Could not auto-start Deepgram")
    try:
        yield
    finally:
        if listen_watch_task is not None:
            listen_watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await listen_watch_task
        if conversation_start_task is not None:
            conversation_start_task.cancel()
            await asyncio.gather(conversation_start_task, return_exceptions=True)
        await services.meeting_recorder.close()
        await services.screenpipe_vector_memory.stop()
        await services.screenpipe_transcriber.stop()
        await services.conversation.close()
        await services.deepgram.stop()
        await services.assistant.close()
        await services.executor.close()
        await services.routing_v2.close()
        await services.capability_index.close()
        if services.qdrant_shadow is not None:
            await services.qdrant_shadow.close()
        await services.qdrant.close()


app = FastAPI(
    title="VoiceLoop",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
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


@app.get("/api/v1/capabilities")
async def capabilities(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    return services.actions.capability_catalog()


@app.post("/api/v1/capabilities/match")
async def match_capabilities(
    payload: CapabilityMatchRequest,
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    try:
        result = await services.capability_index.search(
            payload.text,
            limit=payload.limit,
        )
    except (CapabilityIndexError, EmbeddingUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return result.as_dict()


@app.get("/api/v1/health", response_model=HealthResponse)
async def health(
    services: Annotated[Services, Depends(require_token)],
) -> HealthResponse:
    cloud_health = (
        services.cloud_planner.health()
        if services.cloud_planner is not None
        else asyncio.sleep(0, result=(False, "wyłączony"))
    )
    gemini_health = (
        services.gemini_planner.health()
        if services.gemini_planner is not None
        else asyncio.sleep(0, result=(False, "brak GEMINI_API_KEY"))
    )
    corpus_health = (
        services.corpus_candidates.health()
        if services.settings.corpus_enabled
        else asyncio.sleep(0, result=(False, "wyłączony"))
    )
    (
        (lm_ok, lm_detail),
        (cloud_ok, cloud_detail),
        (gemini_ok, gemini_detail),
        (embeddings_ok, embeddings_detail),
        (qdrant_ok, qdrant_detail),
        (digest_ok, digest_detail),
        (n8n_ok, n8n_detail),
        (screenpipe_ok, screenpipe_detail),
        (web_search_ok, web_search_detail),
        (corpus_ok, corpus_detail),
    ) = await asyncio.gather(
        services.local_planner.health(),
        cloud_health,
        gemini_health,
        services.embeddings.health(),
        services.qdrant.health(),
        services.behavior_digester.health(),
        services.n8n.health(),
        services.screenpipe.health(),
        services.web_search.health(),
        corpus_health,
    )
    dg_ok, dg_detail = services.deepgram.health()
    conversation_ok, conversation_detail = services.conversation.health()
    screenpipe_dg_ok, screenpipe_dg_detail = services.screenpipe_transcriber.health()
    meeting_ok, meeting_detail = services.meeting_recorder.health()
    hume_ok, hume_detail = services.meeting_recorder.emotion_analyzer.health()
    memory_worker_ok, memory_worker_detail = services.screenpipe_vector_memory.health()
    capability_index_ok, capability_index_detail = services.capability_index.health()
    routing_v2_ok, routing_v2_detail = services.routing_v2.health()
    telemetry_ok, telemetry_detail = services.telemetry.health()
    ui_page = services.settings.ui_vision_home_path / "ui.vision.html"
    va_path = services.settings.voiceattack_path
    llm_primary = services.settings.llm_primary.strip().lower()
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
        "gemini_llm": HealthComponent(
            status=(
                "ok"
                if gemini_ok
                else ("error" if services.gemini_planner else "stopped")
            ),
            detail=gemini_detail,
        ),
        "local_embeddings": HealthComponent(
            status="ok" if embeddings_ok else "stopped",
            detail=embeddings_detail,
        ),
        "qdrant": HealthComponent(
            status="ok" if qdrant_ok else "stopped",
            detail=qdrant_detail,
        ),
        "capability_embeddings": HealthComponent(
            status="ok" if capability_index_ok else "stopped",
            detail=capability_index_detail,
        ),
        "routing_v2": HealthComponent(
            status="ok" if routing_v2_ok else "stopped",
            detail=routing_v2_detail,
        ),
        "private_corpus": HealthComponent(
            status="ok" if corpus_ok else "stopped",
            detail=corpus_detail,
        ),
        "behavior_digest": HealthComponent(
            status="ok" if digest_ok else "stopped",
            detail=digest_detail,
        ),
        "n8n": HealthComponent(
            status=(
                "stopped"
                if not services.settings.n8n_enabled
                else ("ok" if n8n_ok else "error")
            ),
            detail=n8n_detail,
        ),
        "deepgram": HealthComponent(
            status="ok" if dg_ok else "stopped",
            detail=dg_detail,
        ),
        "conversation": HealthComponent(
            status="ok" if conversation_ok else "stopped",
            detail=conversation_detail,
        ),
        "conversation_telemetry": HealthComponent(
            status="ok" if telemetry_ok else "stopped",
            detail=telemetry_detail,
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
        "meeting_recording": HealthComponent(
            status="ok" if meeting_ok else "stopped",
            detail=meeting_detail,
        ),
        "hume_emotions": HealthComponent(
            status="ok" if hume_ok else "stopped",
            detail=hume_detail,
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
    primary_ok = lm_ok
    if llm_primary == "gemini":
        primary_ok = gemini_ok or lm_ok
    elif llm_primary in {"cloud", "venice"}:
        primary_ok = cloud_ok or lm_ok
    vector_backend_ok = qdrant_ok or not services.settings.qdrant_enabled
    overall = (
        "ok"
        if primary_ok and screenpipe_ok and vector_backend_ok
        else "degraded"
    )
    return HealthResponse(status=overall, version=__version__, components=components)


@app.get("/api/v1/conversation/traces", response_model=ConversationTraceSnapshot)
async def conversation_traces(
    services: Annotated[Services, Depends(require_token)],
    limit: int = Query(default=20, ge=1, le=200),
) -> ConversationTraceSnapshot:
    return await services.telemetry.snapshot(limit=limit)


@app.get("/api/v1/conversation/quality")
async def conversation_quality(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    return await services.telemetry.quality_report()


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
) -> dict[str, object]:
    if services.meeting_recorder.active:
        await services.deepgram.stop()
        await services.tts.stop()
        await services.assistant.interrupt(end_conversation=True)
        meeting = await services.meeting_recorder.stop()
        return {
            "status": "meeting_finalizing",
            "interrupted": "true",
            "meeting": meeting,
        }
    # Soft barge-in: cut TTS / in-flight plan and listen again.
    services.conversation.clear_block()
    result = await services.conversation.interrupt_speech()
    return {"status": result.get("status", "listening_once"), "interrupted": "true"}


@app.post("/api/v1/conversation/interrupt")
async def interrupt_conversation(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, str]:
    services.conversation.clear_block()
    return await services.conversation.interrupt_speech()


@app.post("/api/v1/conversation/transcribe-only")
async def start_transcribe_only(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    meeting = await services.meeting_recorder.start()
    result = await services.conversation.start_transcribe_only()
    return {**result, "meeting": meeting}


@app.post("/api/v1/meetings/start")
async def start_meeting_recording(
    services: Annotated[Services, Depends(require_token)],
    title: Annotated[str, Query(max_length=500)] = "",
) -> dict[str, object]:
    meeting = await services.meeting_recorder.start(title=title)
    mode = await services.conversation.start_transcribe_only()
    return {**meeting, "conversation_mode": mode["status"]}


@app.post("/api/v1/meetings/stop")
async def stop_meeting_recording(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    await services.deepgram.stop()
    await services.tts.stop()
    await services.assistant.interrupt(end_conversation=True)
    return await services.meeting_recorder.stop()


@app.get("/api/v1/meetings/current")
async def current_meeting_recording(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    return await services.meeting_recorder.current_payload()


@app.get("/api/v1/meetings")
async def list_meeting_recordings(
    services: Annotated[Services, Depends(require_token)],
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[dict[str, object]]:
    return await services.meeting_recorder.list_payloads(limit=limit)


@app.get("/api/v1/meetings/{session_id}")
async def get_meeting_recording(
    session_id: str,
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    meeting = await services.meeting_recorder.get_payload(session_id)
    if meeting is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting recording not found",
        )
    return meeting


@app.post("/api/v1/conversation/start")
async def start_conversation(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, str]:
    if services.meeting_recorder.active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Najpierw zakończ aktywne nagranie spotkania.",
        )
    services.conversation.end_transcribe_only()
    services.conversation.clear_block()
    return await services.conversation.start_conversation()


@app.post("/api/v1/conversation/stop")
async def stop_conversation(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    if services.meeting_recorder.active:
        await services.deepgram.stop()
        return await services.meeting_recorder.stop()
    result = await services.conversation.hard_stop()
    services.conversation.clear_block()
    return result


@app.post("/api/v1/conversation/resume")
async def resume_conversation(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, str]:
    services.conversation.clear_block()
    return await services.conversation.resume_from_pause(source="api")


@app.post("/api/v1/listening/start")
async def start_listening(
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, str]:
    if services.meeting_recorder.active:
        result = await services.conversation.start_transcribe_only()
        return {"status": result.get("status", "transcribe_only")}
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
    await services.deepgram.stop()
    await services.tts.speak(prompt)
    cooldown = max(0.0, services.settings.conversation_cooldown_ms / 1000.0)
    if cooldown:
        await asyncio.sleep(cooldown)
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
    return await services.manual_memory.create(item)


@app.delete("/api/v1/memories/{memory_id}")
async def delete_memory(
    memory_id: int,
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, bool]:
    return {"deleted": await services.manual_memory.delete(memory_id)}


@app.get("/api/v1/memory-candidates", response_model=list[MemoryCandidate])
async def list_memory_candidates(
    services: Annotated[Services, Depends(require_token)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    candidate_status: CandidateStatus | None = CandidateStatus.PENDING,
) -> list[MemoryCandidate]:
    if not services.settings.corpus_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prywatny korpus jest wyłączony.",
        )
    return await services.corpus_candidates.list(
        status=candidate_status,
        limit=limit,
    )


@app.post("/api/v1/memory-candidates", response_model=MemoryCandidate)
async def create_memory_candidate(
    item: MemoryCandidateCreate,
    services: Annotated[Services, Depends(require_token)],
) -> MemoryCandidate:
    if not services.settings.corpus_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prywatny korpus jest wyłączony.",
        )
    block_reason = sensitive_reason(item.proposed_content)
    safe_item = item.model_copy(
        update={
            "status": (
                CandidateStatus.BLOCKED
                if block_reason
                else CandidateStatus.PENDING
            ),
            "block_reason": block_reason,
        }
    )
    return await services.corpus_candidates.upsert(safe_item)


@app.post("/api/v1/memory-candidates/{candidate_id}/approve")
async def approve_memory_candidate(
    candidate_id: str,
    payload: MemoryCandidateApprovalRequest,
    services: Annotated[Services, Depends(require_token)],
) -> dict[str, object]:
    if not services.settings.corpus_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prywatny korpus jest wyłączony.",
        )
    try:
        candidate, memory_item = await services.corpus_candidates.approve(
            candidate_id,
            services.memory,
            expected_content_sha256=payload.content_sha256,
        )
    except CandidateDecisionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return {
        "candidate": candidate.model_dump(mode="json"),
        "memory": memory_item.model_dump(mode="json"),
    }


@app.post(
    "/api/v1/memory-candidates/{candidate_id}/reject",
    response_model=MemoryCandidate,
)
async def reject_memory_candidate(
    candidate_id: str,
    services: Annotated[Services, Depends(require_token)],
) -> MemoryCandidate:
    if not services.settings.corpus_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prywatny korpus jest wyłączony.",
        )
    try:
        return await services.corpus_candidates.reject(candidate_id)
    except CandidateDecisionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@app.get("/api/v1/events", include_in_schema=False)
async def events(
    request: Request,
    services: Annotated[Services, Depends(require_token)],
) -> StreamingResponse:
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
