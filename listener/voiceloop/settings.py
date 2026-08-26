from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LISTENER_DIR = PROJECT_ROOT / "listener"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=LISTENER_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    deepgram_api_key: SecretStr | None = None
    voiceloop_token: SecretStr | None = None

    voiceloop_host: str = "127.0.0.1"
    voiceloop_port: int = 8765
    voiceloop_log_level: str = "INFO"
    voiceloop_data_dir: str = "../data"

    n8n_enabled: bool = False
    n8n_base_url: str = "http://127.0.0.1:5678"
    n8n_webhook_url: str = "http://127.0.0.1:5678/webhook/voice-command-v1"
    n8n_token: SecretStr | None = None
    n8n_timeout_seconds: float = 5.0

    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_api_key: SecretStr = SecretStr("lm-studio")
    lm_studio_model: str | None = None
    lm_studio_timeout_seconds: float = 90.0

    llm_primary: str = "local"
    conversation_context_policy: str = "auto"
    cloud_llm_enabled: bool = False
    cloud_llm_base_url: str | None = None
    cloud_llm_api_key: SecretStr | None = None
    cloud_llm_model: str | None = None

    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-3.5-flash"
    gemini_timeout_seconds: float = 90.0

    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"
    web_search_fallback_provider: str = "duckduckgo"
    web_search_api_key: SecretStr | None = None
    web_search_gemini_model: str = "gemini-3.6-flash"
    web_search_timeout_seconds: float = 12.0
    web_search_max_results: int = 5
    knowledge_tools_enabled: bool = True
    knowledge_tools_max_results: int = 5
    knowledge_tools_timeout_seconds: float = 15.0
    knowledge_tools_cache_ttl_seconds: float = 300.0

    local_embeddings_enabled: bool = True
    local_embeddings_base_url: str | None = None
    local_embeddings_api_key: SecretStr | None = None
    local_embeddings_model: str | None = None
    local_embeddings_timeout_seconds: float = 30.0
    vector_memory_context_limit: int = 8
    # Zmierzone na kolekcji produkcyjnej: najniższy cosinus, jaki nomic-v2-moe
    # w ogóle produkuje dla tych danych, to 0.441, a mediany szumu per oś leżą
    # między 0.567 i 0.647. Próg 0.15 nie odrzucał niczego i nie mógł, a jedna
    # liczba na pięć osi znaczyłaby w każdej co innego. Zostaje 0.0, żeby kod nie
    # obiecywał bramki, której nie ma. RRF działa na rangach, więc jej nie
    # potrzebuje. Absolutne progi trzymamy tam, gdzie są skalibrowane — przy
    # deduplikacji.
    vector_memory_min_score: float = 0.0
    vector_memory_adaptive_query_weights: bool = True
    vector_memory_weight_semantic: float = 0.40
    vector_memory_weight_topic: float = 0.20
    vector_memory_weight_intent: float = 0.15
    vector_memory_weight_decision: float = 0.15
    vector_memory_weight_person_context: float = 0.10
    vector_memory_rrf_k: int = 60
    # Strażnik progów. Powstał, bo trzy progi w tym projekcie stały martwe
    # miesiącami, choć jedna trzecia kodu zajmuje się mierzeniem systemu — pomiar
    # nie miał konsumenta. Doba wystarcza: rozkłady w kolekcji zmieniają się
    # w tempie napływu pamięci, nie zapytań.
    threshold_guard_enabled: bool = True
    threshold_guard_interval_seconds: int = 86400
    threshold_guard_sample: int = 40
    capability_embeddings_enabled: bool = True
    capability_match_limit: int = 5
    capability_match_min_score: float = 0.20
    routing_v2_enabled: bool = True
    routing_v2_shadow_mode: bool = True
    routing_v2_execute: bool = False
    routing_v2_canary_enabled: bool = True
    routing_v2_canary_action_ids: str = (
        "describe_active_window,describe_recent_activity,describe_text_target,"
        "recall,open_url,open_browser,open_folder,open_app"
    )
    routing_v2_candidate_limit: int = 10
    routing_v2_execute_min_score: float = 0.50
    routing_v2_execute_min_margin: float = 0.10
    routing_v2_max_subtasks: int = 12
    routing_v2_shadow_timeout_seconds: float = 5.0
    routing_v2_quality_gate_file: str = "corpus/eval/routing-v2-metrics.json"
    routing_v2_calibration_mode: str = "off"
    routing_v2_calibration_artifact_file: str = (
        "corpus/eval/routing-calibration-artifact-v1.json"
    )
    routing_v2_calibration_store_file: str = (
        "corpus/routing_calibration/observations-v1.db"
    )
    routing_v2_calibration_queue_limit: int = 4096
    qdrant_enabled: bool = True
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "voiceloop_memory"
    qdrant_capability_collection: str = "voiceloop_capabilities_v1"
    qdrant_memory_next_collection: str | None = None
    qdrant_capability_next_collection: str | None = None
    qdrant_timeout_seconds: float = 10.0
    qdrant_dual_write: bool = True
    vector_memory_ttl_days: int = 14
    vector_memory_prune_enabled: bool = False
    vector_memory_prune_interval_seconds: int = 86400
    behavior_digest_enabled: bool = True
    behavior_digest_model: str | None = None
    behavior_digest_timeout_seconds: float = 240.0
    behavior_digest_poll_seconds: int = 300
    behavior_digest_recent_minutes: int = 10
    behavior_digest_max_contexts: int = 24
    behavior_digest_min_confidence: float = 0.65
    corpus_enabled: bool = True
    corpus_style_profile_enabled: bool = False
    corpus_style_profile_file: str = "style/profile-v1.json"
    corpus_routing_margin_threshold: float = 0.05

    screenpipe_enabled: bool = True
    screenpipe_base_url: str = "http://127.0.0.1:3030"
    screenpipe_api_token: SecretStr | None = None
    screenpipe_timeout_seconds: float = 5.0
    screenpipe_recent_window_seconds: int = 90
    screenpipe_history_limit: int = 20
    screenpipe_lookback_days: int = 14
    screenpipe_deepgram_blocked_hosts: str = (
        "youtube.com,www.youtube.com,m.youtube.com,music.youtube.com,youtu.be"
    )
    screenpipe_deepgram_call_hosts: str = (
        "meet.google.com,teams.microsoft.com,zoom.us,discord.com,webex.com"
    )
    screenpipe_deepgram_call_apps: str = (
        "zoom,teams,discord,webex,skype,slack"
    )
    screenpipe_deepgram_enabled: bool = True
    screenpipe_deepgram_poll_seconds: int = 30
    screenpipe_deepgram_meeting_grace_seconds: int = 90
    screenpipe_deepgram_max_file_mb: int = 100
    meeting_recording_poll_seconds: int = 5
    meeting_recording_finalize_seconds: int = 25
    meeting_recording_archive_audio: bool = True
    meeting_recording_audio_chunk_seconds: int = 15
    meeting_recording_output_sample_rate: int = 48000
    screenpipe_vector_memory_enabled: bool = True
    screenpipe_vector_poll_seconds: int = 300
    screenpipe_vector_recent_minutes: int = 10

    voiceattack_exe: str | None = None
    hume_api_key: SecretStr | None = None
    hume_emotion_analysis_enabled: bool = False
    hume_emotion_endpoint: str = "wss://api.hume.ai/v0/evi/chat"
    hume_emotion_timeout_seconds: float = 30.0
    hume_emotion_top_n: int = 3
    hume_emotion_min_score: float = 0.05
    azure_tts_enabled: bool = False
    azure_tts_key: SecretStr | None = None
    azure_tts_region: str | None = None
    azure_tts_voice: str = "pl-PL-ZofiaNeural"
    azure_tts_timeout_seconds: float = 20.0
    tts_rate_percent: int = -20
    tts_pitch_percent: int = -5
    uivision_home: str | None = None
    uivision_timeout_seconds: int = 60

    command_dedupe_seconds: float = 2.0
    command_queue_limit: int = 10
    auto_start_listening: bool = False
    auto_start_conversation: bool = False
    conversation_greeting: str = (
        "Cześć. Możemy porozmawiać albo możesz od razu wydać polecenie. "
        "Możesz też zapytać, co potrafię. W czym mogę pomóc?"
    )
    conversation_cooldown_ms: int = 100
    conversation_barge_in_after_ms: int = 3000
    conversation_stream_reuse_enabled: bool = True
    conversation_hybrid_barge_in_enabled: bool = True
    conversation_hybrid_barge_in_grace_ms: int = 350
    conversation_barge_in_stability_ms: int = 150
    conversation_barge_in_profile: str = "auto"
    conversation_direct_address_after_seconds: float = 30.0
    conversation_ignore_multi_speaker: bool = True
    stt_min_action_confidence: float = 0.75
    deepgram_model: str = "nova-3"
    deepgram_language: str = "pl"
    deepgram_diarization_enabled: bool = True
    deepgram_diarization_model: str = "latest"
    deepgram_endpointing_ms: int = 300
    deepgram_utterance_end_ms: int = 1200
    microphone_device: str | int | None = None
    sample_rate: int = 16000

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def listener_dir(self) -> Path:
        return LISTENER_DIR

    @property
    def panel_dir(self) -> Path:
        return PROJECT_ROOT / "panel"

    @property
    def data_dir(self) -> Path:
        path = Path(self.voiceloop_data_dir)
        if not path.is_absolute():
            path = LISTENER_DIR / path
        return path.resolve()

    @property
    def logs_dir(self) -> Path:
        return PROJECT_ROOT / "logs"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def corpus_style_profile_path(self) -> Path:
        path = Path(self.corpus_style_profile_file)
        if not path.is_absolute():
            path = self.corpus_dir / path
        return path.resolve()

    @property
    def routing_v2_quality_gate_path(self) -> Path:
        path = Path(self.routing_v2_quality_gate_file)
        if not path.is_absolute():
            path = self.data_dir / path
        return path.resolve()

    @property
    def routing_v2_calibration_artifact_path(self) -> Path:
        path = Path(self.routing_v2_calibration_artifact_file)
        if not path.is_absolute():
            path = self.data_dir / path
        return path.resolve()

    @property
    def routing_v2_calibration_store_path(self) -> Path:
        path = Path(self.routing_v2_calibration_store_file)
        if not path.is_absolute():
            path = self.data_dir / path
        return path.resolve()

    @property
    def vector_memory_weights(self) -> dict[str, float]:
        return {
            "semantic": self.vector_memory_weight_semantic,
            "topic": self.vector_memory_weight_topic,
            "intent": self.vector_memory_weight_intent,
            "decision": self.vector_memory_weight_decision,
            "person_context": self.vector_memory_weight_person_context,
        }

    @property
    def ui_vision_home_path(self) -> Path:
        if self.uivision_home:
            return Path(os.path.expandvars(self.uivision_home)).expanduser()
        return Path.home() / "Desktop" / "uivision"

    @property
    def voiceattack_path(self) -> Path:
        if self.voiceattack_exe:
            return Path(os.path.expandvars(self.voiceattack_exe)).expanduser()
        return (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "VoiceAttack"
            / ("VoiceAttack.exe")
        )

    def ensure_local_token(self) -> str:
        if self.voiceloop_token:
            value = self.voiceloop_token.get_secret_value().strip()
            if value:
                return value

        self.data_dir.mkdir(parents=True, exist_ok=True)
        token_path = self.data_dir / "voiceloop.token"
        if token_path.exists():
            value = token_path.read_text(encoding="utf-8").strip()
            if value:
                return value

        value = secrets.token_urlsafe(32)
        token_path.write_text(value, encoding="utf-8")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
