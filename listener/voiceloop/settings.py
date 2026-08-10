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

    n8n_base_url: str = "http://127.0.0.1:5678"
    n8n_webhook_url: str = "http://127.0.0.1:5678/webhook/voice-command-v1"
    n8n_token: SecretStr | None = None
    n8n_timeout_seconds: float = 5.0

    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_api_key: SecretStr = SecretStr("lm-studio")
    lm_studio_model: str | None = None
    lm_studio_timeout_seconds: float = 90.0

    llm_primary: str = "local"
    cloud_llm_enabled: bool = False
    cloud_llm_base_url: str | None = None
    cloud_llm_api_key: SecretStr | None = None
    cloud_llm_model: str | None = None

    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"
    web_search_fallback_provider: str = "duckduckgo"
    web_search_api_key: SecretStr | None = None
    web_search_gemini_model: str = "gemini-3.6-flash"
    web_search_timeout_seconds: float = 6.0
    web_search_max_results: int = 5

    local_embeddings_enabled: bool = True
    local_embeddings_base_url: str | None = None
    local_embeddings_api_key: SecretStr | None = None
    local_embeddings_model: str | None = None
    local_embeddings_timeout_seconds: float = 30.0
    vector_memory_context_limit: int = 8
    qdrant_enabled: bool = True
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "voiceloop_memory"
    qdrant_timeout_seconds: float = 10.0
    qdrant_dual_write: bool = True
    behavior_digest_enabled: bool = True
    behavior_digest_model: str | None = None
    behavior_digest_timeout_seconds: float = 240.0
    behavior_digest_poll_seconds: int = 60
    behavior_digest_recent_minutes: int = 30
    behavior_digest_max_contexts: int = 24

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
    screenpipe_vector_memory_enabled: bool = True
    screenpipe_vector_poll_seconds: int = 120
    screenpipe_vector_recent_minutes: int = 60

    voiceattack_exe: str | None = None
    azure_tts_enabled: bool = False
    azure_tts_key: SecretStr | None = None
    azure_tts_region: str | None = None
    azure_tts_voice: str = "pl-PL-ZofiaNeural"
    azure_tts_timeout_seconds: float = 20.0
    uivision_home: str | None = None
    uivision_timeout_seconds: int = 60

    command_dedupe_seconds: float = 2.0
    command_queue_limit: int = 10
    auto_start_listening: bool = False
    deepgram_model: str = "nova-3"
    deepgram_language: str = "pl"
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
