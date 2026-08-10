from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from .settings import Settings


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip().casefold() for item in value.split(",") if item.strip())


def _hostname(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return (parsed.hostname or "").casefold().rstrip(".")


def _matches_host(host: str, configured: tuple[str, ...]) -> bool:
    return any(host == item or host.endswith(f".{item}") for item in configured)


@dataclass(frozen=True)
class DeepgramAudioDecision:
    allowed: bool
    reason: str


class DeepgramAudioPolicy:
    """Fail-closed routing policy for Screenpipe audio.

    YouTube is a hard deny regardless of meeting signals. Other audio reaches
    Deepgram only when Screenpipe reports a meeting or a configured call
    application/domain is active together with microphone activity.
    """

    def __init__(self, settings: Settings) -> None:
        self.blocked_hosts = _csv_values(settings.screenpipe_deepgram_blocked_hosts)
        self.call_hosts = _csv_values(settings.screenpipe_deepgram_call_hosts)
        self.call_apps = _csv_values(settings.screenpipe_deepgram_call_apps)

    def decide(
        self,
        *,
        app_name: str = "",
        window_name: str = "",
        browser_url: str = "",
        meeting_detected: bool = False,
        microphone_active: bool = False,
    ) -> DeepgramAudioDecision:
        host = _hostname(browser_url)
        combined = f"{app_name} {window_name}".casefold()

        if _matches_host(host, self.blocked_hosts) or "youtube" in combined:
            return DeepgramAudioDecision(False, "youtube_blocked")

        if meeting_detected:
            return DeepgramAudioDecision(True, "screenpipe_meeting")

        if not microphone_active:
            return DeepgramAudioDecision(False, "microphone_inactive")

        if _matches_host(host, self.call_hosts):
            return DeepgramAudioDecision(True, "allowed_call_host")

        if any(app in combined for app in self.call_apps):
            return DeepgramAudioDecision(True, "allowed_call_app")

        return DeepgramAudioDecision(False, "not_a_known_call")
