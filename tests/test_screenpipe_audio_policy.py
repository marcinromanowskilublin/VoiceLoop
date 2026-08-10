from voiceloop.screenpipe_audio_policy import DeepgramAudioPolicy
from voiceloop.settings import Settings


def policy() -> DeepgramAudioPolicy:
    return DeepgramAudioPolicy(Settings())


def test_youtube_is_blocked_even_when_meeting_is_detected() -> None:
    decision = policy().decide(
        app_name="chrome.exe",
        window_name="Wykład - YouTube",
        browser_url="https://www.youtube.com/watch?v=abc",
        meeting_detected=True,
        microphone_active=True,
    )

    assert decision.allowed is False
    assert decision.reason == "youtube_blocked"


def test_youtube_subdomain_is_blocked() -> None:
    decision = policy().decide(
        browser_url="https://music.youtube.com/watch?v=abc",
        microphone_active=True,
    )

    assert decision.allowed is False
    assert decision.reason == "youtube_blocked"


def test_known_browser_call_requires_microphone_activity() -> None:
    blocked = policy().decide(
        browser_url="https://meet.google.com/abc-defg-hij",
        microphone_active=False,
    )
    allowed = policy().decide(
        browser_url="https://meet.google.com/abc-defg-hij",
        microphone_active=True,
    )

    assert blocked.allowed is False
    assert allowed.allowed is True
    assert allowed.reason == "allowed_call_host"


def test_unknown_media_fails_closed() -> None:
    decision = policy().decide(
        app_name="chrome.exe",
        browser_url="https://example.com/podcast",
        microphone_active=True,
    )

    assert decision.allowed is False
    assert decision.reason == "not_a_known_call"
