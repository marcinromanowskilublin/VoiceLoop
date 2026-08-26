import pytest

from voiceloop.hume_emotion import HumeEmotionClient
from voiceloop.settings import Settings


def test_hume_auth_is_sent_in_header_not_uri(tmp_path) -> None:
    client = HumeEmotionClient(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            hume_emotion_analysis_enabled=True,
            hume_api_key="test-key",
            hume_emotion_endpoint=(
                "wss://api.hume.ai/v0/evi/chat"
                "?api_key=legacy&access_token=legacy-token&custom=value"
            ),
        )
    )

    uri = client._endpoint_uri()

    assert "api_key=" not in uri
    assert "access_token=" not in uri
    assert "custom=value" in uri
    assert "verbose_transcription=true" in uri
    assert client._connection_headers() == {"X-Hume-Api-Key": "test-key"}


def test_hume_emotion_parser_extracts_top_prosody_windows(tmp_path) -> None:
    client = HumeEmotionClient(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            hume_emotion_analysis_enabled=True,
            hume_api_key="test",
            hume_emotion_top_n=2,
        )
    )

    windows = client._extract_windows(
        {
            "prosody": {
                "predictions": [
                    {
                        "time": {"begin": 0.0, "end": 1.5},
                        "emotions": [
                            {"name": "Calmness", "score": 0.7},
                            {"name": "Interest", "score": 0.4},
                            {"name": "Anger", "score": 0.1},
                        ],
                    },
                    {
                        "time": {"begin": 1.5, "end": 3.0},
                        "emotions": [
                            {"name": "Joy", "score": 0.8},
                            {"name": "Excitement", "score": 0.5},
                        ],
                    },
                ]
            }
        }
    )

    assert len(windows) == 2
    assert [emotion.name for emotion in windows[0].emotions] == ["Calmness", "Interest"]
    assert windows[0].begin_seconds == 0.0
    assert windows[0].end_seconds == 1.5


def test_hume_emotion_parser_extracts_evi_prosody_scores(tmp_path) -> None:
    client = HumeEmotionClient(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            hume_emotion_analysis_enabled=True,
            hume_api_key="test",
            hume_emotion_top_n=2,
        )
    )

    windows = client._extract_windows(
        {
            "type": "user_message",
            "message": {"role": "user", "content": "dzień dobry"},
            "models": {
                "prosody": {
                    "scores": {
                        "Calmness": 0.72,
                        "Interest": 0.54,
                        "Anger": 0.08,
                    }
                }
            },
        }
    )

    assert len(windows) == 1
    assert [emotion.name for emotion in windows[0].emotions] == ["Calmness", "Interest"]
    assert windows[0].begin_seconds is None
    assert windows[0].end_seconds is None


def test_hume_emotions_are_weighted_for_segment_interval(tmp_path) -> None:
    client = HumeEmotionClient(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            hume_emotion_analysis_enabled=True,
            hume_api_key="test",
            hume_emotion_top_n=2,
        )
    )
    windows = client._extract_windows(
        {
            "prosody": {
                "predictions": [
                    {
                        "time": {"begin": 0.0, "end": 1.0},
                        "emotions": [{"name": "Calmness", "score": 0.8}],
                    },
                    {
                        "time": {"begin": 1.0, "end": 3.0},
                        "emotions": [{"name": "Interest", "score": 0.6}],
                    },
                ]
            }
        }
    )

    emotions = client.emotions_for_interval(
        windows,
        begin_seconds=0.5,
        end_seconds=2.5,
    )

    assert [item["name"] for item in emotions] == ["Interest", "Calmness"]
    assert emotions[0]["score"] == pytest.approx(0.45)
    assert emotions[1]["score"] == pytest.approx(0.2)
