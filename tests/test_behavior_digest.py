from voiceloop.behavior_digest import LocalBehaviorDigestClient
from voiceloop.qdrant_memory import VECTOR_NAMES


def test_fallback_digest_always_builds_five_vector_documents() -> None:
    digest = LocalBehaviorDigestClient.fallback(
        title="Cursor: VoiceLoop",
        content="Użytkownik pracuje nad lokalną pamięcią Qdrant.",
    )

    documents = digest.vector_documents(
        title="Cursor: VoiceLoop",
        raw_content="Użytkownik pracuje nad lokalną pamięcią Qdrant.",
    )

    assert tuple(documents) == VECTOR_NAMES
    assert all(documents[name].strip() for name in VECTOR_NAMES)
    assert "Qdrant" in documents["semantic"]


def test_digest_coerces_qwen_alternative_schema() -> None:
    digest = LocalBehaviorDigestClient._coerce_digest(
        {
            "title": "Aktywność użytkownika",
            "themes": [
                {
                    "name": "VoiceLoop",
                    "description": "Rozwijanie lokalnej pamięci.",
                    "confidence": 0.9,
                }
            ],
            "goals": [{"name": "Wdrożyć Qdrant", "confidence": 0.8}],
            "decisions": [{"name": "Użyć pięciu wektorów", "confidence": 1.0}],
            "relations": [{"name": "Marcin", "description": "użytkownik"}],
            "observations": [
                {
                    "time": "04:00",
                    "description": "Edytowano konfigurację.",
                    "confidence": 0.9,
                }
            ],
        },
        title="Cursor: VoiceLoop",
        source_content="Dane źródłowe.",
    )

    assert digest.topic.startswith("VoiceLoop")
    assert digest.intent == "Wdrożyć Qdrant"
    assert digest.decision == "Użyć pięciu wektorów"
    assert "Marcin" in digest.person_context
    assert digest.observations == ["[04:00] Edytowano konfigurację."]
    assert digest.confidence == 0.9
