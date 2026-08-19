from voiceloop.behavior_digest import DigestedMemory, LocalBehaviorDigestClient
from voiceloop.memory_vectorization import (
    MEMORY_QUERY_DOCUMENTS_VERSION,
    memory_query_documents,
    memory_query_weights,
)
from voiceloop.qdrant_memory import VECTOR_NAMES


def test_fallback_digest_builds_only_evidenced_semantic_document() -> None:
    digest = LocalBehaviorDigestClient.fallback(
        title="Cursor: VoiceLoop",
        content="Użytkownik pracuje nad lokalną pamięcią Qdrant.",
    )

    documents = digest.vector_documents(
        title="Cursor: VoiceLoop",
        raw_content="Użytkownik pracuje nad lokalną pamięcią Qdrant.",
    )

    assert tuple(documents) == ("semantic",)
    assert "Qdrant" in documents["semantic"]


def test_vector_documents_omit_empty_aspects_and_raw_ocr() -> None:
    digest = DigestedMemory(
        summary="Krótki opis pracy.",
        topic="",
        intent="Naprawić indeks.",
        decision="",
        person_context="",
        people=["Marcin"],
        observations=["Indeks zwrócił błąd.", "Krótki opis pracy."],
        confidence=0.9,
    )

    documents = digest.vector_documents(
        title="Tytuł nie jest fallbackiem tematu",
        raw_content="SUROWY_OCR_NIE_MOŻE_TRAFIĆ_DO_EMBEDDINGU " * 200,
    )

    assert tuple(documents) == ("semantic", "intent")
    assert "Indeks zwrócił błąd." in documents["semantic"]
    assert "SUROWY_OCR" not in documents["semantic"]
    assert "Tytuł nie jest fallbackiem" not in "\n".join(documents.values())


def test_five_present_aspects_build_five_distinct_documents() -> None:
    digest = DigestedMemory(
        summary="Podsumowanie.",
        topic="Pamięć",
        intent="Poprawić wyszukiwanie",
        decision="Użyć RRF",
        person_context="Rozmowa z Mikołajem",
    )

    documents = digest.vector_documents(title="VoiceLoop", raw_content="surowe")

    assert tuple(documents) == VECTOR_NAMES
    assert len(set(documents.values())) == len(VECTOR_NAMES)


def test_memory_query_documents_are_versioned_distinct_and_weighted() -> None:
    query = "Jaką decyzję ustaliliśmy w sprawie Qdrant?"

    documents = memory_query_documents(query, version=MEMORY_QUERY_DOCUMENTS_VERSION)
    adaptive = memory_query_weights(query)
    static = memory_query_weights(query, adaptive=False)

    assert tuple(documents) == VECTOR_NAMES
    assert len(set(documents.values())) == len(VECTOR_NAMES)
    assert all(query in document for document in documents.values())
    assert adaptive["decision"] > static["decision"]
    assert sum(adaptive.values()) == 1.0


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
