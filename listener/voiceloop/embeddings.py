from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import SecretStr

from .corpus.local_only import LocalOnlyViolation, require_loopback_url

EMBEDDING_PREFIX_POLICY_VERSION = "nomic-search-prefixes-v1"
EMBEDDING_QUERY_INPUT_KIND = "query"
EMBEDDING_DOCUMENT_INPUT_KIND = "document"
EMBEDDING_QUERY_PREFIX = "search_query:"
EMBEDDING_DOCUMENT_PREFIX = "search_document:"
EmbeddingInputKind = Literal["query", "document"]


class EmbeddingUnavailableError(RuntimeError):
    pass


def with_embedding_prefix(text: str, prefix: str) -> str:
    """Apply an embedding-model prefix once, preserving already-prefixed inputs."""

    clean = text.strip()
    normalized_prefix = prefix.strip()
    if not normalized_prefix.endswith(":"):
        normalized_prefix = f"{normalized_prefix}:"
    if clean.casefold().startswith(normalized_prefix.casefold()):
        return clean
    return f"{normalized_prefix} {clean}"


def embedding_prefix_metadata(kind: EmbeddingInputKind) -> dict[str, str]:
    prefix = (
        EMBEDDING_QUERY_PREFIX if kind == EMBEDDING_QUERY_INPUT_KIND else EMBEDDING_DOCUMENT_PREFIX
    )
    return {
        "embedding_prefix_policy": EMBEDDING_PREFIX_POLICY_VERSION,
        "embedding_input_kind": kind,
        "embedding_prefix": prefix,
    }


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str | None,
        timeout_seconds: float,
        enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.configured_model = model
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        self._resolved_model: str | None = None

    def accepts_private_text(self) -> bool:
        if not self.enabled:
            return False
        try:
            require_loopback_url(self.base_url)
        except LocalOnlyViolation:
            return False
        return True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            key = self.api_key.get_secret_value().strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    async def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączone w konfiguracji"
        try:
            model = await self.resolve_model()
        except EmbeddingUnavailableError as exc:
            return False, str(exc)
        return True, f"model: {model}"

    async def resolve_model(self) -> str:
        if self.configured_model:
            return self.configured_model
        if self._resolved_model:
            return self._resolved_model

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingUnavailableError(f"embedding models unavailable: {exc}") from exc

        models = [
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        embedding_models = [model for model in models if "embed" in model.lower()]
        candidates = embedding_models or models
        if not candidates:
            raise EmbeddingUnavailableError("embedding API works, but no model is loaded")
        self._resolved_model = candidates[0]
        return self._resolved_model

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0] if vectors else []

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_queries([text])
        return vectors[0] if vectors else []

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        queries = [
            with_embedding_prefix(text, EMBEDDING_QUERY_PREFIX)
            for text in texts
            if text.strip()
        ]
        return await self.embed_texts(queries)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        documents = [
            with_embedding_prefix(text, EMBEDDING_DOCUMENT_PREFIX)
            for text in texts
            if text.strip()
        ]
        return await self.embed_texts(documents)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            return []
        clean_texts = [text.strip()[:8000] for text in texts if text.strip()]
        if not clean_texts:
            return []
        model = await self.resolve_model()
        body: dict[str, Any] = {
            "model": model,
            "input": clean_texts,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json=body,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise EmbeddingUnavailableError(
                f"embedding request failed ({exc.response.status_code}): {detail}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingUnavailableError(f"embedding request failed: {exc}") from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingUnavailableError("embedding response has no data list")
        ordered = sorted(
            (item for item in data if isinstance(item, dict)),
            key=lambda item: int(item.get("index", 0)),
        )
        vectors: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding")
            if isinstance(embedding, list):
                vectors.append([float(value) for value in embedding])
        if len(vectors) != len(clean_texts):
            raise EmbeddingUnavailableError("embedding response count mismatch")
        return vectors
