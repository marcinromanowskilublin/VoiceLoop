from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .schema import (
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
    content_hash_for,
)


@dataclass(frozen=True)
class ContextBudgets:
    total: int = 20
    notices: int = 2
    vector: int = 8
    manual: int = 3
    actions: int = 5


class ContextAssembler:
    """Build an ordered pack without letting appended data evict top retrieval hits."""

    def __init__(self, budgets: ContextBudgets | None = None) -> None:
        self.budgets = budgets or ContextBudgets()

    def assemble(
        self,
        *,
        question: str,
        session_id: str | None,
        vector_contexts: Sequence[str] = (),
        manual_contexts: Sequence[str] = (),
        action_summaries: Sequence[str] = (),
        notices: Sequence[str] = (),
        live_items: Sequence[ContextItemV1] = (),
    ) -> ContextPackV1:
        selected: list[ContextItemV1] = []
        seen: set[str] = set()

        def add(item: ContextItemV1) -> None:
            dedupe_key = content_hash_for(item.content)
            if dedupe_key in seen or len(selected) >= self.budgets.total:
                return
            seen.add(dedupe_key)
            selected.append(item)

        for item in live_items:
            add(item)
        self._add_strings(
            add,
            notices[: self.budgets.notices],
            source="system_notice",
            scope=ContextScope.SESSION,
            kind="notice",
            trust=ContextTrust.AUTHORITATIVE,
            reason="runtime_notice",
        )
        self._add_strings(
            add,
            vector_contexts[: self.budgets.vector],
            source="vector_memory",
            scope=ContextScope.EPISODIC,
            kind="retrieval",
            trust=ContextTrust.DERIVED,
            reason="ranked_vector_hit",
        )

        manual_budget = (
            self.budgets.manual
            if vector_contexts
            else min(self.budgets.total, max(self.budgets.manual, 12))
        )
        self._add_strings(
            add,
            manual_contexts[:manual_budget],
            source="manual_memory",
            scope=ContextScope.DURABLE,
            kind="manual",
            trust=ContextTrust.USER_ASSERTED,
            reason="explicit_user_memory",
        )
        self._add_strings(
            add,
            action_summaries[-self.budgets.actions :],
            source="recent_action",
            scope=ContextScope.SESSION,
            kind="action_summary",
            trust=ContextTrust.OBSERVED,
            reason="recent_successful_action",
        )

        counts: dict[str, int] = {}
        for item in selected:
            counts[item.source] = counts.get(item.source, 0) + 1
        return ContextPackV1(
            question=question,
            session_id=session_id,
            items=tuple(selected),
            sources=counts,
        )

    @staticmethod
    def _add_strings(
        add,
        values: Iterable[str],
        *,
        source: str,
        scope: ContextScope,
        kind: str,
        trust: ContextTrust,
        reason: str,
    ) -> None:
        for index, value in enumerate(values):
            clean = str(value or "").strip()
            if not clean:
                continue
            source_id = f"{source}:{content_hash_for(clean)[:20]}:{index}"
            add(
                ContextItemV1(
                    source=source,
                    source_id=source_id,
                    scope=scope,
                    kind=kind,
                    content=clean,
                    trust=trust,
                    confidence=1.0 if trust is ContextTrust.AUTHORITATIVE else 0.7,
                    selection_reason=reason,
                )
            )
