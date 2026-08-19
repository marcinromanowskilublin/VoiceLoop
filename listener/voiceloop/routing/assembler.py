from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import (
    CommandPlan,
    CommandRequest,
    PlanStep,
    ResolutionDecisionV1,
    ResolutionStatusV1,
    RiskLevel,
    SegmentationResultV1,
)
from .resolver import selected_candidate
from .validation import validate_arguments

_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    plan: CommandPlan | None
    blocked_reason: str | None = None


def assemble_plan(
    request: CommandRequest,
    segmentation: SegmentationResultV1,
    decisions: tuple[ResolutionDecisionV1, ...],
    *,
    definitions: list[dict[str, Any]],
    max_steps: int,
) -> AssemblyResult:
    if not segmentation.subtasks:
        return AssemblyResult(None, "no_subtasks")
    if len(segmentation.subtasks) != len(decisions):
        return AssemblyResult(None, "subtask_decision_count_mismatch")
    if len(decisions) > max(1, max_steps):
        return AssemblyResult(None, "too_many_steps")

    definitions_by_id = {
        str(definition.get("id") or ""): definition
        for definition in definitions
        if str(definition.get("id") or "")
    }
    steps: list[PlanStep] = []
    signatures: set[tuple[str, str]] = set()
    previous_step_id: str | None = None
    for subtask, decision in zip(segmentation.subtasks, decisions, strict=True):
        if decision.decision is not ResolutionStatusV1.RESOLVED:
            return AssemblyResult(
                None,
                f"subtask_{subtask.order}_{decision.decision.value}:{decision.reason or 'unknown'}",
            )
        candidate = selected_candidate(decision)
        if candidate is None:
            return AssemblyResult(None, f"subtask_{subtask.order}_missing_candidate")
        definition = definitions_by_id.get(candidate.action_id)
        if definition is None:
            return AssemblyResult(None, f"unknown_action:{candidate.action_id}")
        argument_errors = validate_arguments(
            candidate.extracted_args,
            definition.get("args_schema"),
        )
        if argument_errors:
            return AssemblyResult(None, argument_errors[0])
        try:
            risk = RiskLevel(str(definition.get("risk") or RiskLevel.LOW.value))
        except ValueError:
            return AssemblyResult(None, f"invalid_risk:{candidate.action_id}")
        confirmation_required = bool(definition.get("confirmation_required"))
        if risk is RiskLevel.HIGH:
            confirmation_required = True
        signature = (
            candidate.action_id,
            repr(sorted(candidate.extracted_args.items())),
        )
        if signature in signatures:
            return AssemblyResult(None, f"duplicate_action:{candidate.action_id}")
        signatures.add(signature)
        step = PlanStep(
            action_id=candidate.action_id,
            args=dict(candidate.extracted_args),
            depends_on=[previous_step_id] if previous_step_id else [],
            risk=risk,
            confirmation_required=confirmation_required,
        )
        steps.append(step)
        previous_step_id = step.id

    conflict = _find_conflict(segmentation, steps)
    if conflict is not None:
        return AssemblyResult(None, conflict)

    response_text = _response_text(steps)
    confidence = min(
        segmentation.confidence,
        *(
            max(0.0, min(decision.candidates[0].combined_score, 1.0))
            for decision in decisions
            if decision.candidates
        ),
    )
    plan = CommandPlan(
        request_id=request.request_id,
        intent="task",
        response_text=response_text,
        confidence=confidence,
        steps=steps,
        provider="routing_v2",
    )
    validation_errors = validate_plan(
        plan,
        definitions=definitions,
        max_steps=max_steps,
    )
    if validation_errors:
        return AssemblyResult(None, validation_errors[0])
    return AssemblyResult(plan)


def clarification_plan(
    request: CommandRequest,
    *,
    reason: str,
    question: str | None = None,
) -> CommandPlan:
    return CommandPlan(
        request_id=request.request_id,
        intent="task",
        response_text="Nie wykonuję polecenia, dopóki wszystkie jego części nie są jednoznaczne.",
        confidence=0.0,
        requires_clarification=True,
        clarification_question=question or _clarification_question(reason),
        steps=[],
        provider="routing_v2_guard",
    )


def validate_plan(
    plan: CommandPlan,
    *,
    definitions: list[dict[str, Any]],
    max_steps: int,
) -> list[str]:
    errors: list[str] = []
    if len(plan.steps) > max(1, max_steps):
        errors.append("too_many_steps")
    definitions_by_id = {
        str(definition.get("id") or ""): definition
        for definition in definitions
        if str(definition.get("id") or "")
    }
    seen_ids: set[str] = set()
    for step in plan.steps:
        definition = definitions_by_id.get(step.action_id)
        if definition is None:
            errors.append(f"unknown_action:{step.action_id}")
            continue
        if any(dependency not in seen_ids for dependency in step.depends_on):
            errors.append(f"invalid_dependency:{step.action_id}")
        seen_ids.add(step.id)
        errors.extend(validate_arguments(step.args, definition.get("args_schema")))
        try:
            catalog_risk = RiskLevel(
                str(definition.get("risk") or RiskLevel.LOW.value)
            )
        except ValueError:
            errors.append(f"invalid_risk:{step.action_id}")
            continue
        if _RISK_ORDER[step.risk] < _RISK_ORDER[catalog_risk]:
            errors.append(f"risk_policy_mismatch:{step.action_id}")
        required_confirmation = bool(definition.get("confirmation_required"))
        required_confirmation = required_confirmation or catalog_risk is RiskLevel.HIGH
        if required_confirmation and not step.confirmation_required:
            errors.append(f"confirmation_policy_mismatch:{step.action_id}")
    return list(dict.fromkeys(errors))


def _find_conflict(
    segmentation: SegmentationResultV1,
    steps: list[PlanStep],
) -> str | None:
    operations_by_target: dict[str, set[str]] = {}
    for subtask, step in zip(segmentation.subtasks, steps, strict=True):
        target = subtask.target
        operation = subtask.operation
        if not target or not operation:
            continue
        operations = operations_by_target.setdefault(target, set())
        operations.add(operation)
        if "open" in operations and ("close" in operations or "minimize" in operations):
            return f"conflicting_operations:{target}"
        if step.risk is RiskLevel.HIGH and not step.confirmation_required:
            return f"unconfirmed_high_risk:{step.action_id}"
    return None


def _response_text(steps: list[PlanStep]) -> str:
    actions = ", ".join(step.action_id for step in steps)
    if any(step.confirmation_required for step in steps):
        return (
            f"Przygotowałam pełny plan: {actions}. "
            "Plan wymaga potwierdzenia przed wykonaniem."
        )
    return f"Przygotowałam pełny plan: {actions}."


def _clarification_question(reason: str) -> str:
    if "target_identity_not_supported" in reason:
        return (
            "Nie mam bezpiecznej akcji zamykania okna po samej nazwie. "
            "Czy mam zamknąć okno wskazane kursorem?"
        )
    if "missing_required_argument" in reason:
        return "Brakuje argumentu jednej z czynności. Co dokładnie mam przekazać?"
    if "target_context_missing" in reason:
        return "Rozpoznaję czynność, ale brakuje celu. Co dokładnie mam wskazać?"
    if "no_capability_candidates" in reason:
        return (
            "Tego jeszcze nie wykonuję. Powiedz, jaki efekt chcesz uzyskać, "
            "a podam najbliższą bezpieczną możliwość."
        )
    if "low_stt_confidence" in reason:
        return "Nie dosłyszałam jednej z części. Powtórz proszę całe polecenie."
    if "low_top2_margin" in reason:
        return "Dwie możliwości są zbyt podobne. Którą czynność masz na myśli?"
    if "low_combined_score" in reason or "single_candidate_without_comparator" in reason:
        return (
            "Rozpoznaję możliwą czynność, ale nie mam wystarczającej pewności. "
            "Powiedz krótko: co zrobić i na jakim obiekcie?"
        )
    if "operation_mismatch" in reason or "target_mismatch" in reason:
        return (
            "Rozpoznany cel nie pasuje do dostępnej czynności. "
            "Czy chcesz go otworzyć, opisać, skopiować czy zamknąć?"
        )
    return "Doprecyzuj proszę wszystkie czynności w kolejności wykonania."
