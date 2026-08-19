from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from voiceloop.capability_index import (
    CapabilityDocuments,
    CapabilityMatch,
    CapabilitySearchResult,
    SubtaskCapabilitySearch,
)
from voiceloop.corpus.schema import RoutingV2RuntimeConfig
from voiceloop.models import (
    CommandPlan,
    CommandRequest,
    CommandSource,
    PlanStep,
    ResolutionStatusV1,
    RiskLevel,
    SubtaskEmbeddingV1,
    TranscriptEnvelopeV1,
    TranscriptWordV1,
)
from voiceloop.router import deterministic_plan
from voiceloop.routing.assembler import assemble_plan, validate_plan
from voiceloop.routing.resolver import resolve_subtasks
from voiceloop.routing.segmenter import segment_command
from voiceloop.routing.service import RoutingV2Outcome, RoutingV2Service
from voiceloop.settings import Settings


def definition(
    action_id: str,
    *,
    schema: dict | None = None,
    risk: str = "low",
    confirmation: bool = False,
) -> dict:
    return {
        "id": action_id,
        "label": action_id.replace("_", " "),
        "description": action_id.replace("_", " "),
        "args_schema": schema
        or {"type": "object", "properties": {}, "additionalProperties": False},
        "risk": risk,
        "confirmation_required": confirmation,
        "routing_examples": [],
        "available_in_voiceattack": False,
    }


def search_for(
    subtask,
    matches: list[tuple[str, float, dict[str, float]]],
    *,
    catalog_hash: str = "catalog",
) -> SubtaskCapabilitySearch:
    dimension = 3
    embedding = SubtaskEmbeddingV1(
        subtask_id=subtask.subtask_id,
        semantic=(1.0, 0.0, 0.0),
        intent=(0.0, 1.0, 0.0),
        target_context=(0.0, 0.0, 1.0),
        embedding_model="fake-embedding",
        dimension=dimension,
        normalized_text_sha256=SubtaskEmbeddingV1.text_hash(subtask.normalized_text),
    )
    result = CapabilitySearchResult(
        query_documents=CapabilityDocuments(
            semantic=subtask.text,
            intent=subtask.operation or "",
            target_context=subtask.target or "",
        ),
        matches=[
            CapabilityMatch(
                action_id=action_id,
                label=action_id,
                description=action_id,
                score=score,
                vector_scores=vector_scores,
                risk="low",
                confirmation_required=False,
                available_in_voiceattack=False,
            )
            for action_id, score, vector_scores in matches
        ],
        catalog_hash=catalog_hash,
    )
    return SubtaskCapabilitySearch(
        subtask=subtask,
        embedding=embedding,
        result=result,
    )


def test_transcript_envelope_preserves_word_confidence_and_speakers() -> None:
    envelope = TranscriptEnvelopeV1(
        raw_text="Otwórz Chrome",
        normalized_text="Otwórz Chrome",
        confidence_mean=0.91,
        confidence_min=0.84,
        words=(
            TranscriptWordV1(
                word="otwórz",
                start_seconds=1.0,
                end_seconds=1.4,
                confidence=0.84,
                speaker_id=0,
            ),
            TranscriptWordV1(
                word="chrome",
                start_seconds=1.5,
                end_seconds=1.9,
                confidence=0.98,
                speaker_id=0,
            ),
        ),
        started_at_seconds=1.0,
        ended_at_seconds=1.9,
        speaker_ids=(0,),
        is_final=True,
        speech_final=True,
        model="nova-3",
    )

    request = CommandRequest.from_transcript(envelope)

    assert request.text == "Otwórz Chrome"
    assert request.transcript_confidence == pytest.approx(0.91)
    assert request.effective_transcript_confidence == pytest.approx(0.84)
    assert request.transcript is not None
    assert request.transcript.speaker_ids == (0,)
    assert request.transcript.words[0].confidence == pytest.approx(0.84)


def test_segmenter_preserves_three_ordered_subtasks_and_raw_spans() -> None:
    text = "zamknij UI Vision, otwórz Chrome i YouTube"

    result = segment_command(text)

    assert result.decision.value == "compound"
    assert [subtask.operation for subtask in result.subtasks] == [
        "close",
        "open",
        "open",
    ]
    assert [subtask.target for subtask in result.subtasks] == [
        "UI Vision",
        "browser",
        "YouTube",
    ]
    assert [subtask.order for subtask in result.subtasks] == [0, 1, 2]
    assert result.subtasks[2].text == "otwórz YouTube"
    assert result.subtasks[2].source_text == "YouTube"
    for subtask in result.subtasks:
        assert text[subtask.start_char : subtask.end_char].strip() == subtask.source_text
    assert result.unrecognized_spans == ()


def test_segmenter_does_not_split_conjunction_inside_note_argument() -> None:
    result = segment_command("zapisz notatkę kup mleko i chleb")

    assert result.decision.value == "simple"
    assert len(result.subtasks) == 1
    assert result.subtasks[0].operation == "create_note"
    assert "mleko i chleb" in result.subtasks[0].raw_arguments["tail"]


def test_segmenter_does_not_treat_named_note_content_as_second_action() -> None:
    result = segment_command("zapisz notatkę kup mleko i Chrome")

    assert result.decision.value == "simple"
    assert len(result.subtasks) == 1
    assert result.subtasks[0].operation == "create_note"
    assert "mleko i Chrome" in result.subtasks[0].raw_arguments["tail"]


def test_segmenter_marks_unknown_coordinated_prefix_as_ambiguous() -> None:
    result = segment_command("wyślij e-mail i otwórz Chrome")

    assert result.decision.value == "ambiguous"
    assert result.reason == "unrecognized_command_fragment"
    assert len(result.unrecognized_spans) == 1
    assert result.unrecognized_spans[0].text == "wyślij e-mail"
    assert [subtask.target for subtask in result.subtasks] == [None, "browser"]


def test_segmenter_rejects_adjacent_unsupported_command_without_separator() -> None:
    result = segment_command("wyślij e-mail otwórz Chrome")

    assert result.decision.value == "ambiguous"
    assert result.reason == "missing_separator_before_command"
    assert result.subtasks == ()


def test_segmenter_does_not_split_period_inside_url() -> None:
    result = segment_command("otwórz https://example.com")

    assert result.decision.value == "simple"
    assert len(result.subtasks) == 1


def test_segmenter_keeps_spaced_polish_domain_atomic() -> None:
    result = segment_command("Otwórz onet. Pl")

    assert result.decision.value == "simple"
    assert len(result.subtasks) == 1
    assert result.subtasks[0].target == "url"


def test_segmenter_maps_session_targets() -> None:
    this_pc = segment_command("Otwórz mój komputer")
    whatsapp = segment_command("Otwórz Whatsap")
    devilpage = segment_command("Otwórz stronę devilpage. Pl")

    assert this_pc.subtasks[0].target == "this_pc"
    assert whatsapp.subtasks[0].target == "WhatsApp"
    assert devilpage.subtasks[0].target == "url"


def test_default_canary_includes_session_launch_actions() -> None:
    ids = {
        item.strip()
        for item in Settings().routing_v2_canary_action_ids.split(",")
        if item.strip()
    }

    assert {"open_url", "open_browser", "open_folder", "open_app"} <= ids


def test_segmenter_understands_personal_check_contexts() -> None:
    recent = segment_command(
        "Weź sobie sprawdź w Screenpipe, co robiłem przed chwilą"
    )
    text_target = segment_command("sprawdź gdzie trafi teraz wpisywany tekst")

    assert recent.decision.value == "simple"
    assert recent.subtasks[0].operation == "describe"
    assert text_target.decision.value == "simple"
    assert text_target.subtasks[0].operation == "describe"


def test_segmenter_does_not_turn_third_party_web_query_into_local_activity() -> None:
    result = segment_command("sprawdź co robił Elon Musk")

    assert result.decision.value == "simple"
    assert result.subtasks[0].operation == "search"


def test_segmenter_allows_if_it_means_about() -> None:
    result = segment_command(
        "Szukaj jak przebiegała sesja jeśli chodzi o Rocket Lab"
    )

    assert result.decision.value == "simple"
    assert result.subtasks[0].operation == "search"


def test_segmenter_detects_alias_clause_before_explicit_command() -> None:
    result = segment_command("kalendarz i otwórz Chrome")

    assert result.decision.value == "compound"
    assert [subtask.operation for subtask in result.subtasks] == ["open", "open"]
    assert [subtask.target for subtask in result.subtasks] == [
        "calendar",
        "browser",
    ]
    assert result.subtasks[0].source_text == "kalendarz"
    assert result.subtasks[0].text == "otwórz kalendarz"


def test_segmenter_keeps_api_endpoint_check_atomic() -> None:
    result = segment_command("sprawdź API Deepgram i sprawdź endpoint /v1/listen")

    assert result.decision.value == "simple"
    assert len(result.subtasks) == 1


def test_compound_fast_path_fails_closed() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="zamknij UI Vision, otwórz Chrome i YouTube",
            transcript_confidence=0.96,
        )
    )

    assert plan is not None
    assert plan.provider == "compound_fast_path_guard"
    assert plan.requires_clarification is True
    assert plan.steps == []


def test_compound_fast_path_blocks_alias_clause_too() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="kalendarz i otwórz Chrome",
            transcript_confidence=0.96,
        )
    )

    assert plan is not None
    assert plan.provider == "compound_fast_path_guard"
    assert plan.steps == []


def test_compound_fast_path_blocks_unknown_clause_before_known_action() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="wyślij e-mail i otwórz Chrome",
            transcript_confidence=0.96,
        )
    )

    assert plan is not None
    assert plan.provider == "compound_fast_path_guard"
    assert plan.steps == []


@pytest.mark.parametrize(
    "alias_clause",
    [
        "kalendarz",
        "ostatnia aktywność",
        "czy to pasek adresu",
        "numer do schowka",
        "jakie masz komendy",
        "pulpit",
    ],
)
def test_compound_fast_path_blocks_existing_deterministic_aliases(
    alias_clause,
) -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text=f"{alias_clause} i otwórz Chrome",
            transcript_confidence=0.96,
        )
    )

    assert plan is not None
    assert plan.provider == "compound_fast_path_guard"
    assert plan.steps == []


@pytest.mark.parametrize(
    "known_command",
    [
        "Otwórz kalendarz",
        "Kalendarz",
        "Odpal przeglądarkę",
        "Przeglądarka",
        "Otwórz GPT",
        "Otwórz Gemini",
        "Jakie okno jest aktywne",
        "Czy to dobre pole do pisania",
        "Zwiń aktywne okno",
        "Minimalizuj wszystko",
        "Zminimalizuj aplikację pod kursorem",
        "Kopiuj zaznaczony tekst",
        "Skopiuj numer pod kursorem",
        "Skopiuj tekst pod kursorem",
        "Zaznacz to zdanie pod kursorem",
        "Zaznacz akapit",
        "Wyłącz aplikację którą wskazuję kursorem",
        "Wyszukaj w internecie Python 3.13",
        "Jaka jest pogoda",
        "Zapamiętaj ostatnie źródło 2",
        "Zmień nazwę pod kursorem na Raport Q2",
        "Podsumuj aktywność z ostatnich 2 godzin",
        "Zapisz notatkę kup mleko",
        "Zapamiętaj że wolę krótkie odpowiedzi",
    ],
)
def test_unknown_prefix_cannot_be_dropped_before_any_fast_path_action(
    known_command,
) -> None:
    simple = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text=known_command,
            transcript_confidence=0.96,
        )
    )
    assert simple is not None
    assert simple.steps

    guarded = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text=f"wyślij e-mail i {known_command}",
            transcript_confidence=0.96,
        )
    )

    if guarded is not None:
        assert guarded.provider == "compound_fast_path_guard"
        assert guarded.steps == []


def test_resolver_rejects_named_window_but_resolves_browser_and_youtube() -> None:
    segmentation = segment_command("zamknij UI Vision, otwórz Chrome i YouTube")
    close, browser, youtube = segmentation.subtasks
    definitions = [
        definition(
            "close_window_under_cursor",
            risk="medium",
            confirmation=True,
        ),
        definition("open_browser"),
        definition(
            "open_url",
            schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
    ]
    searches = (
        search_for(
            close,
            [
                (
                    "close_window_under_cursor",
                    0.95,
                    {"semantic": 0.95, "intent": 0.98, "target_context": 0.92},
                ),
                (
                    "open_browser",
                    0.20,
                    {"semantic": 0.20, "intent": 0.10, "target_context": 0.30},
                ),
            ],
        ),
        search_for(
            browser,
            [
                (
                    "open_browser",
                    0.94,
                    {"semantic": 0.94, "intent": 0.96, "target_context": 0.92},
                ),
                (
                    "open_url",
                    0.45,
                    {"semantic": 0.45, "intent": 0.70, "target_context": 0.20},
                ),
            ],
        ),
        search_for(
            youtube,
            [
                (
                    "open_url",
                    0.96,
                    {"semantic": 0.96, "intent": 0.95, "target_context": 0.97},
                ),
                (
                    "open_browser",
                    0.40,
                    {"semantic": 0.40, "intent": 0.70, "target_context": 0.10},
                ),
            ],
        ),
    )

    decisions = resolve_subtasks(
        searches,
        definitions=definitions,
        transcript_confidence=0.96,
        min_score=0.60,
        min_margin=0.10,
        stt_threshold=0.75,
    )

    assert decisions[0].decision is ResolutionStatusV1.UNSUPPORTED
    assert decisions[0].reason == "target_identity_not_supported"
    assert decisions[1].decision is ResolutionStatusV1.RESOLVED
    assert decisions[1].top1_action_id == "open_browser"
    assert decisions[2].decision is ResolutionStatusV1.RESOLVED
    assert decisions[2].top1_action_id == "open_url"
    assert decisions[2].candidates[0].extracted_args == {"url": "https://www.youtube.com"}


def test_resolver_uses_top2_margin_instead_of_guessing() -> None:
    subtask = segment_command("zapamiętaj ostatnie źródło").subtasks[0]
    definitions = [
        definition(
            "remember",
            schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        ),
        definition(
            "remember_last_source",
            schema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "kind": {"type": "string"},
                },
                "required": ["index"],
                "additionalProperties": False,
            },
        ),
    ]
    search = search_for(
        subtask,
        [
            (
                "remember",
                0.90,
                {"semantic": 0.90, "intent": 0.90, "target_context": 0.90},
            ),
            (
                "remember_last_source",
                0.89,
                {"semantic": 0.89, "intent": 0.90, "target_context": 0.88},
            ),
        ],
    )

    decision = resolve_subtasks(
        (search,),
        definitions=definitions,
        transcript_confidence=0.95,
        min_score=0.50,
        min_margin=0.90,
        stt_threshold=0.75,
    )[0]

    assert decision.decision is ResolutionStatusV1.CLARIFY
    assert decision.reason == "low_top2_margin"


def test_resolver_rejects_copy_without_an_explicit_target() -> None:
    subtask = segment_command("kopiuj").subtasks[0]
    decision = resolve_subtasks(
        (
            search_for(
                subtask,
                [
                    (
                        "copy_selected_text",
                        0.95,
                        {
                            "semantic": 0.95,
                            "intent": 0.95,
                            "target_context": 0.95,
                        },
                    ),
                    (
                        "copy_text_under_cursor",
                        0.90,
                        {
                            "semantic": 0.90,
                            "intent": 0.90,
                            "target_context": 0.90,
                        },
                    ),
                ],
            ),
        ),
        definitions=[
            definition("copy_selected_text"),
            definition("copy_text_under_cursor"),
        ],
        transcript_confidence=0.95,
        min_score=0.50,
        min_margin=0.01,
        stt_threshold=0.75,
    )[0]

    assert decision.decision is ResolutionStatusV1.UNSUPPORTED
    assert decision.reason == "selected_text_not_explicit"


def test_resolver_abstains_when_index_returns_only_one_candidate() -> None:
    subtask = segment_command("otwórz Chrome").subtasks[0]
    decision = resolve_subtasks(
        (
            search_for(
                subtask,
                [
                    (
                        "open_browser",
                        0.99,
                        {
                            "semantic": 0.99,
                            "intent": 0.99,
                            "target_context": 0.99,
                        },
                    )
                ],
            ),
        ),
        definitions=[definition("open_browser")],
        transcript_confidence=0.99,
        min_score=0.50,
        min_margin=0.10,
        stt_threshold=0.75,
    )[0]

    assert decision.decision is ResolutionStatusV1.CLARIFY
    assert decision.reason == "single_candidate_without_comparator"
    assert decision.margin_top2 is None


def test_resolver_penalizes_partial_vector_coverage() -> None:
    subtask = segment_command("otwórz Chrome").subtasks[0]
    decision = resolve_subtasks(
        (
            search_for(
                subtask,
                [
                    (
                        "open_browser",
                        2 / 3,
                        {
                            "semantic": 0.90,
                            "intent": 0.90,
                        },
                    )
                ],
            ),
        ),
        definitions=[definition("open_browser")],
        transcript_confidence=0.99,
        min_score=0.50,
        min_margin=0.10,
        stt_threshold=0.75,
    )[0]

    candidate = decision.candidates[0]
    assert candidate.vector_score == pytest.approx(0.60)
    assert candidate.vector_scores == {"semantic": 0.90, "intent": 0.90}
    assert candidate.coverage == pytest.approx(2 / 3)
    assert candidate.missing_vector_names == ("target_context",)
    assert decision.decision is ResolutionStatusV1.CLARIFY
    assert decision.reason == "single_candidate_without_comparator"


def test_resolver_rejects_single_space_candidate() -> None:
    subtask = segment_command("otwórz Chrome").subtasks[0]
    decision = resolve_subtasks(
        (
            search_for(
                subtask,
                [
                    (
                        "open_browser",
                        1 / 3,
                        {"semantic": 0.99},
                    )
                ],
            ),
        ),
        definitions=[definition("open_browser")],
        transcript_confidence=0.99,
        min_score=0.50,
        min_margin=0.10,
        stt_threshold=0.75,
    )[0]

    assert decision.candidates[0].eligible is False
    assert "insufficient_vector_coverage" in decision.candidates[0].rejection_reasons
    assert decision.decision is ResolutionStatusV1.CLARIFY
    assert decision.reason == "insufficient_vector_coverage"


def test_assembler_preserves_order_dependencies_and_catalog_policy() -> None:
    segmentation = segment_command("otwórz Chrome i otwórz YouTube")
    browser, youtube = segmentation.subtasks
    definitions = [
        definition("open_browser"),
        definition(
            "open_url",
            schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
    ]
    decisions = resolve_subtasks(
        (
            search_for(
                browser,
                [
                    (
                        "open_browser",
                        0.95,
                        {"semantic": 0.95, "intent": 0.95, "target_context": 0.95},
                    ),
                    (
                        "open_url",
                        0.30,
                        {"semantic": 0.30, "intent": 0.50, "target_context": 0.10},
                    ),
                ],
            ),
            search_for(
                youtube,
                [
                    (
                        "open_url",
                        0.96,
                        {"semantic": 0.96, "intent": 0.96, "target_context": 0.96},
                    ),
                    (
                        "open_browser",
                        0.25,
                        {"semantic": 0.25, "intent": 0.40, "target_context": 0.10},
                    ),
                ],
            ),
        ),
        definitions=definitions,
        transcript_confidence=0.97,
        min_score=0.60,
        min_margin=0.10,
        stt_threshold=0.75,
    )
    request = CommandRequest(source=CommandSource.API, text="otwórz Chrome i otwórz YouTube")

    assembly = assemble_plan(
        request,
        segmentation,
        decisions,
        definitions=definitions,
        max_steps=12,
    )

    assert assembly.blocked_reason is None
    assert assembly.plan is not None
    assert [step.action_id for step in assembly.plan.steps] == [
        "open_browser",
        "open_url",
    ]
    assert assembly.plan.steps[0].depends_on == []
    assert assembly.plan.steps[1].depends_on == [assembly.plan.steps[0].id]


def test_plan_validator_accepts_stricter_policy_but_rejects_downgrade() -> None:
    safer_plan = CommandPlan(
        request_id="safer",
        intent="task",
        steps=[
            PlanStep(
                action_id="open_browser",
                risk=RiskLevel.HIGH,
                confirmation_required=True,
            )
        ],
    )
    downgraded_plan = CommandPlan(
        request_id="downgraded",
        intent="task",
        steps=[
            PlanStep(
                action_id="open_browser",
                risk=RiskLevel.LOW,
                confirmation_required=False,
            )
        ],
    )

    assert (
        validate_plan(
            safer_plan,
            definitions=[definition("open_browser")],
            max_steps=2,
        )
        == []
    )
    errors = validate_plan(
        downgraded_plan,
        definitions=[
            definition(
                "open_browser",
                risk="high",
                confirmation=True,
            )
        ],
        max_steps=2,
    )
    assert "risk_policy_mismatch:open_browser" in errors
    assert "confirmation_policy_mismatch:open_browser" in errors


@pytest.mark.asyncio
async def test_execute_flag_stays_off_without_local_quality_gate(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        routing_v2_execute=True,
        routing_v2_shadow_mode=True,
        routing_v2_quality_gate_file="missing.json",
    )
    fake_index = SimpleNamespace(catalog_hash="catalog")
    service = RoutingV2Service(
        settings,
        capability_index=fake_index,  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )

    assert service.execution_enabled is False
    assert service.quality_gate.reason == "missing_report"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_request", "expected_reason"),
    [
        (
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="otwórz Chrome",
            ),
            "missing_stt_confidence",
        ),
        (
            CommandRequest.from_transcript(
                TranscriptEnvelopeV1.from_text(
                    "otwórz Chrome",
                    confidence=0.98,
                    speaker_ids=(0, 1),
                ),
                managed_voice_turn=True,
            ),
            "multiple_speakers",
        ),
    ],
)
async def test_routing_service_applies_voice_gate_before_embeddings(
    tmp_path,
    command_request,
    expected_reason,
) -> None:
    class FailingIndex:
        catalog_hash = "catalog"

        async def search_subtasks(self, *_args, **_kwargs):
            raise AssertionError("voice gate must run before embeddings")

    service = RoutingV2Service(
        Settings(voiceloop_data_dir=str(tmp_path)),
        capability_index=FailingIndex(),  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )

    outcome = await service.evaluate(command_request)

    assert outcome.blocked_reason == expected_reason
    assert outcome.searches == ()
    assert outcome.plan is not None
    assert outcome.plan.requires_clarification is True
    assert outcome.plan.steps == []


@pytest.mark.asyncio
async def test_pre_resolver_exits_record_challenge_observation(tmp_path) -> None:
    recorded: list[tuple] = []

    class Index:
        catalog_hash = "catalog"

        async def search_subtasks(self, *_args, **_kwargs):
            return []

    service = RoutingV2Service(
        Settings(voiceloop_data_dir=str(tmp_path), routing_v2_calibration_mode="report_only"),
        capability_index=Index(),  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )
    service.calibration_recorder._started = True
    service.calibration_recorder.enabled = True
    service.calibration_recorder.record = lambda batch: recorded.append(batch)  # type: ignore[method-assign]

    await service.evaluate(
        CommandRequest(source=CommandSource.DEEPGRAM, text="otwórz Chrome")
    )
    assert recorded
    observation = recorded[0][0]
    assert observation.set_role.value == "challenge"
    assert observation.ranking_score is None


def test_execute_flag_requires_matching_complete_quality_report(tmp_path) -> None:
    incomplete_report = tmp_path / "incomplete-routing-v2-metrics.json"
    incomplete_report.write_text(
        json.dumps(
            {
                "quality_gate_passed": True,
                "catalog_coverage": 1.0,
                "expected_action_count": 1,
                "catalog_hash": "catalog",
            }
        ),
        encoding="utf-8",
    )
    incomplete_service = RoutingV2Service(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_execute=True,
            routing_v2_shadow_mode=False,
            routing_v2_quality_gate_file=str(incomplete_report),
        ),
        capability_index=SimpleNamespace(catalog_hash="catalog"),  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )
    runtime_config = RoutingV2RuntimeConfig(
        candidate_limit=10,
        execute_min_score=0.50,
        execute_min_margin=0.10,
        stt_threshold=0.75,
        max_subtasks=12,
        embedding_model="fake-embedding",
        embedding_dimension=3,
        capability_collection="fake-capabilities",
        catalog_hash="catalog",
    )
    report = tmp_path / "routing-v2-metrics.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "quality_gate_passed": True,
                "catalog_coverage": 1.0,
                "sample_count": 4,
                "base_example_count": 2,
                "action_count": 1,
                "expected_action_count": 1,
                "resolved_accuracy": 1.0,
                "topk_recall": 1.0,
                "safe_abstention_recall": 1.0,
                "unsafe_resolution_count": 0,
                "quality_gate_failures": [],
                "catalog_hash": "catalog",
                "runtime_config": runtime_config.model_dump(mode="json"),
                "runtime_fingerprint": runtime_config.fingerprint(),
            }
        ),
        encoding="utf-8",
    )
    runtime_index = SimpleNamespace(
        catalog_hash="catalog",
        embeddings=SimpleNamespace(
            _resolved_model="fake-embedding",
            configured_model="fake-embedding",
        ),
        _dimension=3,
        collection_name="fake-capabilities",
    )
    shadow_service = RoutingV2Service(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_execute=True,
            routing_v2_shadow_mode=True,
            routing_v2_execute_min_margin=0.10,
            routing_v2_quality_gate_file=str(report),
        ),
        capability_index=runtime_index,  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        routing_v2_execute=True,
        routing_v2_shadow_mode=False,
        routing_v2_execute_min_margin=0.10,
        routing_v2_quality_gate_file=str(report),
        routing_v2_canary_action_ids="describe_active_window,recall",
    )
    service = RoutingV2Service(
        settings,
        capability_index=runtime_index,  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )
    canary_allowed_service = RoutingV2Service(
        settings.model_copy(
            update={"routing_v2_canary_action_ids": "open_browser"}
        ),
        capability_index=runtime_index,  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )
    unsafe_canary_service = RoutingV2Service(
        settings.model_copy(
            update={"routing_v2_canary_action_ids": "open_browser"}
        ),
        capability_index=runtime_index,  # type: ignore[arg-type]
        definitions=[definition("open_browser", risk="medium")],
    )
    mismatched_service = RoutingV2Service(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_execute=True,
            routing_v2_shadow_mode=False,
            routing_v2_execute_min_score=0.60,
            routing_v2_execute_min_margin=0.10,
            routing_v2_quality_gate_file=str(report),
        ),
        capability_index=runtime_index,  # type: ignore[arg-type]
        definitions=[definition("open_browser")],
    )

    assert incomplete_service.quality_gate.passed is False
    assert incomplete_service.execution_enabled is False
    assert shadow_service.quality_gate.passed is True
    assert shadow_service.execution_enabled is False
    assert service.quality_gate.passed is True
    assert service.execution_enabled is True
    assert service.canary_enabled is True
    canary_plan = CommandPlan(
        request_id="canary",
        intent="task",
        steps=[PlanStep(action_id="open_browser")],
    )
    assert service.plan_execution_allowed(canary_plan) is False
    assert canary_allowed_service.plan_execution_allowed(canary_plan) is True
    assert unsafe_canary_service.plan_execution_allowed(canary_plan) is False
    assert "canary" in canary_allowed_service.health()[1]
    assert mismatched_service.quality_gate.passed is True
    assert mismatched_service.execution_enabled is False
    assert "runtime_config_mismatch" in mismatched_service.health()[1]
    mismatched_outcome = RoutingV2Outcome(segmentation=segment_command("otwórz Chrome"))
    guard = mismatched_service.activation_guard_plan(
        CommandRequest(text="otwórz Chrome"),
        mismatched_outcome,
    )
    assert guard is not None
    assert guard.provider == "routing_v2_guard"
    assert guard.steps == []


@pytest.mark.asyncio
async def test_routing_service_builds_ordered_shadow_plan_per_subtask(tmp_path) -> None:
    definitions = [
        definition("open_browser"),
        definition(
            "open_url",
            schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
    ]

    class FakeIndex:
        catalog_hash = "catalog"

        async def search_subtasks(self, subtasks, **_kwargs):
            results = []
            for subtask in subtasks:
                if subtask.target == "browser":
                    matches = [
                        (
                            "open_browser",
                            0.96,
                            {
                                "semantic": 0.96,
                                "intent": 0.96,
                                "target_context": 0.96,
                            },
                        ),
                        (
                            "open_url",
                            0.30,
                            {
                                "semantic": 0.30,
                                "intent": 0.50,
                                "target_context": 0.10,
                            },
                        ),
                    ]
                else:
                    matches = [
                        (
                            "open_url",
                            0.97,
                            {
                                "semantic": 0.97,
                                "intent": 0.97,
                                "target_context": 0.97,
                            },
                        ),
                        (
                            "open_browser",
                            0.25,
                            {
                                "semantic": 0.25,
                                "intent": 0.40,
                                "target_context": 0.10,
                            },
                        ),
                    ]
                results.append(search_for(subtask, matches))
            return results

    service = RoutingV2Service(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_execute=False,
        ),
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome i YouTube",
        transcript_confidence=0.97,
    )

    outcome = await service.evaluate(request)

    assert service.execution_enabled is False
    assert outcome.blocked_reason is None
    assert outcome.plan is not None
    assert [step.action_id for step in outcome.plan.steps] == [
        "open_browser",
        "open_url",
    ]
    assert len(outcome.searches) == 2
    payload = service.shadow_payload(request, outcome, legacy_plan=None)
    assert payload["mode"] == "shadow"
    assert payload["execution_enabled"] is False
    assert payload["v2_action_ids"] == ["open_browser", "open_url"]
    assert all("semantic" not in item for item in payload["embeddings"])
