from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence

from .schema import (
    CorpusSplit,
    ExpectedIntent,
    RoutingEvalRecord,
    RoutingMetrics,
    RoutingScoreCard,
    SpeakerStatus,
    SttErrorType,
    StyleEvaluationReport,
    StyleProfile,
    UtteranceRecord,
)

_DIRECT_PATTERN = re.compile(
    r"^\s*(?:otwórz|zamknij|zrób|sprawdź|pokaż|dodaj|usuń|napisz|"
    r"chcę|potrzebuję|zobacz|uruchom)\b",
    re.IGNORECASE,
)
_FORMAL_PATTERN = re.compile(r"\b(?:pan|pani|państwo|proszę pana|proszę pani)\b", re.IGNORECASE)
_INFORMAL_PATTERN = re.compile(r"\b(?:ty|tobie|ci|ciebie|twój|twoja|zrób|powiedz)\b", re.IGNORECASE)
_ROUTING_HOLDOUTS: dict[str, tuple[str, ...]] = {
    "open_calendar": (
        "pokaż mi terminarz na dzisiaj",
        "odpal harmonogram z Windows",
    ),
    "open_browser": (
        "włącz mi zwykłą przeglądarkę",
        "przejdź do programu do internetu",
    ),
    "open_url": (
        "przejdź na YouTube w nowej karcie",
        "wyświetl adres https://example.com",
    ),
    "open_folder": (
        "odpal Eksplorator plików Windows",
        "uruchom widok Ten komputer",
    ),
    "open_app": (
        "wejdź do WhatsApp na tym komputerze",
        "włącz aplikację WhatsApp na pulpicie",
    ),
    "open_chat": (
        "uruchom ogólną stronę czatu",
        "wejdź na dawny czat w przeglądarce",
    ),
    "search_web": (
        "znajdź w sieci aktualny kurs euro",
        "poszukaj online godzin otwarcia apteki",
    ),
    "open_gpt_chat": (
        "wejdź na stronę GPT od OpenAI",
        "zacznij nową rozmowę w Chat GPT",
    ),
    "open_gemini_chat": (
        "wejdź do asystenta Google Gemini",
        "uruchom czat Gemini od Google",
    ),
    "describe_active_window": (
        "w jakim programie teraz jestem",
        "podaj tytuł tego co mam na wierzchu",
    ),
    "minimize_active_window": (
        "schowaj bieżący program na pasek",
        "zwiń to aktywne okno",
    ),
    "minimize_all_windows": (
        "odsłoń cały pulpit",
        "schowaj wszystkie programy z ekranu",
    ),
    "minimize_window_under_cursor": (
        "zwiń program wskazany myszą",
        "schowaj okno znajdujące się pod strzałką",
    ),
    "close_window_under_cursor": (
        "poproś o zamknięcie okna pod myszą",
        "zamknij aplikację wskazaną kursorem",
    ),
    "copy_selected_text": (
        "przerzuć podświetlony fragment do schowka",
        "weź zaznaczenie na clipboard",
    ),
    "copy_text_under_cursor": (
        "weź napis spod strzałki do schowka",
        "skopiuj etykietę wskazywaną myszą",
    ),
    "copy_email_under_cursor": (
        "złap adres poczty spod kursora",
        "skopiuj mail znajdujący się pod myszą",
    ),
    "copy_number_under_cursor": (
        "złap cyfry spod strzałki",
        "weź numer telefonu wskazany myszą",
    ),
    "copy_sentence_under_cursor": (
        "weź całe zdanie spod strzałki",
        "skopiuj frazę wskazywaną kursorem",
    ),
    "select_sentence_under_cursor": (
        "podświetl całe zdanie tam gdzie mysz",
        "obejmij frazę znajdującą się pod kursorem",
    ),
    "select_paragraph_under_cursor": (
        "obejmij cały akapit pod strzałką",
        "podświetl blok tekstu wskazany myszą",
    ),
    "rename_under_cursor": (
        "daj nową nazwę plikowi pod kursorem raport_v2",
        "przemianuj ikonę wskazaną myszą na archiwum",
    ),
    "describe_text_target": (
        "sprawdź czy tutaj można bezpiecznie pisać",
        "powiedz czy kursor wskazuje pole adresu",
    ),
    "paste_text_safe": (
        "wstaw do Cursora tekst zrób krótkie streszczenie",
        "wklej do Gemini treść podaj trzy wnioski",
    ),
    "describe_recent_activity": (
        "podsumuj ekran z ostatnich czterdziestu pięciu minut",
        "opisz programy używane przez ostatnią godzinę",
    ),
    "create_note": (
        "zanotuj w notesie że jutro odbiór wyników",
        "wrzuć do notatnika listę mleko i chleb",
    ),
    "run_uivision_macro": (
        "wykonaj makro demo_notatka.json w UI Vision",
        "uruchom automatyzację voiceloop_notatka.json",
    ),
    "remember": (
        "zachowaj w pamięci że wolę odpowiedzi w punktach",
        "zapamiętaj informację gabinet działa po południu",
    ),
    "remember_last_source": (
        "zachowaj drugi link z poprzedniego szukania",
        "zapamiętaj pierwsze ostatnio znalezione źródło",
    ),
    "recall": (
        "wyciągnij z pamięci informacje o gabinecie",
        "poszukaj w zapiskach hasła krótkie odpowiedzi",
    ),
}
_ROUTING_HOLDOUT_EXPECTED_ARGS: dict[tuple[str, int], dict] = {
    ("open_url", 0): {"url": "https://www.youtube.com"},
    ("open_url", 1): {"url": "https://example.com"},
    ("open_folder", 0): {"folder_id": "this_pc"},
    ("open_folder", 1): {"folder_id": "this_pc"},
    ("open_app", 0): {"app_id": "whatsapp"},
    ("open_app", 1): {"app_id": "whatsapp"},
    ("search_web", 0): {"query": "aktualny kurs euro", "limit": 5},
    ("search_web", 1): {"query": "godzin otwarcia apteki", "limit": 5},
    ("paste_text_safe", 0): {
        "text": "zrób krótkie streszczenie",
        "expected_window": "cursor",
    },
    ("paste_text_safe", 1): {
        "text": "podaj trzy wnioski",
        "expected_window": "gemini",
    },
    ("describe_recent_activity", 0): {"minutes": 45},
    ("describe_recent_activity", 1): {"minutes": 60},
    ("create_note", 0): {"text": "jutro odbiór wyników"},
    ("create_note", 1): {"text": "mleko i chleb"},
    ("run_uivision_macro", 0): {"macro": "demo_notatka.json"},
    ("run_uivision_macro", 1): {"macro": "voiceloop_notatka.json"},
    ("remember", 0): {
        "content": "wolę odpowiedzi w punktach",
        "kind": "fact",
    },
    ("remember", 1): {
        "content": "gabinet działa po południu",
        "kind": "fact",
    },
    ("remember_last_source", 0): {"index": 2, "kind": "web_source"},
    ("remember_last_source", 1): {"index": 1, "kind": "web_source"},
    ("recall", 0): {"query": "informacje o gabinecie"},
    ("recall", 1): {"query": "hasła krótkie odpowiedzi"},
    ("rename_under_cursor", 0): {"new_name": "raport_v2"},
    ("rename_under_cursor", 1): {"new_name": "archiwum"},
}


def routing_holdout_action_ids() -> set[str]:
    return set(_ROUTING_HOLDOUTS)


def build_style_profile(
    records: list[UtteranceRecord],
    *,
    manifest_id: str,
    enabled: bool = False,
) -> StyleProfile:
    eligible = [
        record
        for record in records
        if record.speaker_status is SpeakerStatus.SELF
        and not record.is_near_duplicate
        and not record.quarantine_reason
        and record.split in {None, CorpusSplit.TRAIN}
        and record.text.strip()
    ]
    lengths = sorted(record.word_count for record in eligible)
    low = _percentile(lengths, 0.25) if lengths else 8
    high = _percentile(lengths, 0.75) if lengths else 40
    question_ratio = _ratio(
        sum("?" in record.text for record in eligible),
        len(eligible),
    )
    direct_ratio = _ratio(
        sum(bool(_DIRECT_PATTERN.search(record.text)) for record in eligible),
        len(eligible),
    )
    formal_hits = sum(bool(_FORMAL_PATTERN.search(record.text)) for record in eligible)
    informal_hits = sum(bool(_INFORMAL_PATTERN.search(record.text)) for record in eligible)
    grammatical_form = (
        "pan"
        if formal_hits > informal_hits * 1.5
        else ("ty" if informal_hits > formal_hits * 1.5 else "mixed")
    )
    median = _percentile(lengths, 0.5) if lengths else 20
    detail_tolerance = "low" if median <= 15 else ("high" if median >= 55 else "medium")
    question_style = (
        "few" if question_ratio < 0.15 else ("many" if question_ratio > 0.45 else "clarifying")
    )
    conversation_style = (
        "concise"
        if detail_tolerance == "low"
        else ("max_iq" if detail_tolerance == "high" else "default")
    )
    fingerprint = "|".join(
        (
            manifest_id,
            str(len(eligible)),
            str(sum(lengths)),
            str(low),
            str(high),
            f"{direct_ratio:.6f}",
        )
    )
    return StyleProfile(
        profile_id=hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20],
        built_from_manifest_id=manifest_id,
        utterance_count=len(eligible),
        word_count=sum(lengths),
        preferred_reply_words=(max(5, low), max(max(5, low), high)),
        directness=direct_ratio,
        grammatical_form=grammatical_form,
        detail_tolerance=detail_tolerance,
        question_style=question_style,
        maps_to_conversation_style=conversation_style,
        enabled=enabled,
    )


def evaluate_style_profile(
    profile: StyleProfile,
    records: list[UtteranceRecord],
) -> StyleEvaluationReport:
    holdout = [
        record
        for record in records
        if record.split is CorpusSplit.HOLDOUT
        and record.speaker_status is SpeakerStatus.SELF
        and not record.is_near_duplicate
        and not record.quarantine_reason
    ]
    observed = build_style_profile(
        [record.model_copy(update={"split": None}) for record in holdout],
        manifest_id=f"{profile.built_from_manifest_id}:holdout",
        enabled=False,
    )
    overlap = _interval_overlap(
        profile.preferred_reply_words,
        observed.preferred_reply_words,
    )
    directness_delta = abs(profile.directness - observed.directness)
    form_matches = profile.grammatical_form == observed.grammatical_form
    passes = len(holdout) >= 20 and overlap >= 0.4 and directness_delta <= 0.2 and form_matches
    return StyleEvaluationReport(
        profile_id=profile.profile_id,
        holdout_utterance_count=len(holdout),
        length_interval_overlap=overlap,
        directness_delta=directness_delta,
        grammatical_form_matches=form_matches,
        passes_quality_gate=passes,
    )


def build_routing_eval_records(
    definitions: list[dict],
) -> list[RoutingEvalRecord]:
    records: list[RoutingEvalRecord] = []
    definitions_by_id = {
        str(definition.get("id") or ""): definition
        for definition in definitions
        if str(definition.get("id") or "")
    }
    for action_id, holdouts in _ROUTING_HOLDOUTS.items():
        definition = definitions_by_id.get(action_id)
        if definition is None:
            continue
        training_examples = {
            " ".join(str(example).split()).casefold()
            for example in definition.get("routing_examples") or ()
        }
        for index, raw_example in enumerate(holdouts):
            example = " ".join(str(raw_example).split())
            base_example_id = f"{action_id}:manual:{index}"
            if not example or len(example.split()) > 25:
                continue
            if example.casefold() in training_examples:
                raise ValueError(f"Holdout routingu {action_id} występuje w routing_examples.")
            records.append(
                RoutingEvalRecord(
                    example_id=f"{action_id}:holdout:{index}",
                    base_example_id=base_example_id,
                    gold_text=example,
                    stt_text=example,
                    stt_confidence=0.96,
                    expected_action_id=action_id,
                    expected_step_args=(
                        (_ROUTING_HOLDOUT_EXPECTED_ARGS[(action_id, index)],)
                        if (action_id, index) in _ROUTING_HOLDOUT_EXPECTED_ARGS
                        else ()
                    ),
                    expected_intent=ExpectedIntent.TASK,
                    tags=("manual_holdout", "exact"),
                )
            )
            ascii_variant = _without_diacritics(example)
            if ascii_variant != example:
                records.append(
                    RoutingEvalRecord(
                        example_id=f"{action_id}:holdout-diacritics:{index}",
                        base_example_id=base_example_id,
                        gold_text=example,
                        stt_text=ascii_variant,
                        stt_confidence=0.85,
                        error_type=SttErrorType.FLEXION,
                        expected_action_id=action_id,
                        expected_intent=ExpectedIntent.TASK,
                        tags=("manual_holdout", "without_diacritics"),
                    )
                )
            words = example.split()
            if len(words) >= 3:
                records.append(
                    RoutingEvalRecord(
                        example_id=f"{action_id}:holdout-low-confidence:{index}",
                        base_example_id=base_example_id,
                        gold_text=example,
                        stt_text=" ".join(words[:-1]),
                        stt_confidence=0.55,
                        error_type=SttErrorType.LOW_CONFIDENCE,
                        expected_action_id=action_id,
                        expected_intent=ExpectedIntent.TASK,
                        tags=("manual_holdout", "low_confidence"),
                    )
                )
    if {
        "close_window_under_cursor",
        "open_browser",
        "open_url",
    }.issubset(definitions_by_id):
        records.extend(
            (
                RoutingEvalRecord(
                    example_id="compound:open-browser-youtube",
                    base_example_id="compound:open-browser-youtube",
                    gold_text="otwórz Chrome i YouTube",
                    stt_text="otwórz Chrome i YouTube",
                    stt_confidence=0.96,
                    error_type=SttErrorType.COMPOUND,
                    expected_action_id="open_browser",
                    expected_action_ids=("open_browser", "open_url"),
                    expected_step_args=(
                        {},
                        {"url": "https://www.youtube.com"},
                    ),
                    expected_intent=ExpectedIntent.TASK,
                    compound=True,
                    tags=("manual_holdout", "compound", "exact_plan"),
                ),
                RoutingEvalRecord(
                    example_id="negative:close-window-by-name",
                    gold_text="zamknij UI Vision, otwórz Chrome i YouTube",
                    stt_text="zamknij UI Vision, otwórz Chrome i YouTube",
                    stt_confidence=0.96,
                    error_type=SttErrorType.COMPOUND,
                    expected_intent=ExpectedIntent.AMBIGUOUS,
                    ambiguous=True,
                    compound=True,
                    tags=("manual_holdout", "negative", "target_identity"),
                ),
                RoutingEvalRecord(
                    example_id="negative:missing-separator",
                    gold_text="wyślij e-mail otwórz Chrome",
                    stt_text="wyślij e-mail otwórz Chrome",
                    stt_confidence=0.96,
                    error_type=SttErrorType.COMPOUND,
                    expected_intent=ExpectedIntent.AMBIGUOUS,
                    ambiguous=True,
                    compound=True,
                    tags=("manual_holdout", "negative", "missing_separator"),
                ),
                RoutingEvalRecord(
                    example_id="negative:ambiguous-connector",
                    gold_text="otwórz Chrome albo YouTube",
                    stt_text="otwórz Chrome albo YouTube",
                    stt_confidence=0.96,
                    expected_intent=ExpectedIntent.AMBIGUOUS,
                    ambiguous=True,
                    tags=("manual_holdout", "negative", "ambiguous_connector"),
                ),
                RoutingEvalRecord(
                    example_id="negative:multiple-speakers",
                    gold_text="otwórz Chrome",
                    stt_text="otwórz Chrome",
                    stt_confidence=0.96,
                    expected_intent=ExpectedIntent.AMBIGUOUS,
                    ambiguous=True,
                    speaker_ids=(0, 1),
                    tags=("manual_holdout", "negative", "multiple_speakers"),
                ),
            )
        )
    return records


def score_routing_result(
    example: RoutingEvalRecord,
    matches: Sequence[tuple[str, float]],
    *,
    min_score: float,
    margin_threshold: float,
    stt_threshold: float,
) -> RoutingScoreCard:
    action_ids = tuple(action_id for action_id, _ in matches)
    scores = tuple(float(score) for _, score in matches)
    predicted_top1 = action_ids[0] if action_ids else None
    margin = scores[0] - scores[1] if len(scores) >= 2 else None
    expected_action = example.expected_action_id
    expected_uncertain = example.ambiguous or example.error_type is SttErrorType.LOW_CONFIDENCE
    predicted_uncertain = (
        not scores or scores[0] < min_score or margin is None or margin < margin_threshold
    )
    return RoutingScoreCard(
        example_id=example.example_id,
        base_example_id=example.base_example_id,
        expected_action_id=expected_action,
        predicted_top1=predicted_top1,
        topk_action_ids=action_ids,
        scores=scores,
        margin_top2=margin,
        hit_at_1=bool(expected_action and predicted_top1 == expected_action),
        hit_at_k=bool(expected_action and expected_action in action_ids),
        below_min_score=not scores or scores[0] < min_score,
        stt_gate_blocked=example.stt_confidence < stt_threshold,
        expected_ambiguous=expected_uncertain,
        predicted_ambiguous=predicted_uncertain,
    )


def aggregate_routing_metrics(
    cards: list[RoutingScoreCard],
    *,
    expected_action_ids: set[str] | None = None,
) -> RoutingMetrics:
    count = len(cards)
    margins = [card.margin_top2 for card in cards if card.margin_top2 is not None]
    ambiguity_tp = sum(card.expected_ambiguous and card.predicted_ambiguous for card in cards)
    ambiguity_predicted = sum(card.predicted_ambiguous for card in cards)
    ambiguity_expected = sum(card.expected_ambiguous for card in cards)
    gate_tp = sum(card.expected_ambiguous and card.stt_gate_blocked for card in cards)
    gate_predicted = sum(card.stt_gate_blocked for card in cards)
    top1_accuracy = _ratio(sum(card.hit_at_1 for card in cards), count)
    topk_recall = _ratio(sum(card.hit_at_k for card in cards), count)
    ambiguity_precision = _ratio(ambiguity_tp, ambiguity_predicted)
    ambiguity_recall = _ratio(ambiguity_tp, ambiguity_expected)
    stt_gate_precision = _ratio(gate_tp, gate_predicted)
    stt_gate_recall = _ratio(gate_tp, ambiguity_expected)
    bases_by_action: dict[str, set[str]] = {}
    for card in cards:
        if not card.expected_action_id:
            continue
        bases_by_action.setdefault(card.expected_action_id, set()).add(
            card.base_example_id or card.example_id
        )
    base_example_count = len(
        {base_id for base_ids in bases_by_action.values() for base_id in base_ids}
    )
    action_count = len(bases_by_action)
    expected_actions = expected_action_ids or set(bases_by_action)
    covered_actions = set(bases_by_action) & expected_actions
    catalog_coverage = _ratio(len(covered_actions), len(expected_actions))
    failures: list[str] = []
    if base_example_count < 6:
        failures.append("base_example_count_below_6")
    if action_count < 3:
        failures.append("action_count_below_3")
    if any(len(base_ids) < 2 for base_ids in bases_by_action.values()):
        failures.append("per_action_coverage_below_2")
    if covered_actions != expected_actions:
        failures.append("catalog_action_coverage_incomplete")
    if top1_accuracy < 0.80:
        failures.append("top1_below_0_80")
    if topk_recall < 0.95:
        failures.append("topk_below_0_95")
    if ambiguity_recall < 0.80:
        failures.append("ambiguity_recall_below_0_80")
    if stt_gate_recall < 0.90:
        failures.append("stt_gate_recall_below_0_90")
    return RoutingMetrics(
        sample_count=count,
        base_example_count=base_example_count,
        action_count=action_count,
        expected_action_count=len(expected_actions),
        catalog_coverage=catalog_coverage,
        top1_accuracy=top1_accuracy,
        topk_recall=topk_recall,
        mean_margin=sum(margins) / len(margins) if margins else 0.0,
        ambiguity_precision=ambiguity_precision,
        ambiguity_recall=ambiguity_recall,
        stt_gate_precision=stt_gate_precision,
        stt_gate_recall=stt_gate_recall,
        quality_gate_passed=not failures,
        quality_gate_failures=tuple(failures),
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    if not reference_words:
        return 0.0 if not hypothesis.split() else 1.0
    return _edit_distance(reference_words, hypothesis.split()) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            substitution = previous[column - 1] + (reference_item != hypothesis_item)
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = round((len(values) - 1) * fraction)
    return int(values[max(0, min(index, len(values) - 1))])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _without_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _interval_overlap(first: tuple[int, int], second: tuple[int, int]) -> float:
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    intersection = max(0, right - left)
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 1.0
