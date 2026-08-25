"""Smoke test: syntetyczny transkrypt przechodzi całą drogę do geometrii.

Uruchamiać venvem VoiceLoopa. Używa prawdziwego LM Studio, bo sens tego panelu
polega na mierzeniu realnego modelu, nie atrapy.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from vectorscope.analysis import AnalysisRequest, run_analysis  # noqa: E402
from vectorscope.config import settings, vectorscope_data_dir  # noqa: E402
from vectorscope.fragments import build_fragments  # noqa: E402
from vectorscope.store import RecordingStore, transcript_hash  # noqa: E402

WORDS = [
    ("Pacjent", 0.00, 0.40),
    ("zgłasza", 0.40, 0.80),
    ("silny", 0.80, 1.10),
    ("lęk.", 1.10, 1.50),
    ("Mówi", 2.10, 2.40),
    ("o", 2.40, 2.50),
    ("narastającym", 2.50, 3.10),
    ("niepokoju.", 3.10, 3.70),
    ("Boli", 4.30, 4.60),
    ("go", 4.60, 4.70),
    ("głowa", 4.70, 5.10),
    ("od", 5.10, 5.25),
    ("tygodnia.", 5.25, 5.80),
    ("Ustaliliśmy", 6.40, 7.00),
    ("zwiększenie", 7.00, 7.60),
    ("dawki", 7.60, 7.95),
    ("leku.", 7.95, 8.40),
]


def synthetic_transcript() -> dict:
    words = [
        {
            "text": text,
            "start": start,
            "end": end,
            "confidence": 0.95,
            "speaker": 0,
        }
        for text, start, end in WORDS
    ]
    text = " ".join(item["text"] for item in words)
    return {
        "text": text,
        "model": "nova-3",
        "language": "pl",
        "words": words,
        "sentences": [
            "Pacjent zgłasza silny lęk.",
            "Mówi o narastającym niepokoju.",
            "Boli go głowa od tygodnia.",
            "Ustaliliśmy zwiększenie dawki leku.",
        ],
        "utterances": [
            {"transcript": text, "start": 0.0, "end": 8.4, "speaker": 0},
        ],
        "raw": {},
    }


async def main() -> int:
    active = settings()
    store = RecordingStore(vectorscope_data_dir(active) / "_smoke")

    for meta in store.list_recordings():
        store.delete(meta.id)

    payload = synthetic_transcript()
    meta = store.create(
        payload=b"RIFF-atrapa-nie-audio",
        mime="audio/webm",
        label="smoke-pacjent",
        duration_seconds=8.4,
        microphone_processing=False,
        upload_ms=1.0,
    )
    store.write_transcript(meta.id, payload)
    meta.transcript_status = "ok"
    meta.transcript_hash = transcript_hash(payload)
    meta.word_count = len(payload["words"])
    meta.text_preview = payload["text"][:300]
    store.write_meta(meta)

    print(f"nagranie: {meta.id}")
    print(f"transcript_hash: {meta.transcript_hash}")

    fragments = build_fragments(
        transcript=payload,
        recording_id=meta.id,
        recording_label=meta.label,
    )
    counts: dict[str, int] = {}
    for fragment in fragments:
        counts[fragment.level] = counts.get(fragment.level, 0) + 1
    print(f"fragmenty po poziomach: {counts}")

    sample = [item for item in fragments if item.level == "phrase"][:4]
    for item in sample:
        print(
            f"  fraza {item.id}: '{item.text}' "
            f"[{item.start_ms}-{item.end_ms} ms] rodzic={item.parent_id}"
        )
    words_sample = [item for item in fragments if item.level == "word"][:3]
    for item in words_sample:
        print(f"  slowo {item.id}: '{item.text}' rodzic={item.parent_id}")

    for levels in (["word"], ["sentence"], ["word", "phrase", "sentence"]):
        result = await run_analysis(
            AnalysisRequest(
                recording_ids=[meta.id],
                levels=levels,
                neighbours=3,
                threshold=0.15,
                projection="mds",
                include_anchors=True,
            ),
            store,
        )
        if not result.get("ok"):
            print(f"BLAD dla {levels}: {result.get('message')}")
            return 1
        distortion = result["distortion"]
        print(
            f"\npoziomy={levels} fragmentow={result['fragment_count']} "
            f"wymiar={result['dimension']} model={result['model']}"
        )
        print(
            f"  krawedzie={result['edge_stats']['shown']} "
            f"hierarchia={len(result['hierarchy_edges'])} "
            f"czasy={result['timings_ms']}"
        )
        trust = distortion["trustworthiness"]
        cont = distortion["continuity"]
        print(
            "  trustworthiness="
            + (f"{trust:.4f}" if trust is not None else "n/d")
            + ", continuity="
            + (f"{cont:.4f}" if cont is not None else "n/d")
            + f", stress={distortion['stress']:.4f}"
        )
        for warning in result["warnings"]:
            print(f"  UWAGA: {warning}")

    print("\nkotwice skali:")
    for anchor in result["anchors"]:
        print(f"  {anchor['relation']:22s} {anchor['cosine']:.4f}  ({anchor['key']})")

    print("\nnajblizsi sasiedzi pierwszych fragmentow:")
    for index in range(min(3, result["fragment_count"])):
        fragment = result["fragments"][index]
        neighbours = result["neighbours"][index][:3]
        rendered = ", ".join(
            f"{item['text']}={item['cosine']:.3f}" for item in neighbours
        )
        print(f"  '{fragment['text']}' -> {rendered}")

    meta = store.read_meta(meta.id)
    print(f"\nembedding_runs w meta: {list(meta.embedding_runs.keys())}")
    print(f"czasy etapow: {meta.timings_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
