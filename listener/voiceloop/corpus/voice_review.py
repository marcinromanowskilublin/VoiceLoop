from __future__ import annotations

import html
import json
from collections.abc import Mapping

from ..models import TranscriptEnvelopeV1
from .schema import VoiceEvalSampleV1


def render_voice_annotation_review(
    samples: list[VoiceEvalSampleV1],
    *,
    download_filename: str = "annotations-v1.jsonl",
    storage_namespace: str = "all",
    prefill_envelopes: Mapping[str, TranscriptEnvelopeV1] | None = None,
) -> str:
    prefills = prefill_envelopes or {}
    payload = [
        {
            "sample_id": sample.sample_id,
            "audio": sample.audio.relative_path if sample.audio else "",
            "audio_clip_sha256": sample.audio.clip_sha256 if sample.audio else "",
            "duration_seconds": (
                round(sample.audio.duration_seconds, 3) if sample.audio else 0.0
            ),
            "split": sample.split.value if sample.split else "",
            "captured_start": sample.provenance.captured_start.isoformat(),
            "quality_tags": list(sample.tags),
            "initial": _annotation_prefill(prefills.get(sample.sample_id)),
        }
        for sample in samples
    ]
    dataset_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    filename_json = json.dumps(download_filename, ensure_ascii=False)
    storage_key_json = json.dumps(
        f"voiceloop-voice-annotations-v1:{storage_namespace}",
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VoiceLoop — ręczna adnotacja Voice Eval V1</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #111827; color: #e5e7eb; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 12px 20px;
      background: #111827ee; border-bottom: 1px solid #374151; }}
    main {{ max-width: 1050px; margin: auto; padding: 16px; }}
    .card {{ margin: 12px 0; padding: 16px; border: 1px solid #374151;
      border-radius: 12px; background: #1f2937; }}
    .meta {{ color: #9ca3af; font-size: 12px; word-break: break-all; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr));
      gap: 10px; margin-top: 10px; }}
    label {{ display: grid; gap: 4px; font-size: 13px; }}
    input, select, textarea {{ box-sizing: border-box; width: 100%; padding: 8px;
      border: 1px solid #4b5563; border-radius: 6px; background: #111827;
      color: #f9fafb; }}
    textarea {{ min-height: 72px; }}
    audio {{ width: 100%; margin-top: 8px; }}
    button {{ padding: 9px 14px; border: 0; border-radius: 7px; cursor: pointer;
      background: #2563eb; color: white; }}
    .danger {{ background: #b91c1c; }}
    .done {{ border-color: #059669; }}
    .check {{ display: flex; align-items: center; gap: 8px; }}
    .check input {{ width: auto; }}
  </style>
</head>
<body>
<header>
  <strong>VoiceLoop Voice Eval V1</strong>
  <span id="progress"></span>
  <button id="export">Pobierz annotations-v1.jsonl</button>
  <button id="clear" class="danger">Wyczyść lokalny formularz</button>
</header>
<main>
  <p>Audio pozostaje lokalne. Formularz zapisuje wersję roboczą wyłącznie w
  localStorage tej przeglądarki. Eksport wymaga potwierdzenia własnego mówcy.</p>
  <div id="samples"></div>
</main>
<script>
const samples = {dataset_json};
const storageKey = {storage_key_json};
const state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
const defaults = Object.fromEntries(
  samples.map(sample => [sample.sample_id, sample.initial || {{}}])
);
const intents = ["question","conversation","task","ambiguous","cancellation","barge_in"];
const outcomes = ["respond","execute","clarify","reject","ignore"];

function field(id, name, fallback = "") {{
  return state[id]?.[name] ?? defaults[id]?.[name] ?? fallback;
}}
function save(id, name, value) {{
  state[id] = state[id] || {{}};
  state[id][name] = value;
  localStorage.setItem(storageKey, JSON.stringify(state));
  updateProgress();
}}
function options(values, current) {{
  return '<option value=""></option>' + values.map(
    value => `<option ${{value === current ? "selected" : ""}}>${{value}}</option>`
  ).join("");
}}
function escapeText(value) {{
  const node = document.createElement("span");
  node.textContent = value ?? "";
  return node.innerHTML;
}}
function render() {{
  document.getElementById("samples").innerHTML = samples.map((sample, index) => {{
    const id = sample.sample_id;
    const checked = field(id, "speaker_confirmed", false) ? "checked" : "";
    return `<section class="card" id="card-${{id}}">
      <div><strong>${{index + 1}}/${{samples.length}}</strong>
        · ${{escapeText(sample.split)}} · ${{sample.duration_seconds}} s</div>
      <div class="meta">${{escapeText(id)}} · ${{escapeText(sample.captured_start)}}
        · ${{escapeText(sample.quality_tags.join(", "))}}</div>
      <audio controls preload="metadata" src="${{encodeURI(sample.audio)}}"></audio>
      <div class="grid">
        <label>Tekst dosłowny
          <textarea data-name="literal_text">${{escapeText(field(id,"literal_text"))}}</textarea>
        </label>
        <label>Tekst z interpunkcją
          <textarea data-name="punctuated_text">${{escapeText(
            field(id,"punctuated_text")
          )}}</textarea>
        </label>
        <label>Intencja
          <select data-name="intent">${{options(intents, field(id,"intent"))}}</select>
        </label>
        <label>Oczekiwany wynik
          <select data-name="expected_outcome">${{options(
            outcomes, field(id,"expected_outcome")
          )}}</select>
        </label>
        <label>Tagi prozodii (po przecinku)
          <input data-name="prosody_tags" value="${{escapeText(field(id,"prosody_tags"))}}">
        </label>
        <label>Nazwy własne (po przecinku)
          <input data-name="proper_names" value="${{escapeText(field(id,"proper_names"))}}">
        </label>
        <label>Oczekiwane action_id (JSON)
          <input data-name="expected_action_ids"
            value="${{escapeText(field(id,"expected_action_ids","[]"))}}">
        </label>
        <label>Oczekiwane args kroków (JSON)
          <input data-name="expected_step_args"
            value="${{escapeText(field(id,"expected_step_args","[]"))}}">
        </label>
        <label>Anotator
          <input data-name="annotator" value="${{escapeText(field(id,"annotator"))}}">
        </label>
        <label class="check"><input type="checkbox" data-name="expected_abstention"
          ${{field(id,"expected_abstention",false) ? "checked" : ""}}>
          Oczekiwana abstencja</label>
        <label class="check"><input type="checkbox" data-name="speaker_confirmed" ${{checked}}>
          Potwierdzam: to mój głos</label>
      </div>
    </section>`;
  }}).join("");
  document.querySelectorAll("[data-name]").forEach(node => {{
    const card = node.closest(".card");
    const id = card.id.replace("card-", "");
    node.addEventListener("change", () => {{
      save(id, node.dataset.name, node.type === "checkbox" ? node.checked : node.value);
    }});
  }});
  updateProgress();
}}
function csvList(value) {{
  return String(value || "").split(",").map(item => item.trim()).filter(Boolean);
}}
function updateProgress() {{
  const done = samples.filter(sample => {{
    const item = state[sample.sample_id] || {{}};
    const ready = item.literal_text && item.punctuated_text && item.intent
      && item.expected_outcome && item.annotator && item.speaker_confirmed;
    document.getElementById(`card-${{sample.sample_id}}`)?.classList.toggle("done", !!ready);
    return ready;
  }}).length;
  document.getElementById("progress").textContent = ` · gotowe ${{done}}/${{samples.length}} `;
}}
document.getElementById("export").addEventListener("click", () => {{
  try {{
    const rows = samples.map(sample => {{
      const item = state[sample.sample_id] || {{}};
      if (!item.literal_text || !item.punctuated_text || !item.intent
          || !item.expected_outcome || !item.annotator || !item.speaker_confirmed) {{
        throw new Error(`Niekompletna próbka: ${{sample.sample_id}}`);
      }}
      return {{
        schema_version: 1,
        sample_id: sample.sample_id,
        audio_clip_sha256: sample.audio_clip_sha256,
        literal_text: item.literal_text,
        punctuated_text: item.punctuated_text,
        intent: item.intent,
        prosody_tags: csvList(item.prosody_tags),
        proper_names: csvList(item.proper_names),
        speaker_role: "self",
        speaker_confirmed: true,
        expected_outcome: item.expected_outcome,
        expected_action_ids: JSON.parse(item.expected_action_ids || "[]"),
        expected_step_args: JSON.parse(item.expected_step_args || "[]"),
        expected_abstention: !!item.expected_abstention,
        annotator: item.annotator,
        approved_at: new Date().toISOString()
      }};
    }});
    const body = rows.map(row => JSON.stringify(row)).join("\\n") + "\\n";
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([body], {{type:"application/x-ndjson"}}));
    link.download = {filename_json};
    link.click();
    URL.revokeObjectURL(link.href);
  }} catch (error) {{
    alert(error.message);
  }}
}});
document.getElementById("clear").addEventListener("click", () => {{
  if (confirm("Usunąć lokalną wersję roboczą formularza?")) {{
    localStorage.removeItem(storageKey);
    location.reload();
  }}
}});
render();
</script>
</body>
</html>
"""


def _annotation_prefill(envelope: TranscriptEnvelopeV1 | None) -> dict[str, object]:
    if envelope is None:
        return {}
    literal = " ".join(word.word for word in envelope.words).strip()
    return {
        "literal_text": literal or envelope.normalized_text,
        "punctuated_text": envelope.raw_text,
        "deepgram_confidence": envelope.confidence_mean,
    }


def annotation_review_title(sample: VoiceEvalSampleV1) -> str:
    return html.escape(
        f"{sample.sample_id} — "
        f"{sample.audio.duration_seconds if sample.audio else 0.0:.2f}s"
    )
