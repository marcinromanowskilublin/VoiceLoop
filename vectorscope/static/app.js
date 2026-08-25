"use strict";

/* Vectorscope — warstwa widoku.
 *
 * Zasada, która rządzi tym plikiem: krawędzie, cosinusy i dendrogram pochodzą
 * z pełnych 768 wymiarów i są traktowane jako prawda. Współrzędne 2D są tylko
 * ilustracją i wszędzie, gdzie się pojawiają, towarzyszy im miara zniekształcenia.
 */

const LEVEL_LABELS = {
  word: "słowa",
  phrase: "frazy",
  sentence: "zdania",
  utterance: "wypowiedzi",
};

const LEVEL_SHAPES = {
  word: "ellipse",
  phrase: "round-rectangle",
  sentence: "diamond",
  utterance: "hexagon",
  anchor: "star",
  reference: "triangle",
};

const LEVEL_SIZES = {
  word: 20,
  phrase: 27,
  sentence: 33,
  utterance: 39,
  anchor: 24,
  reference: 24,
};

const GROUP_COLOURS = [
  "#4da3ff", "#37d39b", "#f5b445", "#a97bff",
  "#f2617a", "#4fd1d9", "#c9d24f", "#ff8f5a",
];

const ANCHOR_COLOUR = "#f5b445";
const REFERENCE_COLOUR = "#a97bff";

const PLOT_FONT = { family: "Inter, Segoe UI, system-ui, sans-serif", size: 11, color: "#8f9db4" };

const PLOT_LAYOUT = {
  paper_bgcolor: "#121826",
  plot_bgcolor: "#0d131f",
  font: PLOT_FONT,
  margin: { l: 52, r: 18, t: 28, b: 44 },
  showlegend: false,
  hoverlabel: { bgcolor: "#161d2b", bordercolor: "#222d40", font: { color: "#e6ecf5", size: 11 } },
  xaxis: { gridcolor: "#1a2231", zerolinecolor: "#222d40", linecolor: "#222d40" },
  yaxis: { gridcolor: "#1a2231", zerolinecolor: "#222d40", linecolor: "#222d40" },
};

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

const state = {
  config: null,
  recordings: [],
  selected: new Set(),
  levels: new Set(["word"]),
  analysis: null,
  scaleFloor: null,
  scaleFloorPending: false,
  colourByGroup: new Map(),
  cy: null,
  recorder: null,
  chunks: [],
  startedAt: 0,
  timer: null,
  stream: null,
};

/* ---------------------------------------------------------------- pomocnicze */

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function toast(message, kind = "") {
  const host = $("#toast");
  const item = document.createElement("div");
  item.className = `toast-item ${kind ? `is-${kind}` : ""}`;
  item.textContent = message;
  host.appendChild(item);
  setTimeout(() => item.remove(), kind === "error" ? 8000 : 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { detail: text };
  }
  if (!response.ok) {
    throw new Error(payload?.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

const num = (value, digits = 3) =>
  value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : Number(value).toFixed(digits);

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatClock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

function formatMs(value) {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));

function truncate(text, limit = 64) {
  const value = String(text ?? "");
  return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
}

/* --------------------------------------------------------------- diagnostyka */

async function loadHealth() {
  try {
    const health = await api("/api/health");
    for (const [service, info] of Object.entries(health)) {
      const pill = $(`.pill[data-service="${service}"]`);
      if (!pill) continue;
      pill.classList.remove("pill-idle", "pill-ok", "pill-bad");
      pill.classList.add(info.ok ? "pill-ok" : "pill-bad");
      pill.title = info.detail || "";
    }
  } catch (error) {
    toast(`Nie udało się sprawdzić usług: ${error.message}`, "error");
  }
}

async function loadConfig() {
  state.config = await api("/api/config");

  const levelHost = $("#levels");
  levelHost.innerHTML = "";
  for (const level of state.config.levels) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `level-chip ${state.levels.has(level) ? "is-on" : ""}`;
    chip.textContent = LEVEL_LABELS[level] || level;
    chip.addEventListener("click", () => {
      if (state.levels.has(level)) {
        if (state.levels.size === 1) {
          toast("Zostaw przynajmniej jeden poziom.");
          return;
        }
        state.levels.delete(level);
      } else {
        state.levels.add(level);
      }
      chip.classList.toggle("is-on");
    });
    levelHost.appendChild(chip);
  }

  const prefixSelect = $("#prefix");
  prefixSelect.innerHTML = "";
  const prefixLabels = {
    search_document: "search_document: (ścieżka zapisu VoiceLoopa)",
    search_query: "search_query: (ścieżka pytania)",
    none: "bez prefiksu (stara ścieżka Screenpipe)",
  };
  for (const prefix of state.config.prefixes) {
    const option = document.createElement("option");
    option.value = prefix;
    option.textContent = prefixLabels[prefix] || prefix;
    prefixSelect.appendChild(option);
  }

  $("#analyze-hint").textContent =
    `Model: ${state.config.embeddings.configured_model} · limit ${state.config.context_tokens} tokenów · ` +
    `maks. ${state.config.max_fragments} fragmentów na wykres.`;

  renderThresholds(state.config.thresholds);
}

/* ---------------------------------------------------------------- nagrywanie */

function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function startRecording() {
  const processing = $("#record-processing").checked;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: processing,
        noiseSuppression: processing,
        autoGainControl: processing,
      },
    });
  } catch (error) {
    toast(`Brak dostępu do mikrofonu: ${error.message}`, "error");
    return;
  }

  const mimeType = pickMimeType();
  state.recorder = new MediaRecorder(state.stream, {
    ...(mimeType ? { mimeType } : {}),
    audioBitsPerSecond: 128000,
  });
  state.chunks = [];
  state.recorder.ondataavailable = (event) => {
    if (event.data && event.data.size) state.chunks.push(event.data);
  };
  state.recorder.onstop = handleRecordingStop;
  state.recorder.start(250);
  state.startedAt = performance.now();

  $("#record-button").classList.add("is-recording");
  $("#record-label").textContent = "Zatrzymaj";
  state.timer = setInterval(() => {
    $("#record-timer").textContent = formatClock((performance.now() - state.startedAt) / 1000);
  }, 200);
}

function stopRecording() {
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  clearInterval(state.timer);
  $("#record-button").classList.remove("is-recording");
  $("#record-label").textContent = "Nagrywaj";
}

async function handleRecordingStop() {
  const duration = (performance.now() - state.startedAt) / 1000;
  const type = state.recorder.mimeType || "audio/webm";
  const blob = new Blob(state.chunks, { type });
  state.chunks = [];
  $("#record-timer").textContent = "00:00";

  if (blob.size < 1024) {
    toast("Nagranie jest puste — sprawdź mikrofon.", "error");
    return;
  }

  const label = $("#record-name").value.trim();
  const params = new URLSearchParams({
    duration: duration.toFixed(2),
    processing: $("#record-processing").checked ? "1" : "0",
  });
  if (label) params.set("label", label);

  try {
    const created = await api(`/api/recordings?${params.toString()}`, {
      method: "POST",
      headers: { "Content-Type": type },
      body: blob,
    });
    toast(`Zapisano ${created.label} (${formatBytes(created.size_bytes)}).`, "ok");
    $("#record-name").value = "";
    await loadRecordings();
    await transcribe(created.id);
  } catch (error) {
    toast(`Zapis nie udał się: ${error.message}`, "error");
  }
}

/* ------------------------------------------------------------------ nagrania */

async function loadRecordings() {
  const payload = await api("/api/recordings");
  state.recordings = payload.items;
  const known = new Set(state.recordings.map((item) => item.id));
  for (const id of Array.from(state.selected)) {
    if (!known.has(id)) state.selected.delete(id);
  }
  renderRecordings();
}

function renderRecordings() {
  const host = $("#recordings");
  host.innerHTML = "";

  if (!state.recordings.length) {
    host.innerHTML = '<p class="empty">Brak nagrań. Naciśnij „Nagrywaj”.</p>';
    return;
  }

  for (const item of state.recordings) {
    const row = document.createElement("div");
    row.className = `recording ${state.selected.has(item.id) ? "is-selected" : ""}`;

    const statusClass =
      item.transcript_status === "ok"
        ? "status-ok"
        : item.transcript_status === "error"
          ? "status-error"
          : "status-pending";
    const statusText =
      item.transcript_status === "ok"
        ? `${item.word_count} słów`
        : item.transcript_status === "error"
          ? "błąd"
          : "bez transkrypcji";

    row.innerHTML = `
      <div class="recording-top">
        <input type="checkbox" ${state.selected.has(item.id) ? "checked" : ""}>
        <span class="recording-name" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span>
        <span class="status ${statusClass}">${statusText}</span>
      </div>
      <div class="recording-meta">
        ${item.duration_seconds ? `${formatClock(item.duration_seconds)} · ` : ""}${formatBytes(item.size_bytes)}${item.microphone_processing ? " · filtr mikrofonu" : ""}
      </div>
      ${item.text_preview ? `<div class="recording-preview">${escapeHtml(truncate(item.text_preview, 120))}</div>` : ""}
      ${item.transcript_error ? `<div class="recording-preview" style="color:var(--bad)">${escapeHtml(truncate(item.transcript_error, 140))}</div>` : ""}
      <div class="recording-actions">
        <audio controls preload="none" src="/api/recordings/${item.id}/audio" style="height:28px;flex:1;min-width:0"></audio>
        <button class="btn btn-ghost btn-small" data-action="transcribe">${item.transcript_status === "ok" ? "Ponów" : "Transkrybuj"}</button>
        <button class="btn btn-ghost btn-small btn-danger" data-action="delete">Usuń</button>
      </div>`;

    row.querySelector('input[type="checkbox"]').addEventListener("change", (event) => {
      if (event.target.checked) state.selected.add(item.id);
      else state.selected.delete(item.id);
      row.classList.toggle("is-selected", event.target.checked);
    });

    row.querySelector('[data-action="transcribe"]').addEventListener("click", (event) => {
      transcribe(item.id, event.target);
    });

    row.querySelector('[data-action="delete"]').addEventListener("click", async () => {
      if (!confirm(`Usunąć „${item.label}” razem z audio i wektorami?`)) return;
      try {
        await api(`/api/recordings/${item.id}`, { method: "DELETE" });
        state.selected.delete(item.id);
        await loadRecordings();
      } catch (error) {
        toast(`Nie udało się usunąć: ${error.message}`, "error");
      }
    });

    host.appendChild(row);
  }
}

async function transcribe(recordingId, button = null) {
  const original = button ? button.textContent : null;
  if (button) {
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>';
  }
  try {
    const result = await api(`/api/recordings/${recordingId}/transcribe`, { method: "POST" });
    toast(`Transkrypcja gotowa: ${result.word_count} słów, ${result.sentence_count} zdań.`, "ok");
    state.selected.add(recordingId);
  } catch (error) {
    toast(`Deepgram: ${error.message}`, "error");
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = original;
    }
    await loadRecordings();
  }
}

/* ------------------------------------------------------------------- analiza */

async function runAnalysis() {
  if (!state.selected.size) {
    toast("Zaznacz przynajmniej jedno nagranie.");
    return;
  }

  const button = $("#analyze");
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Liczę…';

  const payload = {
    recording_ids: Array.from(state.selected),
    levels: Array.from(state.levels),
    prefix: $("#prefix").value,
    neighbours: Number($("#neighbours").value),
    threshold: Number($("#threshold").value),
    projection: $("#projection").value,
    include_anchors: $("#include-anchors").checked,
    merge_identical: $("#merge-identical").checked,
    reference_texts: $("#reference-texts").value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
  };

  try {
    const result = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!result.ok) {
      toast(result.message || "Analiza nie zwróciła danych.", "error");
      return;
    }
    state.analysis = result;
    assignColours(result.groups);
    renderGraph();
    renderScale();
    renderMatrix();
    renderProjection();
    for (const warning of result.warnings || []) toast(warning);
    await loadRecordings();
  } catch (error) {
    toast(`Analiza: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Analizuj";
  }
}

function assignColours(groups) {
  state.colourByGroup = new Map();
  let index = 0;
  for (const group of groups) {
    if (group.id === "__kotwice__") state.colourByGroup.set(group.id, ANCHOR_COLOUR);
    else if (group.id === "__referencja__") state.colourByGroup.set(group.id, REFERENCE_COLOUR);
    else state.colourByGroup.set(group.id, GROUP_COLOURS[index++ % GROUP_COLOURS.length]);
  }
}

/* --------------------------------------------------------------------- graf */

function renderGraph() {
  const result = state.analysis;
  const levelSummary = Object.entries(result.level_counts)
    .map(([level, count]) => `${count} ${LEVEL_LABELS[level] || level}`)
    .join(", ");

  $("#graph-summary").innerHTML =
    `<strong>${result.fragment_count}</strong> fragmentów (${escapeHtml(levelSummary)}) · ` +
    `${result.edge_stats.shown} krawędzi z ${result.edge_stats.candidates} kandydujących · ` +
    `model ${escapeHtml(result.model)}, ${result.dimension}D · ` +
    `embedding ${formatMs(result.timings_ms.embedding)}, geometria ${formatMs(result.timings_ms.geometry)}. ` +
    `Cosinusy policzone w ${result.dimension} wymiarach — pozycja węzła jest tylko układem rysunku.`;

  const nodes = result.fragments.map((fragment) => ({
    data: {
      id: String(fragment.index),
      label: truncate(fragment.text, 28),
      text: fragment.text,
      level: fragment.level,
      group: fragment.recording_id,
      colour: state.colourByGroup.get(fragment.recording_id) || "#4da3ff",
      shape: LEVEL_SHAPES[fragment.level] || "ellipse",
      size: LEVEL_SIZES[fragment.level] || 22,
    },
  }));

  const edges = result.edges.map((edge, index) => ({
    data: {
      id: `e${index}`,
      source: String(edge.source),
      target: String(edge.target),
      cosine: edge.cosine,
      label: edge.cosine.toFixed(2),
      kind: "similarity",
    },
  }));

  const hierarchy = (result.hierarchy_edges || []).map((edge, index) => ({
    data: {
      id: `h${index}`,
      source: String(edge.source),
      target: String(edge.target),
      kind: "hierarchy",
    },
  }));

  if (state.cy) state.cy.destroy();

  state.cy = cytoscape({
    container: $("#cy"),
    elements: { nodes, edges: [...edges, ...hierarchy] },
    minZoom: 0.15,
    maxZoom: 4,
    wheelSensitivity: 0.25,
    style: [
      {
        selector: "node",
        style: {
          "background-color": "data(colour)",
          shape: "data(shape)",
          width: "data(size)",
          height: "data(size)",
          "border-width": 1.5,
          "border-color": "#0b0f16",
          label: "data(label)",
          color: "#c9d4e5",
          "font-size": 9,
          "font-family": "Inter, Segoe UI, sans-serif",
          "text-valign": "bottom",
          "text-margin-y": 3,
          "text-max-width": 90,
          "text-wrap": "ellipsis",
          "text-outline-width": 2,
          "text-outline-color": "#0a0e15",
        },
      },
      {
        selector: 'edge[kind="similarity"]',
        style: {
          "curve-style": "straight",
          width: "mapData(cosine, 0.4, 1, 0.5, 3.6)",
          "line-color": "mapData(cosine, 0.4, 1, #1e2c3f, #4da3ff)",
          opacity: 0.75,
          "font-size": 8,
          color: "#7f8ea5",
          "text-outline-width": 2,
          "text-outline-color": "#0a0e15",
        },
      },
      {
        selector: 'edge[kind="hierarchy"]',
        style: {
          "curve-style": "bezier",
          width: 1,
          "line-color": "#3a4457",
          "line-style": "dashed",
          opacity: 0.5,
          "target-arrow-shape": "triangle",
          "target-arrow-color": "#3a4457",
          "arrow-scale": 0.6,
        },
      },
      {
        selector: "node:selected",
        style: { "border-width": 3, "border-color": "#ffffff", "font-size": 11, color: "#ffffff" },
      },
      { selector: ".dimmed", style: { opacity: 0.12 } },
      { selector: ".spotlight", style: { opacity: 1, "line-color": "#f5b445", width: 3 } },
    ],
    layout: {
      name: "fcose",
      animate: false,
      randomize: true,
      quality: "proof",
      nodeSeparation: 90,
      nodeRepulsion: 9000,
      idealEdgeLength: (edge) =>
        edge.data("kind") === "hierarchy" ? 55 : 45 + 300 * (1 - edge.data("cosine")),
      edgeElasticity: (edge) => (edge.data("kind") === "hierarchy" ? 0.1 : 0.45),
      gravity: 0.3,
      numIter: 3200,
    },
  });

  state.cy.on("tap", "node", (event) => showInspector(Number(event.target.id())));
  state.cy.on("tap", (event) => {
    if (event.target === state.cy) {
      state.cy.elements().removeClass("dimmed spotlight");
      $("#inspector").innerHTML =
        '<p class="subtle">Kliknij węzeł, żeby zobaczyć fragment, jego rodziców i sąsiadów.</p>';
    }
  });

  applyEdgeLabels();
  applyHierarchyVisibility();
  renderLegend();
}

function applyEdgeLabels() {
  if (!state.cy) return;
  const show = $("#show-edge-labels").checked;
  state.cy.style().selector('edge[kind="similarity"]').style({ label: show ? "data(label)" : "" }).update();
}

function applyHierarchyVisibility() {
  if (!state.cy) return;
  const show = $("#show-hierarchy").checked;
  state.cy.edges('[kind="hierarchy"]').style("display", show ? "element" : "none");
}

function renderLegend() {
  const result = state.analysis;
  const host = $("#graph-legend");
  const groups = result.groups
    .map((group) => {
      const colour = state.colourByGroup.get(group.id) || "#4da3ff";
      return `<span class="legend-group"><i class="legend-swatch" style="background:${colour}"></i>${escapeHtml(group.label)} (${group.count})</span>`;
    })
    .join("");

  const levels = Object.keys(result.level_counts)
    .filter((level) => LEVEL_LABELS[level])
    .map((level) => `<span class="legend-group"><i class="legend-shape ${level}"></i>${LEVEL_LABELS[level]}</span>`)
    .join("");

  host.innerHTML = `${groups}${levels}
    <span class="legend-group"><i class="legend-swatch" style="background:${ANCHOR_COLOUR}"></i>kotwice skali</span>
    <span class="legend-group">grubość krawędzi = wartość cosinusa</span>
    <span class="legend-group">linia przerywana = rodzic → dziecko</span>`;
}

function showInspector(index) {
  const result = state.analysis;
  const fragment = result.fragments[index];
  const neighbours = result.neighbours[index] || [];

  const chain = [];
  let current = fragment;
  const byId = new Map(result.fragments.map((item) => [item.id, item]));
  const guard = new Set();
  while (current?.parent_id && !guard.has(current.parent_id)) {
    guard.add(current.parent_id);
    const parent = byId.get(current.parent_id);
    if (parent) {
      chain.push({ level: parent.level, text: parent.text });
      current = parent;
    } else {
      if (current.parent_text) chain.push({ level: "rodzic", text: current.parent_text });
      break;
    }
  }

  const time =
    fragment.start_ms === null || fragment.start_ms === undefined
      ? "brak czasów"
      : `${(fragment.start_ms / 1000).toFixed(2)} – ${(fragment.end_ms / 1000).toFixed(2)} s`;

  $("#inspector").innerHTML = `
    <h4>${escapeHtml(fragment.text)}</h4>
    <p class="subtle">${LEVEL_LABELS[fragment.level] || fragment.level} · ${escapeHtml(fragment.recording_label)}</p>
    <dl class="kv">
      <dt>id</dt><dd>${escapeHtml(fragment.id)}</dd>
      <dt>czas</dt><dd>${time}</dd>
      <dt>słów</dt><dd>${fragment.word_count}</dd>
      ${fragment.speaker !== null && fragment.speaker !== undefined ? `<dt>mówca</dt><dd>${fragment.speaker}</dd>` : ""}
      ${fragment.merged_ids?.length > 1 ? `<dt>scalone</dt><dd>${fragment.merged_ids.length} wystąpień</dd>` : ""}
    </dl>
    ${chain.length ? `<div class="chain">${chain
      .map((item) => `<div class="chain-item"><span>${LEVEL_LABELS[item.level] || item.level}</span><p>${escapeHtml(truncate(item.text, 90))}</p></div>`)
      .join("")}</div>` : ""}
    <h3 style="margin-top:10px">Najbliżsi sąsiedzi w 768D</h3>
    ${neighbours
      .map(
        (item) => `<div class="neighbour">
          <span class="neighbour-text" title="${escapeHtml(item.text)}">${escapeHtml(truncate(item.text, 26))}</span>
          <span class="neighbour-score">${num(item.cosine)}</span>
        </div>`,
      )
      .join("")}
    ${neighbours.length ? `<p class="hint" style="margin-top:8px">${escapeHtml(interpretCosine(neighbours[0].cosine))}</p>` : ""}`;

  const node = state.cy.getElementById(String(index));
  const neighbourhood = node.closedNeighborhood();
  state.cy.elements().addClass("dimmed");
  neighbourhood.removeClass("dimmed");
  node.connectedEdges('[kind="similarity"]').removeClass("dimmed").addClass("spotlight");
}

function interpretCosine(value) {
  const anchors = state.analysis?.anchors || [];
  if (!anchors.length) return "";
  const sorted = [...anchors].sort((left, right) => right.cosine - left.cosine);
  let closest = sorted[0];
  for (const anchor of sorted) {
    if (Math.abs(anchor.cosine - value) < Math.abs(closest.cosine - value)) closest = anchor;
  }
  const floor = anchors.find((item) => item.key === "bez_zwiazku");
  const above = floor ? value - floor.cosine : null;
  return (
    `${num(value, 2)} to poziom „${closest.relation}” (${num(closest.cosine, 2)})` +
    (above !== null ? `, czyli ${num(above, 2)} nad dnem skali tego modelu.` : ".")
  );
}

/* -------------------------------------------------------------- skala progów */

function renderScale() {
  const result = state.analysis;
  renderAnchors(result.anchors);
  renderThresholds(result.thresholds, state.scaleFloor);
  renderVerdict(result);
  renderHistogram(result);
  loadScaleFloor();
}

/**
 * Dno skali z rozkładu, nie z jednej kotwicy.
 *
 * Pojedyncza para „bez związku" pokazuje, jak wygląda brak podobieństwa, ale
 * nie mówi, gdzie leży dno — to próba o liczności jeden. Pomiar na korpusie
 * dziedzin liczy ponad dwa tysiące par i dopiero na tym można ustawiać próg.
 */
async function loadScaleFloor() {
  if (state.scaleFloor || state.scaleFloorPending) return;
  state.scaleFloorPending = true;
  try {
    const measured = await api("/api/scale-floor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (measured.ok) {
      state.scaleFloor = measured;
      if (state.analysis) {
        renderThresholds(state.analysis.thresholds, state.scaleFloor);
        renderVerdict(state.analysis);
      }
    }
  } catch (error) {
    /* Werdykt zostaje przy kotwicach i sam o tym mówi. */
  } finally {
    state.scaleFloorPending = false;
  }
}

function renderAnchors(anchors) {
  const host = $("#anchors");
  if (!anchors?.length) {
    host.innerHTML = '<p class="empty">Włącz kotwice skali i uruchom analizę.</p>';
    return;
  }
  host.innerHTML = anchors
    .map(
      (anchor) => `<div class="anchor-row">
        <div>
          <div class="label">${escapeHtml(anchor.relation)}</div>
          <div class="pair">„${escapeHtml(anchor.left)}” ↔ „${escapeHtml(anchor.right)}”</div>
        </div>
        <div class="value">${num(anchor.cosine)}</div>
        <div class="anchor-bar"><i style="width:${Math.max(0, Math.min(100, anchor.cosine * 100))}%"></i></div>
      </div>`,
    )
    .join("");
}

const THRESHOLD_POSITIONS = {
  ponizej_dna: ["nie odrzuca niczego", "var(--bad)"],
  wewnatrz_szumu: ["wewnątrz szumu", "var(--warn, #d08b28)"],
  na_krawedzi_szumu: ["na krawędzi szumu", "var(--warn, #d08b28)"],
  powyzej_szumu: ["ponad szumem", "var(--good)"],
};

function renderThresholds(thresholds, measured = null) {
  const host = $("#thresholds");
  if (!thresholds?.length) {
    host.innerHTML = '<p class="empty">Brak progów.</p>';
    return;
  }
  const positions = new Map(
    (measured?.thresholds || []).map((item) => [item.key, item.position]),
  );
  host.innerHTML = thresholds
    .map((item) => {
      const position = positions.get(item.key);
      const [text, colour] = THRESHOLD_POSITIONS[position] || [];
      const badge = text
        ? ` <em style="color:${colour};font-style:normal">${text}</em>`
        : "";
      return `<div class="threshold-row">
        <div>
          <div class="label">${escapeHtml(item.label)}${badge}</div>
          <div class="origin">${escapeHtml(item.key)} · ${escapeHtml(item.origin)}</div>
        </div>
        <div class="value">${num(item.value, 2)}</div>
      </div>`;
    })
    .join("");
}

function renderVerdict(result) {
  const host = $("#verdict");
  const minScore = (result.thresholds || []).find((item) => item.key === "vector_memory_min_score");
  if (!minScore) {
    host.className = "verdict";
    host.innerHTML = '<p class="subtle">Brak progu vector_memory_min_score w ustawieniach.</p>';
    return;
  }

  const measured = state.scaleFloor;
  if (!measured) {
    host.className = "verdict";
    host.innerHTML = `<p class="subtle">Mierzę dno skali na korpusie ośmiu dziedzin…
      Kotwice pokazują, <em>co</em> znaczy dana wysokość cosinusa, ale pojedyncza para
      nie wystarczy, żeby orzec, gdzie leży dno.</p>`;
    return;
  }

  const retrieval = measured.retrieval;
  const floor = retrieval.floor;
  const signal = retrieval.signal;
  const position = (measured.thresholds || []).find((item) => item.key === minScore.key)?.position;
  const cost = retrieval.signal_below_floor_p95;
  const overlapping = retrieval.separation < 0.15;

  const headline = {
    ponizej_dna: "Próg odcięcia nie odcina niczego",
    wewnatrz_szumu: "Próg leży w środku szumu",
    na_krawedzi_szumu: "Próg ociera się o szum",
    powyzej_szumu: "Próg mieści się ponad szumem",
  }[position] || "Położenie progu";

  host.className = `verdict ${position === "powyzej_szumu" ? "is-fine" : "is-alarm"}`;
  host.innerHTML = `
    <div class="verdict-title">${headline}</div>
    <p>
      Dno zmierzone na <strong>${retrieval.pairs_unrelated}</strong> parach z różnych dziedzin,
      w układzie takim jak w retrievalu: zapytanie z prefiksem <code>search_query</code>
      kontra dokument z <code>search_document</code>. Mediana szumu wynosi
      <span class="num">${num(floor.p50)}</span>, 95. percentyl <span class="num">${num(floor.p95)}</span>,
      maksimum <span class="num">${num(floor.max)}</span>. Próg
      <code>${escapeHtml(minScore.key)}</code> = <span class="num">${num(minScore.value, 2)}</span>.
    </p>
    <p>
      Pary faktycznie powiązane mają medianę <span class="num">${num(signal.p50)}</span>, czyli tylko
      <strong>${num(retrieval.separation)}</strong> nad szumem. Ten sam tekst po obu stronach
      dostaje <span class="num">${num(retrieval.identical_text.p50)}</span>, nie 1.000 — prefiksy
      rozsuwają nawet identyczną treść.
    </p>
    ${overlapping ? `<p><strong>Rozkłady niemal się pokrywają.</strong> Podniesienie progu do
      95. percentyla szumu (<span class="num">${num(floor.p95)}</span>) wycięłoby
      <strong>${num(cost * 100, 0)}%</strong> par prawdziwie powiązanych. Na tym modelu żaden
      pojedynczy próg cosinusa nie rozdzieli sygnału od szumu — skuteczniejsze jest ograniczanie
      liczby wyników i sortowanie niż odcinanie wartością.</p>` : ""}
    <p class="hint">Kotwice obok pokazują, <em>co</em> znaczy dana wysokość cosinusa.
      Dno skali bierze się jednak z rozkładu, nie z jednej pary.</p>`;
}

function renderHistogram(result) {
  const centres = result.histogram.centres || [];
  const counts = result.histogram.counts || [];
  if (!centres.length) {
    Plotly.purge("histogram");
    return;
  }

  const populated = centres.filter((_, index) => counts[index] > 0);
  const low = populated.length ? Math.min(...populated) - 0.06 : -1;

  const shapes = [];
  const annotations = [];

  for (const threshold of result.thresholds || []) {
    if (!threshold.key.includes("min_score") && !threshold.key.includes("min_confidence")) continue;
    if (threshold.value < low) continue;
    shapes.push({
      type: "line",
      x0: threshold.value,
      x1: threshold.value,
      yref: "paper",
      y0: 0,
      y1: 1,
      line: { color: "#f2617a", width: 1.4, dash: "dot" },
    });
    annotations.push({
      x: threshold.value,
      yref: "paper",
      y: 1.03,
      text: `${threshold.key.replace(/_min_(score|confidence)/, "")} ${num(threshold.value, 2)}`,
      showarrow: false,
      font: { size: 9, color: "#f2617a" },
      textangle: -35,
      xanchor: "left",
    });
  }

  for (const anchor of result.anchors || []) {
    shapes.push({
      type: "line",
      x0: anchor.cosine,
      x1: anchor.cosine,
      yref: "paper",
      y0: 0,
      y1: 0.88,
      line: { color: "#f5b445", width: 1.2 },
    });
    annotations.push({
      x: anchor.cosine,
      yref: "paper",
      y: 0.9,
      text: anchor.relation,
      showarrow: false,
      font: { size: 9, color: "#f5b445" },
      textangle: -35,
      xanchor: "left",
    });
  }

  Plotly.react(
    "histogram",
    [
      {
        type: "bar",
        x: centres,
        y: counts,
        marker: { color: "#2f5f8f", line: { width: 0 } },
        hovertemplate: "cosinus %{x:.3f}<br>%{y} par<extra></extra>",
      },
    ],
    {
      ...PLOT_LAYOUT,
      margin: { l: 52, r: 18, t: 46, b: 44 },
      title: { text: "Rozkład wszystkich par fragmentów", font: { size: 12, color: "#8f9db4" }, x: 0 },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: "cosinus", font: PLOT_FONT }, range: [low, 1.02] },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: { text: "liczba par", font: PLOT_FONT } },
      shapes,
      annotations,
      bargap: 0.04,
    },
    PLOT_CONFIG,
  );
}

/* ------------------------------------------------------------------ macierz */

function renderMatrix() {
  const result = state.analysis;
  const dendrogram = result.dendrogram;

  $("#matrix-note").textContent =
    `${dendrogram.metric}. Kolejność wierszy pochodzi z dendrogramu, więc bloki na przekątnej to realne grupy.`;

  const traces = dendrogram.segments.map((segment) => ({
    type: "scatter",
    mode: "lines",
    x: segment.x,
    y: segment.y,
    line: { color: "#4da3ff", width: 1.1 },
    hoverinfo: "skip",
  }));

  Plotly.react(
    "dendrogram",
    traces,
    {
      ...PLOT_LAYOUT,
      margin: { l: 52, r: 18, t: 30, b: 20 },
      title: { text: "Hierarchia grup (linkage średni, 768D)", font: { size: 12, color: "#8f9db4" }, x: 0 },
      xaxis: { ...PLOT_LAYOUT.xaxis, showticklabels: false, range: [-1, dendrogram.order.length] },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: { text: "odległość scalenia", font: PLOT_FONT } },
    },
    PLOT_CONFIG,
  );

  if (!result.heatmap) {
    Plotly.purge("heatmap");
    $("#heatmap").innerHTML =
      `<p class="empty">Heatmapa wyłączona przy ${result.fragment_count} fragmentach — macierz byłaby nieczytelna. Zawęź wybór albo włącz scalanie identycznych tekstów.</p>`;
    return;
  }

  const labels = result.heatmap.labels.map((text) => truncate(text, 22));
  Plotly.react(
    "heatmap",
    [
      {
        type: "heatmap",
        z: result.heatmap.matrix,
        x: labels,
        y: labels,
        colorscale: [
          [0, "#0d131f"], [0.45, "#16324a"], [0.7, "#1f5f86"],
          [0.85, "#3d9bd4"], [1, "#9fe0ff"],
        ],
        zmin: Math.min(...result.heatmap.matrix.flat()),
        zmax: 1,
        colorbar: { thickness: 10, tickfont: PLOT_FONT, outlinewidth: 0 },
        hovertemplate: "%{y}<br>%{x}<br>cosinus %{z:.3f}<extra></extra>",
      },
    ],
    {
      ...PLOT_LAYOUT,
      margin: { l: 130, r: 18, t: 30, b: 130 },
      title: { text: "Macierz cosinusów w kolejności dendrogramu", font: { size: 12, color: "#8f9db4" }, x: 0 },
      xaxis: { ...PLOT_LAYOUT.xaxis, tickangle: -45, tickfont: { ...PLOT_FONT, size: 9 }, showgrid: false },
      yaxis: { ...PLOT_LAYOUT.yaxis, tickfont: { ...PLOT_FONT, size: 9 }, showgrid: false, autorange: "reversed" },
    },
    PLOT_CONFIG,
  );
}

/* ------------------------------------------------------------------- rzut 2D */

function renderProjection() {
  const result = state.analysis;
  const distortion = result.distortion;

  const cards = [
    metricCard(
      "Trustworthiness",
      distortion.trustworthiness,
      `k=${distortion.metric_neighbours}`,
      [0.9, 0.75],
    ),
    metricCard("Continuity", distortion.continuity, `k=${distortion.metric_neighbours}`, [0.9, 0.75]),
    metricCard("Stress Kruskala", distortion.stress, "0 = brak zniekształcenia", [0.1, 0.2], true),
  ];

  if (result.projection.explained_variance_total !== null) {
    cards.push(
      metricCard(
        "Wyjaśniona wariancja",
        result.projection.explained_variance_total,
        `PC1 ${num(result.projection.explained_variance?.[0], 3)} · PC2 ${num(result.projection.explained_variance?.[1], 3)}`,
        [0.5, 0.25],
      ),
    );
  }

  $("#distortion").innerHTML = cards.join("");

  const groups = new Map();
  result.fragments.forEach((fragment, index) => {
    const key = fragment.recording_id;
    if (!groups.has(key)) groups.set(key, { label: fragment.recording_label, indices: [] });
    groups.get(key).indices.push(index);
  });

  const traces = Array.from(groups.entries()).map(([key, group]) => ({
    type: "scatter",
    mode: "markers+text",
    name: group.label,
    x: group.indices.map((index) => result.projection.coords[index][0]),
    y: group.indices.map((index) => result.projection.coords[index][1]),
    text: group.indices.map((index) => truncate(result.fragments[index].text, 18)),
    customdata: group.indices.map((index) => [
      result.fragments[index].text,
      LEVEL_LABELS[result.fragments[index].level] || result.fragments[index].level,
      distortion.per_unit_preservation[index],
    ]),
    textposition: "top center",
    textfont: { size: 8, color: "#67748a" },
    marker: {
      size: group.indices.map((index) => 6 + 6 * (distortion.per_unit_preservation[index] ?? 0)),
      color: state.colourByGroup.get(key) || "#4da3ff",
      line: { width: 0.5, color: "#0b0f16" },
    },
    hovertemplate:
      "%{customdata[0]}<br>%{customdata[1]}<br>zachowanie sąsiedztwa %{customdata[2]:.2f}<extra></extra>",
  }));

  Plotly.react(
    "scatter",
    traces,
    {
      ...PLOT_LAYOUT,
      showlegend: true,
      legend: { font: PLOT_FONT, bgcolor: "rgba(0,0,0,0)", orientation: "h", y: 1.06 },
      margin: { l: 52, r: 18, t: 52, b: 44 },
      title: {
        text: `${result.projection.method.toUpperCase()} — obraz poglądowy, wielkość punktu = ile sąsiedztwa przetrwało rzut`,
        font: { size: 12, color: "#8f9db4" },
        x: 0,
      },
      xaxis: { ...PLOT_LAYOUT.xaxis, zeroline: false },
      yaxis: { ...PLOT_LAYOUT.yaxis, zeroline: false, scaleanchor: "x", scaleratio: 1 },
    },
    PLOT_CONFIG,
  );

  const shepard = distortion.shepard;
  const maximum = Math.max(...shepard.high, ...shepard.low, 0.001);
  Plotly.react(
    "shepard",
    [
      {
        type: "scatter",
        mode: "markers",
        x: shepard.high,
        y: shepard.low,
        marker: { size: 3, color: "#4da3ff", opacity: 0.35 },
        hovertemplate: "768D %{x:.3f}<br>2D %{y:.3f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [0, maximum],
        y: [0, maximum],
        line: { color: "#f5b445", width: 1.2, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    {
      ...PLOT_LAYOUT,
      margin: { l: 52, r: 18, t: 46, b: 44 },
      title: {
        text: `Diagram Sheparda${shepard.sampled ? " (próbka par)" : ""}`,
        font: { size: 12, color: "#8f9db4" },
        x: 0,
      },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: "odległość w 768D", font: PLOT_FONT } },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: { text: "odległość na rzucie", font: PLOT_FONT } },
    },
    PLOT_CONFIG,
  );
}

function metricCard(label, value, note, [good, warn], lowerIsBetter = false) {
  let tone = "";
  if (value !== null && value !== undefined) {
    if (lowerIsBetter) tone = value <= good ? "is-good" : value <= warn ? "is-warn" : "is-bad";
    else tone = value >= good ? "is-good" : value >= warn ? "is-warn" : "is-bad";
  }
  return `<div class="metric ${tone}">
    <div class="metric-label">${escapeHtml(label)}</div>
    <div class="metric-value">${num(value)}</div>
    <div class="metric-note">${escapeHtml(note)}</div>
  </div>`;
}

/* --------------------------------------------------------------- VoiceLoop */

async function runDiagnostics() {
  const query = $("#diagnose-query").value.trim();
  if (!query) {
    toast("Wpisz pytanie, które ma trafić do pamięci.");
    return;
  }
  const button = $("#diagnose");
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>';
  try {
    const result = await api("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    renderDiagnostics(result);
  } catch (error) {
    toast(`Diagnostyka: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Odpytaj pamięć";
  }
}

function renderDiagnostics(result) {
  const host = $("#diagnose-result");

  if (!result.ok) {
    host.innerHTML = `<div class="note is-bad">${escapeHtml(result.message || "Pamięć nie odpowiedziała.")}</div>`;
    return;
  }

  const qdrant = result.qdrant || {};
  const live = qdrant.enabled && qdrant.healthy;

  const header = live
    ? `<div class="note is-good">Odpytano realnego Qdranta — kolekcja <code>${escapeHtml(qdrant.collection || "?")}</code>,
        ${result.axes.length} osi osobno oraz fuzja RRF (k=${result.rrf_k}, próg ${num(result.min_score, 2)},
        wagi ${result.adaptive_weights ? "adaptacyjne" : "stałe"}).</div>`
    : `<div class="note is-bad">Qdrant niedostępny (${escapeHtml(qdrant.detail || "brak szczegółów")}).
        Ścieżka zapasowa w SQLite ma <strong>jeden</strong> wektor, więc pięciu osi tam nie ma —
        ten wynik nie odwzorowuje produkcji.</div>`;

  const warnings = (result.warnings || [])
    .map((item) => `<div class="note">${escapeHtml(item)}</div>`)
    .join("");

  const documents = result.query_documents || {};
  const distinct = new Set(Object.values(documents)).size;
  const axisWarning =
    distinct <= 1 && result.axes.length > 1
      ? `<div class="note is-bad">Wszystkie osie dostały <strong>identyczny tekst zapytania</strong>,
         więc pięć named vectors to pięć kopii jednego wektora. Fuzja RRF nie wnosi wtedy żadnej
         nowej informacji, a mnoży koszt przez ${result.axes.length}.</div>`
      : `<div class="note is-good">Każda oś dostała inny tekst zapytania (${distinct} z ${result.axes.length}),
         więc to są realnie różne przestrzenie, a nie kopie.</div>`;

  const axes = result.axes
    .map((name) => {
      const hits = (result.per_axis || {})[name] || [];
      return `<div class="axis">
        <div class="axis-head">
          <span class="axis-name">${escapeHtml(name)}</span>
          <span class="axis-weight">waga ${num(result.weights?.[name], 2)}</span>
        </div>
        <div class="axis-query">${escapeHtml(truncate(documents[name] || "", 160))}</div>
        ${hits.length
          ? hits
              .map(
                (hit) => `<div class="axis-hit">
                  <span class="text" title="${escapeHtml(hit.title || hit.content)}">${escapeHtml(truncate(hit.title || hit.content, 40))}</span>
                  <span class="score">${num(hit.cosine)}</span>
                </div>`,
              )
              .join("")
          : '<p class="hint">Brak trafień powyżej progu na tej osi.</p>'}
      </div>`;
    })
    .join("");

  const fused = (result.fused || [])
    .map((hit, index) => {
      const axesHit = (hit.axes_hit || [])
        .map((name) => `${name} ${num(hit.evidence?.[name]?.score, 2)}`)
        .join(" · ");
      return `<div class="fused-row">
        <span class="rank">${index + 1}</span>
        <span>
          ${escapeHtml(truncate(hit.title || hit.content, 90))}
          ${axesHit ? `<div class="axis-weight" style="margin-top:2px">wciągnęły: ${escapeHtml(axesHit)}</div>` : ""}
        </span>
        <span><span class="source">${escapeHtml(hit.source || "?")}</span> <span class="score">${num(hit.fusion_score, 5)}</span></span>
      </div>`;
    })
    .join("");

  const fallback = result.fallback?.available
    ? `<h3>Ścieżka zapasowa (SQLite, jedna oś)</h3>
       <div class="fused">${result.fallback.items
         .map(
           (item, index) => `<div class="fused-row">
             <span class="rank">${index + 1}</span>
             <span>${escapeHtml(truncate(item.title || item.content, 90))}</span>
             <span><span class="source">${escapeHtml(item.source || "?")}</span> <span class="score">${num(item.cosine)}</span></span>
           </div>`,
         )
         .join("")}</div>`
    : result.fallback
      ? `<div class="note">Ścieżka zapasowa też niedostępna: ${escapeHtml(result.fallback.reason || "?")}</div>`
      : "";

  host.innerHTML = `
    ${header}
    ${axisWarning}
    ${warnings}
    <h3>Pięć osi osobno — każda widzi coś innego</h3>
    <div class="axes">${axes}</div>
    <h3>Po fuzji RRF — to trafia do modelu</h3>
    <div class="fused">${fused || '<p class="empty">Fuzja nie zwróciła nic powyżej progu.</p>'}</div>
    ${fallback}`;
}

async function runPrefixCheck() {
  const button = $("#prefix-check");
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>';
  try {
    const result = await api("/api/prefix-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    renderPrefixCheck(result);
  } catch (error) {
    toast(`Test prefiksów: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Zmierz wpływ prefiksów";
  }
}

function renderPrefixCheck(result) {
  const host = $("#prefix-result");
  if (!result.ok) {
    host.innerHTML = `<div class="note is-bad">${escapeHtml(result.message || "Test nie zwrócił danych.")}</div>`;
    return;
  }

  const displacement = result.displacement || {};
  const identity = displacement.same_text_across_prefixes || {};
  const official = (result.conventions || []).find((item) => item.key === "official");
  const legacy = (result.conventions || []).find((item) => item.key === "legacy_raw");
  const gap = official && legacy ? Math.abs(official.hits_at_1 - legacy.hits_at_1) : 0;
  const severe = gap > Math.max(1, Math.floor(result.probe_count / 20));

  const notes = (result.interpretation || [])
    .map((item) => `<div class="note ${severe ? "is-bad" : "is-good"}">${escapeHtml(item)}</div>`)
    .join("");

  const cards = [
    metricCard("Ten sam tekst, inny prefiks", identity.median, "1.000 = prefiks nic nie zmienia", [0.99, 0.95]),
    metricCard("Konwencja nomica: trafienia", official ? official.hit_at_1 : null, `${official?.hits_at_1 ?? "—"} z ${result.probe_count} sond`, [0.9, 0.75]),
    metricCard("Stara ścieżka: trafienia", legacy ? legacy.hit_at_1 : null, `${legacy?.hits_at_1 ?? "—"} z ${result.probe_count} sond`, [0.9, 0.75]),
  ].join("");

  const rows = (result.conventions || [])
    .map(
      (row) => `<div class="prefix-row">
        <div class="probe">${escapeHtml(row.label)}
          <span class="origin">zapytanie: ${escapeHtml(row.query_prefix)} · dokument: ${escapeHtml(row.document_prefix)}</span>
        </div>
        <div class="numbers">
          <span>trafienie 1.: <b>${row.hits_at_1}/${row.of_total}</b></span>
          <span>MRR: <b>${num(row.mrr)}</b></span>
          <span>margines: <b>${num(row.margin.median)}</b></span>
          <span>cosinus par: <b>${num(row.correct_pair_cosine.median)}</b></span>
          <span>najgorsza pozycja: <b>${row.worst_rank}</b></span>
        </div>
      </div>`,
    )
    .join("");

  host.innerHTML = `
    <div class="metrics">${cards}</div>
    ${notes}
    <p class="hint">${escapeHtml(result.method_note || "")}
      Model: ${escapeHtml(result.model)}, ${result.dimension}D, ${result.probe_count} sond.</p>
    ${rows}`;
}

/* ---------------------------------------------------------------- inicjacja */

function bindEvents() {
  $("#record-button").addEventListener("click", () => {
    if (state.recorder && state.recorder.state === "recording") stopRecording();
    else startRecording();
  });

  $("#refresh-recordings").addEventListener("click", () => loadRecordings());
  $("#analyze").addEventListener("click", runAnalysis);
  $("#diagnose").addEventListener("click", runDiagnostics);
  $("#prefix-check").addEventListener("click", runPrefixCheck);

  $("#diagnose-query").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runDiagnostics();
  });

  $("#neighbours").addEventListener("input", (event) => {
    $("#neighbours-value").textContent = event.target.value;
  });
  $("#threshold").addEventListener("input", (event) => {
    $("#threshold-value").textContent = Number(event.target.value).toFixed(2);
  });

  $("#show-edge-labels").addEventListener("change", applyEdgeLabels);
  $("#show-hierarchy").addEventListener("change", applyHierarchyVisibility);
  $("#graph-refit").addEventListener("click", () => state.cy?.fit(undefined, 40));

  $("#record-processing").addEventListener("change", (event) => {
    $("#record-hint").textContent = event.target.checked
      ? "Filtry przeglądarki włączone: transkrypcja opisuje sygnał po obróbce, nie surowy głos."
      : "Domyślnie surowy sygnał: bez redukcji szumu i bez AGC, żeby transkrypcja opisywała twój głos, a nie filtr przeglądarki.";
  });

  for (const tab of $$(".tab")) {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((item) => item.classList.remove("is-active"));
      $$(".panel").forEach((item) => item.classList.remove("is-active"));
      tab.classList.add("is-active");
      $(`.panel[data-panel="${tab.dataset.tab}"]`).classList.add("is-active");
      if (tab.dataset.tab === "graph") state.cy?.resize();
      else {
        for (const id of ["histogram", "dendrogram", "heatmap", "scatter", "shepard"]) {
          const element = document.getElementById(id);
          if (element?.data) Plotly.Plots.resize(element);
        }
      }
    });
  }
}

async function boot() {
  cytoscape.use(window.cytoscapeFcose);
  bindEvents();
  try {
    await loadConfig();
    await loadRecordings();
  } catch (error) {
    toast(`Nie udało się wczytać konfiguracji: ${error.message}`, "error");
  }
  loadHealth();
}

boot();
