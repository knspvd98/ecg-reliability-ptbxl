/* ECG Reliability Research Workbench — Interactive Workstation Controller */
(function () {
  "use strict";

  const MODEL_LABELS = {
    xgboost: "XGBoost",
    simple_cnn: "SimpleCNN",
    resnet1d: "ResNet1D",
    inceptiontime: "InceptionTime",
    tcn: "TCN",
    transformer: "Transformer",
  };

  const PERTURB_LABELS = {
    baseline_wander_mild: "Baseline wander (mild)",
    baseline_wander_moderate: "Baseline wander (moderate)",
    baseline_wander_severe: "Baseline wander (severe)",
    powerline_50hz_mild: "Power-line 50 Hz (mild)",
    powerline_60hz_moderate: "Power-line 60 Hz (moderate)",
    muscle_noise_18db: "EMG muscle noise (18 dB)",
    muscle_noise_12db: "EMG muscle noise (12 dB)",
    muscle_noise_6db: "EMG muscle noise (6 dB)",
    electrode_motion_moderate: "Electrode motion transient",
    lead_dropout_single_v1: "Lead dropout (V1)",
    lead_dropout_group_limb: "Lead dropout (limb group)",
    lead_dropout_group_precordial: "Lead dropout (precordial)",
    amplitude_scaling_attenuated: "Amplitude scaling (0.75×)",
    amplitude_scaling_amplified: "Amplitude scaling (1.25×)",
  };

  const PERTURB_UI = {
    none: [],
    baseline_wander: [
      { key: "amplitude", label: "Drift amplitude", type: "number", step: "0.05", value: 0.45 },
      { key: "frequency_hz", label: "Frequency (Hz)", type: "number", step: "0.05", value: 0.33 },
    ],
    powerline_interference: [
      { key: "amplitude", label: "Interference amplitude", type: "number", step: "0.05", value: 0.25 },
      { key: "frequency_hz", label: "Mains frequency (Hz)", type: "number", step: "10", value: 50 },
    ],
    muscle_noise: [{ key: "snr_db", label: "Target SNR (dB, lower = noisier)", type: "number", step: "3", value: 12 }],
    electrode_motion: [
      { key: "amplitude", label: "Impulse amplitude", type: "number", step: "0.1", value: 0.85 },
      { key: "n_events", label: "Transient events", type: "number", step: "1", value: 3 },
    ],
    gaussian_noise: [{ key: "std", label: "Noise std dev", type: "number", step: "0.05", value: 0.25 }],
    lead_dropout_single: [{ key: "lead_index", label: "Lead index (0=I … 6=V1 … 11=V6)", type: "number", step: "1", value: 6 }],
    lead_dropout_group: [{ key: "group", label: "Group (limb or precordial)", type: "text", value: "precordial" }],
    amplitude_scaling: [{ key: "scale", label: "Gain factor", type: "number", step: "0.05", value: 0.75 }],
  };

  const PREFS_KEY = "ecg_reliability_prefs_v2";
  function loadPrefs() {
    try {
      return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") || {};
    } catch {
      return {};
    }
  }
  function savePrefs(patch) {
    try {
      const next = { ...loadPrefs(), ...patch };
      localStorage.setItem(PREFS_KEY, JSON.stringify(next));
    } catch {
      // Ignore storage quota errors
    }
  }

  const prefs = loadPrefs();
  const state = {
    runs: [],
    runDetailCache: {},
    activeRunId: null,
    activeTab: prefs.activeTab || "signals",
    // Signals
    sigSplit: "test",
    signalsList: [],
    activeEcgId: null,
    signalData: null,
    perturbedData: null,
    sigTheme: prefs.sigTheme || "paper",
    sigLayout: prefs.sigLayout || "stacked",
    // Experiments
    expSearch: "",
    expTier: "all",
    expStatus: "all",
    expSortKey: "created_at_utc",
    expSortDir: "desc",
    expCompareIds: new Set(),
    comparisonsData: null,
    activeArtifactPath: null,
    // Reliability
    relClass: "MI",
    relMethod: "best",
    relBinning: "uniform",
    reliabilityData: null,
    // Robustness
    robMetric: "delta_macro_auroc",
    robSeverity: "all",
    robActivePert: null,
    robProbeEcgId: null,
    robustnessData: null,
    // Explainability
    xaiEcgId: null,
    xaiLabel: "MI",
    xaiMethod: "ig",
    xaiHighlightLead: null,
    explainabilityData: null,
  };

  const $ = (id) => document.getElementById(id);
  const workspaceHost = document.querySelector(".ws-host");
  let workspaceRequest = 0;
  let signalRequest = 0;
  let robustnessProbeRequest = 0;
  let artifactRequest = 0;
  const esc = (s) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d));

  function modelDisplayName(raw) {
    return MODEL_LABELS[raw] || raw || "Unknown";
  }

  function shortRunId(runId) {
    if (!runId) return "—";
    const parts = String(runId).split("_");
    return parts[parts.length - 1] || runId.slice(-6);
  }

  function tierMeta(tier) {
    if (tier === "full_campaign") return { label: "Full PTB-XL campaign", dot: "dot-ok", short: "Full campaign" };
    if (tier === "bounded_real") return { label: "Bounded real PTB-XL", dot: "dot-accent", short: "Bounded real" };
    if (tier === "synthetic_smoke") return { label: "Synthetic smoke", dot: "dot-warn", short: "Synthetic" };
    return { label: String(tier || "Unknown"), dot: "", short: "Unknown" };
  }

  function statusMeta(status) {
    if (status === "completed") return { label: "Completed", dot: "dot-ok", cls: "text-ok" };
    if (status === "failed") return { label: "Failed", dot: "dot-err", cls: "text-err" };
    if (status === "interrupted") return { label: "Interrupted", dot: "dot-warn", cls: "text-warn" };
    return { label: "Running", dot: "dot-accent", cls: "text-accent" };
  }

  async function apiGet(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function setError(msg) {
    const bar = $("globalError");
    if (!msg) {
      bar.classList.add("hidden");
      $("globalErrorMsg").textContent = "";
      return;
    }
    $("globalErrorMsg").textContent = msg;
    bar.classList.remove("hidden");
  }

  function setWorkspaceStatus(title, detail = "", busy = false) {
    workspaceHost.setAttribute("aria-busy", String(busy));
    workspaceHost.classList.toggle("has-workspace-status", Boolean(title));
    $("workspaceStatus").classList.toggle("hidden", !title);
    $("workspaceStatusTitle").textContent = title || "";
    $("workspaceStatusDetail").textContent = detail;
    $("workspaceStatusDetail").classList.toggle("hidden", !detail);
  }

  function workspaceLoadError(title, detail) {
    if (workspaceHost.getAttribute("aria-busy") === "true") return { title, detail };
    setError(`${title} · ${detail}`);
  }

  function openPopover(title, html) {
    $("refPopoverTitle").textContent = title;
    $("refPopoverBody").innerHTML = html;
    $("refPopover").classList.remove("hidden");
  }
  function closePopover() {
    $("refPopover").classList.add("hidden");
  }

  function bindSegGroup(containerId, onChange) {
    const root = $(containerId);
    if (!root) return;
    root.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        root.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        onChange(btn.dataset.val);
      });
    });
  }

  function setSegValue(containerId, val) {
    const root = $(containerId);
    if (!root) return;
    root.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.val === String(val)));
  }

  function bindSubtabs(scopeEl) {
    scopeEl.querySelectorAll(".subtabs").forEach((bar) => {
      const parent = bar.parentElement;
      bar.querySelectorAll(".subtab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          bar.querySelectorAll(".subtab-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          const targetId = `subtab-${btn.dataset.subtab}`;
          parent.querySelectorAll(":scope > .subtab-panel").forEach((p) => p.classList.toggle("active", p.id === targetId));
        });
      });
    });
  }

  // ---------- CANVAS VIEWERS ----------
  const signalViewer = new window.ECGCanvasViewer($("signalCanvas"), {
    theme: state.sigTheme,
    layout: state.sigLayout,
    onCaliper: (c) => {
      let line = `${c.leadName}  t=${c.tSec.toFixed(2)}s (n=${c.sampleIdx})  clean=${c.cleanVal.toFixed(3)} ${c.unitLabel}`;
      if (c.pertVal !== null) {
        const diff = c.pertVal - c.cleanVal;
        line += `  pert=${c.pertVal.toFixed(3)} (Δ=${diff >= 0 ? "+" : ""}${diff.toFixed(3)})`;
      }
      if (c.delta) {
        line += `  |  Caliper Δt=${c.delta.dtMs.toFixed(0)} ms${c.delta.bpm ? ` (${c.delta.bpm.toFixed(0)} bpm)` : ""}  ΔV=${c.delta.dv >= 0 ? "+" : ""}${c.delta.dv.toFixed(3)} ${c.unitLabel}`;
      }
      $("sigCaliperReadout").textContent = line;
    },
    onViewChange: ({ zoom }) => {
      setSegValue("sigZoomSeg", zoom);
      updateSignalScale();
      if (signalPanner) signalPanner.render();
    },
  });
  const signalPanner = new window.ECGPanner($("signalPanner"), signalViewer);

  const robPreviewViewer = new window.ECGCanvasViewer($("robPreviewCanvas"), {
    theme: "paper",
    layout: "stacked",
    subsetLeads: [1, 3, 7, 10], // II, aVR, V2, V5
  });

  const xaiViewer = new window.ECGCanvasViewer($("xaiCanvas"), {
    theme: "paper",
    layout: "stacked",
    onCaliper: (c) => {
      const attrStr = c.attrVal !== null ? `  attr=${c.attrVal >= 0 ? "+" : ""}${c.attrVal.toFixed(5)}` : "";
      $("xaiCaliperReadout").textContent = `${c.leadName}  t=${c.tSec.toFixed(2)}s  amp=${c.cleanVal.toFixed(3)} ${c.unitLabel}${attrStr}`;
    },
  });

  // ---------- GLOBAL RUN MANAGEMENT ----------
  function updateChromeContext() {
    const run = state.runs.find((r) => r.run_id === state.activeRunId);
    if (!run) {
      $("topTierDot").className = "status-dot";
      $("topTierText").textContent = "No active run";
      $("sbTierDot").className = "status-dot";
      $("sbDatasetText").textContent = "No active dataset";
      $("sbRunText").textContent = "—";
      return;
    }
    const tm = tierMeta(run.execution_tier);
    const sm = statusMeta(run.status);
    $("topTierDot").className = `status-dot ${tm.dot}`;
    $("topTierText").textContent = `${tm.label} · ${sm.label}`;
    $("sbTierDot").className = `status-dot ${tm.dot}`;
    $("sbDatasetText").textContent = `${tm.label}${run.execution_tier === "synthetic_smoke" ? " (synthetic engineering data — not clinical PTB-XL)" : ""}`;
    $("sbRunText").textContent = `${modelDisplayName(run.model_name)} · ${shortRunId(run.run_id)}${run.macro_auroc != null ? ` · AUROC ${fmt(run.macro_auroc, 3)}` : ""}`;
  }

  function selectRun(runId) {
    if (state.activeRunId !== runId) {
      state.activeRunId = runId;
      state.activeEcgId = null;
      state.xaiEcgId = null;
      state.xaiHighlightLead = null;
      xaiViewer.setView({ highlightLead: null });
    }
    $("globalRunSelect").value = runId;
    updateChromeContext();
  }

  async function loadRuns() {
    setError(null);
    ++workspaceRequest;
    setWorkspaceStatus("Loading experiment registry…", "", true);
    const select = $("globalRunSelect");
    select.disabled = true;
    try {
      const data = await apiGet("/api/runs");
      state.runs = data.runs || [];
      if (!state.runs.length) {
        state.activeRunId = null;
        select.innerHTML = `<option value="">No runs in outputs/runs/</option>`;
        updateChromeContext();
        setWorkspaceStatus("No experiment runs", "Generate local artifacts with ecg-reliability run-suite --config configs/debug.yaml --tier synthetic_smoke.");
        return;
      }
      select.innerHTML = state.runs
        .map((r) => {
          const prefix = r.status === "completed" ? "" : `[${r.status}] `;
          const label = `${prefix}${modelDisplayName(r.model_name)} · ${shortRunId(r.run_id)}`;
          return `<option value="${esc(r.run_id)}" title="${esc(r.run_id)}">${esc(label)}</option>`;
        })
        .join("");
      if (!state.activeRunId || !state.runs.some((r) => r.run_id === state.activeRunId)) {
        const preferred = state.runs.find((r) => r.status === "completed" && r.model_name === "resnet1d") || state.runs.find((r) => r.status === "completed") || state.runs[0];
        selectRun(preferred.run_id);
      }
      select.value = state.activeRunId;
      select.disabled = false;
      updateChromeContext();
      await loadActiveWorkspace();
    } catch (err) {
      setWorkspaceStatus("Unable to load experiment registry", err.message);
    }
  }

  function switchWorkspace(tabName) {
    state.activeTab = tabName;
    $("chartTooltip").classList.add("hidden");
    savePrefs({ activeTab: tabName });
    document.querySelectorAll(".ws-nav-btn").forEach((b) => {
      const isAct = b.dataset.tab === tabName;
      b.classList.toggle("active", isAct);
      b.setAttribute("aria-selected", String(isAct));
    });
    document.querySelectorAll(".ws-view").forEach((v) => v.classList.toggle("active", v.id === `view-${tabName}`));
    loadActiveWorkspace();
  }

  async function loadActiveWorkspace() {
    if (!state.activeRunId || $("globalRunSelect").disabled) return;
    const request = ++workspaceRequest;
    const run = state.runs.find((item) => item.run_id === state.activeRunId);
    setError(null);
    if (run?.status === "failed" && state.activeTab !== "experiments") {
      const failure = run.failure_info;
      setError(["Experiment failed", failure?.stage, failure?.message].filter(Boolean).join(" · "));
    }
    const tab = state.activeTab;
    const loaders = {
      signals: loadSignalWorkspace,
      experiments: loadExperimentWorkspace,
      reliability: loadReliabilityWorkspace,
      robustness: loadRobustnessWorkspace,
      explainability: loadExplainabilityWorkspace,
    };
    const label = document.querySelector(`.ws-nav-btn[data-tab="${tab}"]`).textContent;
    setWorkspaceStatus(`Loading ${label.toLowerCase()}…`, "", true);
    const issue = await loaders[tab]();
    if (request !== workspaceRequest) return;
    if (issue) setWorkspaceStatus(issue.title, issue.detail);
    else setWorkspaceStatus(null);
  }

  // ==================== 1. SIGNALS WORKSPACE ====================
  function populateLeadPicker() {
    const names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
    $("sigLeadSelect").innerHTML = names.map((n, i) => `<option value="${i}" ${i === 1 ? "selected" : ""}>Lead ${n}</option>`).join("");
  }

  function renderPerturbInputs() {
    const type = $("sigPerturbType").value;
    const fields = PERTURB_UI[type] || [];
    $("sigPerturbParams").innerHTML = fields
      .map(
        (f) => `<div class="form-field"><label for="perturb-${esc(f.key)}">${esc(f.label)}</label><input id="perturb-${esc(f.key)}" class="ctrl-input" data-pkey="${esc(f.key)}" type="${f.type}" step="${f.step || ""}" value="${esc(f.value)}"/></div>`
      )
      .join("");
  }

  async function loadSignalWorkspace() {
    const request = ++signalRequest;
    const run = state.activeRunId;
    const split = state.sigSplit;
    try {
      const list = await apiGet(`/api/runs/${encodeURIComponent(run)}/signals?split=${encodeURIComponent(split)}&limit=120`);
      if (request !== signalRequest || run !== state.activeRunId) return;
      state.signalsList = list.records || [];
      const sel = $("sigRecordSelect");
      if (state.activeEcgId != null && !state.signalsList.some((r) => r.ecg_id === state.activeEcgId)) {
        // A linked failure or disagreement can be outside the first page of records.
        state.signalsList.unshift({ ecg_id: state.activeEcgId });
      }
      sel.replaceChildren(...state.signalsList.map((r) => {
        const details = r.true_labels ? ` · P${r.patient_id ?? "—"} · ${r.true_labels.join("+") || "—"}` : "";
        return new Option(`#${r.ecg_id}${details}`, r.ecg_id);
      }));
      if (!state.signalsList.length) {
        clearSignalRecord(`No records in ${split} split`);
        return;
      }
      state.activeEcgId ??= state.signalsList[0].ecg_id;
      sel.value = String(state.activeEcgId);
      await loadSignalRecord(state.activeEcgId);
    } catch (err) {
      if (request !== signalRequest || run !== state.activeRunId) return;
      state.signalsList = [];
      $("sigRecordSelect").replaceChildren();
      clearSignalRecord("Waveforms unavailable", err.message);
    }
  }

  function clearSignalRecord(title, detail = "") {
    state.signalData = null;
    state.perturbedData = null;
    signalViewer.setData({ signal: null });
    signalPanner.render();
    $("sigScaleReadout").textContent = "";
    $("sigCaliperReadout").textContent = title;
    $("sigMetaBody").innerHTML = `<strong>${esc(title)}</strong>${detail ? `<div class="muted">${esc(detail)}</div>` : ""}`;
    for (const id of ["sigPredBody", "sigSqiSummary", "sigSqiTable", "sigPerturbDeltaBody"]) $(id).replaceChildren();
    $("sigApplyPerturbBtn").disabled = true;
    $("sigRecordSelect").disabled = !state.signalsList.length;
    $("sigPrevBtn").disabled = state.signalsList.length < 2;
    $("sigNextBtn").disabled = state.signalsList.length < 2;
  }

  function updateSignalScale() {
    if (!state.signalData) return;
    const m = state.signalData.metadata;
    const unit = m.amplitude_unit === "mV" ? "mV" : "Normalized SD";
    const zoom = signalViewer.zoom === 1 ? "" : ` · ${fmt(signalViewer.zoom, 2)}× zoom`;
    $("sigScaleReadout").textContent = `${m.duration_seconds.toFixed(1)} s @ ${m.sampling_rate_hz} Hz · ${unit}${zoom}`;
  }

  async function loadSignalRecord(ecgId) {
    const request = ++signalRequest;
    const run = state.activeRunId;
    state.activeEcgId = Number(ecgId);
    state.perturbedData = null;
    $("sigApplyPerturbBtn").disabled = true;
    try {
      const detail = await apiGet(`/api/runs/${encodeURIComponent(run)}/signal/${ecgId}`);
      if (request !== signalRequest || run !== state.activeRunId) return;
      state.signalData = detail;
      $("sigApplyPerturbBtn").disabled = false;
      $("sigRecordSelect").disabled = false;
      $("sigPrevBtn").disabled = state.signalsList.length < 2;
      $("sigNextBtn").disabled = state.signalsList.length < 2;
      $("sigCaliperReadout").textContent = "Hover for sample readout · Wheel to zoom · Drag to pan";
      const m = detail.metadata;
      const unitShort = m.amplitude_unit === "mV" ? "mV" : "SD";
      updateSignalScale();

      signalViewer.setData({
        signal: detail.signal,
        perturbed: null,
        leadNames: m.lead_names,
        samplingRate: m.sampling_rate_hz,
        unitLabel: unitShort,
      });
      signalPanner.render();
      renderSignalInspector(detail, null);
    } catch (err) {
      if (request === signalRequest && run === state.activeRunId) clearSignalRecord(`ECG #${ecgId} unavailable`, err.message);
    }
  }

  async function applySignalPerturbation() {
    if (!state.activeRunId || state.activeEcgId == null || !state.signalData) return;
    const detail = state.signalData;
    const request = signalRequest;
    const run = state.activeRunId;
    const type = $("sigPerturbType").value;
    if (type === "none") {
      state.perturbedData = null;
      signalViewer.setData({
        signal: state.signalData.signal,
        perturbed: null,
        leadNames: state.signalData.metadata.lead_names,
        samplingRate: state.signalData.metadata.sampling_rate_hz,
        unitLabel: state.signalData.metadata.amplitude_unit === "mV" ? "mV" : "SD",
      });
      renderSignalInspector(state.signalData, null);
      return;
    }
    const params = {};
    $("sigPerturbParams")
      .querySelectorAll("input[data-pkey]")
      .forEach((inp) => {
        params[inp.dataset.pkey] = inp.type === "number" ? Number(inp.value) : inp.value;
      });
    try {
      setError(null);
      const res = await apiPost(`/api/runs/${encodeURIComponent(run)}/perturb`, {
        ecg_id: state.activeEcgId,
        perturbation: type,
        params,
      });
      if (request !== signalRequest || state.signalData !== detail || run !== state.activeRunId || type !== $("sigPerturbType").value) return;
      state.perturbedData = res;
      signalViewer.setData({
        signal: state.signalData.signal,
        perturbed: res.perturbed_signal,
        leadNames: state.signalData.metadata.lead_names,
        samplingRate: state.signalData.metadata.sampling_rate_hz,
        unitLabel: state.signalData.metadata.amplitude_unit === "mV" ? "mV" : "SD",
      });
      renderSignalInspector(state.signalData, res);
    } catch (err) {
      if (request === signalRequest && run === state.activeRunId) setError(`Perturbation inference failed: ${err.message}`);
    }
  }

  function renderSignalInspector(detail, pert) {
    const m = detail.metadata;
    const gtHtml = (m.true_labels || []).length
      ? `<span class="gt-list">${m.true_labels.map((l) => `<span class="gt-token gt-token-active">${esc(l)}</span>`).join("")}</span>`
      : `<span class="muted">None</span>`;
    $("sigMetaBody").innerHTML = `
      <h3 class="insp-heading"><span>Record metadata</span><span class="muted">${esc(m.split)} · fold ${m.strat_fold ?? "—"}</span></h3>
      <dl class="dl-compact">
        <dt>ECG / Patient</dt><dd class="tnum">#${m.ecg_id} / ${m.patient_id ?? "—"}</dd>
        <dt>Ground truth</dt><dd>${gtHtml}</dd>
        <dt>Sampling</dt><dd>${m.sampling_rate_hz} Hz · ${m.num_samples} samples</dd>
        <dt>Amplitude unit</dt><dd>${m.amplitude_unit === "mV" ? "mV (physical)" : "SD (z-normalized)"}</dd>
        <dt>Fingerprint</dt><dd class="mono" title="${esc(m.preprocessing_fingerprint)}">${esc((m.preprocessing_fingerprint || "").slice(0, 14))}</dd>
      </dl>`;

    const cleanPred = (pert && pert.clean_prediction) || detail.predictions;
    const pertPred = pert ? pert.perturbed_prediction : null;
    const labels = ["NORM", "MI", "STTC", "CD", "HYP"];

    if (!cleanPred) {
      $("sigPredBody").innerHTML = `<h3 class="insp-heading">Model output</h3><div class="muted">Saved predictions are available on the test split, or run a live perturbation probe.</div>`;
    } else {
      const cProbs = cleanPred.calibrated_probabilities || cleanPred.raw_probabilities || {};
      const pProbs = pertPred ? pertPred.calibrated_probabilities || pertPred.raw_probabilities || {} : null;
      const cd = cleanPred.selective_decision;
      const pd = pertPred && pertPred.selective_decision;

      let decLine = "";
      if (cd) {
        const cCls = cd.action === "ACCEPT" ? "text-ok" : "text-err";
        const pStr = pd
          ? ` → <span class="${pd.action === "ACCEPT" ? "text-ok" : "text-err"}">${pd.action === "ACCEPT" ? "Accepted" : "Abstained"}</span> (u=${fmt(pd.uncertainty, 3)})`
          : "";
        decLine = `<div style="margin-bottom:8px;font-size:12px;" class="tnum">Selective policy: <strong class="${cCls}">${cd.action === "ACCEPT" ? "Accepted" : "Abstained"}</strong> (u=${fmt(cd.uncertainty, 3)})${pStr}</div>`;
      }

      const rows = labels
        .map((lbl) => {
          const isTrue = (m.true_labels || []).includes(lbl);
          const cv = Number(cProbs[lbl] ?? 0);
          const pv = pProbs ? Number(pProbs[lbl] ?? 0) : null;
          const delta = pv !== null ? pv - cv : null;
          return `<tr>
            <td><span class="${isTrue ? "gt-token gt-token-active" : "secondary"}">${lbl}</span></td>
            <td class="num">${fmt(cv, 3)}</td>
            <td style="width:72px;"><div class="prob-track"><div class="prob-fill" style="width:${(cv * 100).toFixed(0)}%"></div></div></td>
            <td class="num ${delta !== null && Math.abs(delta) >= 0.05 ? "text-err" : "muted"}">${delta !== null ? `${delta >= 0 ? "+" : ""}${fmt(delta, 3)}` : "—"}</td>
          </tr>`;
        })
        .join("");

      $("sigPredBody").innerHTML = `
        <h3 class="insp-heading"><span>Model probabilities</span><span class="muted">${esc(cleanPred.calibration_method || "raw")}</span></h3>
        ${decLine}
        <table class="sci-table">
          <thead><tr><th>Class</th><th class="num">Clean</th><th>Profile</th><th class="num">Δ Pert</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    }

    // Perturb summary block inside Perturb subtab
    if (!pert) {
      $("sigPerturbDeltaBody").innerHTML = `<div class="muted">Select a perturbation above and click <strong>Apply &amp; infer</strong> to overlay the degraded waveform and measure probability and selective-decision shifts.</div>`;
    } else {
      $("sigPerturbDeltaBody").innerHTML = `
        <h3 class="insp-heading">Active perturbation telemetry</h3>
        <dl class="dl-compact">
          <dt>Empirical SNR</dt><dd class="tnum">${pert.empirical_snr_db !== null ? `${fmt(pert.empirical_snr_db, 1)} dB` : "—"}</dd>
          <dt>Mean basSQI</dt><dd class="tnum">${fmt(pert.clean_sqi.mean_basSQI, 3)} → ${fmt(pert.perturbed_sqi.mean_basSQI, 3)}</dd>
          <dt>Flatline leads</dt><dd class="tnum">${pert.perturbed_sqi.n_flatline_leads}</dd>
        </dl>
        <p class="muted" style="margin:8px 0 0;font-size:11px;">Traces: dark = original, vermilion = perturbed. Synthetic stress test only.</p>`;
    }

    // Quality (SQI) subtab
    const activeSqi = (pert && pert.perturbed_sqi) || detail.sqi;
    if (activeSqi) {
      $("sigSqiSummary").innerHTML = `
        <h3 class="insp-heading"><span>Aggregate SQI</span><span class="muted">${pert ? "Perturbed" : "Original"}</span></h3>
        <dl class="dl-compact">
          <dt>Mean kSQI</dt><dd class="tnum">${fmt(activeSqi.mean_kSQI, 2)}</dd>
          <dt>Mean basSQI</dt><dd class="tnum">${fmt(activeSqi.mean_basSQI, 3)}</dd>
          <dt>Mean pSQI</dt><dd class="tnum">${fmt(activeSqi.mean_pSQI, 3)}</dd>
          <dt>Flatlines</dt><dd class="tnum">${activeSqi.n_flatline_leads} / ${activeSqi.num_leads}</dd>
        </dl>`;
      $("sigSqiTable").innerHTML = `
        <table class="sci-table">
          <thead><tr><th>Lead</th><th class="num">kSQI</th><th class="num">basSQI</th><th class="num">HF</th><th>State</th></tr></thead>
          <tbody>${activeSqi.per_lead
            .map((r) => {
              const dot = r.quality_grade === "acceptable" ? "dot-ok" : r.quality_grade === "unacceptable" ? "dot-err" : "dot-warn";
              return `<tr class="clickable" data-leadidx="${r.lead_index}" title="Click to focus lead ${esc(r.lead)}">
                <td><strong>${esc(r.lead)}</strong></td>
                <td class="num">${fmt(r.kSQI, 2)}</td>
                <td class="num">${fmt(r.basSQI, 2)}</td>
                <td class="num">${fmt(r.hf_noise_ratio, 2)}</td>
                <td><span class="status-dot ${dot}" style="margin-right:5px;"></span>${esc(r.quality_grade)}</td>
              </tr>`;
            })
            .join("")}</tbody>
        </table>`;
      $("sigSqiTable").querySelectorAll("tr[data-leadidx]").forEach((tr) => {
        tr.addEventListener("click", () => {
          const idx = Number(tr.dataset.leadidx);
          state.sigLayout = "single";
          setSegValue("sigLayoutSeg", "single");
          $("sigLeadSelect").classList.remove("hidden");
          $("sigLeadSelect").value = String(idx);
          signalViewer.setView({ layout: "single", focusLead: idx });
        });
      });
    }
  }

  // ==================== 2. EXPERIMENTS WORKSPACE ====================
  async function getRunDetailCached(runId) {
    if (!state.runDetailCache[runId]) {
      state.runDetailCache[runId] = await apiGet(`/api/runs/${encodeURIComponent(runId)}`);
    }
    return state.runDetailCache[runId];
  }

  async function loadExperimentWorkspace() {
    const run = state.activeRunId;
    const request = workspaceRequest;
    try {
      const [detail, comps] = await Promise.all([getRunDetailCached(run), apiGet("/api/comparisons")]);
      if (request !== workspaceRequest || run !== state.activeRunId) return;
      state.comparisonsData = comps;
      state.runs = comps.runs || state.runs;
      renderExperimentsTable();
      await renderExperimentsLower(detail, comps, request);
    } catch (err) {
      if (request === workspaceRequest && run === state.activeRunId) return workspaceLoadError("Unable to load experiment details", err.message);
    }
  }

  function renderExperimentsTable() {
    const q = state.expSearch.trim().toLowerCase();
    let rows = [...state.runs].filter((r) => {
      if (state.expTier !== "all" && r.execution_tier !== state.expTier) return false;
      if (state.expStatus !== "all" && r.status !== state.expStatus) return false;
      if (q) {
        const hay = `${r.model_name} ${r.run_id} ${r.selected_calibration_method || ""} ${r.execution_tier}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    const dir = state.expSortDir === "asc" ? 1 : -1;
    rows.sort((a, b) => {
      const av = a[state.expSortKey];
      const bv = b[state.expSortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });

    $("expCompareCount").textContent = `${state.expCompareIds.size} selected for comparison`;

    if (!rows.length) {
      $("expRunsTableWrap").innerHTML = `<div class="state-empty"><strong>No runs match current filters</strong></div>`;
      return;
    }

    const th = (key, label, num = false) => {
      const arrow = state.expSortKey === key ? (state.expSortDir === "asc" ? " ↑" : " ↓") : "";
      return `<th class="sortable ${num ? "num" : ""}" data-sort="${key}">${label}${arrow}</th>`;
    };

    $("expRunsTableWrap").innerHTML = `
      <table class="sci-table">
        <thead>
          <tr>
            <th style="width:28px;"></th>
            ${th("model_name", "Model / Run")}
            ${th("status", "Status")}
            ${th("execution_tier", "Tier")}
            ${th("macro_auroc", "AUROC", true)}
            ${th("macro_auprc", "AUPRC", true)}
            ${th("cal_macro_ece", "Cal ECE", true)}
            ${th("cal_macro_brier", "Cal Brier", true)}
            ${th("selected_calibration_method", "Calibrator")}
            ${th("total_parameters", "Params", true)}
            ${th("created_at_utc", "Created (UTC)")}
          </tr>
        </thead>
        <tbody>${rows
          .map((r) => {
            const isSel = r.run_id === state.activeRunId;
            const checked = state.expCompareIds.has(r.run_id) ? "checked" : "";
            const sm = statusMeta(r.status);
            const tm = tierMeta(r.execution_tier);
            const created = (r.created_at_utc || "").replace("T", " ").slice(0, 16);
            const params = r.total_parameters ? Number(r.total_parameters).toLocaleString() : "—";
            return `<tr class="clickable ${isSel ? "selected" : ""}" data-runid="${esc(r.run_id)}">
              <td><input type="checkbox" class="exp-cmp-chk" data-cmpid="${esc(r.run_id)}" ${checked} aria-label="Compare ${esc(r.model_name)}"/></td>
              <td title="${esc(r.run_id)}"><strong>${esc(modelDisplayName(r.model_name))}</strong> <span class="tnum muted">${esc(shortRunId(r.run_id))}</span></td>
              <td><span class="status-dot ${sm.dot}" style="margin-right:5px;"></span><span class="${sm.cls}">${sm.label}</span></td>
              <td class="secondary"><span class="status-dot ${tm.dot}" style="margin-right:5px;"></span>${tm.short}</td>
              <td class="num">${fmt(r.macro_auroc, 4)}</td>
              <td class="num">${fmt(r.macro_auprc, 4)}</td>
              <td class="num">${fmt(r.cal_macro_ece, 4)}</td>
              <td class="num">${fmt(r.cal_macro_brier, 4)}</td>
              <td class="secondary">${esc(r.selected_calibration_method || "—")}</td>
              <td class="num muted">${params}</td>
              <td class="tnum muted">${esc(created)}</td>
            </tr>`;
          })
          .join("")}</tbody>
      </table>`;

    $("expRunsTableWrap").querySelectorAll("th[data-sort]").forEach((el) => {
      el.addEventListener("click", () => {
        const key = el.dataset.sort;
        if (state.expSortKey === key) state.expSortDir = state.expSortDir === "asc" ? "desc" : "asc";
        else {
          state.expSortKey = key;
          state.expSortDir = "desc";
        }
        renderExperimentsTable();
      });
    });

    $("expRunsTableWrap").querySelectorAll(".exp-cmp-chk").forEach((chk) => {
      chk.addEventListener("click", (ev) => ev.stopPropagation());
      chk.addEventListener("change", () => {
        if (chk.checked) state.expCompareIds.add(chk.dataset.cmpid);
        else state.expCompareIds.delete(chk.dataset.cmpid);
        $("expCompareCount").textContent = `${state.expCompareIds.size} selected for comparison`;
        loadExperimentWorkspace();
      });
    });

    $("expRunsTableWrap").querySelectorAll("tr[data-runid]").forEach((tr) => {
      tr.addEventListener("click", () => {
        selectRun(tr.dataset.runid);
        loadActiveWorkspace();
      });
    });
  }

  async function renderExperimentsLower(detail, comps, request) {
    const m = detail.manifest;
    const history = (m.metrics && m.metrics.training_history) || (detail.training_runtime_report && detail.training_runtime_report.epoch_history) || [];

    // Multi-run convergence overlay when >= 2 runs are checked
    const compareSeries = [];
    if (state.expCompareIds.size >= 2) {
      for (const rid of state.expCompareIds) {
        try {
          const d = await getRunDetailCached(rid);
          const h = (d.manifest.metrics && d.manifest.metrics.training_history) || [];
          if (h.length) {
            compareSeries.push({ label: `${modelDisplayName(d.manifest.model_identity.model_name)} (${shortRunId(rid)})`, history: h });
          }
        } catch {
          // skip
        }
      }
    }
    if (request !== workspaceRequest || m.run_id !== state.activeRunId) return;
    window.WorkbenchCharts.renderTrainingHistoryChart($("expHistoryChart"), history, compareSeries);

    // Run comparison diff table
    const cmpIds = state.expCompareIds.size >= 2 ? Array.from(state.expCompareIds) : state.runs.slice(0, 6).map((r) => r.run_id);
    const cmpRuns = cmpIds.map((id) => state.runs.find((r) => r.run_id === id)).filter(Boolean);
    const cmpMetrics = [
      ["Execution tier", (r) => tierMeta(r.execution_tier).short],
      ["Status", (r) => statusMeta(r.status).label],
      ["Macro AUROC", (r) => fmt(r.macro_auroc, 4)],
      ["Macro AUPRC", (r) => fmt(r.macro_auprc, 4)],
      ["Macro F1", (r) => fmt(r.macro_f1, 4)],
      ["Raw ECE", (r) => fmt(r.raw_macro_ece, 4)],
      ["Calibrated ECE", (r) => fmt(r.cal_macro_ece, 4)],
      ["Adaptive ECE", (r) => fmt(r.cal_macro_adaptive_ece, 4)],
      ["Calibrated Brier", (r) => fmt(r.cal_macro_brier, 4)],
      ["Selected calibrator", (r) => r.selected_calibration_method || "—"],
    ];
    $("expCompareBody").innerHTML = cmpRuns.length
      ? `<table class="sci-table">
          <thead><tr><th>Metric</th>${cmpRuns.map((r) => `<th class="num">${esc(modelDisplayName(r.model_name))} <span class="muted">${shortRunId(r.run_id)}</span></th>`).join("")}</tr></thead>
          <tbody>${cmpMetrics.map(([k, fn]) => `<tr><td class="secondary">${k}</td>${cmpRuns.map((r) => `<td class="num">${esc(fn(r))}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>`
      : `<div class="state-empty"><strong>No runs available to compare</strong></div>`;

    // Paired bootstrap table
    const paired = (comps.paired_comparisons && comps.paired_comparisons.paired_comparisons) || [];
    $("expPairedBody").innerHTML = paired.length
      ? `<table class="sci-table">
          <thead><tr><th>Pair (A − B)</th><th>Metric</th><th class="num">Δ Point</th><th class="num">95% CI</th><th class="num">p (2-sided)</th><th class="num">N_eff</th></tr></thead>
          <tbody>${paired
            .map(
              (p) => `<tr>
                <td><strong>${esc(modelDisplayName(p.model_a))}</strong> − ${esc(modelDisplayName(p.model_b))}</td>
                <td class="secondary">${esc(p.metric)}</td>
                <td class="num ${p.delta_point_estimate > 0 ? "text-ok" : p.delta_point_estimate < 0 ? "text-err" : ""}">${p.delta_point_estimate >= 0 ? "+" : ""}${fmt(p.delta_point_estimate, 4)}</td>
                <td class="num">[${fmt(p.delta_ci_low, 4)}, ${fmt(p.delta_ci_high, 4)}]</td>
                <td class="num">${fmt(p.p_value_two_sided, 3)}</td>
                <td class="num muted" title="${esc(p.ci_definition)}">${p.effective_sample_size} (${p.resampling_unit === "patient_cluster" ? "patients" : "records"})</td>
              </tr>`
            )
            .join("")}</tbody>
        </table>`
      : `<div class="state-empty"><strong>No paired bootstrap comparisons</strong><span>Run a multi-model benchmark suite to compute paired patient-cluster intervals.</span></div>`;

    // Provenance inspector
    const hw = m.hardware_software_metadata || {};
    const cv = m.code_version || {};
    const ds = m.dataset_provenance || {};
    const ov = ds.patient_overlap_report || {};
    const ckpts = Object.entries(m.checkpoints || {})
      .map(([k, v]) => `${k}: ${(v.sha256 || "—").slice(0, 12)}`)
      .join(" · ");
    const failBlock = m.failure_info
      ? `<p class="run-failure"><strong class="text-err">Failed in ${esc(m.failure_info.stage)}</strong> · ${esc(m.failure_info.exception_type)} · ${esc(m.failure_info.message)}</p>`
      : "";
    $("expProvenanceBody").innerHTML = `
      <dl class="dl-compact">
        <dt>Full run ID</dt><dd>${esc(m.run_id)}</dd>
        <dt>Execution tier</dt><dd>${tierMeta(m.execution_tier).label} · claim eligible: ${m.scientific_claim_eligible ? "yes" : "no"}</dd>
        <dt>Code version</dt><dd><code>${esc((cv.git_commit || "n/a").slice(0, 12))}</code> (${esc(cv.git_branch || "?")}${cv.git_dirty ? ", dirty" : ""})</dd>
        <dt>Preprocessing</dt><dd class="mono">${esc(m.preprocessing_fingerprint || "—")}</dd>
        <dt>Dataset splits</dt><dd>${esc(ds.dataset_identity || ds.manifest_path || "—")} · train ${ds.split_counts?.train ?? "—"} / val ${ds.split_counts?.val ?? "—"} / test ${ds.split_counts?.test ?? "—"}</dd>
        <dt>Patient leakage</dt><dd>${ov.patient_leakage_checked ? (ov.has_patient_leakage ? '<span class="text-err">Leakage detected</span>' : '<span class="text-ok">Verified disjoint (0 overlap)</span>') : "Not checked"}</dd>
        <dt>Random seed</dt><dd>${esc(m.seeds?.project_seed)} · deterministic: ${String(m.seeds?.deterministic_torch)}</dd>
        <dt>Checkpoints</dt><dd class="mono">${esc(ckpts || "none")}</dd>
        <dt>Runtime</dt><dd>Python ${esc(hw.python_version)} · PyTorch ${esc(hw.packages?.torch)} · ${esc(hw.platform)}</dd>
      </dl>${failBlock}`;

    // Artifacts tree
    const files = detail.files || [];
    $("expArtifactList").innerHTML = files.length
      ? `<table class="sci-table"><tbody>${files
          .map(
            (f) => `<tr class="clickable ${state.activeArtifactPath === f.path ? "selected" : ""}" data-artpath="${esc(f.path)}">
              <td class="artifact-path">${esc(f.path)}</td>
              <td class="num muted" style="font-size:10.5px;">${f.size_bytes > 1024 ? `${(f.size_bytes / 1024).toFixed(1)} KB` : `${f.size_bytes} B`}</td>
            </tr>`
          )
          .join("")}</tbody></table>`
      : `<div class="state-empty">No files</div>`;
    $("expArtifactList").querySelectorAll("tr[data-artpath]").forEach((tr) => {
      tr.addEventListener("click", () => loadArtifactPreview(tr.dataset.artpath));
    });
    if (files.length) {
      const file = files.find((f) => f.path === state.activeArtifactPath) || files.find((f) => f.path === "run_manifest.json") || files[0];
      void loadArtifactPreview(file.path);
    } else {
      $("expArtifactPreview").textContent = "No artifacts available.";
    }

    // Config snapshot
    try {
      const cfgArt = await apiGet(`/api/runs/${encodeURIComponent(m.run_id)}/artifact?path=config_snapshot.yaml`);
      if (request === workspaceRequest && m.run_id === state.activeRunId) $("expConfigPre").textContent = cfgArt.content || "";
    } catch {
      if (request === workspaceRequest && m.run_id === state.activeRunId) $("expConfigPre").textContent = JSON.stringify(m.config_snapshot || {}, null, 2);
    }
  }

  async function loadArtifactPreview(relPath) {
    const run = state.activeRunId;
    const request = ++artifactRequest;
    state.activeArtifactPath = relPath;
    $("expArtifactPreview").textContent = "Loading artifact…";
    $("expArtifactList").querySelectorAll("tr[data-artpath]").forEach((tr) => tr.classList.toggle("selected", tr.dataset.artpath === relPath));
    try {
      const data = await apiGet(`/api/runs/${encodeURIComponent(run)}/artifact?path=${encodeURIComponent(relPath)}`);
      if (request !== artifactRequest || run !== state.activeRunId) return;
      const content = data.content ?? JSON.stringify(data, null, 2);
      $("expArtifactPreview").textContent = data.truncated ? `${content}\n\n[Preview truncated]` : content;
    } catch (err) {
      if (request === artifactRequest && run === state.activeRunId) $("expArtifactPreview").textContent = err.message;
    }
  }

  // ==================== 3. RELIABILITY WORKSPACE ====================
  async function loadReliabilityWorkspace() {
    const run = state.activeRunId;
    const request = workspaceRequest;
    const { relMethod, relBinning } = state;
    try {
      const qs = new URLSearchParams({ method: relMethod, binning: relBinning });
      const rel = await apiGet(`/api/runs/${encodeURIComponent(run)}/reliability?${qs.toString()}`);
      if (request !== workspaceRequest || run !== state.activeRunId || relMethod !== state.relMethod || relBinning !== state.relBinning) return;
      state.reliabilityData = rel;
      renderReliabilityView();
    } catch (err) {
      if (request === workspaceRequest && run === state.activeRunId) return workspaceLoadError("Unable to load reliability analysis", err.message);
    }
  }

  function renderReliabilityView() {
    const rel = state.reliabilityData;
    if (!rel) return;
    const rawM = rel.raw_test_metrics || {};
    const calT = rel.calibration_test_reports || {};
    const best = calT.selected_method_from_val || (rel.calibration_val_reports && rel.calibration_val_reports.best_method) || "raw";
    const activeMethod = rel.active_calibration_method || best;
    const activeReport = calT[activeMethod] || {};
    const rawCal = calT.raw || {};
    const eceMetric = rel.binning === "adaptive" ? "macro_adaptive_ece" : "macro_ece";

    $("relInlineSummary").innerHTML = `
      <span>AUROC <strong>${fmt(rawM.macro_auroc, 4)}</strong></span>
      <span>AUPRC <strong>${fmt(rawM.macro_auprc, 4)}</strong></span>
      <span>ECE <strong>${fmt(rawCal[eceMetric], 4)} → ${fmt(activeReport[eceMetric], 4)}</strong></span>
      <span>Brier <strong>${fmt(activeReport.macro_brier, 4)}</strong></span>
      <span>${rel.selective_results ? (rel.selective_results.coverage_monotonic ? '<span class="text-ok">Monotonic ✓</span>' : '<span class="text-warn">Non-monotonic</span>') : ""}</span>`;

    const cls = state.relClass;
    $("relDiagTitle").textContent = `Reliability diagram · ${cls} (held-out test)`;
    const bins = rel.reliability_bins?.per_class?.[cls];
    $("relDiagMeta").textContent = bins?.calibrated ? `${activeMethod} · ${rel.binning} bins` : `${activeMethod} bins unavailable`;
    $("relHistTitle").textContent = `Predicted probability distribution · ${cls}`;
    const conflict = rel.selective_results?.norm_conflict_report;
    const policy = rel.selective_results ? `Saved policy (${rel.selective_results.calibration_method || best})` : "";
    $("relSelMeta").textContent = [policy, conflict ? `NORM conflicts resolved: ${conflict.n_changed_norm_decisions}` : ""].filter(Boolean).join(" · ");

    window.WorkbenchCharts.renderReliabilityDiagram($("relDiagramChart"), bins?.raw, activeMethod === "raw" ? null : bins?.calibrated, {
      className: cls,
      methodLabel: activeMethod,
    });
    window.WorkbenchCharts.renderRiskCoverageChart($("relRiskCoverageChart"), rel.selective_results && rel.selective_results.curves);
    window.WorkbenchCharts.renderConfidenceHistogram($("relConfidenceChart"), rel.confidence_distributions && rel.confidence_distributions[cls], cls);

    // Per-class table (click row -> updates selected superclass across charts!)
    const labels = ["NORM", "MI", "STTC", "CD", "HYP"];
    const perClass = rawM.per_class || {};
    const eceList = activeReport.ece_per_class || [];
    const aeceList = activeReport.adaptive_ece_per_class || [];
    const brierList = activeReport.brier_per_class || [];
    $("relClassTable").innerHTML = `
      <table class="sci-table">
        <thead><tr><th>Superclass</th><th class="num">AUROC</th><th class="num">AUPRC</th><th class="num">ECE</th><th class="num">Adaptive ECE</th><th class="num">Brier</th></tr></thead>
        <tbody>${labels
          .map(
            (lbl, idx) => `<tr class="clickable ${lbl === cls ? "selected" : ""}" data-relcls="${lbl}">
              <td><strong>${lbl}</strong></td>
              <td class="num">${fmt(perClass[lbl]?.auroc, 4)}</td>
              <td class="num">${fmt(perClass[lbl]?.auprc, 4)}</td>
              <td class="num">${fmt(eceList[idx], 4)}</td>
              <td class="num">${fmt(aeceList[idx], 4)}</td>
              <td class="num">${fmt(brierList[idx], 4)}</td>
            </tr>`
          )
          .join("")}</tbody>
      </table>`;
    $("relClassTable").querySelectorAll("tr[data-relcls]").forEach((tr) => {
      tr.addEventListener("click", () => {
        state.relClass = tr.dataset.relcls;
        setSegValue("relClassSeg", state.relClass);
        renderReliabilityView();
      });
    });

    // Calibration methods comparison
    const valReports = rel.calibration_val_reports || {};
    const methods = Object.keys(calT).filter((k) => k !== "selected_method_from_val");
    $("relCalMethodsTable").innerHTML = methods.length
      ? `<table class="sci-table">
          <thead><tr><th>Method</th><th class="num">Val ECE (fit)</th><th class="num">Test ECE</th><th class="num">Test Adaptive ECE</th><th class="num">Test Brier</th></tr></thead>
          <tbody>${methods
            .map(
              (m) => `<tr class="clickable ${m === activeMethod ? "selected" : ""}" data-calmethod="${esc(m)}">
                <td>${esc(m)}${m === best ? ' <span class="text-accent">· selected on val</span>' : ""}</td>
                <td class="num muted">${fmt(valReports[m]?.macro_ece, 4)}</td>
                <td class="num">${fmt(calT[m]?.macro_ece, 4)}</td>
                <td class="num">${fmt(calT[m]?.macro_adaptive_ece, 4)}</td>
                <td class="num">${fmt(calT[m]?.macro_brier, 4)}</td>
              </tr>`
            )
            .join("")}</tbody>
        </table>`
      : `<div class="state-empty">Calibration comparison unavailable</div>`;
    $("relCalMethodsTable").querySelectorAll("tr[data-calmethod]").forEach((tr) => {
      tr.addEventListener("click", () => {
        state.relMethod = tr.dataset.calmethod;
        $("relMethodSelect").value = state.relMethod;
        loadReliabilityWorkspace();
      });
    });

    // Bootstrap CIs
    const boot = rel.bootstrap_ci || {};
    const bKeys = Object.keys(boot);
    $("relBootstrapTable").innerHTML = bKeys.length
      ? `<table class="sci-table">
          <thead><tr><th>Metric</th><th class="num">Point</th><th class="num">95% CI</th><th class="num">Valid / Req</th><th class="num">N_eff</th><th>Unit &amp; Definition</th></tr></thead>
          <tbody>${bKeys
            .map((k) => {
              const b = boot[k];
              return `<tr>
                <td><strong>${esc(k)}</strong></td>
                <td class="num">${fmt(b.point_estimate, 4)}</td>
                <td class="num">[${fmt(b.ci_low, 4)}, ${fmt(b.ci_high, 4)}]</td>
                <td class="num">${b.n_bootstrap_valid}/${b.n_bootstrap_requested}${b.n_bootstrap_invalid > 0 ? ` (${b.n_bootstrap_invalid} invalid)` : ""}</td>
                <td class="num">${b.effective_sample_size}</td>
                <td class="secondary">${esc(b.ci_definition)}</td>
              </tr>`;
            })
            .join("")}</tbody>
        </table>`
      : `<div class="state-empty">No bootstrap confidence intervals recorded</div>`;
  }

  // ==================== 4. ROBUSTNESS WORKSPACE ====================
  async function loadRobustnessWorkspace() {
    const run = state.activeRunId;
    const request = workspaceRequest;
    try {
      const [rob, sigList] = await Promise.all([
        apiGet(`/api/runs/${encodeURIComponent(run)}/robustness`),
        apiGet(`/api/runs/${encodeURIComponent(run)}/signals?split=test&limit=40`).catch((err) => ({ error: err.message })),
      ]);
      if (request !== workspaceRequest || run !== state.activeRunId) return;
      state.robustnessData = rob;
      const records = sigList.records || [];
      const sel = $("robProbeRecordSelect");
      sel.innerHTML = records.map((r) => `<option value="${r.ecg_id}">#${r.ecg_id} (${(r.true_labels || []).join("+")})</option>`).join("");
      if (!records.some((r) => r.ecg_id === state.robProbeEcgId)) {
        state.robProbeEcgId = records[0]?.ecg_id ?? null;
      }
      sel.value = String(state.robProbeEcgId);
      sel.disabled = !records.length;
      $("robOpenInSignalsBtn").disabled = !records.length;

      renderRobustnessView();
      if (state.robProbeEcgId == null && sigList.error) {
        $("robProbeDetails").textContent = sigList.error;
      }
    } catch (err) {
      if (request === workspaceRequest && run === state.activeRunId) return workspaceLoadError("Unable to load robustness analysis", err.message);
    }
  }

  function mergedRobustnessRows() {
    const rob = state.robustnessData;
    if (!rob || !rob.available || !rob.results) return [];
    const specs = rob.results.perturbation_specs || {};
    const decMap = {};
    for (const d of (rob.results.decision_layer && rob.results.decision_layer.decision_delta_summary) || []) {
      decMap[d.perturbation] = d;
    }
    return (rob.results.delta_summary || [])
      .map((d) => {
        const sp = specs[d.perturbation] || {};
        const dec = decMap[d.perturbation] || {};
        return {
          ...d,
          ...dec,
          severity: sp.severity || "custom",
          units: sp.units || "signal_units",
          spec: sp,
          displayName: PERTURB_LABELS[d.perturbation] || d.perturbation,
        };
      })
      .filter((r) => state.robSeverity === "all" || r.severity === state.robSeverity);
  }

  function renderRobustnessView() {
    const rows = mergedRobustnessRows();
    $("robBarChart").parentElement.style.setProperty("--rob-chart-height", `${60 + rows.length * 23}px`);
    $("robMatrixCount").textContent = `${rows.length} conditions`;
    if (!rows.length) {
      ++robustnessProbeRequest;
      $("robBarChart").innerHTML = `<div class="state-empty"><strong>No robustness data available</strong></div>`;
      $("robFullTable").innerHTML = `<div class="state-empty">No perturbations available for this run or filter.</div>`;
      $("robProbeTitle").textContent = "Live waveform and model response";
      $("robProbeDetails").textContent = "A saved perturbation condition and test record are required.";
      robPreviewViewer.setData({ signal: null });
      return;
    }
    if (!state.robActivePert || !rows.some((r) => r.perturbation === state.robActivePert)) {
      // Pick the condition with largest absolute AUROC drop by default
      const worst = [...rows].sort((a, b) => Number(a.delta_macro_auroc) - Number(b.delta_macro_auroc))[0];
      state.robActivePert = worst.perturbation;
    }

    const selectPert = (name) => {
      state.robActivePert = name;
      renderRobustnessView();
    };

    window.WorkbenchCharts.renderRobustnessBars($("robBarChart"), rows, state.robMetric, {
      activePerturbation: state.robActivePert,
      onSelect: selectPert,
    });

    $("robFullTable").innerHTML = `
      <table class="sci-table">
        <thead><tr><th>Perturbation</th><th>Severity</th><th>Units</th><th class="num">Δ AUROC</th><th class="num">Δ AUPRC</th><th class="num">Δ ECE</th><th class="num">Δ Harm</th><th class="num">Δ Abstain</th></tr></thead>
        <tbody>${rows
          .map((r) => {
            const sevDot = r.severity === "severe" ? "dot-err" : r.severity === "moderate" ? "dot-warn" : "dot-ok";
            const isSel = r.perturbation === state.robActivePert;
            return `<tr class="clickable ${isSel ? "selected" : ""}" data-pert="${esc(r.perturbation)}">
              <td><strong>${esc(r.displayName)}</strong></td>
              <td class="secondary"><span class="status-dot ${sevDot}" style="margin-right:5px;"></span>${esc(r.severity)}</td>
              <td class="muted" title="${esc(r.units)}">${esc(r.units.startsWith("signal_units") ? "SD or mV" : r.units.replaceAll("_", " "))}</td>
              <td class="num ${r.delta_macro_auroc < -0.001 ? "text-err" : ""}">${r.delta_macro_auroc >= 0 ? "+" : ""}${fmt(r.delta_macro_auroc, 4)}</td>
              <td class="num ${r.delta_macro_auprc < -0.001 ? "text-err" : ""}">${r.delta_macro_auprc >= 0 ? "+" : ""}${fmt(r.delta_macro_auprc, 4)}</td>
              <td class="num ${r.delta_macro_ece > 0.001 ? "text-err" : ""}">${r.delta_macro_ece >= 0 ? "+" : ""}${fmt(r.delta_macro_ece, 4)}</td>
              <td class="num ${r.delta_expected_harm_with_abstain > 0.01 ? "text-err" : ""}">${r.delta_expected_harm_with_abstain != null ? `${r.delta_expected_harm_with_abstain >= 0 ? "+" : ""}${fmt(r.delta_expected_harm_with_abstain, 3)}` : "—"}</td>
              <td class="num">${r.delta_abstain_rate != null ? `${r.delta_abstain_rate >= 0 ? "+" : ""}${fmt(r.delta_abstain_rate, 3)}` : "—"}</td>
            </tr>`;
          })
          .join("")}</tbody>
      </table>`;
    $("robFullTable").querySelectorAll("tr[data-pert]").forEach((tr) => {
      tr.addEventListener("click", () => selectPert(tr.dataset.pert));
    });

    runRobustnessLiveProbe();
  }

  async function runRobustnessLiveProbe() {
    const request = ++robustnessProbeRequest;
    const run = state.activeRunId;
    if (!state.robActivePert || state.robProbeEcgId == null || !state.robustnessData) {
      $("robProbeTitle").textContent = "Live probe unavailable";
      $("robProbeDetails").textContent = "No test records available.";
      robPreviewViewer.setData({ signal: null });
      return;
    }
    const specs = (state.robustnessData.results && state.robustnessData.results.perturbation_specs) || {};
    const spec = specs[state.robActivePert] || {};
    const label = PERTURB_LABELS[state.robActivePert] || state.robActivePert;
    $("robProbeTitle").textContent = `Live probe · ${label}`;
    $("robProbeDetails").textContent = "Running probe…";

    try {
      const [sigDetail, pertRes] = await Promise.all([
        apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/signal/${state.robProbeEcgId}`),
        apiPost(`/api/runs/${encodeURIComponent(state.activeRunId)}/perturb`, {
          ecg_id: state.robProbeEcgId,
          perturbation: state.robActivePert,
          params: spec,
        }),
      ]);
      if (request !== robustnessProbeRequest || run !== state.activeRunId) return;
      const m = sigDetail.metadata;
      robPreviewViewer.setData({
        signal: sigDetail.signal,
        perturbed: pertRes.perturbed_signal,
        leadNames: m.lead_names,
        samplingRate: m.sampling_rate_hz,
        unitLabel: m.amplitude_unit === "mV" ? "mV" : "SD",
      });

      const cp = pertRes.clean_prediction;
      const pp = pertRes.perturbed_prediction;
      const cd = cp && cp.selective_decision;
      const pd = pp && pp.selective_decision;
      const labels = ["NORM", "MI", "STTC", "CD", "HYP"];
      const probRows = cp
        ? labels
            .map((l) => {
              const a = Number(cp.calibrated_probabilities[l] ?? 0);
              const b = Number(pp.calibrated_probabilities[l] ?? 0);
              const d = b - a;
              return `<tr>
                <td><strong>${l}</strong></td>
                <td class="num">${fmt(a, 3)}</td>
                <td class="num">${fmt(b, 3)}</td>
                <td class="num ${Math.abs(d) >= 0.05 ? "text-err" : "muted"}">${d >= 0 ? "+" : ""}${fmt(d, 3)}</td>
              </tr>`;
            })
            .join("")
        : "";

      $("robProbeDetails").innerHTML = `
        <dl class="dl-compact" style="margin-bottom:10px;">
          <dt>Condition</dt><dd>${esc(label)} (${esc(spec.severity || "custom")})</dd>
          <dt>Probabilities</dt><dd>${cp ? esc(cp.calibration_method || "Raw") : "Unavailable"}</dd>
          <dt>Empirical SNR</dt><dd class="tnum">${pertRes.empirical_snr_db != null ? `${fmt(pertRes.empirical_snr_db, 1)} dB` : "—"}</dd>
          <dt>SQI basSQI</dt><dd class="tnum">${fmt(pertRes.clean_sqi.mean_basSQI, 3)} → ${fmt(pertRes.perturbed_sqi.mean_basSQI, 3)} (flatlines: ${pertRes.perturbed_sqi.n_flatline_leads})</dd>
          <dt>Selective action</dt><dd>${cd && pd ? `<span class="${cd.action === "ACCEPT" ? "text-ok" : "text-err"}">${cd.action === "ACCEPT" ? "Accepted" : "Abstained"}</span> → <strong class="${pd.action === "ACCEPT" ? "text-ok" : "text-err"}">${pd.action === "ACCEPT" ? "Accepted" : "Abstained"}</strong>` : "—"}</dd>
        </dl>
        ${
          probRows
            ? `<table class="sci-table"><thead><tr><th>Class</th><th class="num">Clean P</th><th class="num">Perturbed P</th><th class="num">ΔP</th></tr></thead><tbody>${probRows}</tbody></table>`
            : ""
        }`;
    } catch (err) {
      if (request !== robustnessProbeRequest || run !== state.activeRunId) return;
      robPreviewViewer.setData({ signal: null });
      $("robProbeDetails").innerHTML = `<strong>Probe unavailable</strong><div class="muted">${esc(err.message)}</div>`;
    }
  }

  // ==================== 5. EXPLAINABILITY WORKSPACE ====================
  async function loadExplainabilityWorkspace() {
    const run = state.activeRunId;
    const request = workspaceRequest;
    const { xaiEcgId, xaiLabel, xaiMethod } = state;
    try {
      const qs = new URLSearchParams({ method: xaiMethod });
      if (xaiEcgId != null) qs.set("ecg_id", String(xaiEcgId));
      if (xaiLabel) qs.set("target_label", xaiLabel);
      const [xai, sigList] = await Promise.all([
        apiGet(`/api/runs/${encodeURIComponent(run)}/explainability?${qs.toString()}`),
        apiGet(`/api/runs/${encodeURIComponent(run)}/signals?split=test&limit=80`).catch(() => ({ records: [] })),
      ]);
      if (request !== workspaceRequest || run !== state.activeRunId || xaiEcgId !== state.xaiEcgId ||
          xaiLabel !== state.xaiLabel || xaiMethod !== state.xaiMethod) return;
      state.explainabilityData = xai;

      const records = sigList.records || [];
      const recSel = $("xaiRecordSelect");
      recSel.replaceChildren(...records.map((r) => new Option(`#${r.ecg_id} · ${(r.true_labels || []).join("+") || "—"}`, r.ecg_id)));
      recSel.disabled = !records.length;
      $("xaiOpenSignalBtn").disabled = !records.length;
      const canSelectAttribution = xai.model_name !== "xgboost" && Boolean(records.length || xai.selected_attribution);
      for (const button of document.querySelectorAll("#xaiLabelSeg button, #xaiMethodSeg button")) {
        button.disabled = !canSelectAttribution;
      }

      if (xai.selected_attribution) {
        const sa = xai.selected_attribution;
        state.xaiEcgId = sa.ecg_id;
        state.xaiLabel = sa.target_label;
        if (!records.some((r) => r.ecg_id === sa.ecg_id)) recSel.add(new Option(`#${sa.ecg_id}`, sa.ecg_id));
        recSel.value = String(sa.ecg_id);
        setSegValue("xaiLabelSeg", sa.target_label);
        const methodLabel = { ig: "IG", gx: "Grad × Input", occlusion: "Occlusion" }[sa.method];
        $("xaiCanvasTitle").textContent = `ECG #${sa.ecg_id} · Target ${sa.target_label} (${methodLabel})`;
        $("xaiSourceMeta").textContent = sa.source === "saved_artifact" ? "Saved artifact" : "Computed from checkpoint";
        $("xaiCaliperReadout").textContent = "Hover waveform for signed attribution and amplitude";
        xaiViewer.setData({
          signal: sa.signal,
          attribution: sa.attribution,
          leadNames: sa.metadata.lead_names,
          samplingRate: sa.metadata.sampling_rate_hz,
          unitLabel: sa.metadata.amplitude_unit === "mV" ? "mV" : "SD",
        });
        const renderImportance = () => {
          window.WorkbenchCharts.renderLeadImportanceChart($("xaiLeadImpChart"), sa.lead_importance, {
            highlightLead: state.xaiHighlightLead,
            onSelectLead: (idx) => {
              state.xaiHighlightLead = state.xaiHighlightLead === idx ? null : idx;
              xaiViewer.setView({ highlightLead: state.xaiHighlightLead });
              renderImportance();
            },
          });
        };
        renderImportance();
      } else {
        const tabular = xai.model_name === "xgboost";
        state.xaiEcgId = records.some((r) => r.ecg_id === xaiEcgId) ? xaiEcgId : null;
        if (state.xaiEcgId == null && records.length) recSel.insertBefore(new Option("Select a record", ""), recSel.firstChild);
        recSel.value = state.xaiEcgId == null ? "" : String(state.xaiEcgId);
        $("xaiOpenSignalBtn").disabled = state.xaiEcgId == null;
        $("xaiCanvasTitle").textContent = "Waveform attribution unavailable";
        $("xaiSourceMeta").textContent = tabular ? "XGBoost uses engineered lead features" : "";
        $("xaiCaliperReadout").textContent = "";
        xaiViewer.setData({ signal: null });
        let explanation = "No test records or saved attributions for this run.";
        if (tabular) explanation = "Waveform gradients require a neural model.";
        else if (records.length) explanation = "Choose a test record to compute attribution from an available checkpoint.";
        $("xaiLeadImpChart").innerHTML = `<div class="state-empty">${esc(xai.attribution_unavailable_reason || explanation)}</div>`;
      }

      renderXaiSanity(xai.sanity_checks);
      renderXaiBottomDock(xai);
    } catch (err) {
      if (request === workspaceRequest && run === state.activeRunId) return workspaceLoadError("Unable to load explainability analysis", err.message);
    }
  }

  function renderXaiSanity(sanity) {
    const checks = (sanity && sanity.checks) || [];
    if (!checks.length) {
      $("xaiSanityMeta").textContent = "Not recorded";
      $("xaiSanityBody").innerHTML = `<div class="state-empty"><strong>No sanity checks recorded</strong><span>Run explainability on a PyTorch model to execute randomization and deletion checks.</span></div>`;
      return;
    }
    const c = checks[0];
    const comp = c.completeness || {};
    const rand = c.parameter_randomization || {};
    const faith = c.faithfulness || {};
    const passIcon = rand.passes_sensitivity_check ? '<span class="text-ok">✓ sensitive</span>' : '<span class="text-warn">! high residual corr</span>';
    $("xaiSanityMeta").textContent = `ECG #${c.ecg_id} · ${c.target_label}`;
    $("xaiSanityBody").innerHTML = `
      <dl class="dl-compact" style="margin-bottom:8px;">
        <dt>IG completeness</dt><dd class="tnum">abs ${fmt(comp.abs_error, 4)} · rel ${fmt(comp.rel_error, 4)}</dd>
        <dt>Head rand. ρ / cos</dt><dd class="tnum">${fmt(rand.head_randomized_spearman, 3)} / ${fmt(rand.head_randomized_cosine, 3)}</dd>
        <dt>Full rand. ρ / cos</dt><dd class="tnum">${fmt(rand.full_randomized_spearman, 3)} / ${fmt(rand.full_randomized_cosine, 3)} · ${passIcon}</dd>
        <dt>Threshold</dt><dd class="muted">|ρ|, |cos| &lt; 0.50 on full randomization</dd>
        <dt>Deletion margin</dt><dd class="tnum">${faith.faithfulness_margin >= 0 ? "+" : ""}${fmt(faith.faithfulness_margin, 3)} (rand − top-k AUC)</dd>
      </dl>
      <div id="xaiFaithChartHost" style="height:168px;background:#fff;border:1px solid var(--hairline-soft);"></div>`;
    window.WorkbenchCharts.renderFaithfulnessChart($("xaiFaithChartHost"), faith);
  }

  function renderXaiBottomDock(xai) {
    // 1. Misclassified Failures
    const fa = xai.failure_analysis;
    const fns = (fa && fa.high_confidence_false_negatives) || [];
    const fps = (fa && fa.high_confidence_false_positives) || [];
    const allErrs = [...fns, ...fps];
    $("xaiFailTabBtn").textContent = `Misclassified failures (${allErrs.length})`;

    if (!fa) {
      $("xaiFailuresTable").innerHTML = `<div class="state-empty">Failure analysis artifact unavailable.</div>`;
    } else if (!allErrs.length) {
      $("xaiFailuresTable").innerHTML = `<div class="state-empty"><strong>Zero misclassified records</strong><span>Exact-match accuracy ${fmt(fa.exact_match_accuracy, 3)} across ${fa.n_records} test records.</span></div>`;
    } else {
      $("xaiFailuresTable").innerHTML = `
        <table class="sci-table">
          <thead><tr><th>Record</th><th>Error type</th><th>Class</th><th class="num">Pred P</th><th>Ground truth</th><th class="num">Harm</th><th>Actions</th></tr></thead>
          <tbody>${allErrs
            .map((e) => {
              const isFn = e.error_type === "false_negative";
              const typeHtml = isFn
                ? `<span class="text-err">− False negative</span>`
                : `<span class="text-warn">+ False positive</span>`;
              return `<tr class="clickable" data-fail-ecg="${e.ecg_id}" data-fail-lbl="${esc(e.label)}">
                <td class="tnum"><strong>#${e.ecg_id}</strong></td>
                <td>${typeHtml}</td>
                <td><strong>${esc(e.label)}</strong></td>
                <td class="num">${fmt(e.predicted_prob, 3)}</td>
                <td class="secondary">${esc((e.true_labels || []).join(", ") || "—")}</td>
                <td class="num">${fmt(e.realized_harm, 2)}</td>
                <td>
                  <button class="link-btn" data-act="xai" data-ecg="${e.ecg_id}" data-lbl="${esc(e.label)}">Attribution</button>
                  <span class="muted"> · </span>
                  <button class="link-btn" data-act="sig" data-ecg="${e.ecg_id}">Waveform</button>
                </td>
              </tr>`;
            })
            .join("")}</tbody>
        </table>`;
      $("xaiFailuresTable").querySelectorAll("button[data-act]").forEach((btn) => {
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          const eid = Number(btn.dataset.ecg);
          if (btn.dataset.act === "xai") {
            state.xaiEcgId = eid;
            state.xaiLabel = btn.dataset.lbl;
            loadExplainabilityWorkspace();
          } else {
            openRecordInSignals(eid);
          }
        });
      });
    }

    // 2. Cross-model disagreement
    const dis = xai.model_disagreement;
    const cases = (dis && dis.top_disagreement_cases) || [];
    $("xaiDisagreeTabBtn").textContent = `Cross-model disagreement (${cases.length})`;
    if (!cases.length) {
      $("xaiDisagreementTable").innerHTML = `<div class="state-empty">Run a multi-model suite to surface cross-model disagreements.</div>`;
    } else {
      const models = dis.models || [];
      $("xaiDisagreementTable").innerHTML = `
        <table class="sci-table">
          <thead><tr><th>Record</th><th>Disputed</th><th class="num">Prob σ</th><th>Truth</th>${models.map((m) => `<th class="num">${esc(modelDisplayName(m))}</th>`).join("")}<th>Inspect</th></tr></thead>
          <tbody>${cases
            .slice(0, 15)
            .map(
              (c) => `<tr>
                <td class="tnum"><strong>#${c.ecg_id}</strong></td>
                <td><strong>${esc(c.most_disputed_label)}</strong></td>
                <td class="num text-warn">${fmt(c.max_prob_std, 3)}</td>
                <td class="secondary">${esc((c.true_labels || []).join(", ") || "—")}</td>
                ${models.map((m) => `<td class="num">${fmt(c.model_probabilities[m]?.[c.most_disputed_label], 2)}</td>`).join("")}
                <td>
                  <button class="link-btn" data-dis-xai="${c.ecg_id}" data-dis-lbl="${esc(c.most_disputed_label)}">Attr</button>
                  <span class="muted"> · </span>
                  <button class="link-btn" data-dis-sig="${c.ecg_id}">Wave</button>
                </td>
              </tr>`
            )
            .join("")}</tbody>
        </table>`;
      $("xaiDisagreementTable").querySelectorAll("button[data-dis-xai]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.xaiEcgId = Number(btn.dataset.disXai);
          state.xaiLabel = btn.dataset.disLbl;
          loadExplainabilityWorkspace();
        });
      });
      $("xaiDisagreementTable").querySelectorAll("button[data-dis-sig]").forEach((btn) => {
        btn.addEventListener("click", () => openRecordInSignals(Number(btn.dataset.disSig)));
      });
    }

    // 3. Stability telemetry
    const stab = xai.stability_rows || [];
    $("xaiStabilityTable").innerHTML = stab.length
      ? `<table class="sci-table">
          <thead><tr><th>Record</th><th>Class</th><th class="num">IG–G×I cos</th><th class="num">IG–Noise cos</th><th class="num">IG–Dropout cos</th><th class="num">Top-k overlap</th><th class="num">Lead entropy</th></tr></thead>
          <tbody>${stab
            .map(
              (s) => `<tr>
                <td class="tnum">#${s.ecg_id}</td>
                <td><strong>${esc(s.target_label)}</strong></td>
                <td class="num">${fmt(s.ig_gx_cosine, 3)}</td>
                <td class="num">${fmt(s.ig_noise_cosine, 3)}</td>
                <td class="num">${fmt(s.ig_dropout_cosine, 3)}</td>
                <td class="num">${fmt(s.ig_noise_topk_overlap, 3)}</td>
                <td class="num">${fmt(s.lead_entropy, 3)}</td>
              </tr>`
            )
            .join("")}</tbody>
        </table>`
      : `<div class="state-empty">No stability telemetry recorded.</div>`;
  }

  function openRecordInSignals(ecgId) {
    state.sigSplit = "test";
    state.activeEcgId = Number(ecgId);
    setSegValue("sigSplitSeg", "test");
    switchWorkspace("signals");
  }

  // ==================== EVENT WIRING & KEYBOARD SHORTCUTS ====================
  document.querySelectorAll(".ws-nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchWorkspace(btn.dataset.tab));
  });
  bindSubtabs(document);

  $("globalRunSelect").addEventListener("change", () => {
    selectRun($("globalRunSelect").value);
    loadActiveWorkspace();
  });

  $("globalErrorDismiss").addEventListener("click", () => setError(null));
  $("refPopoverClose").addEventListener("click", closePopover);
  $("refPopover").addEventListener("click", (ev) => {
    if (ev.target === $("refPopover")) closePopover();
  });

  // Contextual Scope & Reference popovers
  $("scopeInfoBtn").addEventListener("click", () => {
    const run = state.runs.find((r) => r.run_id === state.activeRunId);
    const tier = run ? tierMeta(run.execution_tier).label : "None";
    openPopover(
      "Methodological Scope & Limitations",
      `<h4>Research instrument — not a medical device</h4>
      <p>This software is intended exclusively for reproducible machine learning reliability research on 12-lead ECG benchmarks. It has not undergone clinical validation or regulatory review and must not be used for diagnosis, triage, or clinical care.</p>
      <h4>Execution tiers</h4>
      <ul>
        <li><strong>Synthetic smoke</strong> (currently: ${esc(tier)}): CPU engineering fixtures with synthetic PQRST waveforms; metrics verify pipeline correctness only.</li>
        <li><strong>Bounded real</strong>: Subset sanity gate on real PTB-XL v1.0.3 waveforms.</li>
        <li><strong>Full campaign</strong>: Complete official PTB-XL patient-disjoint folds (1–8 train, 9 val, 10 test).</li>
      </ul>
      <h4>Key safeguards</h4>
      <ul>
        <li>Calibration, thresholds, and model selection use validation fold 9 only.</li>
        <li>Per-lead z-normalization converts amplitudes to dimensionless SD units; disable normalization for raw physical mV voltage studies.</li>
        <li>Attributions quantify local model sensitivity, not physiological causality.</li>
      </ul>`
    );
  });

  $("robNstdbBtn").addEventListener("click", () => {
    const ref = (state.robustnessData && state.robustnessData.nstdb_reference) || {};
    const nr = ref.noise_records || {};
    openPopover(
      "MIT-BIH Noise Stress Test Database Reference",
      `<h4>${esc(ref.database || "MIT-BIH NSTDB")}</h4>
      <p>Native sampling rate: <strong>${esc(ref.native_sampling_rate_hz || 360)} Hz</strong> · Canonical SNR levels: <strong>${esc((ref.canonical_snr_db_levels || []).join(", "))} dB</strong></p>
      <p><a href="${esc(ref.physionet_url || "https://physionet.org/content/nstdb/1.0.0/")}" target="_blank" rel="noreferrer">${esc(ref.physionet_url || "https://physionet.org/content/nstdb/1.0.0/")}</a></p>
      <h4>Reference noise records</h4>
      <ul>
        <li><strong>bw</strong>: ${esc(nr.bw || "Baseline wander")}</li>
        <li><strong>ma</strong>: ${esc(nr.ma || "Muscle artifact (EMG)")}</li>
        <li><strong>em</strong>: ${esc(nr.em || "Electrode motion artifact")}</li>
      </ul>
      <h4>Scientific caveat</h4>
      <p>${esc(ref.scientific_caveat || "Synthetic perturbations evaluate model sensitivity and do not establish clinical robustness.")}</p>`
    );
  });

  $("xaiMethodsRefBtn").addEventListener("click", () => {
    const guides = (state.explainabilityData && state.explainabilityData.method_guides) || [];
    openPopover(
      "Attribution Methods & Scientific Limitations",
      guides
        .map((g) => `<h4>${esc(g.method)}</h4><p>${esc(g.measures)}</p><p class="muted"><strong>Limitation:</strong> ${esc(g.limitations)}</p>`)
        .join("") || "<p>Attribution reference unavailable.</p>"
    );
  });

  // Signals controls
  bindSegGroup("sigSplitSeg", (val) => {
    state.sigSplit = val;
    state.activeEcgId = null;
    loadSignalWorkspace();
  });
  $("sigRecordSelect").addEventListener("change", () => loadSignalRecord(Number($("sigRecordSelect").value)));
  $("sigPrevBtn").addEventListener("click", () => stepSignalRecord(-1));
  $("sigNextBtn").addEventListener("click", () => stepSignalRecord(1));
  bindSegGroup("sigLayoutSeg", (val) => {
    state.sigLayout = val;
    savePrefs({ sigLayout: val });
    $("sigLeadSelect").classList.toggle("hidden", val !== "single");
    signalViewer.setView({ layout: val });
  });
  $("sigLeadSelect").addEventListener("change", () => signalViewer.setView({ focusLead: Number($("sigLeadSelect").value) }));
  bindSegGroup("sigZoomSeg", (val) => {
    signalViewer.setView({ zoom: Number(val) });
    updateSignalScale();
    signalPanner.render();
  });
  $("sigGainSelect").addEventListener("change", () => signalViewer.setView({ gainMode: $("sigGainSelect").value }));
  bindSegGroup("sigThemeSeg", (val) => {
    state.sigTheme = val;
    savePrefs({ sigTheme: val });
    signalViewer.setView({ theme: val });
  });
  $("sigCaliperBtn").addEventListener("click", () => {
    const next = !signalViewer.caliperMode;
    $("sigCaliperBtn").classList.toggle("ctrl-btn-primary", next);
    $("sigCaliperBtn").setAttribute("aria-pressed", String(next));
    $("sigCaliperReadout").textContent = next ? "Select two sample points on the waveform" : "Hover for sample readout · Wheel to zoom · Drag to pan";
    signalViewer.setView({ caliperMode: next });
  });
  $("sigInspectorToggle").addEventListener("click", () => {
    const collapsed = $("sigInspector").classList.toggle("collapsed");
    $("sigInspectorToggle").setAttribute("aria-pressed", String(!collapsed));
    signalViewer.render();
    signalPanner.render();
  });
  $("sigPerturbType").addEventListener("change", renderPerturbInputs);
  $("sigApplyPerturbBtn").addEventListener("click", applySignalPerturbation);
  $("sigResetPerturbBtn").addEventListener("click", () => {
    $("sigPerturbType").value = "none";
    renderPerturbInputs();
    applySignalPerturbation();
  });

  function stepSignalRecord(delta) {
    if (!state.signalsList.length) return;
    const idx = state.signalsList.findIndex((r) => r.ecg_id === state.activeEcgId);
    const nextIdx = (idx + delta + state.signalsList.length) % state.signalsList.length;
    const nextId = state.signalsList[nextIdx].ecg_id;
    $("sigRecordSelect").value = String(nextId);
    loadSignalRecord(nextId);
  }

  // Experiments controls
  $("expSearchInput").addEventListener("input", () => {
    state.expSearch = $("expSearchInput").value;
    renderExperimentsTable();
  });
  bindSegGroup("expTierFilter", (val) => {
    state.expTier = val;
    renderExperimentsTable();
  });
  bindSegGroup("expStatusFilter", (val) => {
    state.expStatus = val;
    renderExperimentsTable();
  });
  $("expClearCompareBtn").addEventListener("click", () => {
    state.expCompareIds.clear();
    loadExperimentWorkspace();
  });
  $("expRefreshBtn").addEventListener("click", () => {
    state.runDetailCache = {};
    loadRuns();
  });

  // Reliability controls
  bindSegGroup("relClassSeg", (val) => {
    state.relClass = val;
    renderReliabilityView();
  });
  $("relMethodSelect").addEventListener("change", () => {
    state.relMethod = $("relMethodSelect").value;
    loadReliabilityWorkspace();
  });
  bindSegGroup("relBinningSeg", (val) => {
    state.relBinning = val;
    loadReliabilityWorkspace();
  });

  // Robustness controls
  bindSegGroup("robMetricSeg", (val) => {
    state.robMetric = val;
    renderRobustnessView();
  });
  bindSegGroup("robSeveritySeg", (val) => {
    state.robSeverity = val;
    renderRobustnessView();
  });
  $("robProbeRecordSelect").addEventListener("change", () => {
    state.robProbeEcgId = Number($("robProbeRecordSelect").value);
    runRobustnessLiveProbe();
  });
  $("robOpenInSignalsBtn").addEventListener("click", () => {
    if (state.robProbeEcgId != null) openRecordInSignals(state.robProbeEcgId);
  });

  // Explainability controls
  $("xaiRecordSelect").addEventListener("change", () => {
    const value = $("xaiRecordSelect").value;
    state.xaiEcgId = value === "" ? null : Number(value);
    loadExplainabilityWorkspace();
  });
  bindSegGroup("xaiLabelSeg", (val) => {
    state.xaiLabel = val;
    loadExplainabilityWorkspace();
  });
  bindSegGroup("xaiMethodSeg", (val) => {
    state.xaiMethod = val;
    loadExplainabilityWorkspace();
  });
  $("xaiOpenSignalBtn").addEventListener("click", () => {
    if (state.xaiEcgId != null) openRecordInSignals(state.xaiEcgId);
  });

  // Global keyboard shortcuts (disabled when typing in inputs)
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      closePopover();
      return;
    }
    const tag = (ev.target && ev.target.tagName) || "";
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || ev.metaKey || ev.ctrlKey || ev.altKey ||
        workspaceHost.getAttribute("aria-busy") === "true" || workspaceHost.classList.contains("has-workspace-status")) return;
    const tabs = ["signals", "experiments", "reliability", "robustness", "explainability"];
    if (ev.key >= "1" && ev.key <= "5") {
      switchWorkspace(tabs[Number(ev.key) - 1]);
    } else if (state.activeTab === "signals") {
      if (ev.key === "ArrowLeft") stepSignalRecord(-1);
      if (ev.key === "ArrowRight") stepSignalRecord(1);
      if (ev.key === "+" || ev.key === "=") {
        signalViewer.setView({ zoom: signalViewer.zoom * 1.5 });
        setSegValue("sigZoomSeg", signalViewer.zoom);
        updateSignalScale();
        signalPanner.render();
      }
      if (ev.key === "-") {
        signalViewer.setView({ zoom: signalViewer.zoom / 1.5 });
        setSegValue("sigZoomSeg", signalViewer.zoom);
        updateSignalScale();
        signalPanner.render();
      }
    }
  });

  // Boot
  populateLeadPicker();
  renderPerturbInputs();
  setSegValue("sigThemeSeg", state.sigTheme);
  setSegValue("sigLayoutSeg", state.sigLayout);
  if (state.sigLayout === "single") $("sigLeadSelect").classList.remove("hidden");
  switchWorkspace(document.getElementById(`view-${state.activeTab}`) ? state.activeTab : "signals");
  loadRuns();
})();
