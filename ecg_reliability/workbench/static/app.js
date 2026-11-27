/* ECG Reliability Research Workbench Interactive Controller */
(function () {
  "use strict";

  const state = {
    runs: [],
    activeRunId: null,
    activeTab: "signals",
    signalsList: [],
    activeEcgId: null,
    signalData: null,
    perturbedData: null,
    reliabilityData: null,
    robustnessData: null,
    explainabilityData: null,
    comparisonsData: null,
  };

  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d));
  const esc = (s) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

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

  function showAlert(msg) {
    const el = $("globalAlert");
    if (!msg) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    el.classList.remove("hidden");
  }

  function tierBadgeInfo(tier) {
    if (tier === "full_campaign") return { cls: "badge-tier-full", text: "FULL PTB-XL CAMPAIGN" };
    if (tier === "bounded_real") return { cls: "badge-tier-real", text: "BOUNDED REAL PTB-XL" };
    if (tier === "synthetic_smoke") return { cls: "badge-tier-synthetic", text: "SYNTHETIC SMOKE DATA" };
    return { cls: "badge-neutral", text: String(tier || "UNKNOWN").toUpperCase() };
  }

  // Initialize ECG Canvas viewers
  const signalViewer = new window.ECGCanvasViewer($("signalCanvas"), {
    onCaliper: (c) => {
      let text = `${c.leadName} | t=${c.tSec.toFixed(2)}s (n=${c.sampleIdx}) | clean=${c.cleanVal.toFixed(3)} ${c.unitLabel}`;
      if (c.pertVal !== null) {
        const diff = c.pertVal - c.cleanVal;
        text += ` | pert=${c.pertVal.toFixed(3)} (Δ=${diff >= 0 ? "+" : ""}${diff.toFixed(3)})`;
      }
      $("sigCaliperReadout").textContent = text;
    },
    onViewChange: ({ zoom, startSec }) => {
      $("sigZoomRange").value = zoom.toFixed(1);
      $("sigZoomLabel").textContent = `${zoom.toFixed(1)}×`;
      $("sigPanRange").value = startSec.toFixed(2);
      $("sigPanLabel").textContent = `${startSec.toFixed(2)} s`;
    },
  });

  const xaiViewer = new window.ECGCanvasViewer($("xaiCanvas"));

  // Perturbation parameter presets
  const PERTURB_UI = {
    none: [],
    baseline_wander: [
      { key: "amplitude", label: "Drift Amplitude", type: "number", step: "0.05", value: 0.45 },
      { key: "frequency_hz", label: "Frequency (Hz)", type: "number", step: "0.05", value: 0.33 },
    ],
    powerline_interference: [
      { key: "amplitude", label: "AC Amplitude", type: "number", step: "0.05", value: 0.25 },
      { key: "frequency_hz", label: "Mains Freq (Hz)", type: "number", step: "10", value: 50 },
    ],
    muscle_noise: [
      { key: "snr_db", label: "Target SNR (dB, lower=noisier)", type: "number", step: "3", value: 12 },
    ],
    electrode_motion: [
      { key: "amplitude", label: "Impulse Amplitude", type: "number", step: "0.1", value: 0.85 },
      { key: "n_events", label: "Transient Events", type: "number", step: "1", value: 3 },
    ],
    gaussian_noise: [
      { key: "std", label: "Noise Std Dev", type: "number", step: "0.05", value: 0.25 },
    ],
    lead_dropout_single: [
      { key: "lead_index", label: "Lead Index (0=I .. 6=V1 .. 11=V6)", type: "number", step: "1", value: 6 },
    ],
    lead_dropout_group: [
      { key: "group", label: "Group (limb or precordial)", type: "text", value: "precordial" },
    ],
    amplitude_scaling: [
      { key: "scale", label: "Gain Scale Factor", type: "number", step: "0.05", value: 0.7 },
    ],
  };

  function renderPerturbInputs() {
    const type = $("sigPerturbType").value;
    const fields = PERTURB_UI[type] || [];
    $("sigPerturbParams").innerHTML = fields
      .map(
        (f) => `<label class="field"><span>${esc(f.label)}</span><input data-pkey="${esc(f.key)}" type="${f.type}" step="${f.step || ""}" value="${esc(f.value)}"/></label>`
      )
      .join("");
  }

  function populateLeadDropdown() {
    const names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
    $("sigLeadSelect").innerHTML = names.map((n, i) => `<option value="${i}" ${i === 1 ? "selected" : ""}>Lead ${n} (idx ${i})</option>`).join("");
  }

  async function loadRuns() {
    showAlert(null);
    try {
      const data = await apiGet("/api/runs");
      state.runs = data.runs || [];
      const select = $("globalRunSelect");
      if (!state.runs.length) {
        select.innerHTML = `<option value="">No recorded runs in outputs/runs/</option>`;
        showAlert("No experiment runs found in outputs/runs/. Run `ecg-reliability run-suite --tier synthetic_smoke` to populate real artifacts.");
        return;
      }
      select.innerHTML = state.runs
        .map((r) => {
          const tag = r.status === "completed" ? "OK" : String(r.status || "?").toUpperCase();
          return `<option value="${esc(r.run_id)}">[${tag}] ${esc(r.model_name)} · ${esc(r.run_id)}</option>`;
        })
        .join("");

      if (!state.activeRunId || !state.runs.some((r) => r.run_id === state.activeRunId)) {
        const preferred = state.runs.find((r) => r.status === "completed") || state.runs[0];
        state.activeRunId = preferred.run_id;
      }
      select.value = state.activeRunId;
      updateTierBadge();
      await loadActiveTab();
    } catch (err) {
      showAlert(`Failed to load runs: ${err.message}`);
    }
  }

  function updateTierBadge() {
    const run = state.runs.find((r) => r.run_id === state.activeRunId);
    const info = tierBadgeInfo(run ? run.execution_tier : null);
    const badge = $("tierBadge");
    badge.className = `badge ${info.cls}`;
    badge.textContent = info.text;
  }

  async function loadActiveTab() {
    if (!state.activeRunId) return;
    showAlert(null);
    if (state.activeTab === "signals") await loadSignalExplorer();
    if (state.activeTab === "experiments") await loadExperimentExplorer();
    if (state.activeTab === "reliability") await loadReliabilityAnalysis();
    if (state.activeTab === "robustness") await loadRobustnessLab();
    if (state.activeTab === "explainability") await loadExplainability();
  }

  // ---------- 1. SIGNAL EXPLORER ----------
  async function loadSignalExplorer() {
    const split = $("sigSplitSelect").value;
    try {
      const list = await apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/signals?split=${encodeURIComponent(split)}&limit=120`);
      state.signalsList = list.records || [];
      const sel = $("sigRecordSelect");
      sel.innerHTML = state.signalsList
        .map((rec) => {
          const lbls = (rec.true_labels || []).join("+") || "NONE";
          const pid = rec.patient_id !== null ? `P${rec.patient_id}` : "P—";
          return `<option value="${rec.ecg_id}">ECG #${rec.ecg_id} (${pid}) · [${lbls}]</option>`;
        })
        .join("");
      if (!state.signalsList.length) {
        $("sigMetaStrip").textContent = "No signals found in this split.";
        return;
      }
      if (!state.activeEcgId || !state.signalsList.some((r) => r.ecg_id === state.activeEcgId)) {
        state.activeEcgId = state.signalsList[0].ecg_id;
      }
      sel.value = String(state.activeEcgId);
      await loadSingleSignal(state.activeEcgId);
    } catch (err) {
      showAlert(`Signal explorer error: ${err.message}`);
    }
  }

  async function loadSingleSignal(ecgId) {
    state.activeEcgId = Number(ecgId);
    state.perturbedData = null;
    try {
      const detail = await apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/signal/${ecgId}`);
      state.signalData = detail;
      const m = detail.metadata;
      const unitStr = m.amplitude_unit === "mV" ? "mV (physical)" : "SD (per-lead z-normalized)";
      $("sigTitle").textContent = `ECG #${m.ecg_id} · ${m.num_leads}-Lead Waveform (${m.duration_seconds.toFixed(1)} s @ ${m.sampling_rate_hz} Hz)`;
      $("sigMetaStrip").textContent =
        `Split: ${m.split} | Fold: ${m.strat_fold ?? "—"} | Patient: ${m.patient_id ?? "—"} | True: [${(m.true_labels || []).join(", ") || "—"}] | Fingerprint: ${(m.preprocessing_fingerprint || "").slice(0, 12)}`;
      $("sigUnitNote").textContent = `Amplitude units: ${unitStr} · Paper grid: 25 mm/s (40 ms / 200 ms)`;

      signalViewer.setData({
        signal: detail.signal,
        perturbed: null,
        leadNames: m.lead_names,
        samplingRate: m.sampling_rate_hz,
        unitLabel: m.amplitude_unit === "mV" ? "mV" : "SD",
      });
      renderSignalPredictions(detail.predictions, null, m.true_labels || []);
      renderSignalSqi(detail.sqi, null);
    } catch (err) {
      showAlert(`Failed to load ECG #${ecgId}: ${err.message}`);
    }
  }

  async function applyLivePerturbation() {
    if (!state.activeRunId || !state.activeEcgId) return;
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
      renderSignalPredictions(state.signalData.predictions, null, state.signalData.metadata.true_labels || []);
      renderSignalSqi(state.signalData.sqi, null);
      return;
    }
    const params = {};
    $("sigPerturbParams")
      .querySelectorAll("input[data-pkey]")
      .forEach((inp) => {
        params[inp.dataset.pkey] = inp.type === "number" ? Number(inp.value) : inp.value;
      });
    try {
      showAlert(null);
      const res = await apiPost(`/api/runs/${encodeURIComponent(state.activeRunId)}/perturb`, {
        ecg_id: state.activeEcgId,
        perturbation: type,
        params,
      });
      state.perturbedData = res;
      signalViewer.setData({
        signal: state.signalData.signal,
        perturbed: res.perturbed_signal,
        leadNames: state.signalData.metadata.lead_names,
        samplingRate: state.signalData.metadata.sampling_rate_hz,
        unitLabel: state.signalData.metadata.amplitude_unit === "mV" ? "mV" : "SD",
      });
      renderSignalPredictions(res.clean_prediction || state.signalData.predictions, res.perturbed_prediction, state.signalData.metadata.true_labels || [], res.empirical_snr_db);
      renderSignalSqi(res.clean_sqi, res.perturbed_sqi);
    } catch (err) {
      showAlert(`Live perturbation failed: ${err.message}`);
    }
  }

  function renderSignalPredictions(cleanPred, pertPred, trueLabels, snrDb = null) {
    const el = $("sigPredictionPanel");
    if (!cleanPred) {
      el.innerHTML = `<p class="muted">No saved test predictions for this ECG (select a Test split record or apply a live perturbation probe).</p>`;
      return;
    }
    const labels = ["NORM", "MI", "STTC", "CD", "HYP"];
    const cProbs = cleanPred.calibrated_probabilities || cleanPred.raw_probabilities || {};
    const pProbs = pertPred ? pertPred.calibrated_probabilities || pertPred.raw_probabilities || {} : null;

    let decHtml = "";
    if (cleanPred.selective_decision) {
      const cd = cleanPred.selective_decision;
      const pd = pertPred && pertPred.selective_decision ? pertPred.selective_decision : null;
      decHtml = `<div style="margin-bottom:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
        <span class="badge ${cd.action === "ACCEPT" ? "badge-ok" : "badge-fail"}">Clean: ${cd.action} (u=${fmt(cd.uncertainty, 3)})</span>
        ${pd ? `<span class="badge ${pd.action === "ACCEPT" ? "badge-ok" : "badge-fail"}">Perturbed: ${pd.action} (u=${fmt(pd.uncertainty, 3)})</span>` : ""}
        ${snrDb !== null ? `<span class="badge badge-neutral">Empirical SNR: ${fmt(snrDb, 1)} dB</span>` : ""}
      </div>`;
    }

    const rows = labels
      .map((lbl) => {
        const isTrue = trueLabels.includes(lbl);
        const cv = Number(cProbs[lbl] ?? 0);
        const pv = pProbs ? Number(pProbs[lbl] ?? 0) : null;
        const delta = pv !== null ? pv - cv : null;
        return `<tr>
          <td><span class="badge ${isTrue ? "badge-label-pos" : "badge-label"}">${lbl}${isTrue ? " ✓" : ""}</span></td>
          <td class="num">${fmt(cv, 3)}</td>
          <td><div class="prob-bar-track"><div class="prob-bar-fill" style="width:${(cv * 100).toFixed(1)}%"></div></div></td>
          ${
            pv !== null
              ? `<td class="num">${fmt(pv, 3)}</td><td class="num ${Math.abs(delta) > 0.05 ? "neg-delta" : ""}">${delta >= 0 ? "+" : ""}${fmt(delta, 3)}</td>`
              : `<td class="num muted">—</td><td class="num muted">—</td>`
          }
        </tr>`;
      })
      .join("");

    el.innerHTML = `${decHtml}
      <table class="data-table">
        <thead><tr><th>Superclass</th><th class="num">Clean P</th><th style="width:110px">Profile</th><th class="num">Perturbed P</th><th class="num">ΔP</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function renderSignalSqi(cleanSqi, pertSqi) {
    if (!cleanSqi) {
      $("sigSqiPanel").innerHTML = `<p class="muted">SQI not available.</p>`;
      return;
    }
    $("sigSqiSummary").textContent = pertSqi
      ? `Clean basSQI=${fmt(cleanSqi.mean_basSQI, 2)} → Pert=${fmt(pertSqi.mean_basSQI, 2)} | Flatlines: ${pertSqi.n_flatline_leads}`
      : `Mean kSQI=${fmt(cleanSqi.mean_kSQI, 2)} · basSQI=${fmt(cleanSqi.mean_basSQI, 2)} · pSQI=${fmt(cleanSqi.mean_pSQI, 2)}`;

    const active = pertSqi || cleanSqi;
    const rows = active.per_lead
      .map(
        (r) => `<tr>
          <td class="mono"><strong>${esc(r.lead)}</strong></td>
          <td class="num">${fmt(r.kSQI, 2)}</td>
          <td class="num">${fmt(r.basSQI, 2)}</td>
          <td class="num">${fmt(r.pSQI, 2)}</td>
          <td class="num">${fmt(r.hf_noise_ratio, 2)}</td>
          <td><span class="badge ${r.quality_grade === "acceptable" ? "badge-ok" : r.quality_grade === "unacceptable" ? "badge-fail" : "badge-tier-synthetic"}">${esc(r.quality_grade)}</span></td>
        </tr>`
      )
      .join("");
    $("sigSqiPanel").innerHTML = `
      <table class="data-table">
        <thead><tr><th>Lead</th><th class="num">kSQI</th><th class="num">basSQI</th><th class="num">pSQI</th><th class="num">HF Ratio</th><th>Grade</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  // ---------- 2. EXPERIMENT EXPLORER ----------
  async function loadExperimentExplorer() {
    try {
      const [detail, comps] = await Promise.all([
        apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}`),
        apiGet("/api/comparisons"),
      ]);
      state.comparisonsData = comps;

      const runRows = (comps.runs || [])
        .map((r) => {
          const selected = r.run_id === state.activeRunId ? "selected-row" : "";
          const stCls = r.status === "completed" ? "badge-ok" : r.status === "failed" ? "badge-fail" : "badge-tier-synthetic";
          const tInfo = tierBadgeInfo(r.execution_tier);
          return `<tr class="selectable ${selected}" data-runid="${esc(r.run_id)}">
            <td class="mono">${esc(r.run_id)}</td>
            <td><span class="badge ${stCls}">${esc(r.status)}</span></td>
            <td><span class="badge ${tInfo.cls}">${esc(r.execution_tier)}</span></td>
            <td class="mono"><strong>${esc(r.model_name)}</strong></td>
            <td class="num">${fmt(r.macro_auroc, 4)}</td>
            <td class="num">${fmt(r.macro_auprc, 4)}</td>
            <td class="num">${fmt(r.cal_macro_ece, 4)}</td>
            <td class="num">${fmt(r.cal_macro_brier, 4)}</td>
            <td class="mono">${esc(r.selected_calibration_method || "—")}</td>
          </tr>`;
        })
        .join("");
      $("runsTableWrap").innerHTML = `
        <table class="data-table">
          <thead><tr><th>Run ID</th><th>Status</th><th>Tier</th><th>Model</th><th class="num">AUROC</th><th class="num">AUPRC</th><th class="num">Cal ECE</th><th class="num">Cal Brier</th><th>Calibrator</th></tr></thead>
          <tbody>${runRows}</tbody>
        </table>`;
      $("runsTableWrap").querySelectorAll("tr[data-runid]").forEach((tr) => {
        tr.addEventListener("click", () => {
          state.activeRunId = tr.dataset.runid;
          $("globalRunSelect").value = state.activeRunId;
          updateTierBadge();
          loadExperimentExplorer();
        });
      });

      const m = detail.manifest;
      const history = (m.metrics && m.metrics.training_history) || (detail.training_runtime_report && detail.training_runtime_report.epoch_history) || [];
      window.WorkbenchCharts.renderTrainingHistoryChart($("expHistoryChart"), history);

      const hw = m.hardware_software_metadata || {};
      const cv = m.code_version || {};
      const ds = m.dataset_provenance || {};
      const ckpts = Object.entries(m.checkpoints || {})
        .map(([k, v]) => `${k}: ${(v.sha256 || "").slice(0, 12)}`)
        .join(" | ");
      let failHtml = "";
      if (m.failure_info) {
        failHtml = `<div class="alert-banner" style="margin-top:10px;"><strong>Failure in stage "${esc(m.failure_info.stage)}":</strong> ${esc(m.failure_info.exception_type)}: ${esc(m.failure_info.message)}</div>`;
      }
      $("expProvenancePanel").innerHTML = `
        <div class="kv-grid">
          <div class="kv-key">Run ID</div><div class="kv-val">${esc(m.run_id)}</div>
          <div class="kv-key">Execution Tier</div><div class="kv-val">${esc(m.execution_tier)} (claim_eligible=${String(m.scientific_claim_eligible)})</div>
          <div class="kv-key">Git Commit / Branch</div><div class="kv-val">${esc((cv.git_commit || "n/a").slice(0, 12))} (${esc(cv.git_branch || "?")}, dirty=${String(cv.git_dirty)})</div>
          <div class="kv-key">Preprocessing SHA-256</div><div class="kv-val">${esc(m.preprocessing_fingerprint || "—")}</div>
          <div class="kv-key">Dataset / Splits</div><div class="kv-val">${esc(ds.dataset_identity || ds.manifest_path || "—")} · ${esc(JSON.stringify(ds.split_counts || {}))}</div>
          <div class="kv-key">Patient Leakage Check</div><div class="kv-val">${esc(JSON.stringify(ds.patient_overlap_report || {}))}</div>
          <div class="kv-key">Seed / Deterministic</div><div class="kv-val">seed=${esc((m.seeds || {}).project_seed)} · deterministic=${String((m.seeds || {}).deterministic_torch)}</div>
          <div class="kv-key">Checkpoints (SHA-256)</div><div class="kv-val">${esc(ckpts || "none")}</div>
          <div class="kv-key">Environment</div><div class="kv-val">Python ${esc(hw.python_version)} · PyTorch ${esc((hw.packages || {}).torch)} · ${esc(hw.platform)}</div>
        </div>${failHtml}`;

      const paired = (comps.paired_comparisons && comps.paired_comparisons.paired_comparisons) || [];
      if (!paired.length) {
        $("expPairedPanel").innerHTML = `<p class="muted">Run a multi-model benchmark comparison to generate paired patient-cluster bootstrap statistics.</p>`;
      } else {
        const prows = paired
          .map(
            (p) => `<tr>
              <td class="mono">${esc(p.model_a)} vs ${esc(p.model_b)}</td>
              <td class="mono">${esc(p.metric)}</td>
              <td class="num">${p.delta_point_estimate >= 0 ? "+" : ""}${fmt(p.delta_point_estimate, 4)}</td>
              <td class="num">[${fmt(p.delta_ci_low, 4)}, ${fmt(p.delta_ci_high, 4)}]</td>
              <td class="num">${fmt(p.p_value_two_sided, 3)}</td>
              <td class="mono">${esc(p.resampling_unit)} (n=${p.effective_sample_size})</td>
            </tr>`
          )
          .join("");
        $("expPairedPanel").innerHTML = `<table class="data-table">
          <thead><tr><th>Comparison (A − B)</th><th>Metric</th><th class="num">Δ Point</th><th class="num">95% CI</th><th class="num">p (2-sided)</th><th>Unit</th></tr></thead>
          <tbody>${prows}</tbody></table>`;
      }

      const files = detail.files || [];
      const fSel = $("artifactFileSelect");
      fSel.innerHTML = files.map((f) => `<option value="${esc(f.path)}">${esc(f.path)} (${f.size_bytes} B)</option>`).join("");
      if (files.length) await loadArtifactFile(files[0].path);
    } catch (err) {
      showAlert(`Experiment explorer error: ${err.message}`);
    }
  }

  async function loadArtifactFile(relPath) {
    try {
      const data = await apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/artifact?path=${encodeURIComponent(relPath)}`);
      $("artifactContentPre").textContent = data.content !== undefined ? data.content : JSON.stringify(data, null, 2);
    } catch (err) {
      $("artifactContentPre").textContent = `Error reading artifact: ${err.message}`;
    }
  }

  // ---------- 3. RELIABILITY ANALYSIS ----------
  async function loadReliabilityAnalysis() {
    try {
      const rel = await apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/reliability`);
      state.reliabilityData = rel;
      const rawM = rel.raw_test_metrics || {};
      const calT = rel.calibration_test_reports || {};
      const best = calT.selected_method_from_val || (rel.calibration_val_reports && rel.calibration_val_reports.best_method) || "raw";
      const bestReport = calT[best] || calT.raw || {};
      const rawCal = calT.raw || {};

      $("relKpiStrip").innerHTML = [
        { k: "Macro AUROC (Test)", v: fmt(rawM.macro_auroc, 4), s: `Abn AUROC ${fmt(rawM.abnormal_macro_auroc, 4)}` },
        { k: "Macro AUPRC (Test)", v: fmt(rawM.macro_auprc, 4), s: `Macro F1 ${fmt(rawM.macro_f1, 4)}` },
        { k: "Macro ECE (Raw → Cal)", v: `${fmt(rawCal.macro_ece, 4)} → ${fmt(bestReport.macro_ece, 4)}`, s: `Selected on Val: ${best}` },
        { k: "Adaptive ECE (Cal)", v: fmt(bestReport.macro_adaptive_ece, 4), s: `Brier: ${fmt(bestReport.macro_brier, 4)}` },
        {
          k: "Selective Monotonic",
          v: rel.selective_results ? String(rel.selective_results.coverage_monotonic) : "—",
          s: rel.selective_results && rel.selective_results.norm_conflict_report ? `NORM conflicts fixed: ${rel.selective_results.norm_conflict_report.n_changed_norm_decisions}` : "",
        },
      ]
        .map((c) => `<div class="kpi-card"><div class="kpi-label">${esc(c.k)}</div><div class="kpi-value">${esc(c.v)}</div><div class="kpi-sub">${esc(c.s)}</div></div>`)
        .join("");

      renderReliabilityClassCharts();

      const curves = (rel.selective_results && rel.selective_results.curves) || [];
      window.WorkbenchCharts.renderRiskCoverageChart($("relRiskCoverageChart"), curves);

      // Per-class table
      const labels = ["NORM", "MI", "STTC", "CD", "HYP"];
      const perClass = rawM.per_class || {};
      const eceList = bestReport.ece_per_class || [];
      const aeceList = bestReport.adaptive_ece_per_class || [];
      const brierList = bestReport.brier_per_class || [];
      $("relClassTable").innerHTML = `<table class="data-table">
        <thead><tr><th>Superclass</th><th class="num">AUROC</th><th class="num">AUPRC</th><th class="num">ECE</th><th class="num">Adaptive ECE</th><th class="num">Brier</th></tr></thead>
        <tbody>${labels
          .map(
            (lbl, idx) => `<tr>
              <td class="mono"><strong>${lbl}</strong></td>
              <td class="num">${fmt(perClass[lbl] && perClass[lbl].auroc, 4)}</td>
              <td class="num">${fmt(perClass[lbl] && perClass[lbl].auprc, 4)}</td>
              <td class="num">${fmt(eceList[idx], 4)}</td>
              <td class="num">${fmt(aeceList[idx], 4)}</td>
              <td class="num">${fmt(brierList[idx], 4)}</td>
            </tr>`
          )
          .join("")}</tbody></table>`;

      // Calibration methods comparison
      const valReports = rel.calibration_val_reports || {};
      const methods = Object.keys(calT).filter((k) => k !== "selected_method_from_val");
      $("relCalMethodsTable").innerHTML = methods.length
        ? `<table class="data-table">
            <thead><tr><th>Method</th><th class="num">Val ECE</th><th class="num">Test ECE</th><th class="num">Test Adaptive ECE</th><th class="num">Test Brier</th></tr></thead>
            <tbody>${methods
              .map(
                (m) => `<tr class="${m === best ? "selected-row" : ""}">
                  <td class="mono">${esc(m)}${m === best ? " ★" : ""}</td>
                  <td class="num">${fmt(valReports[m] && valReports[m].macro_ece, 4)}</td>
                  <td class="num">${fmt(calT[m] && calT[m].macro_ece, 4)}</td>
                  <td class="num">${fmt(calT[m] && calT[m].macro_adaptive_ece, 4)}</td>
                  <td class="num">${fmt(calT[m] && calT[m].macro_brier, 4)}</td>
                </tr>`
              )
              .join("")}</tbody></table>`
        : `<p class="muted">Calibration reports unavailable.</p>`;

      // Bootstrap CI table
      const boot = rel.bootstrap_ci || {};
      const bootKeys = Object.keys(boot);
      $("relBootstrapTable").innerHTML = bootKeys.length
        ? `<table class="data-table">
            <thead><tr><th>Metric</th><th class="num">Point</th><th class="num">95% CI</th><th class="num">Valid / Req</th><th class="num">N_eff</th><th>Definition</th></tr></thead>
            <tbody>${bootKeys
              .map((k) => {
                const b = boot[k];
                return `<tr>
                  <td class="mono"><strong>${esc(k)}</strong></td>
                  <td class="num">${fmt(b.point_estimate, 4)}</td>
                  <td class="num">[${fmt(b.ci_low, 4)}, ${fmt(b.ci_high, 4)}]</td>
                  <td class="num">${b.n_bootstrap_valid}/${b.n_bootstrap_requested} (inv=${b.n_bootstrap_invalid})</td>
                  <td class="num">${b.effective_sample_size}</td>
                  <td class="mono">${esc(b.ci_definition)}</td>
                </tr>`;
              })
              .join("")}</tbody></table>`
        : `<p class="muted">No bootstrap confidence intervals recorded for this run.</p>`;
    } catch (err) {
      showAlert(`Reliability analysis error: ${err.message}`);
    }
  }

  function renderReliabilityClassCharts() {
    const rel = state.reliabilityData;
    if (!rel) return;
    const cls = $("relClassSelect").value;
    const perClassBins = rel.reliability_bins && rel.reliability_bins.per_class && rel.reliability_bins.per_class[cls];
    window.WorkbenchCharts.renderReliabilityDiagram($("relDiagramChart"), perClassBins && perClassBins.raw, perClassBins && perClassBins.calibrated, cls);
    const dist = rel.confidence_distributions && rel.confidence_distributions[cls];
    window.WorkbenchCharts.renderConfidenceHistogram($("relConfidenceChart"), dist, cls);
  }

  // ---------- 4. ROBUSTNESS LAB ----------
  async function loadRobustnessLab() {
    try {
      const rob = await apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/robustness`);
      state.robustnessData = rob;
      const ref = rob.nstdb_reference || {};
      $("robCaveatBanner").innerHTML = `<strong>Scientific Scope Notice:</strong> ${esc(ref.scientific_caveat || "Synthetic perturbation stress tests measure sensitivity and do not represent clinical robustness claims.")}`;

      if (!rob.available || !rob.results) {
        $("robBarChart").innerHTML = `<p class="muted" style="padding:16px">Robustness evaluation not available for this run.</p>`;
        $("robDecisionTable").innerHTML = `<p class="muted">Unavailable.</p>`;
        $("robFullTable").innerHTML = `<p class="muted">Unavailable.</p>`;
      } else {
        const deltas = rob.results.delta_summary || [];
        const metricKey = $("robMetricSelect").value;
        window.WorkbenchCharts.renderRobustnessBars($("robBarChart"), deltas, metricKey);

        const decDeltas = (rob.results.decision_layer && rob.results.decision_layer.decision_delta_summary) || [];
        $("robDecisionTable").innerHTML = decDeltas.length
          ? `<table class="data-table">
              <thead><tr><th>Perturbation</th><th class="num">Δ Exp Harm</th><th class="num">Δ Abstain</th><th class="num">Δ Acc Risk</th><th class="num">Δ FN Harm</th></tr></thead>
              <tbody>${decDeltas
                .map(
                  (d) => `<tr>
                    <td class="mono">${esc(d.perturbation)}</td>
                    <td class="num ${d.delta_expected_harm_with_abstain > 0 ? "neg-delta" : "pos-delta"}">${d.delta_expected_harm_with_abstain >= 0 ? "+" : ""}${fmt(d.delta_expected_harm_with_abstain, 4)}</td>
                    <td class="num">${d.delta_abstain_rate >= 0 ? "+" : ""}${fmt(d.delta_abstain_rate, 3)}</td>
                    <td class="num">${d.delta_accepted_risk >= 0 ? "+" : ""}${fmt(d.delta_accepted_risk, 4)}</td>
                    <td class="num">${d.delta_weighted_fn_harm >= 0 ? "+" : ""}${fmt(d.delta_weighted_fn_harm, 4)}</td>
                  </tr>`
                )
                .join("")}</tbody></table>`
          : `<p class="muted">Decision-layer robustness unavailable for this run.</p>`;

        const specs = rob.results.perturbation_specs || {};
        $("robFullTable").innerHTML = `<table class="data-table">
          <thead><tr><th>Perturbation</th><th>Severity</th><th>Units</th><th class="num">Δ AUROC</th><th class="num">Δ AUPRC</th><th class="num">Δ ECE</th><th class="num">Δ Brier</th></tr></thead>
          <tbody>${deltas
            .map((d) => {
              const sp = specs[d.perturbation] || {};
              return `<tr>
                <td class="mono"><strong>${esc(d.perturbation)}</strong></td>
                <td><span class="badge ${sp.severity === "severe" ? "badge-fail" : sp.severity === "moderate" ? "badge-tier-synthetic" : "badge-label"}">${esc(sp.severity || "custom")}</span></td>
                <td class="mono">${esc(sp.units || "signal_units")}</td>
                <td class="num ${d.delta_macro_auroc < 0 ? "neg-delta" : ""}">${d.delta_macro_auroc >= 0 ? "+" : ""}${fmt(d.delta_macro_auroc, 4)}</td>
                <td class="num ${d.delta_macro_auprc < 0 ? "neg-delta" : ""}">${d.delta_macro_auprc >= 0 ? "+" : ""}${fmt(d.delta_macro_auprc, 4)}</td>
                <td class="num ${d.delta_macro_ece > 0 ? "neg-delta" : ""}">${d.delta_macro_ece >= 0 ? "+" : ""}${fmt(d.delta_macro_ece, 4)}</td>
                <td class="num ${d.delta_macro_brier > 0 ? "neg-delta" : ""}">${d.delta_macro_brier >= 0 ? "+" : ""}${fmt(d.delta_macro_brier, 4)}</td>
              </tr>`;
            })
            .join("")}</tbody></table>`;
      }

      const nr = ref.noise_records || {};
      $("robNstdbPanel").innerHTML = `
        <div class="kv-grid">
          <div class="kv-key">Reference Database</div><div class="kv-val">${esc(ref.database)} (${esc(ref.native_sampling_rate_hz)} Hz native)</div>
          <div class="kv-key">PhysioNet URL</div><div class="kv-val"><a href="${esc(ref.physionet_url)}" target="_blank" rel="noreferrer">${esc(ref.physionet_url)}</a></div>
          <div class="kv-key">Canonical SNR Grid</div><div class="kv-val">${esc((ref.canonical_snr_db_levels || []).join(", "))} dB</div>
          <div class="kv-key">bw / ma / em</div><div class="kv-val">${esc(nr.bw || "")} | ${esc(nr.ma || "")} | ${esc(nr.em || "")}</div>
        </div>`;
    } catch (err) {
      showAlert(`Robustness lab error: ${err.message}`);
    }
  }

  // ---------- 5. EXPLAINABILITY & FAILURES ----------
  async function loadExplainability(overrideParams = null) {
    try {
      const qs = new URLSearchParams();
      if (overrideParams && overrideParams.ecg_id) qs.set("ecg_id", String(overrideParams.ecg_id));
      if (overrideParams && overrideParams.target_label) qs.set("target_label", overrideParams.target_label);
      if (overrideParams && overrideParams.method) qs.set("method", overrideParams.method);
      const xai = await apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/explainability?${qs.toString()}`);
      state.explainabilityData = xai;

      // Populate record picker from test split or saved attributions
      if (!state.signalsList.length) {
        const list = await apiGet(`/api/runs/${encodeURIComponent(state.activeRunId)}/signals?split=test&limit=80`);
        state.signalsList = list.records || [];
      }
      const recSel = $("xaiRecordSelect");
      recSel.innerHTML = state.signalsList
        .map((r) => `<option value="${r.ecg_id}">ECG #${r.ecg_id} [${(r.true_labels || []).join("+")}]</option>`)
        .join("");

      if (xai.selected_attribution) {
        const sa = xai.selected_attribution;
        recSel.value = String(sa.ecg_id);
        $("xaiLabelSelect").value = sa.target_label;
        $("xaiMethodSelect").value = sa.method;
        $("xaiSourceNote").textContent = `Source: ${sa.source} · Target: ${sa.target_label}`;
        xaiViewer.setData({
          signal: sa.signal,
          attribution: sa.attribution,
          leadNames: sa.metadata.lead_names,
          samplingRate: sa.metadata.sampling_rate_hz,
          unitLabel: sa.metadata.amplitude_unit === "mV" ? "mV" : "SD",
        });
        window.WorkbenchCharts.renderLeadImportanceChart($("xaiLeadImpChart"), sa.lead_importance);
      } else {
        $("xaiSourceNote").textContent = "Attributions unavailable for this model type (e.g. tabular XGBoost uses engineered summary features rather than raw waveform gradients).";
        window.WorkbenchCharts.renderLeadImportanceChart($("xaiLeadImpChart"), null);
      }

      // Sanity checks panel
      const checks = (xai.sanity_checks && xai.sanity_checks.checks) || [];
      if (!checks.length) {
        $("xaiSanityPanel").innerHTML = `<p class="muted">No attribution sanity checks recorded for this run.</p>`;
      } else {
        const first = checks[0];
        const comp = first.completeness || {};
        const rand = first.parameter_randomization || {};
        const faith = first.faithfulness || {};
        $("xaiSanityPanel").innerHTML = `
          <div class="kv-grid" style="margin-bottom:10px;">
            <div class="kv-key">Checked Sample</div><div class="kv-val">ECG #${esc(first.ecg_id)} (${esc(first.target_label)}, ${esc(first.method)})</div>
            <div class="kv-key">IG Completeness Error</div><div class="kv-val">abs=${fmt(comp.abs_error, 4)} · rel=${fmt(comp.rel_error, 4)}</div>
            <div class="kv-key">Adebayo Head Rand Corr</div><div class="kv-val">Spearman=${fmt(rand.head_randomized_spearman, 3)} · Cosine=${fmt(rand.head_randomized_cosine, 3)}</div>
            <div class="kv-key">Adebayo Full Rand Corr</div><div class="kv-val">Spearman=${fmt(rand.full_randomized_spearman, 3)} · Cosine=${fmt(rand.full_randomized_cosine, 3)} <span class="badge ${rand.passes_sensitivity_check ? "badge-ok" : "badge-fail"}">${rand.passes_sensitivity_check ? "PASS" : "WARN"}</span></div>
            <div class="kv-key">Faithfulness Margin</div><div class="kv-val">${fmt(faith.faithfulness_margin, 4)} (Random Del AUC − Top-k Del AUC)</div>
          </div>
          <div id="xaiFaithChartWrap" class="chart-box" style="min-height:190px;"></div>`;
        window.WorkbenchCharts.renderFaithfulnessChart($("xaiFaithChartWrap"), faith);
      }

      // Method guides
      $("xaiGuidesPanel").innerHTML = (xai.method_guides || [])
        .map(
          (g) => `<div class="guide-card">
            <h4>${esc(g.method)}</h4>
            <p><strong>Measures:</strong> ${esc(g.measures)}</p>
            <p class="muted"><strong>Limitations:</strong> ${esc(g.limitations)}</p>
          </div>`
        )
        .join("");

      // Failure analysis
      const fa = xai.failure_analysis;
      if (!fa) {
        $("xaiFailuresPanel").innerHTML = `<p class="muted">Failure analysis artifact not found.</p>`;
      } else {
        const errs = [...(fa.high_confidence_false_negatives || []).slice(0, 8), ...(fa.high_confidence_false_positives || []).slice(0, 8)];
        $("xaiFailuresPanel").innerHTML = `
          <p class="mono muted" style="margin:0 0 8px;">Records with any error: ${fa.n_records_with_any_error}/${fa.n_records} · Exact-match accuracy: ${fmt(fa.exact_match_accuracy, 3)} · Abstained: ${fa.n_abstained}</p>
          ${
            errs.length
              ? `<table class="data-table">
                  <thead><tr><th>ECG</th><th>Type</th><th>Label</th><th class="num">Pred P</th><th>True Labels</th><th class="num">Harm</th></tr></thead>
                  <tbody>${errs
                    .map(
                      (e) => `<tr>
                        <td class="mono">#${e.ecg_id}</td>
                        <td><span class="badge ${e.error_type === "false_negative" ? "badge-fail" : "badge-tier-synthetic"}">${esc(e.error_type)}</span></td>
                        <td class="mono"><strong>${esc(e.label)}</strong></td>
                        <td class="num">${fmt(e.predicted_prob, 3)}</td>
                        <td class="mono">${esc((e.true_labels || []).join(","))}</td>
                        <td class="num">${fmt(e.realized_harm, 2)}</td>
                      </tr>`
                    )
                    .join("")}</tbody></table>`
              : `<p class="muted">Zero high-confidence false negatives/positives on this split.</p>`
          }`;
      }

      // Model disagreement
      const dis = xai.model_disagreement;
      if (!dis || !(dis.top_disagreement_cases || []).length) {
        $("xaiDisagreementPanel").innerHTML = `<p class="muted">Run a multi-model benchmark comparison to inspect cross-model disagreement cases.</p>`;
      } else {
        const models = dis.models || [];
        $("xaiDisagreementPanel").innerHTML = `
          <p class="mono muted" style="margin:0 0 8px;">Shared test ECGs: ${dis.shared_ecg_count} across [${esc(models.join(", "))}]</p>
          <table class="data-table">
            <thead><tr><th>ECG</th><th>Disputed Label</th><th class="num">Prob Std</th><th>True</th><th>Per-Model Probabilities</th></tr></thead>
            <tbody>${(dis.top_disagreement_cases || [])
              .slice(0, 10)
              .map((c) => {
                const summary = models
                  .map((m) => `${m}:${fmt(c.model_probabilities[m] && c.model_probabilities[m][c.most_disputed_label], 2)}`)
                  .join(" · ");
                return `<tr>
                  <td class="mono">#${c.ecg_id}</td>
                  <td class="mono"><strong>${esc(c.most_disputed_label)}</strong></td>
                  <td class="num">${fmt(c.max_prob_std, 3)}</td>
                  <td class="mono">${esc((c.true_labels || []).join(","))}</td>
                  <td class="mono" style="font-size:11px;">${esc(summary)}</td>
                </tr>`;
              })
              .join("")}</tbody></table>`;
      }
    } catch (err) {
      showAlert(`Explainability error: ${err.message}`);
    }
  }

  // ---------- EVENT WIRING ----------
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".workspace-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      state.activeTab = btn.dataset.tab;
      $(`tab-${state.activeTab}`).classList.add("active");
      loadActiveTab();
    });
  });

  $("globalRunSelect").addEventListener("change", () => {
    state.activeRunId = $("globalRunSelect").value;
    state.activeEcgId = null;
    updateTierBadge();
    loadActiveTab();
  });

  $("sigSplitSelect").addEventListener("change", () => loadSignalExplorer());
  $("sigRecordSelect").addEventListener("change", () => loadSingleSignal(Number($("sigRecordSelect").value)));
  $("sigLayoutSelect").addEventListener("change", () => {
    const layout = $("sigLayoutSelect").value;
    $("singleLeadField").style.opacity = layout === "single" ? "1" : "0.45";
    signalViewer.setView({ layout });
  });
  $("sigThemeSelect").addEventListener("change", () => signalViewer.setView({ theme: $("sigThemeSelect").value }));
  $("sigLeadSelect").addEventListener("change", () => signalViewer.setView({ focusLead: Number($("sigLeadSelect").value), layout: "single" }));
  $("sigZoomRange").addEventListener("input", () => {
    const z = Number($("sigZoomRange").value);
    $("sigZoomLabel").textContent = `${z.toFixed(1)}×`;
    signalViewer.setView({ zoom: z });
  });
  $("sigPanRange").addEventListener("input", () => {
    const s = Number($("sigPanRange").value);
    $("sigPanLabel").textContent = `${s.toFixed(2)} s`;
    signalViewer.setView({ startSec: s });
  });
  $("sigPerturbType").addEventListener("change", renderPerturbInputs);
  $("sigApplyPerturbBtn").addEventListener("click", applyLivePerturbation);
  $("sigResetPerturbBtn").addEventListener("click", () => {
    $("sigPerturbType").value = "none";
    renderPerturbInputs();
    applyLivePerturbation();
  });

  $("refreshRunsBtn").addEventListener("click", loadRuns);
  $("artifactFileSelect").addEventListener("change", () => loadArtifactFile($("artifactFileSelect").value));
  $("relClassSelect").addEventListener("change", renderReliabilityClassCharts);
  $("robMetricSelect").addEventListener("change", () => {
    if (state.robustnessData && state.robustnessData.results) {
      window.WorkbenchCharts.renderRobustnessBars($("robBarChart"), state.robustnessData.results.delta_summary || [], $("robMetricSelect").value);
    }
  });
  $("xaiLoadBtn").addEventListener("click", () => {
    loadExplainability({
      ecg_id: Number($("xaiRecordSelect").value),
      target_label: $("xaiLabelSelect").value,
      method: $("xaiMethodSelect").value,
    });
  });

  window.addEventListener("resize", () => {
    signalViewer.render();
    xaiViewer.render();
  });

  // Boot
  populateLeadDropdown();
  renderPerturbInputs();
  $("singleLeadField").style.opacity = "0.45";
  loadRuns();
})();
