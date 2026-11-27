/* Unified Interactive Scientific SVG Visualization Engine
   Coherent typography (tabular sans-serif), hairline grids, colorblind-safe Okabe-Ito
   & Paul Tol palettes, interactive floating tooltips, and click-linked callbacks. */
(function (global) {
  "use strict";

  const OKABE_ITO = ["#1d4f91", "#c4452b", "#1d7044", "#b47812", "#6b4c9a", "#2b8a9c"];

  const esc = (s) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d));

  function showTooltip(ev, title, rows) {
    const el = document.getElementById("chartTooltip");
    if (!el) return;
    el.innerHTML = `<div class="tt-title">${esc(title)}</div>${rows
      .map(([k, v]) => `<div class="tt-row"><span>${esc(k)}</span><span>${esc(v)}</span></div>`)
      .join("")}`;
    el.classList.remove("hidden");
    const pad = 12;
    const x = Math.min(window.innerWidth - el.offsetWidth - pad, ev.clientX + 14);
    const y = Math.min(window.innerHeight - el.offsetHeight - pad, ev.clientY + 14);
    el.style.left = `${Math.max(pad, x)}px`;
    el.style.top = `${Math.max(pad, y)}px`;
  }

  function hideTooltip() {
    const el = document.getElementById("chartTooltip");
    if (el) el.classList.add("hidden");
  }

  function bindTooltips(container, entries) {
    hideTooltip();
    container.querySelectorAll("[data-tt-idx]").forEach((node) => {
      const idx = Number(node.dataset.ttIdx);
      const entry = entries[idx];
      if (!entry) return;
      node.style.cursor = entry.onClick ? "pointer" : "default";
      node.setAttribute("tabindex", "0");
      node.setAttribute("aria-label", [entry.title, ...entry.rows.map(([label, value]) => `${label}: ${value}`)].join(", "));
      node.addEventListener("mousemove", (ev) => showTooltip(ev, entry.title, entry.rows));
      node.addEventListener("mouseleave", hideTooltip);
      node.addEventListener("focus", () => {
        const rect = node.getBoundingClientRect();
        showTooltip({ clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 }, entry.title, entry.rows);
      });
      node.addEventListener("blur", hideTooltip);
      if (entry.onClick) {
        node.setAttribute("role", "button");
        node.addEventListener("click", () => entry.onClick());
        node.addEventListener("keydown", (event) => {
          if (event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          entry.onClick();
          container.querySelector(`[data-tt-idx="${idx}"]`)?.focus();
        });
      }
    });
  }

  function emptyState(container, title, detail = "") {
    container.innerHTML = `<div class="state-empty"><strong>${esc(title)}</strong>${detail ? `<span>${esc(detail)}</span>` : ""}</div>`;
  }

  function renderReliabilityDiagram(container, rawBins, calBins, { className = "MI", methodLabel = "Calibrated" } = {}) {
    if (!rawBins?.length && !calBins?.length) {
      return emptyState(container, `No reliability bins for ${className}`, "Run calibration evaluation to populate held-out reliability curves.");
    }
    const W = 540, H = 265, L = 48, R = 18, T = 28, B = 38;
    const pw = W - L - R, ph = H - T - B;
    const sx = (v) => L + v * pw;
    const sy = (v) => T + (1 - v) * ph;

    const toPts = (bins) =>
      (bins || [])
        .filter((b) => b.count > 0 && b.confidence !== null && b.accuracy !== null)
        .map((b) => ({
          x: sx(b.confidence),
          y: sy(b.accuracy),
          conf: Number(b.confidence),
          acc: Number(b.accuracy),
          count: Number(b.count),
          left: Number(b.bin_left),
          right: Number(b.bin_right),
        }));

    const rawPts = toPts(rawBins);
    const calPts = toPts(calBins);
    const maxCount = Math.max(1, ...(calBins || rawBins).map((b) => b.count || 0));

    let grid = "";
    for (let i = 0; i <= 5; i++) {
      const v = i * 0.2;
      const y = sy(v), x = sx(v);
      grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#edf0f4"/>`;
      grid += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" stroke="#edf0f4"/>`;
      grid += `<text x="${L - 7}" y="${y + 3.5}" text-anchor="end" font-size="10.5" fill="#6c7789">${v.toFixed(1)}</text>`;
      grid += `<text x="${x}" y="${H - B + 14}" text-anchor="middle" font-size="10.5" fill="#6c7789">${v.toFixed(1)}</text>`;
    }

    let bars = "";
    for (const b of calBins || rawBins) {
      if (!b.count) continue;
      const bx = sx(b.bin_left);
      const bw = Math.max(3, sx(b.bin_right) - bx - 1.5);
      const bh = (b.count / maxCount) * (ph * 0.22);
      bars += `<rect x="${bx + 0.75}" y="${H - B - bh}" width="${bw}" height="${bh}" fill="#dce4ef" opacity="0.65"/>`;
    }

    const pathOf = (pts) => pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    const tooltips = [];
    const drawPts = (pts, color, seriesName) =>
      pts
        .map((p) => {
          const ttIdx = tooltips.length;
          tooltips.push({
            title: `${seriesName} · ${className} [${p.left.toFixed(2)}, ${p.right.toFixed(2)}]`,
            rows: [
              ["Mean predicted P", p.conf.toFixed(4)],
              ["Empirical positive rate", p.acc.toFixed(4)],
              ["Abs. gap |acc − conf|", Math.abs(p.acc - p.conf).toFixed(4)],
              ["Bin sample count", String(p.count)],
            ],
          });
          return `<circle data-tt-idx="${ttIdx}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.2" fill="${color}" stroke="#ffffff" stroke-width="1.2"/>`;
        })
        .join("");

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        ${grid}
        ${bars}
        <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#9aa5b6" stroke-dasharray="4,4" stroke-width="1.1"/>
        ${rawPts.length ? `<path d="${pathOf(rawPts)}" fill="none" stroke="#8c98aa" stroke-width="1.6" stroke-dasharray="4,3"/>${drawPts(rawPts, "#6c7789", "Raw")}` : ""}
        ${calPts.length ? `<path d="${pathOf(calPts)}" fill="none" stroke="#1d4f91" stroke-width="2.1"/>${drawPts(calPts, "#1d4f91", methodLabel)}` : ""}
        <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#b8c2d0"/>
        <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#b8c2d0"/>
        <text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#434d5e">Mean predicted probability (${esc(className)})</text>
        <text x="13" y="${T + ph / 2}" text-anchor="middle" font-size="11" fill="#434d5e" transform="rotate(-90 13 ${T + ph / 2})">Empirical frequency</text>
        <g transform="translate(${L}, 12)" font-size="10.5" fill="#434d5e">
          ${calPts.length ? `<line x1="0" y1="0" x2="14" y2="0" stroke="#1d4f91" stroke-width="2.1"/><text x="19" y="3.5">${esc(methodLabel)}</text>` : ""}
          ${rawPts.length ? `<line x1="128" y1="0" x2="142" y2="0" stroke="#8c98aa" stroke-width="1.6" stroke-dasharray="4,3"/><text x="147" y="3.5">Raw</text>` : ""}
          <line x1="186" y1="0" x2="200" y2="0" stroke="#9aa5b6" stroke-dasharray="4,4" stroke-width="1.1"/><text x="205" y="3.5">Perfect calibration</text>
        </g>
      </svg>`;
    bindTooltips(container, tooltips);
  }

  function renderRiskCoverageChart(container, curves) {
    if (!curves || !curves.length) {
      return emptyState(container, "Selective prediction unavailable", "Run selective-policy evaluation to populate risk–coverage operating points.");
    }
    const sorted = [...curves].sort((a, b) => Number(a.coverage) - Number(b.coverage));
    const W = 540, H = 265, L = 52, R = 18, T = 28, B = 38;
    const pw = W - L - R, ph = H - T - B;
    const xs = sorted.map((c) => Number(c.coverage));
    const risks = sorted.map((c) => c.accepted_risk);
    const harms = sorted.map((c) => c.expected_harm_with_abstain);
    const minX = Math.max(0, Math.min(...xs, 0.85) - 0.04);
    const maxX = 1.02;
    const maxY = Math.max(0.12, ...risks, ...harms) * 1.18;

    const sx = (v) => L + ((v - minX) / Math.max(maxX - minX, 0.05)) * pw;
    const sy = (v) => T + (1 - v / maxY) * ph;

    let grid = "";
    for (let i = 0; i <= 4; i++) {
      const yv = (maxY * i) / 4;
      const y = sy(yv);
      grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#edf0f4"/>`;
      grid += `<text x="${L - 7}" y="${y + 3.5}" text-anchor="end" font-size="10.5" fill="#6c7789">${yv.toFixed(2)}</text>`;
    }
    const xTicks = [0.85, 0.9, 0.95, 1.0].filter((t) => t >= minX - 0.01);
    for (const tv of xTicks) {
      const x = sx(tv);
      grid += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" stroke="#edf0f4"/>`;
      grid += `<text x="${x}" y="${H - B + 14}" text-anchor="middle" font-size="10.5" fill="#6c7789">${(tv * 100).toFixed(0)}%</text>`;
    }

    const tooltips = [];
    const lineOf = (vals, color) => {
      const path = sorted.map((c, i) => vals[i] == null ? "" :
        `${i === 0 || vals[i - 1] == null ? "M" : "L"}${sx(c.coverage).toFixed(1)},${sy(vals[i]).toFixed(1)}`).join(" ");
      return `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`;
    };

    let markers = "";
    sorted.forEach((c, i) => {
      const x = sx(c.coverage);
      const yRisk = sy(risks[i]);
      const yHarm = sy(harms[i]);
      const ttIdx = tooltips.length;
      tooltips.push({
        title: `Target coverage ${(Number(c.target_coverage) * 100).toFixed(0)}%`,
        rows: [
          ["Achieved coverage", `${(Number(c.coverage) * 100).toFixed(1)}%`],
          ["Abstain rate", `${(Number(c.abstain_rate) * 100).toFixed(1)}%`],
          ["Accepted risk", fmt(c.accepted_risk, 4)],
          ["Expected harm (w/ abstain)", fmt(c.expected_harm_with_abstain, 4)],
          ["Weighted FN / FP harm", `${fmt(c.weighted_fn_harm, 3)} / ${fmt(c.weighted_fp_harm, 3)}`],
        ],
      });
      if (harms[i] != null) markers += `<circle data-tt-idx="${ttIdx}" cx="${x.toFixed(1)}" cy="${yHarm.toFixed(1)}" r="4" fill="#c4452b" stroke="#fff" stroke-width="1"/>`;
      if (risks[i] != null) markers += `<circle data-tt-idx="${ttIdx}" cx="${x.toFixed(1)}" cy="${yRisk.toFixed(1)}" r="6.2" fill="none" stroke="#1d4f91" stroke-width="2"/>`;
    });

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        ${grid}
        ${sorted.length > 1 ? lineOf(harms, "#c4452b") + lineOf(risks, "#1d4f91") : ""}
        ${markers}
        <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#b8c2d0"/>
        <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#b8c2d0"/>
        <text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#434d5e">Achieved coverage (accepted fraction)</text>
        <text x="13" y="${T + ph / 2}" text-anchor="middle" font-size="11" fill="#434d5e" transform="rotate(-90 13 ${T + ph / 2})">Weighted harm / risk</text>
        <g transform="translate(${L}, 12)" font-size="10.5" fill="#434d5e">
          <circle cx="5" cy="0" r="4.5" fill="none" stroke="#1d4f91" stroke-width="1.8"/><text x="14" y="3.5">Accepted risk</text>
          <circle cx="110" cy="0" r="3.5" fill="#c4452b"/><text x="119" y="3.5">Expected harm (with abstention)</text>
        </g>
      </svg>`;
    bindTooltips(container, tooltips);
  }

  function renderConfidenceHistogram(container, dist, className) {
    if (!dist || !dist.positive_counts) {
      return emptyState(container, `No probability histogram for ${className}`);
    }
    const W = 480, H = 230, L = 44, R = 16, T = 28, B = 36;
    const pw = W - L - R, ph = H - T - B;
    const pos = dist.positive_counts;
    const neg = dist.negative_counts;
    const nBins = pos.length;
    const maxY = Math.max(1, ...pos, ...neg);
    const groupW = pw / nBins;
    const tooltips = [];

    let bars = "";
    for (let i = 0; i < nBins; i++) {
      const gx = L + i * groupW;
      const bw = Math.max(4, groupW * 0.42);
      const hNeg = (neg[i] / maxY) * ph;
      const hPos = (pos[i] / maxY) * ph;
      const ttIdx = tooltips.length;
      tooltips.push({
        title: `${className} probability [${(i / nBins).toFixed(1)}, ${((i + 1) / nBins).toFixed(1)}]`,
        rows: [
          ["True positives (y = 1)", String(pos[i])],
          ["True negatives (y = 0)", String(neg[i])],
        ],
      });
      bars += `<rect data-tt-idx="${ttIdx}" x="${gx + 2}" y="${H - B - hNeg}" width="${bw}" height="${hNeg}" fill="#b8c4d4"/>`;
      bars += `<rect data-tt-idx="${ttIdx}" x="${gx + 2 + bw}" y="${H - B - hPos}" width="${bw}" height="${hPos}" fill="#1d4f91"/>`;
      if (i % 2 === 0 || i === nBins - 1) {
        bars += `<text x="${gx + groupW / 2}" y="${H - B + 14}" text-anchor="middle" font-size="10" fill="#6c7789">${(i / nBins).toFixed(1)}</text>`;
      }
    }

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        <line x1="${L}" y1="${T}" x2="${W - R}" y2="${T}" stroke="#edf0f4"/>
        <line x1="${L}" y1="${T + ph / 2}" x2="${W - R}" y2="${T + ph / 2}" stroke="#edf0f4"/>
        ${bars}
        <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#b8c2d0"/>
        <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#b8c2d0"/>
        <text x="${L - 7}" y="${T + 4}" text-anchor="end" font-size="10" fill="#6c7789">${maxY}</text>
        <text x="${L - 7}" y="${H - B + 3}" text-anchor="end" font-size="10" fill="#6c7789">0</text>
        <text x="${L + pw / 2}" y="${H - 5}" text-anchor="middle" font-size="11" fill="#434d5e">Predicted probability (${esc(className)})</text>
        <g transform="translate(${L}, 12)" font-size="10.5" fill="#434d5e">
          <rect x="0" y="-7" width="9" height="9" fill="#1d4f91"/><text x="13" y="1">Positive (y = 1)</text>
          <rect x="115" y="-7" width="9" height="9" fill="#b8c4d4"/><text x="128" y="1">Negative (y = 0)</text>
        </g>
      </svg>`;
    bindTooltips(container, tooltips);
  }

  function renderTrainingHistoryChart(container, primaryHistory, compareSeries = []) {
    const hasCompare = compareSeries && compareSeries.length > 1;
    if (!hasCompare && (!primaryHistory || !primaryHistory.length)) {
      return emptyState(container, "No epoch convergence curve", "Tabular baselines (XGBoost) and failed pre-epoch runs do not produce neural epoch histories.");
    }

    const W = 560, H = 250, L = 48, R = hasCompare ? 18 : 48, T = 30, B = 36;
    const pw = W - L - R, ph = H - T - B;
    const tooltips = [];

    if (hasCompare) {
      const maxEp = Math.max(1, ...compareSeries.flatMap((s) => s.history.map((h) => Number(h.epoch))));
      const sx = (e) => L + (maxEp === 1 ? pw / 2 : ((e - 1) / (maxEp - 1)) * pw);
      const sy = (v) => T + (1 - Math.max(0, Math.min(1, v))) * ph;
      let grid = "";
      for (let i = 0; i <= 4; i++) {
        const v = i * 0.25;
        const y = sy(v);
        grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#edf0f4"/>`;
        grid += `<text x="${L - 7}" y="${y + 3.5}" text-anchor="end" font-size="10" fill="#6c7789">${v.toFixed(2)}</text>`;
      }
      let lines = "";
      let legend = "";
      compareSeries.forEach((series, sIdx) => {
        const color = OKABE_ITO[sIdx % OKABE_ITO.length];
        const pts = series.history;
        if (!pts.length) return;
        const d = pts.map((h, i) => `${i === 0 ? "M" : "L"}${sx(h.epoch).toFixed(1)},${sy(h.val_macro_auroc).toFixed(1)}`).join(" ");
        lines += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>`;
        pts.forEach((h) => {
          const ttIdx = tooltips.length;
          tooltips.push({
            title: `${series.label} · Epoch ${h.epoch}`,
            rows: [
              ["Val Macro AUROC", fmt(h.val_macro_auroc, 4)],
              ["Train Loss", fmt(h.train_loss, 4)],
              ["Val Loss", fmt(h.val_loss, 4)],
            ],
          });
          lines += `<circle data-tt-idx="${ttIdx}" cx="${sx(h.epoch).toFixed(1)}" cy="${sy(h.val_macro_auroc).toFixed(1)}" r="3.6" fill="${color}" stroke="#fff"/>`;
        });
        legend += `<g transform="translate(${L + sIdx * 112}, 13)"><line x1="0" y1="0" x2="12" y2="0" stroke="${color}" stroke-width="2"/><text x="16" y="3.5" font-size="10.5" fill="#434d5e">${esc(series.label)}</text></g>`;
      });

      container.innerHTML = `
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMin meet">
          ${grid}
          ${lines}
          <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#b8c2d0"/>
          <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#b8c2d0"/>
          <text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#434d5e">Epoch (1 – ${maxEp}) · Multi-run Val AUROC overlay</text>
          ${legend}
        </svg>`;
      bindTooltips(container, tooltips);
      return;
    }

    const epochs = primaryHistory.map((h) => Number(h.epoch));
    const maxEp = Math.max(1, ...epochs);
    const losses = primaryHistory.map((h) => Number(h.train_loss ?? 0));
    const valLosses = primaryHistory.map((h) => Number(h.val_loss ?? h.train_loss ?? 0));
    const aurocs = primaryHistory.map((h) => Number(h.val_macro_auroc ?? 0));
    const maxLoss = Math.max(0.15, ...losses, ...valLosses) * 1.15;

    const sx = (e) => L + (epochs.length === 1 ? pw / 2 : ((e - 1) / Math.max(maxEp - 1, 1)) * pw);
    const syLoss = (v) => T + (1 - v / maxLoss) * ph;
    const syAuc = (v) => T + (1 - Math.max(0, Math.min(1, v))) * ph;

    let grid = "";
    for (let i = 0; i <= 4; i++) {
      const y = T + (ph * i) / 4;
      grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#edf0f4"/>`;
    }
    epochs.forEach((ep) => {
      const x = sx(ep);
      grid += `<text x="${x}" y="${H - B + 14}" text-anchor="middle" font-size="10" fill="#6c7789">${ep}</text>`;
    });

    const mkLine = (vals, syFn, color, dash = "") =>
      `<path d="${epochs.map((e, i) => `${i === 0 ? "M" : "L"}${sx(e).toFixed(1)},${syFn(vals[i]).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="1.9" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;

    let dots = "";
    primaryHistory.forEach((h, i) => {
      const ttIdx = tooltips.length;
      tooltips.push({
        title: `Epoch ${h.epoch}`,
        rows: [
          ["Train loss", fmt(h.train_loss, 4)],
          ["Val loss", fmt(h.val_loss, 4)],
          ["Val Macro AUROC", fmt(h.val_macro_auroc, 4)],
          ["Val Macro AUPRC", fmt(h.val_macro_auprc, 4)],
          ["Val Macro Brier", fmt(h.val_macro_brier, 4)],
        ],
      });
      const x = sx(h.epoch).toFixed(1);
      dots += `<circle data-tt-idx="${ttIdx}" cx="${x}" cy="${syLoss(losses[i]).toFixed(1)}" r="3.6" fill="#141922" stroke="#fff"/>`;
      dots += `<circle data-tt-idx="${ttIdx}" cx="${x}" cy="${syLoss(valLosses[i]).toFixed(1)}" r="3.6" fill="#6c7789" stroke="#fff"/>`;
      dots += `<circle data-tt-idx="${ttIdx}" cx="${x}" cy="${syAuc(aurocs[i]).toFixed(1)}" r="3.8" fill="#1d4f91" stroke="#fff"/>`;
    });

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMin meet">
        ${grid}
        ${mkLine(losses, syLoss, "#141922")}
        ${mkLine(valLosses, syLoss, "#6c7789", "4,3")}
        ${mkLine(aurocs, syAuc, "#1d4f91")}
        ${dots}
        <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#b8c2d0"/>
        <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#b8c2d0"/>
        <line x1="${W - R}" y1="${T}" x2="${W - R}" y2="${H - B}" stroke="#b8c2d0"/>
        <text x="${L - 7}" y="${T + 4}" text-anchor="end" font-size="10" fill="#434d5e">${maxLoss.toFixed(2)}</text>
        <text x="${L - 7}" y="${H - B + 3}" text-anchor="end" font-size="10" fill="#434d5e">0.00</text>
        <text x="${W - R + 7}" y="${T + 4}" text-anchor="start" font-size="10" fill="#1d4f91">1.00</text>
        <text x="${W - R + 7}" y="${H - B + 3}" text-anchor="start" font-size="10" fill="#1d4f91">0.00</text>
        <text x="${L + pw / 2}" y="${H - 5}" text-anchor="middle" font-size="11" fill="#434d5e">Epoch</text>
        <g transform="translate(${L}, 13)" font-size="10.5" fill="#434d5e">
          <line x1="0" y1="0" x2="14" y2="0" stroke="#141922" stroke-width="2"/><text x="19" y="3.5">Train loss (left)</text>
          <line x1="122" y1="0" x2="136" y2="0" stroke="#6c7789" stroke-width="1.8" stroke-dasharray="4,3"/><text x="141" y="3.5">Val loss (left)</text>
          <line x1="232" y1="0" x2="246" y2="0" stroke="#1d4f91" stroke-width="2"/><text x="251" y="3.5" fill="#1d4f91">Val AUROC (right)</text>
        </g>
      </svg>`;
    bindTooltips(container, tooltips);
  }

  function renderRobustnessBars(container, rows, metricKey, { activePerturbation = null, onSelect = null } = {}) {
    if (!rows || !rows.length) {
      return emptyState(container, "No perturbations match the active filter");
    }
    const rowH = 21;
    const W = 600, L = 196, R = 68, T = 14, B = 16;
    const H = Math.max(180, T + B + rows.length * rowH);
    const pw = W - L - R;
    const vals = rows.map((r) => Number(r[metricKey] ?? 0));
    const maxAbs = Math.max(0.02, ...vals.map((v) => Math.abs(v))) * 1.15;
    const zeroX = L + pw / 2;
    const sx = (v) => zeroX + (v / maxAbs) * (pw / 2);
    const lowerIsBetter = metricKey.includes("ece") || metricKey.includes("brier") || metricKey.includes("harm");
    const tooltips = [];

    let svgRows = `<line x1="${zeroX}" y1="${T}" x2="${zeroX}" y2="${H - B}" stroke="#9aa5b6" stroke-width="1"/>`;
    rows.forEach((item, idx) => {
      const y = T + idx * rowH;
      const v = item[metricKey];
      const degraded = lowerIsBetter ? v > 0.0005 : v < -0.0005;
      const improved = lowerIsBetter ? v < -0.0005 : v > 0.0005;
      const color = degraded ? "#c4452b" : improved ? "#1d7044" : "#8c98aa";
      const bx = Math.min(zeroX, sx(v));
      const bw = Math.max(2, Math.abs(sx(v) - zeroX));
      const isActive = activePerturbation === item.perturbation;
      const ttIdx = tooltips.length;
      tooltips.push({
        title: item.displayName || item.perturbation,
        rows: [
          ["Severity", item.severity || "custom"],
          ["Δ Macro AUROC", fmt(item.delta_macro_auroc, 4)],
          ["Δ Macro AUPRC", fmt(item.delta_macro_auprc, 4)],
          ["Δ Macro ECE", fmt(item.delta_macro_ece, 4)],
          ["Δ Expected Harm", fmt(item.delta_expected_harm_with_abstain, 4)],
        ],
        onClick: onSelect ? () => onSelect(item.perturbation) : null,
      });

      svgRows += `<g data-tt-idx="${ttIdx}">`;
      if (isActive) {
        svgRows += `<rect x="4" y="${y}" width="${W - 8}" height="${rowH}" fill="#e8f0fb"/>`;
      }
      svgRows += `<rect x="4" y="${y}" width="${W - 8}" height="${rowH}" fill="transparent"/>`;
      svgRows += `<text x="${L - 10}" y="${y + 14}" text-anchor="end" font-size="11" fill="${isActive ? "#1d4f91" : "#2d3748"}" font-weight="${isActive ? "600" : "400"}">${esc(item.displayName || item.perturbation)}</text>`;
      if (v != null) svgRows += `<rect x="${bx.toFixed(1)}" y="${y + 5}" width="${bw.toFixed(1)}" height="11" fill="${color}" opacity="0.86"/>`;
      svgRows += `<text x="${W - R + 8}" y="${y + 14}" font-size="10.5" fill="${color}">${v != null && v >= 0 ? "+" : ""}${fmt(v, 4)}</text>`;
      svgRows += `</g>`;
    });

    container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${svgRows}</svg>`;
    bindTooltips(container, tooltips);
  }

  function renderLeadImportanceChart(container, leadImportance, { highlightLead = null, onSelectLead = null } = {}) {
    if (!leadImportance || !leadImportance.length) {
      return emptyState(container, "Lead importance unavailable");
    }
    const W = 320, H = 205, L = 34, R = 46, T = 8, B = 8;
    const pw = W - L - R;
    const rowH = (H - T - B) / leadImportance.length;
    const maxShare = Math.max(0.05, ...leadImportance.map((d) => d.share));
    const tooltips = [];
    let html = "";
    leadImportance.forEach((d, idx) => {
      const y = T + idx * rowH;
      const bw = Math.max(2, (d.share / maxShare) * pw);
      const isHi = highlightLead === idx;
      const ttIdx = tooltips.length;
      tooltips.push({
        title: `Lead ${d.lead}`,
        rows: [
          ["Normalized |attribution| share", `${(d.share * 100).toFixed(2)}%`],
          ["Sum |attribution|", fmt(d.abs_sum, 4)],
        ],
        onClick: onSelectLead ? () => onSelectLead(idx) : null,
      });
      html += `<g data-tt-idx="${ttIdx}">`;
      if (isHi) html += `<rect x="2" y="${y}" width="${W - 4}" height="${rowH}" fill="#e8f0fb"/>`;
      html += `<rect x="2" y="${y}" width="${W - 4}" height="${rowH}" fill="transparent"/>`;
      html += `<text x="${L - 6}" y="${y + rowH * 0.72}" text-anchor="end" font-size="10.5" font-weight="${isHi ? "600" : "500"}" fill="#2d3748">${esc(d.lead)}</text>`;
      html += `<rect x="${L}" y="${y + 3}" width="${bw.toFixed(1)}" height="${Math.max(6, rowH - 6)}" fill="${isHi ? "#c4452b" : "#1d4f91"}" opacity="0.86"/>`;
      html += `<text x="${L + bw + 5}" y="${y + rowH * 0.72}" font-size="10" fill="#434d5e">${(d.share * 100).toFixed(1)}%</text>`;
      html += `</g>`;
    });
    container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${html}</svg>`;
    bindTooltips(container, tooltips);
  }

  function renderFaithfulnessChart(container, faith) {
    if (!faith || !faith.fractions) return emptyState(container, "No deletion/insertion curve");
    const W = 310, H = 165, L = 34, R = 12, T = 24, B = 32;
    const pw = W - L - R, ph = H - T - B;
    const sx = (f) => L + f * pw;
    const sy = (p) => T + (1 - Math.max(0, Math.min(1, p))) * ph;
    const mk = (arr, color, dash = "") =>
      `<path d="${faith.fractions.map((f, i) => `${i === 0 ? "M" : "L"}${sx(f).toFixed(1)},${sy(arr[i]).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="1.8" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;

    let xTicks = "";
    for (let i = 0; i <= 5; i++) {
      const f = i * 0.2;
      const x = sx(f);
      xTicks += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" stroke="#f1f3f6"/>`;
      xTicks += `<text x="${x}" y="${H - B + 12}" text-anchor="middle" font-size="9.5" fill="#6c7789">${f.toFixed(1)}</text>`;
    }
    const tooltips = [];
    let pts = "";
    faith.fractions.forEach((f, i) => {
      const ttIdx = tooltips.length;
      tooltips.push({
        title: `Masked fraction ${(f * 100).toFixed(0)}%`,
        rows: [
          ["Top-k deletion P", fmt(faith.deletion_probs[i], 3)],
          ["Random control P", fmt(faith.random_deletion_probs[i], 3)],
          ["Top-k insertion P", fmt(faith.insertion_probs[i], 3)],
        ],
      });
      pts += `<circle data-tt-idx="${ttIdx}" cx="${sx(f).toFixed(1)}" cy="${sy(faith.deletion_probs[i]).toFixed(1)}" r="3.2" fill="#c4452b"/>`;
      pts += `<circle data-tt-idx="${ttIdx}" cx="${sx(f).toFixed(1)}" cy="${sy(faith.insertion_probs[i]).toFixed(1)}" r="3.2" fill="#1d7044"/>`;
    });

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        ${xTicks}
        <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#b8c2d0"/>
        <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#b8c2d0"/>
        ${mk(faith.deletion_probs, "#c4452b")}
        ${mk(faith.random_deletion_probs, "#8c98aa", "3,3")}
        ${mk(faith.insertion_probs, "#1d7044")}
        ${pts}
        <text x="${L - 5}" y="${T + 4}" text-anchor="end" font-size="9.5" fill="#6c7789">1.0</text>
        <text x="${L - 5}" y="${H - B + 3}" text-anchor="end" font-size="9.5" fill="#6c7789">0.0</text>
        <text x="${L + pw / 2}" y="${H - 4}" text-anchor="middle" font-size="10" fill="#434d5e">Fraction masked</text>
        <g transform="translate(${L}, 12)" font-size="9.5" fill="#434d5e">
          <text x="0" y="0" fill="#c4452b">Del ${faith.deletion_auc.toFixed(2)}</text>
          <text x="64" y="0" fill="#6c7789">Rand ${faith.random_deletion_auc.toFixed(2)}</text>
          <text x="136" y="0" fill="#1d7044">Ins ${faith.insertion_auc.toFixed(2)}</text>
        </g>
      </svg>`;
    bindTooltips(container, tooltips);
  }

  global.WorkbenchCharts = {
    renderReliabilityDiagram,
    renderRiskCoverageChart,
    renderConfidenceHistogram,
    renderTrainingHistoryChart,
    renderRobustnessBars,
    renderLeadImportanceChart,
    renderFaithfulnessChart,
  };
})(window);
