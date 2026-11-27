/* Crisp Scientific SVG Chart Renderers (Zero External Dependencies) */
(function (global) {
  "use strict";

  function esc(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function emptyState(container, msg) {
    container.innerHTML = `<div class="muted mono" style="padding:20px;text-align:center;">${esc(msg)}</div>`;
  }

  function renderReliabilityDiagram(container, rawBins, calBins, className) {
    if (!rawBins || !rawBins.length) {
      return emptyState(container, `Reliability bins unavailable for ${className}.`);
    }
    const W = 540, H = 290, L = 52, R = 20, T = 24, B = 42;
    const pw = W - L - R, ph = H - T - B;
    const sx = (v) => L + v * pw;
    const sy = (v) => T + (1 - v) * ph;

    const validPts = (bins) =>
      (bins || [])
        .filter((b) => b.count > 0 && b.confidence !== null && b.accuracy !== null)
        .map((b) => ({ x: sx(b.confidence), y: sy(b.accuracy), count: b.count, left: b.bin_left, right: b.bin_right }));

    const rawPts = validPts(rawBins);
    const calPts = validPts(calBins);
    const maxCount = Math.max(1, ...(calBins || rawBins).map((b) => b.count || 0));

    const pathOf = (pts) => pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

    let grid = "";
    for (let i = 0; i <= 5; i++) {
      const v = i * 0.2;
      const y = sy(v);
      const x = sx(v);
      grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#e2e8f0" stroke-width="1"/>`;
      grid += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" stroke="#e2e8f0" stroke-width="1"/>`;
      grid += `<text x="${L - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#475569" font-family="monospace">${v.toFixed(1)}</text>`;
      grid += `<text x="${x}" y="${H - B + 15}" text-anchor="middle" font-size="10" fill="#475569" font-family="monospace">${v.toFixed(1)}</text>`;
    }

    // Subtle sample density bars at bottom
    let bars = "";
    for (const b of calBins || rawBins) {
      if (!b.count) continue;
      const bx = sx(b.bin_left);
      const bw = Math.max(2, sx(b.bin_right) - bx - 2);
      const bh = (b.count / maxCount) * (ph * 0.22);
      bars += `<rect x="${bx + 1}" y="${H - B - bh}" width="${bw}" height="${bh}" fill="rgba(148,163,184,0.28)"/>`;
    }

    const dots = (pts, color) =>
      pts.map((p) => `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.6" fill="${color}" stroke="#fff" stroke-width="1"/>`).join("");

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reliability diagram for ${esc(className)}">
        ${grid}
        ${bars}
        <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#64748b" stroke-dasharray="5,4" stroke-width="1.2"/>
        ${rawPts.length ? `<path d="${pathOf(rawPts)}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,3"/>${dots(rawPts, "#64748b")}` : ""}
        ${calPts.length ? `<path d="${pathOf(calPts)}" fill="none" stroke="#1d4ed8" stroke-width="2.2"/>${dots(calPts, "#1d4ed8")}` : ""}
        <rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="#94a3b8" stroke-width="1"/>
        <text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#0f172a">Mean Predicted Probability (${esc(className)})</text>
        <text x="14" y="${T + ph / 2}" text-anchor="middle" font-size="11" fill="#0f172a" transform="rotate(-90 14 ${T + ph / 2})">Empirical Positive Frequency</text>
        <g transform="translate(${L + 10}, ${T + 8})" font-size="10.5" font-family="monospace">
          <line x1="0" y1="5" x2="18" y2="5" stroke="#1d4ed8" stroke-width="2.2"/><text x="23" y="8" fill="#0f172a">Calibrated</text>
          <line x1="100" y1="5" x2="118" y2="5" stroke="#64748b" stroke-width="2" stroke-dasharray="4,3"/><text x="123" y="8" fill="#475569">Raw</text>
          <line x1="164" y1="5" x2="182" y2="5" stroke="#64748b" stroke-dasharray="4,3"/><text x="187" y="8" fill="#475569">Ideal y=x</text>
        </g>
      </svg>`;
  }

  function renderRiskCoverageChart(container, curves) {
    if (!curves || !curves.length) {
      return emptyState(container, "Selective prediction curves not available in this run.");
    }
    const sorted = [...curves].sort((a, b) => a.coverage - b.coverage);
    const W = 540, H = 290, L = 56, R = 20, T = 24, B = 42;
    const pw = W - L - R, ph = H - T - B;
    const xs = sorted.map((c) => Number(c.coverage));
    const risks = sorted.map((c) => Number(c.accepted_risk ?? 0));
    const harms = sorted.map((c) => Number(c.expected_harm_with_abstain ?? 0));
    const minX = Math.max(0, Math.min(...xs) - 0.05);
    const maxX = Math.min(1.02, Math.max(...xs) + 0.02);
    const maxY = Math.max(0.1, ...risks, ...harms) * 1.15;

    const sx = (v) => L + ((v - minX) / Math.max(maxX - minX, 0.05)) * pw;
    const sy = (v) => T + (1 - v / maxY) * ph;

    let grid = "";
    for (let i = 0; i <= 4; i++) {
      const yv = (maxY * i) / 4;
      const y = sy(yv);
      grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#e2e8f0"/>`;
      grid += `<text x="${L - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#475569" font-family="monospace">${yv.toFixed(2)}</text>`;
    }
    for (const c of sorted) {
      const x = sx(c.coverage);
      grid += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" stroke="#f1f5f9"/>`;
      grid += `<text x="${x}" y="${H - B + 15}" text-anchor="middle" font-size="10" fill="#475569" font-family="monospace">${(c.coverage * 100).toFixed(0)}%</text>`;
    }

    const line = (vals, color, hollow = false) => {
      const d = sorted.map((c, i) => `${i === 0 ? "M" : "L"}${sx(c.coverage).toFixed(1)},${sy(vals[i]).toFixed(1)}`).join(" ");
      const pts = sorted
        .map((c, i) =>
          hollow
            ? `<circle cx="${sx(c.coverage).toFixed(1)}" cy="${sy(vals[i]).toFixed(1)}" r="6" fill="none" stroke="${color}" stroke-width="2.2"/>`
            : `<circle cx="${sx(c.coverage).toFixed(1)}" cy="${sy(vals[i]).toFixed(1)}" r="3.6" fill="${color}" stroke="#fff" stroke-width="1"/>`
        )
        .join("");
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.2"/>${pts}`;
    };

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Risk-Coverage and Expected Harm">
        ${grid}
        ${line(harms, "#d97706", false)}
        ${line(risks, "#1d4ed8", true)}
        <rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="#94a3b8"/>
        <text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#0f172a">Achieved Coverage (Non-Abstained Fraction)</text>
        <text x="14" y="${T + ph / 2}" text-anchor="middle" font-size="11" fill="#0f172a" transform="rotate(-90 14 ${T + ph / 2})">Weighted Harm / Risk</text>
        <g transform="translate(${L + 10}, ${T + 8})" font-size="10.5" font-family="monospace">
          <line x1="0" y1="5" x2="18" y2="5" stroke="#1d4ed8" stroke-width="2.2"/><text x="23" y="8" fill="#0f172a">Accepted Risk</text>
          <line x1="128" y1="5" x2="146" y2="5" stroke="#d97706" stroke-width="2.2"/><text x="151" y="8" fill="#0f172a">Expected Harm (w/ Abstain)</text>
        </g>
      </svg>`;
  }

  function renderConfidenceHistogram(container, dist, className) {
    if (!dist || !dist.positive_counts) {
      return emptyState(container, `Confidence distribution unavailable for ${className}.`);
    }
    const W = 540, H = 270, L = 48, R = 16, T = 34, B = 38;
    const pw = W - L - R, ph = H - T - B;
    const pos = dist.positive_counts;
    const neg = dist.negative_counts;
    const nBins = pos.length;
    const maxY = Math.max(1, ...pos, ...neg);
    const groupW = pw / nBins;

    let bars = "";
    for (let i = 0; i < nBins; i++) {
      const gx = L + i * groupW;
      const bw = Math.max(4, groupW * 0.4);
      const hNeg = (neg[i] / maxY) * ph;
      const hPos = (pos[i] / maxY) * ph;
      bars += `<rect x="${gx + 2}" y="${H - B - hNeg}" width="${bw}" height="${hNeg}" fill="rgba(100,116,139,0.55)"/>`;
      bars += `<rect x="${gx + 2 + bw}" y="${H - B - hPos}" width="${bw}" height="${hPos}" fill="rgba(29,78,216,0.82)"/>`;
      if (i % 2 === 0) {
        bars += `<text x="${gx + groupW / 2}" y="${H - B + 14}" text-anchor="middle" font-size="9.5" fill="#475569" font-family="monospace">${(i / nBins).toFixed(1)}</text>`;
      }
    }
    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Confidence histogram for ${esc(className)}">
        <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#94a3b8"/>
        <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#94a3b8"/>
        ${bars}
        <text x="${L - 8}" y="${T + 8}" text-anchor="end" font-size="10" fill="#475569" font-family="monospace">${maxY}</text>
        <text x="${L - 8}" y="${H - B}" text-anchor="end" font-size="10" fill="#475569" font-family="monospace">0</text>
        <text x="${L + pw / 2}" y="${H - 5}" text-anchor="middle" font-size="11" fill="#0f172a">Predicted Calibrated Probability (${esc(className)})</text>
        <g transform="translate(${L + 4}, 10)" font-size="10.5" font-family="monospace">
          <rect x="0" y="0" width="10" height="10" fill="rgba(29,78,216,0.82)"/><text x="14" y="9" fill="#0f172a">True Positive (y=1)</text>
          <rect x="160" y="0" width="10" height="10" fill="rgba(100,116,139,0.55)"/><text x="174" y="9" fill="#475569">True Negative (y=0)</text>
        </g>
      </svg>`;
  }

  function renderTrainingHistoryChart(container, history) {
    if (!history || !history.length) {
      return emptyState(container, "No epoch training history recorded for this run (e.g. tabular XGBoost or failed before epoch 1).");
    }
    const W = 540, H = 260, L = 52, R = 52, T = 24, B = 38;
    const pw = W - L - R, ph = H - T - B;
    const epochs = history.map((h) => Number(h.epoch));
    const maxEp = Math.max(1, ...epochs);
    const losses = history.map((h) => Number(h.train_loss ?? 0));
    const valLosses = history.map((h) => Number(h.val_loss ?? h.train_loss ?? 0));
    const aurocs = history.map((h) => Number(h.val_macro_auroc ?? 0));
    const maxLoss = Math.max(0.1, ...losses, ...valLosses) * 1.15;

    const sx = (e) => L + (history.length === 1 ? pw / 2 : ((e - 1) / Math.max(maxEp - 1, 1)) * pw);
    const syLoss = (v) => T + (1 - v / maxLoss) * ph;
    const syAuc = (v) => T + (1 - Math.max(0, Math.min(1, v))) * ph;

    const mkLine = (vals, syFn, color, dash = "") => {
      const d = epochs.map((e, i) => `${i === 0 ? "M" : "L"}${sx(e).toFixed(1)},${syFn(vals[i]).toFixed(1)}`).join(" ");
      const pts = epochs.map((e, i) => `<circle cx="${sx(e).toFixed(1)}" cy="${syFn(vals[i]).toFixed(1)}" r="3.5" fill="${color}" stroke="#fff"/>`).join("");
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ""}/>${pts}`;
    };

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training history">
        <rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="#cbd5e1"/>
        ${mkLine(losses, syLoss, "#0f172a")}
        ${mkLine(valLosses, syLoss, "#64748b", "4,3")}
        ${mkLine(aurocs, syAuc, "#1d4ed8")}
        <text x="${L - 8}" y="${T + 8}" text-anchor="end" font-size="10" fill="#0f172a" font-family="monospace">${maxLoss.toFixed(2)}</text>
        <text x="${L - 8}" y="${H - B}" text-anchor="end" font-size="10" fill="#0f172a" font-family="monospace">0.00</text>
        <text x="${W - R + 8}" y="${T + 8}" text-anchor="start" font-size="10" fill="#1d4ed8" font-family="monospace">1.00</text>
        <text x="${W - R + 8}" y="${H - B}" text-anchor="start" font-size="10" fill="#1d4ed8" font-family="monospace">0.00</text>
        <text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#0f172a">Epoch (1 – ${maxEp})</text>
        <g transform="translate(${L + 8}, ${T - 10})" font-size="10" font-family="monospace">
          <line x1="0" y1="0" x2="14" y2="0" stroke="#0f172a" stroke-width="2"/><text x="18" y="3">Train Loss (L)</text>
          <line x1="114" y1="0" x2="128" y2="0" stroke="#64748b" stroke-width="2" stroke-dasharray="4,3"/><text x="132" y="3">Val Loss (L)</text>
          <line x1="214" y1="0" x2="228" y2="0" stroke="#1d4ed8" stroke-width="2"/><text x="232" y="3" fill="#1d4ed8">Val AUROC (R)</text>
        </g>
      </svg>`;
  }

  function renderRobustnessBars(container, deltaSummary, metricKey) {
    if (!deltaSummary || !deltaSummary.length) {
      return emptyState(container, "No robustness perturbation deltas available in this run.");
    }
    const rowH = 22;
    const W = 560, L = 195, R = 64, T = 20, B = 24;
    const H = Math.max(180, T + B + deltaSummary.length * rowH);
    const pw = W - L - R;
    const vals = deltaSummary.map((r) => Number(r[metricKey] ?? 0));
    const maxAbs = Math.max(0.02, ...vals.map((v) => Math.abs(v))) * 1.15;
    const zeroX = L + pw / 2;
    const sx = (v) => zeroX + (v / maxAbs) * (pw / 2);

    const lowerIsBetter = metricKey.includes("ece") || metricKey.includes("brier");
    let rows = `<line x1="${zeroX}" y1="${T}" x2="${zeroX}" y2="${H - B}" stroke="#475569" stroke-width="1"/>`;
    deltaSummary.forEach((item, idx) => {
      const y = T + idx * rowH;
      const v = Number(item[metricKey] ?? 0);
      const degraded = lowerIsBetter ? v > 0.0005 : v < -0.0005;
      const color = degraded ? "#dc2626" : "#15803d";
      const bx = Math.min(zeroX, sx(v));
      const bw = Math.max(1.5, Math.abs(sx(v) - zeroX));
      rows += `<text x="${L - 8}" y="${y + 14}" text-anchor="end" font-size="10.5" fill="#0f172a" font-family="monospace">${esc(item.perturbation)}</text>`;
      rows += `<rect x="${bx.toFixed(1)}" y="${y + 4}" width="${bw.toFixed(1)}" height="12" fill="${color}" opacity="0.82"/>`;
      rows += `<text x="${W - R + 6}" y="${y + 14}" font-size="10" fill="${color}" font-family="monospace">${v >= 0 ? "+" : ""}${v.toFixed(4)}</text>`;
    });

    container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Robustness deltas">${rows}</svg>`;
  }

  function renderLeadImportanceChart(container, leadImportance) {
    if (!leadImportance || !leadImportance.length) {
      return emptyState(container, "Lead attribution shares unavailable.");
    }
    const W = 280, H = 300, L = 38, R = 48, T = 20, B = 16;
    const pw = W - L - R;
    const rowH = (H - T - B) / leadImportance.length;
    const maxShare = Math.max(0.05, ...leadImportance.map((d) => d.share));
    let html = `<text x="${L}" y="12" font-size="10.5" font-weight="bold" fill="#0f172a" font-family="monospace">Per-Lead |Attribution| Share</text>`;
    leadImportance.forEach((d, idx) => {
      const y = T + idx * rowH;
      const bw = (d.share / maxShare) * pw;
      html += `<text x="${L - 6}" y="${y + rowH * 0.7}" text-anchor="end" font-size="10" font-family="monospace" fill="#0f172a">${esc(d.lead)}</text>`;
      html += `<rect x="${L}" y="${y + 3}" width="${bw.toFixed(1)}" height="${Math.max(6, rowH - 6)}" fill="#1d4ed8" opacity="0.82"/>`;
      html += `<text x="${L + bw + 5}" y="${y + rowH * 0.7}" font-size="9.5" font-family="monospace" fill="#475569">${(d.share * 100).toFixed(1)}%</text>`;
    });
    container.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${html}</svg>`;
  }

  function renderFaithfulnessChart(container, faith) {
    if (!faith || !faith.fractions) return emptyState(container, "Faithfulness curve unavailable.");
    const W = 480, H = 212, L = 46, R = 16, T = 24, B = 42;
    const pw = W - L - R, ph = H - T - B;
    const sx = (f) => L + f * pw;
    const sy = (p) => T + (1 - Math.max(0, Math.min(1, p))) * ph;
    const mk = (arr, color, dash = "") =>
      `<path d="${faith.fractions.map((f, i) => `${i === 0 ? "M" : "L"}${sx(f).toFixed(1)},${sy(arr[i]).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;
    let xTicks = "";
    for (let i = 0; i <= 5; i++) {
      const f = i * 0.2;
      const x = sx(f);
      xTicks += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" stroke="#f1f5f9"/>`;
      xTicks += `<text x="${x}" y="${H - B + 14}" text-anchor="middle" font-size="9.5" fill="#475569" font-family="monospace">${f.toFixed(1)}</text>`;
    }
    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}">
        ${xTicks}
        <rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="#cbd5e1"/>
        ${mk(faith.deletion_probs, "#dc2626")}
        ${mk(faith.random_deletion_probs, "#64748b", "4,3")}
        ${mk(faith.insertion_probs, "#15803d")}
        <text x="${L + pw / 2}" y="${H - 5}" text-anchor="middle" font-size="10.5" fill="#0f172a">Fraction of Top-Attributed Samples Perturbed</text>
        <text x="${L - 6}" y="${T + 8}" text-anchor="end" font-size="9.5" font-family="monospace">1.0</text>
        <text x="${L - 6}" y="${H - B}" text-anchor="end" font-size="9.5" font-family="monospace">0.0</text>
        <g transform="translate(${L + 6}, ${T - 8})" font-size="9.5" font-family="monospace">
          <text x="0" y="0" fill="#dc2626">■ Deletion (AUC ${faith.deletion_auc.toFixed(3)})</text>
          <text x="155" y="0" fill="#64748b">■ Random (${faith.random_deletion_auc.toFixed(3)})</text>
          <text x="295" y="0" fill="#15803d">■ Insertion (${faith.insertion_auc.toFixed(3)})</text>
        </g>
      </svg>`;
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
