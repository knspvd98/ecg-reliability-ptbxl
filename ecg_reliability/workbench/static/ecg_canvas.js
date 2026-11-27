/* High-DPI 12-Lead Scientific ECG Waveform Canvas Renderer */
(function (global) {
  "use strict";

  const THEMES = {
    paper: {
      bg: "#fffcfa",
      minorGrid: "#f4dad7",
      majorGrid: "#e2b2ad",
      border: "#cbd5e1",
      clean: "#111827",
      perturbed: "#dc2626",
      labelBg: "rgba(255, 252, 250, 0.88)",
      labelText: "#0f172a",
      axisText: "#475569",
      cursor: "#1d4ed8",
    },
    scope: {
      bg: "#0b1220",
      minorGrid: "#172538",
      majorGrid: "#263c59",
      border: "#334155",
      clean: "#38bdf8",
      perturbed: "#f97316",
      labelBg: "rgba(11, 18, 32, 0.85)",
      labelText: "#f1f5f9",
      axisText: "#94a3b8",
      cursor: "#facc15",
    },
  };

  class ECGCanvasViewer {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.onCaliper = options.onCaliper || null;
      this.onViewChange = options.onViewChange || null;

      this.signal = null;        // [12][T]
      this.perturbed = null;     // [12][T] or null
      this.attribution = null;   // [12][T] or null
      this.leadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
      this.samplingRate = 100;
      this.unitLabel = "SD";
      this.layout = "stacked";   // "stacked" | "grid4x3" | "single"
      this.focusLead = 1;
      this.theme = "paper";
      this.zoom = 1.0;
      this.startSec = 0.0;
      this.hoverX = null;
      this.hoverY = null;
      this._dragging = false;
      this._dragOriginX = 0;
      this._dragOriginStart = 0;
      this._boxes = [];

      this._bindEvents();
    }

    get durationSec() {
      if (!this.signal || !this.signal.length) return 10.0;
      return this.signal[0].length / this.samplingRate;
    }

    get windowSec() {
      return Math.max(0.5, this.durationSec / Math.max(this.zoom, 1.0));
    }

    setData({ signal, perturbed = null, attribution = null, leadNames, samplingRate = 100, unitLabel = "SD" }) {
      this.signal = signal;
      this.perturbed = perturbed;
      this.attribution = attribution;
      if (leadNames && leadNames.length) this.leadNames = leadNames;
      this.samplingRate = samplingRate;
      this.unitLabel = unitLabel;
      this.clampView();
      this.render();
    }

    setView({ layout, focusLead, theme, zoom, startSec }) {
      if (layout !== undefined) this.layout = layout;
      if (focusLead !== undefined) this.focusLead = Number(focusLead);
      if (theme !== undefined) this.theme = theme;
      if (zoom !== undefined) this.zoom = Number(zoom);
      if (startSec !== undefined) this.startSec = Number(startSec);
      this.clampView();
      this.render();
    }

    clampView() {
      this.zoom = Math.max(1.0, Math.min(8.0, this.zoom));
      const maxStart = Math.max(0.0, this.durationSec - this.windowSec);
      this.startSec = Math.max(0.0, Math.min(maxStart, this.startSec));
    }

    _bindEvents() {
      this.canvas.addEventListener("mousemove", (ev) => {
        const rect = this.canvas.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        if (this._dragging) {
          const dx = x - this._dragOriginX;
          const activeBox = this._boxes[0];
          const w = activeBox ? activeBox.w : rect.width;
          const deltaSec = -(dx / Math.max(w, 1)) * this.windowSec;
          this.startSec = this._dragOriginStart + deltaSec;
          this.clampView();
          if (this.onViewChange) this.onViewChange({ zoom: this.zoom, startSec: this.startSec });
        }
        this.hoverX = x;
        this.hoverY = y;
        this.render();
        this._emitCaliper(x, y);
      });

      this.canvas.addEventListener("mouseleave", () => {
        this._dragging = false;
        this.hoverX = null;
        this.hoverY = null;
        this.render();
      });

      this.canvas.addEventListener("mousedown", (ev) => {
        const rect = this.canvas.getBoundingClientRect();
        this._dragging = true;
        this._dragOriginX = ev.clientX - rect.left;
        this._dragOriginStart = this.startSec;
      });

      window.addEventListener("mouseup", () => {
        this._dragging = false;
      });

      this.canvas.addEventListener("wheel", (ev) => {
        if (!this.signal) return;
        ev.preventDefault();
        const factor = ev.deltaY < 0 ? 1.25 : 0.8;
        this.zoom = Math.max(1.0, Math.min(8.0, this.zoom * factor));
        this.clampView();
        if (this.onViewChange) this.onViewChange({ zoom: this.zoom, startSec: this.startSec });
        this.render();
      }, { passive: false });
    }

    _emitCaliper(x, y) {
      if (!this.onCaliper || !this.signal) return;
      const hit = this._boxes.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
      if (!hit) return;
      const frac = Math.max(0, Math.min(1, (x - hit.x) / Math.max(hit.w, 1)));
      const tSec = hit.tStart + frac * (hit.tEnd - hit.tStart);
      const sampleIdx = Math.max(0, Math.min(this.signal[0].length - 1, Math.round(tSec * this.samplingRate)));
      const cleanVal = this.signal[hit.leadIndex][sampleIdx];
      const pertVal = this.perturbed ? this.perturbed[hit.leadIndex][sampleIdx] : null;
      const attrVal = this.attribution ? this.attribution[hit.leadIndex][sampleIdx] : null;
      this.onCaliper({
        leadName: this.leadNames[hit.leadIndex],
        tSec,
        sampleIdx,
        cleanVal,
        pertVal,
        attrVal,
        unitLabel: this.unitLabel,
      });
    }

    render() {
      const dpr = window.devicePixelRatio || 1;
      const cssWidth = this.canvas.clientWidth || 1000;
      const cssHeight = this.layout === "single" ? 290 : this.layout === "grid4x3" ? 460 : 620;
      this.canvas.width = Math.round(cssWidth * dpr);
      this.canvas.height = Math.round(cssHeight * dpr);
      this.canvas.style.height = `${cssHeight}px`;

      const ctx = this.ctx;
      ctx.save();
      ctx.scale(dpr, dpr);

      const pal = THEMES[this.theme] || THEMES.paper;
      ctx.fillStyle = pal.bg;
      ctx.fillRect(0, 0, cssWidth, cssHeight);

      if (!this.signal || !this.signal.length) {
        ctx.fillStyle = pal.axisText;
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText("No ECG waveform loaded.", 20, 32);
        ctx.restore();
        return;
      }

      const padLeft = 44;
      const padRight = 28;
      const padTop = 12;
      const padBottom = 26;
      const plotW = cssWidth - padLeft - padRight;
      const plotH = cssHeight - padTop - padBottom;

      this._boxes = [];
      if (this.layout === "single") {
        this._boxes.push({
          leadIndex: Math.min(this.focusLead, this.signal.length - 1),
          x: padLeft,
          y: padTop,
          w: plotW,
          h: plotH,
          tStart: this.startSec,
          tEnd: this.startSec + this.windowSec,
        });
      } else if (this.layout === "grid4x3") {
        const cols = 4;
        const rows = 3;
        const colW = plotW / cols;
        const rowH = plotH / rows;
        const segDur = this.windowSec / cols;
        for (let c = 0; c < cols; c++) {
          for (let r = 0; r < rows; r++) {
            const leadIndex = c * 3 + r;
            if (leadIndex >= this.signal.length) continue;
            this._boxes.push({
              leadIndex,
              x: padLeft + c * colW,
              y: padTop + r * rowH,
              w: colW,
              h: rowH,
              tStart: this.startSec + c * segDur,
              tEnd: this.startSec + (c + 1) * segDur,
            });
          }
        }
      } else {
        const n = this.signal.length;
        const rowH = plotH / n;
        for (let i = 0; i < n; i++) {
          this._boxes.push({
            leadIndex: i,
            x: padLeft,
            y: padTop + i * rowH,
            w: plotW,
            h: rowH,
            tStart: this.startSec,
            tEnd: this.startSec + this.windowSec,
          });
        }
      }

      // Compute shared amplitude scale for cross-lead comparability
      let maxAbs = 2.5;
      for (const box of this._boxes) {
        const arr = this.signal[box.leadIndex];
        for (let i = 0; i < arr.length; i += 4) {
          const v = Math.abs(arr[i]);
          if (v > maxAbs) maxAbs = v;
        }
        if (this.perturbed) {
          const parr = this.perturbed[box.leadIndex];
          for (let i = 0; i < parr.length; i += 4) {
            const v = Math.abs(parr[i]);
            if (v > maxAbs) maxAbs = v;
          }
        }
      }
      const yHalfSpan = Math.min(Math.max(maxAbs * 1.15, 2.5), 9.0);

      let maxAttr = 1e-6;
      if (this.attribution) {
        for (let l = 0; l < this.attribution.length; l++) {
          for (let i = 0; i < this.attribution[l].length; i++) {
            const a = Math.abs(this.attribution[l][i]);
            if (a > maxAttr) maxAttr = a;
          }
        }
      }

      for (const box of this._boxes) {
        this._drawBox(ctx, box, pal, yHalfSpan, maxAttr);
      }

      // Bottom time axis ticks
      ctx.fillStyle = pal.axisText;
      ctx.font = "10.5px ui-monospace, monospace";
      const t0 = this.startSec;
      const t1 = this.startSec + this.windowSec;
      const tickStep = this.windowSec <= 2 ? 0.2 : this.windowSec <= 5 ? 0.5 : 1.0;
      const firstTick = Math.ceil(t0 / tickStep) * tickStep;
      for (let t = firstTick; t <= t1 + 1e-6; t += tickStep) {
        const x = padLeft + ((t - t0) / Math.max(t1 - t0, 1e-6)) * plotW;
        const label = `${t.toFixed(1)}s`;
        const drawX = Math.min(x - 10, cssWidth - 34);
        ctx.fillText(label, Math.max(padLeft, drawX), cssHeight - 8);
      }

      ctx.restore();
    }

    _drawBox(ctx, box, pal, yHalfSpan, maxAttr) {
      const { leadIndex, x, y, w, h, tStart, tEnd } = box;
      const spanSec = Math.max(tEnd - tStart, 1e-6);

      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();

      // Standard ECG paper vertical grid: 0.04s minor (1mm @ 25mm/s), 0.20s major (5mm @ 25mm/s)
      const minorSec = 0.04;
      const majorSec = 0.20;
      const pxPerMinor = (minorSec / spanSec) * w;
      if (pxPerMinor >= 4.5) {
        ctx.strokeStyle = pal.minorGrid;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        const firstM = Math.ceil(tStart / minorSec) * minorSec;
        for (let t = firstM; t <= tEnd; t += minorSec) {
          const gx = x + ((t - tStart) / spanSec) * w;
          ctx.moveTo(gx, y);
          ctx.lineTo(gx, y + h);
        }
        ctx.stroke();
      }

      ctx.strokeStyle = pal.majorGrid;
      ctx.lineWidth = 0.85;
      ctx.beginPath();
      const firstMaj = Math.ceil(tStart / majorSec) * majorSec;
      for (let t = firstMaj; t <= tEnd; t += majorSec) {
        const gx = x + ((t - tStart) / spanSec) * w;
        ctx.moveTo(gx, y);
        ctx.lineTo(gx, y + h);
      }
      // Horizontal baseline + +/- half-scale lines
      const yMid = y + h / 2;
      ctx.moveTo(x, yMid);
      ctx.lineTo(x + w, yMid);
      ctx.stroke();

      // Attribution heatmap overlay behind waveform
      const iStart = Math.max(0, Math.floor(tStart * this.samplingRate));
      const iEnd = Math.min(this.signal[0].length - 1, Math.ceil(tEnd * this.samplingRate));
      if (this.attribution && this.attribution[leadIndex]) {
        const attrLead = this.attribution[leadIndex];
        const barW = Math.max(1.5, w / Math.max(iEnd - iStart, 1) + 0.5);
        for (let i = iStart; i <= iEnd; i++) {
          const norm = attrLead[i] / maxAttr;
          if (Math.abs(norm) < 0.08) continue;
          const px = x + ((i / this.samplingRate - tStart) / spanSec) * w;
          const alpha = Math.min(0.65, Math.abs(norm) * 0.72);
          ctx.fillStyle = norm >= 0 ? `rgba(220, 38, 38, ${alpha})` : `rgba(29, 78, 216, ${alpha})`;
          ctx.fillRect(px - barW / 2, y + 1, barW, h - 2);
        }
      }

      // Draw original signal trace
      const drawTrace = (arr, color, lineWidth, dashed = false) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashed ? [4, 2] : []);
        ctx.beginPath();
        let started = false;
        for (let i = iStart; i <= iEnd; i++) {
          const px = x + ((i / this.samplingRate - tStart) / spanSec) * w;
          const py = yMid - (arr[i] / yHalfSpan) * (h * 0.44);
          if (!started) {
            ctx.moveTo(px, py);
            started = true;
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      };

      drawTrace(this.signal[leadIndex], pal.clean, this.layout === "single" ? 1.5 : 1.15, false);
      if (this.perturbed && this.perturbed[leadIndex]) {
        drawTrace(this.perturbed[leadIndex], pal.perturbed, this.layout === "single" ? 1.35 : 1.05, false);
      }

      // Hover caliper vertical crosshair
      if (this.hoverX !== null && this.hoverX >= x && this.hoverX <= x + w) {
        ctx.strokeStyle = pal.cursor;
        ctx.lineWidth = 0.9;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(this.hoverX, y);
        ctx.lineTo(this.hoverX, y + h);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.restore();

      // Box divider border + crisp lead label pill
      ctx.strokeStyle = pal.border;
      ctx.lineWidth = 0.8;
      ctx.strokeRect(x, y, w, h);

      const label = this.leadNames[leadIndex] || `L${leadIndex + 1}`;
      ctx.fillStyle = pal.labelBg;
      ctx.fillRect(x + 3, y + 3, 30, 15);
      ctx.fillStyle = pal.labelText;
      ctx.font = "bold 10.5px ui-monospace, monospace";
      ctx.fillText(label, x + 6, y + 14);

      if (this.layout === "single") {
        ctx.fillStyle = pal.axisText;
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(`+${yHalfSpan.toFixed(1)} ${this.unitLabel}`, 4, y + 14);
        ctx.fillText(`0`, 16, yMid + 3);
        ctx.fillText(`-${yHalfSpan.toFixed(1)} ${this.unitLabel}`, 4, y + h - 4);
      } else if (x <= 50) {
        ctx.fillStyle = pal.axisText;
        ctx.font = "bold 10px ui-monospace, monospace";
        ctx.fillText(label, 8, yMid + 4);
      }
    }
  }

  global.ECGCanvasViewer = ECGCanvasViewer;
})(window);
