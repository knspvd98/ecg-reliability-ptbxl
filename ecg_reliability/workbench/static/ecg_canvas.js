/* Precision High-DPI 12-Lead ECG Canvas, Overview Panner & Dual-Point Caliper */
(function (global) {
  "use strict";

  const THEMES = {
    paper: {
      bg: "#fffdfb",
      gutterBg: "#faf7f5",
      minorGrid: "#f4dedb",
      majorGrid: "#e4bfb9",
      divider: "#e5d5d2",
      clean: "#181d26",
      perturbed: "#c4452b",
      labelText: "#2d3748",
      axisText: "#5a6578",
      cursor: "#1d4f91",
      caliperFill: "rgba(29, 79, 145, 0.08)",
      caliperStroke: "#1d4f91",
      calPulse: "#6b7280",
    },
    clean: {
      bg: "#ffffff",
      gutterBg: "#f8f9fb",
      minorGrid: "#f1f3f7",
      majorGrid: "#dde2ea",
      divider: "#e2e6ee",
      clean: "#141922",
      perturbed: "#c4452b",
      labelText: "#2d3748",
      axisText: "#647084",
      cursor: "#1d4f91",
      caliperFill: "rgba(29, 79, 145, 0.08)",
      caliperStroke: "#1d4f91",
      calPulse: "#718096",
    },
    scope: {
      bg: "#0e141f",
      gutterBg: "#0a0f18",
      minorGrid: "#172233",
      majorGrid: "#25354d",
      divider: "#1f2e44",
      clean: "#e2ecf8",
      perturbed: "#f59e0b",
      labelText: "#cbd5e1",
      axisText: "#8a9bb3",
      cursor: "#60a5fa",
      caliperFill: "rgba(96, 165, 250, 0.14)",
      caliperStroke: "#60a5fa",
      calPulse: "#64748b",
    },
  };

  class ECGCanvasViewer {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.onCaliper = options.onCaliper || null;
      this.onViewChange = options.onViewChange || null;
      this.subsetLeads = options.subsetLeads || null; // e.g. [1, 3, 7, 10]

      this.signal = null;
      this.perturbed = null;
      this.attribution = null;
      this.leadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
      this.samplingRate = 100;
      this.unitLabel = "SD";
      this.layout = options.layout || "stacked"; // stacked | grid4x3 | single
      this.focusLead = 1;
      this.theme = options.theme || "paper";
      this.gainMode = "auto"; // auto | numeric half-span
      this.zoom = 1.0;
      this.startSec = 0.0;
      this.caliperMode = false;
      this.caliperA = null; // { tSec, leadIndex, val }
      this.caliperB = null;
      this.hoverX = null;
      this.hoverY = null;
      this._dragging = false;
      this._dragStartX = 0;
      this._dragOriginStart = 0;
      this._boxes = [];
      this._highlightLead = null;

      this._bindEvents();
      if (typeof ResizeObserver !== "undefined" && this.canvas.parentElement) {
        this._ro = new ResizeObserver(() => this.render());
        this._ro.observe(this.canvas.parentElement);
      }
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
      this.caliperA = null;
      this.caliperB = null;
      this.clampView();
      this.render();
    }

    setView({ layout, focusLead, theme, zoom, startSec, gainMode, caliperMode, highlightLead }) {
      if (layout !== undefined) this.layout = layout;
      if (focusLead !== undefined) this.focusLead = Number(focusLead);
      if (theme !== undefined) this.theme = theme;
      if (zoom !== undefined) this.zoom = Number(zoom);
      if (startSec !== undefined) this.startSec = Number(startSec);
      if (gainMode !== undefined) this.gainMode = gainMode;
      if (caliperMode !== undefined) {
        this.caliperMode = Boolean(caliperMode);
        if (!this.caliperMode) {
          this.caliperA = null;
          this.caliperB = null;
        }
      }
      if (highlightLead !== undefined) this._highlightLead = highlightLead;
      this.clampView();
      this.render();
    }

    clampView() {
      this.zoom = Math.max(1.0, Math.min(8.0, this.zoom));
      const maxStart = Math.max(0.0, this.durationSec - this.windowSec);
      this.startSec = Math.max(0.0, Math.min(maxStart, this.startSec));
    }

    _hitBox(x, y) {
      return this._boxes.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
    }

    _bindEvents() {
      this.canvas.addEventListener("mousemove", (ev) => {
        const rect = this.canvas.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        if (this._dragging && !this.caliperMode) {
          const dx = x - this._dragStartX;
          const refBox = this._boxes[0];
          const w = refBox ? refBox.w : rect.width;
          this.startSec = this._dragOriginStart - (dx / Math.max(w, 1)) * this.windowSec;
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
        if (!this.signal) return;
        const rect = this.canvas.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        if (this.caliperMode) {
          const hit = this._hitBox(x, y);
          if (!hit) return;
          const frac = Math.max(0, Math.min(1, (x - hit.x) / Math.max(hit.w, 1)));
          const tSec = hit.tStart + frac * (hit.tEnd - hit.tStart);
          const idx = Math.max(0, Math.min(this.signal[0].length - 1, Math.round(tSec * this.samplingRate)));
          const pt = { tSec, leadIndex: hit.leadIndex, val: this.signal[hit.leadIndex][idx] };
          if (!this.caliperA || (this.caliperA && this.caliperB)) {
            this.caliperA = pt;
            this.caliperB = null;
          } else {
            this.caliperB = pt;
          }
          this.render();
          this._emitCaliper(x, y);
          return;
        }
        this._dragging = true;
        this._dragStartX = x;
        this._dragOriginStart = this.startSec;
      });

      window.addEventListener("mouseup", () => {
        this._dragging = false;
      });

      this.canvas.addEventListener(
        "wheel",
        (ev) => {
          if (!this.signal) return;
          ev.preventDefault();
          const factor = ev.deltaY < 0 ? 1.25 : 0.8;
          this.zoom = Math.max(1.0, Math.min(8.0, this.zoom * factor));
          this.clampView();
          if (this.onViewChange) this.onViewChange({ zoom: this.zoom, startSec: this.startSec });
          this.render();
        },
        { passive: false }
      );
    }

    _emitCaliper(x, y) {
      if (!this.onCaliper || !this.signal) return;
      const hit = this._hitBox(x, y);
      if (!hit) return;
      const frac = Math.max(0, Math.min(1, (x - hit.x) / Math.max(hit.w, 1)));
      const tSec = hit.tStart + frac * (hit.tEnd - hit.tStart);
      const sampleIdx = Math.max(0, Math.min(this.signal[0].length - 1, Math.round(tSec * this.samplingRate)));
      const cleanVal = this.signal[hit.leadIndex][sampleIdx];
      const pertVal = this.perturbed ? this.perturbed[hit.leadIndex][sampleIdx] : null;
      const attrVal = this.attribution ? this.attribution[hit.leadIndex][sampleIdx] : null;
      let delta = null;
      if (this.caliperA && this.caliperB) {
        const dtMs = Math.abs(this.caliperB.tSec - this.caliperA.tSec) * 1000;
        const dv = this.caliperB.val - this.caliperA.val;
        const bpm = dtMs > 120 ? 60000 / dtMs : null;
        delta = { dtMs, dv, bpm };
      }
      this.onCaliper({
        leadName: this.leadNames[hit.leadIndex],
        tSec,
        sampleIdx,
        cleanVal,
        pertVal,
        attrVal,
        unitLabel: this.unitLabel,
        delta,
      });
    }

    render() {
      const parent = this.canvas.parentElement;
      const cssWidth = Math.max(300, (parent ? parent.clientWidth : this.canvas.clientWidth) || 900);
      const cssHeight = Math.max(220, (parent ? parent.clientHeight : this.canvas.clientHeight) || 460);
      const dpr = window.devicePixelRatio || 1;
      if (this.canvas.width !== Math.round(cssWidth * dpr) || this.canvas.height !== Math.round(cssHeight * dpr)) {
        this.canvas.width = Math.round(cssWidth * dpr);
        this.canvas.height = Math.round(cssHeight * dpr);
      }

      const ctx = this.ctx;
      ctx.save();
      ctx.scale(dpr, dpr);

      const pal = THEMES[this.theme] || THEMES.paper;
      ctx.fillStyle = pal.bg;
      ctx.fillRect(0, 0, cssWidth, cssHeight);

      if (!this.signal || !this.signal.length) {
        ctx.fillStyle = pal.axisText;
        ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillText("No ECG waveform loaded.", 20, 28);
        ctx.restore();
        return;
      }

      const padLeft = 42;
      const padRight = 24;
      const padTop = 6;
      const padBottom = 22;
      const plotW = cssWidth - padLeft - padRight;
      const plotH = cssHeight - padTop - padBottom;

      // Left gutter background
      ctx.fillStyle = pal.gutterBg;
      ctx.fillRect(0, 0, padLeft, cssHeight - padBottom);
      ctx.strokeStyle = pal.divider;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft + 0.5, 0);
      ctx.lineTo(padLeft + 0.5, cssHeight - padBottom);
      ctx.stroke();

      const leadIndices = this.subsetLeads
        ? this.subsetLeads.filter((i) => i < this.signal.length)
        : Array.from({ length: this.signal.length }, (_, i) => i);

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
          showInlineLabel: false,
        });
      } else if (this.layout === "grid4x3" && !this.subsetLeads) {
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
              showInlineLabel: c > 0, // Col 0 uses gutter label only (no duplication!)
            });
          }
        }
      } else {
        const n = leadIndices.length;
        const rowH = plotH / n;
        leadIndices.forEach((leadIndex, rowIdx) => {
          this._boxes.push({
            leadIndex,
            x: padLeft,
            y: padTop + rowIdx * rowH,
            w: plotW,
            h: rowH,
            tStart: this.startSec,
            tEnd: this.startSec + this.windowSec,
            showInlineLabel: false,
          });
        });
      }

      let yHalfSpan;
      if (this.gainMode !== "auto" && Number.isFinite(Number(this.gainMode))) {
        yHalfSpan = Number(this.gainMode);
      } else {
        let maxAbs = 2.4;
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
        yHalfSpan = Math.min(Math.max(maxAbs * 1.12, 2.4), 8.5);
      }

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

      // Bottom time axis
      ctx.fillStyle = pal.gutterBg;
      ctx.fillRect(0, cssHeight - padBottom, cssWidth, padBottom);
      ctx.strokeStyle = pal.divider;
      ctx.beginPath();
      ctx.moveTo(0, cssHeight - padBottom + 0.5);
      ctx.lineTo(cssWidth, cssHeight - padBottom + 0.5);
      ctx.stroke();

      ctx.fillStyle = pal.axisText;
      ctx.font = "10.5px -apple-system, BlinkMacSystemFont, sans-serif";
      const t0 = this.startSec;
      const t1 = this.startSec + this.windowSec;
      const tickStep = this.windowSec <= 1.5 ? 0.2 : this.windowSec <= 4 ? 0.5 : 1.0;
      const firstTick = Math.ceil(t0 / tickStep) * tickStep;
      for (let t = firstTick; t <= t1 + 1e-6; t += tickStep) {
        const x = padLeft + ((t - t0) / Math.max(t1 - t0, 1e-6)) * plotW;
        const label = `${t.toFixed(1)} s`;
        const w = ctx.measureText(label).width;
        const drawX = Math.max(padLeft + 2, Math.min(x - w / 2, cssWidth - w - 6));
        ctx.fillText(label, drawX, cssHeight - 7);
      }

      // Unit scale note in bottom-left corner
      ctx.fillStyle = pal.axisText;
      ctx.font = "600 10px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText(this.unitLabel, 8, cssHeight - 7);

      ctx.restore();
    }

    _drawBox(ctx, box, pal, yHalfSpan, maxAttr) {
      const { leadIndex, x, y, w, h, tStart, tEnd, showInlineLabel } = box;
      const spanSec = Math.max(tEnd - tStart, 1e-6);
      const yMid = y + h / 2;

      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();

      // Highlighted lead subtle wash
      if (this._highlightLead === leadIndex) {
        ctx.fillStyle = "rgba(29, 79, 145, 0.05)";
        ctx.fillRect(x, y, w, h);
      }

      // 40 ms minor grid (1 mm @ 25 mm/s)
      const minorSec = 0.04;
      const majorSec = 0.20;
      const pxPerMinor = (minorSec / spanSec) * w;
      if (pxPerMinor >= 4.2) {
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

      // 200 ms major grid (5 mm @ 25 mm/s) + baseline
      ctx.strokeStyle = pal.majorGrid;
      ctx.lineWidth = 0.85;
      ctx.beginPath();
      const firstMaj = Math.ceil(tStart / majorSec) * majorSec;
      for (let t = firstMaj; t <= tEnd; t += majorSec) {
        const gx = x + ((t - tStart) / spanSec) * w;
        ctx.moveTo(gx, y);
        ctx.lineTo(gx, y + h);
      }
      ctx.moveTo(x, yMid);
      ctx.lineTo(x + w, yMid);
      ctx.stroke();

      // Standard 1.0-unit × 0.20s calibration reference pulse on first box (when not scrolled)
      if (box === this._boxes[0] && tStart < 0.02) {
        const pulseH = (1.0 / yHalfSpan) * (h * 0.44);
        const pw = Math.max(5, Math.min(14, (0.16 / spanSec) * w));
        ctx.strokeStyle = pal.calPulse;
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.moveTo(x + 3, yMid);
        ctx.lineTo(x + 5, yMid);
        ctx.lineTo(x + 5, yMid - pulseH);
        ctx.lineTo(x + 5 + pw, yMid - pulseH);
        ctx.lineTo(x + 5 + pw, yMid);
        ctx.lineTo(x + 7 + pw, yMid);
        ctx.stroke();
      }

      // Colorblind-safe signed attribution overlay (positive ABOVE baseline in vermilion, negative BELOW in cerulean)
      const iStart = Math.max(0, Math.floor(tStart * this.samplingRate));
      const iEnd = Math.min(this.signal[0].length - 1, Math.ceil(tEnd * this.samplingRate));
      if (this.attribution && this.attribution[leadIndex]) {
        const attrLead = this.attribution[leadIndex];
        const barW = Math.max(1.8, w / Math.max(iEnd - iStart, 1) + 0.5);
        for (let i = iStart; i <= iEnd; i++) {
          const norm = attrLead[i] / maxAttr;
          if (Math.abs(norm) < 0.07) continue;
          const px = x + ((i / this.samplingRate - tStart) / spanSec) * w;
          const alpha = Math.min(0.62, 0.14 + Math.abs(norm) * 0.52);
          const bandH = (h * 0.45) * Math.min(1, 0.35 + Math.abs(norm) * 0.65);
          if (norm >= 0) {
            ctx.fillStyle = `rgba(196, 69, 43, ${alpha})`;
            ctx.fillRect(px - barW / 2, yMid - bandH, barW, bandH);
          } else {
            ctx.fillStyle = `rgba(30, 95, 143, ${alpha})`;
            ctx.fillRect(px - barW / 2, yMid, barW, bandH);
          }
        }
      }

      // Dual-point caliper shaded measurement span
      if (this.caliperA) {
        const ax = x + ((this.caliperA.tSec - tStart) / spanSec) * w;
        if (this.caliperB) {
          const bx = x + ((this.caliperB.tSec - tStart) / spanSec) * w;
          ctx.fillStyle = pal.caliperFill;
          ctx.fillRect(Math.min(ax, bx), y, Math.abs(bx - ax), h);
          ctx.strokeStyle = pal.caliperStroke;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(bx, y);
          ctx.lineTo(bx, y + h);
          ctx.stroke();
        }
        ctx.strokeStyle = pal.caliperStroke;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, y);
        ctx.lineTo(ax, y + h);
        ctx.stroke();
      }

      // Waveform traces
      const drawTrace = (arr, color, lineWidth) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineJoin = "round";
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
      };

      drawTrace(this.signal[leadIndex], pal.clean, this.layout === "single" ? 1.45 : 1.1);
      if (this.perturbed && this.perturbed[leadIndex]) {
        drawTrace(this.perturbed[leadIndex], pal.perturbed, this.layout === "single" ? 1.3 : 1.0);
      }

      // Hover crosshair
      if (this.hoverX !== null && this.hoverX >= x && this.hoverX <= x + w) {
        ctx.strokeStyle = pal.cursor;
        ctx.lineWidth = 0.85;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(this.hoverX, y);
        ctx.lineTo(this.hoverX, y + h);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.restore();

      // Subtle hairline separator between rows/columns
      ctx.strokeStyle = pal.divider;
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, y + h + 0.5);
      ctx.lineTo(x + w, y + h + 0.5);
      if (this.layout === "grid4x3") {
        ctx.moveTo(x + w + 0.5, y);
        ctx.lineTo(x + w + 0.5, y + h);
      }
      ctx.stroke();

      // Single authoritative lead label (gutter for col 0/stacked, quiet inline for cols 1..3 in 4x3)
      const label = this.leadNames[leadIndex] || `L${leadIndex + 1}`;
      ctx.fillStyle = pal.labelText;
      ctx.font = "600 11px -apple-system, BlinkMacSystemFont, sans-serif";
      if (showInlineLabel) {
        ctx.fillText(label, x + 6, y + 14);
      } else {
        ctx.fillText(label, 9, yMid + 4);
      }

      if (this.layout === "single") {
        ctx.fillStyle = pal.axisText;
        ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillText(`+${yHalfSpan.toFixed(1)}`, 6, y + 14);
        ctx.fillText(`-${yHalfSpan.toFixed(1)}`, 6, y + h - 5);
      }
    }
  }

  /* MATLAB Signal Analyzer-style Full-Record Overview Panner */
  class ECGPanner {
    constructor(canvas, viewer) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.viewer = viewer;
      this._dragging = false;
      this._bind();
      if (typeof ResizeObserver !== "undefined" && canvas.parentElement) {
        this._ro = new ResizeObserver(() => this.render());
        this._ro.observe(canvas.parentElement);
      }
    }

    _bind() {
      const updateFromClientX = (clientX) => {
        if (!this.viewer.signal) return;
        const rect = this.canvas.getBoundingClientRect();
        const padLeft = 42;
        const padRight = 24;
        const plotW = Math.max(10, rect.width - padLeft - padRight);
        const frac = Math.max(0, Math.min(1, (clientX - rect.left - padLeft) / plotW));
        const centerSec = frac * this.viewer.durationSec;
        this.viewer.setView({ startSec: centerSec - this.viewer.windowSec / 2 });
        if (this.viewer.onViewChange) {
          this.viewer.onViewChange({ zoom: this.viewer.zoom, startSec: this.viewer.startSec });
        }
        this.render();
      };

      this.canvas.addEventListener("mousedown", (ev) => {
        this._dragging = true;
        updateFromClientX(ev.clientX);
      });
      window.addEventListener("mousemove", (ev) => {
        if (this._dragging) updateFromClientX(ev.clientX);
      });
      window.addEventListener("mouseup", () => {
        this._dragging = false;
      });
    }

    render() {
      const parent = this.canvas.parentElement;
      const cssWidth = Math.max(300, (parent ? parent.clientWidth : 900) || 900);
      const cssHeight = Math.max(36, (parent ? parent.clientHeight : 44) || 44);
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = Math.round(cssWidth * dpr);
      this.canvas.height = Math.round(cssHeight * dpr);

      const ctx = this.ctx;
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.fillStyle = "#fafbfc";
      ctx.fillRect(0, 0, cssWidth, cssHeight);

      if (!this.viewer.signal || !this.viewer.signal.length) {
        ctx.restore();
        return;
      }

      const padLeft = 42;
      const padRight = 24;
      const plotW = cssWidth - padLeft - padRight;
      const leadIdx = Math.min(1, this.viewer.signal.length - 1);
      const arr = this.viewer.signal[leadIdx];

      ctx.fillStyle = "#6c7789";
      ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText("II", 12, cssHeight / 2 + 3);

      let maxAbs = 2.0;
      for (let i = 0; i < arr.length; i += 2) {
        const v = Math.abs(arr[i]);
        if (v > maxAbs) maxAbs = v;
      }

      ctx.strokeStyle = "#526074";
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const px = padLeft + (i / Math.max(arr.length - 1, 1)) * plotW;
        const py = cssHeight / 2 - (arr[i] / (maxAbs * 1.15)) * (cssHeight * 0.4);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Viewport selection window
      const dur = this.viewer.durationSec;
      const wx = padLeft + (this.viewer.startSec / dur) * plotW;
      const ww = Math.max(10, (this.viewer.windowSec / dur) * plotW);
      ctx.fillStyle = "rgba(29, 79, 145, 0.13)";
      ctx.fillRect(wx, 2, ww, cssHeight - 4);
      ctx.strokeStyle = "#1d4f91";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(wx + 0.5, 2.5, ww - 1, cssHeight - 5);

      ctx.restore();
    }
  }

  global.ECGCanvasViewer = ECGCanvasViewer;
  global.ECGPanner = ECGPanner;
})(window);
