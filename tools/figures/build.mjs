#!/usr/bin/env node
// Generates every SVG in docs/assets: the logo, the two block diagrams, and
// the result plots. No dependencies; run with `node tools/figures/build.mjs`.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { bridgeTiming, expandAB, p128HostTimerCandSec, p128Throughput } from "./data.mjs";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "docs", "assets");

const SANS = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif";
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

const C = {
  surface: "#ffffff",
  panel: "#fafaf8",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  rule: "#c3c2b7",
  accent: "#2a78d6",
  accentWash: "#eef5fd",
  // Candidate identities in the batch strip (validated categorical order).
  candidates: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
};

// ---------------------------------------------------------------------------
// SVG primitives
// ---------------------------------------------------------------------------

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const n = (v) => Number(v.toFixed(2));

function svg(w, h, title, body, { background = C.surface } = {}) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-labelledby="title">`,
    `<title id="title">${esc(title)}</title>`,
    background ? `<rect width="${w}" height="${h}" fill="${background}"/>` : "",
    body,
    "</svg>",
    "",
  ].join("\n");
}

// `raw: true` lets callers pass pre-built <tspan> markup (for subscripts).
function text(x, y, s, o = {}) {
  const { size = 13, weight = 400, fill = C.ink, anchor = "start", family = SANS, raw = false, extra = "" } = o;
  const content = raw ? s : esc(s);
  return `<text x="${n(x)}" y="${n(y)}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}"${extra}>${content}</text>`;
}

const sub = (s) => `<tspan font-size="0.75em" dy="0.3em">${esc(s)}</tspan><tspan dy="-0.3em"></tspan>`;

function rect(x, y, w, h, { fill = C.surface, stroke = C.rule, sw = 1, rx = 8 } = {}) {
  return `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

const ARROW_DEFS = `<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="${C.ink2}"/></marker>
<marker id="arrow-accent" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="${C.accent}"/></marker>
</defs>`;

function line(d, { accent = false, head = true, sw = 1.25 } = {}) {
  const color = accent ? C.accent : C.ink2;
  const marker = head ? ` marker-end="url(#${accent ? "arrow-accent" : "arrow"})"` : "";
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round"${marker}/>`;
}

// A labelled box. `label` may be raw markup when `raw` is set.
function node(x, y, w, h, label, detail, o = {}) {
  const { accent = false, mono = false, raw = false } = o;
  const style = accent ? { fill: C.accentWash, stroke: C.accent, sw: 1.25 } : {};
  const cx = x + w / 2;
  const cy = y + h / 2;
  const lines = detail ? (Array.isArray(detail) ? detail : [detail]) : [];
  const block = 16 + lines.length * 14;
  const firstY = cy - block / 2 + 12;
  let out = rect(x, y, w, h, style);
  out += text(cx, firstY, label, { anchor: "middle", weight: 600, family: mono ? MONO : SANS, raw });
  lines.forEach((d, i) => {
    out += text(cx, firstY + 16 + i * 14, d, { anchor: "middle", size: 11, fill: C.ink2 });
  });
  return out;
}

function plus(cx, cy, { accent = false } = {}) {
  const color = accent ? C.accent : C.ink2;
  return (
    `<circle cx="${cx}" cy="${cy}" r="11" fill="${accent ? C.accentWash : C.surface}" stroke="${color}" stroke-width="1.25"/>` +
    `<path d="M${cx - 5},${cy} H${cx + 5} M${cx},${cy - 5} V${cy + 5}" stroke="${color}" stroke-width="1.5" stroke-linecap="round"/>`
  );
}

// ---------------------------------------------------------------------------
// Logo: a base point, a cloud of deterministic perturbations, three selected.
// Dot positions come from the same murmur3 finalizer style hash the kernels use.
// ---------------------------------------------------------------------------

function fmix32(h) {
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}
const unit = (seed, i) => fmix32((fmix32(i + 1) ^ seed) >>> 0) / 2 ** 32;

function markBody({ cx, cy, ink, dot, accent }) {
  const count = 24;
  const golden = Math.PI * (3 - Math.sqrt(5));
  const pts = [];
  for (let i = 1; i <= count; i++) {
    const r = 8.5 + 16.5 * Math.sqrt(i / count) + (unit(7, i) - 0.5) * 2;
    const a = i * golden - Math.PI / 2 + (unit(11, i) - 0.5) * 0.3;
    pts.push({ i, x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), r: 1.45 + unit(3, i) * 0.9 });
  }
  const chosen = new Set([16, 17, 23]);
  let out = "";
  for (const p of pts) {
    if (chosen.has(p.i)) {
      out += `<path d="M${cx},${cy} L${n(p.x)},${n(p.y)}" stroke="${accent}" stroke-width="1.3" stroke-linecap="round" opacity="0.6"/>`;
    }
  }
  for (const p of pts) {
    const on = chosen.has(p.i);
    out += `<circle cx="${n(p.x)}" cy="${n(p.y)}" r="${n(on ? 2.9 : p.r)}" fill="${on ? accent : dot}"/>`;
  }
  out += `<circle cx="${cx}" cy="${cy}" r="4.6" fill="${ink}"/>`;
  return out;
}

function logo(theme) {
  const ink = theme === "dark" ? "#f0efec" : C.ink;
  const dot = theme === "dark" ? "#6f6d66" : "#c3c2b7";
  const accent = theme === "dark" ? "#3987e5" : C.accent;
  const body =
    markBody({ cx: 32, cy: 32, ink, dot, accent }) +
    text(74, 43, "optimus", { size: 31, weight: 600, fill: ink, extra: ` letter-spacing="-0.6"` });
  return svg(206, 64, "Optimus", body, { background: null });
}

function mark() {
  return svg(64, 64, "Optimus", markBody({ cx: 32, cy: 32, ink: C.ink, dot: "#c3c2b7", accent: C.accent }), {
    background: null,
  });
}

// ---------------------------------------------------------------------------
// Diagram 1: where the perturbation enters one decoder layer.
// ---------------------------------------------------------------------------

function blockDiagram() {
  const W = 960;
  const H = 318;
  const midY = 190;
  const boxH = 52;
  const gap = 22;
  const parts = [
    { id: "norm1", w: 84, label: "RMSNorm" },
    { id: "qkv", w: 104, label: "qkv_proj", detail: "W·x", mono: true },
    { id: "add", w: 22, sum: true, accent: true },
    { id: "attn", w: 128, label: "Attention", detail: "KV cache per row" },
    { id: "o", w: 84, label: "o_proj", mono: true },
    { id: "res1", w: 22, sum: true },
    { id: "norm2", w: 84, label: "RMSNorm" },
    { id: "mlp", w: 128, label: "MLP", detail: "gate · up → down" },
    { id: "res2", w: 22, sum: true },
  ];
  let x = 48;
  const at = {};
  for (const p of parts) {
    at[p.id] = { x, w: p.w, cx: x + p.w / 2 };
    x += p.w + gap;
  }

  let body = ARROW_DEFS;
  body += text(24, 30, "One decoder layer, qv target preset", { size: 12, fill: C.ink2 });

  // Main path.
  body += text(22, midY + 4.5, "x", { family: MONO, weight: 600 });
  body += line(`M34,${midY} H${at.norm1.x - 2}`);
  for (const p of parts) {
    const a = at[p.id];
    if (p.sum) {
      body += plus(a.cx, midY, { accent: p.accent });
    } else {
      body += node(a.x, midY - boxH / 2, p.w, boxH, p.label, p.detail, { mono: p.mono });
    }
  }
  for (let i = 0; i < parts.length - 1; i++) {
    const a = at[parts[i].id];
    const b = at[parts[i + 1].id];
    body += line(`M${a.x + a.w},${midY} H${b.x - 2}`);
  }
  const last = at.res2;
  body += line(`M${last.x + last.w},${midY} H${last.x + last.w + 22}`);
  body += text(last.x + last.w + 28, midY + 4.5, "out", { size: 12, fill: C.ink2 });

  // Residual stream.
  const r1 = midY + 58;
  const r2 = midY + 64;
  body += line(`M40,${midY} V${r1} H${at.res1.cx} V${midY + 13}`);
  const tap2 = at.res1.x + at.res1.w + gap / 2;
  body += line(`M${tap2},${midY} V${r2} H${at.res2.cx} V${midY + 13}`);
  body += text(at.attn.cx, r1 + 15, "residual", { size: 11, fill: C.muted, anchor: "middle" });

  // Perturbation branch: shrink with the shared basis, expand with the candidate field.
  const tapX = at.norm1.x + at.norm1.w + gap / 2;
  const branchY = 52;
  const branchH = 50;
  const b1 = { x: tapX - 30, w: 136 };
  const b2 = { x: b1.x + b1.w + 22, w: 230 };
  body += node(b1.x, branchY, b1.w, branchH, `z = Q·x`, "shared basis, rank r", { accent: true, mono: true });
  body += node(b2.x, branchY, b2.w, branchH, `β·G${sub("c")}·z`, "per candidate, generated in-kernel", {
    accent: true,
    mono: true,
    raw: true,
  });
  body += `<circle cx="${tapX}" cy="${midY}" r="2.6" fill="${C.accent}"/>`;
  body += line(`M${tapX},${midY - 3} V${branchY + branchH + 2}`, { accent: true });
  body += line(`M${b1.x + b1.w},${branchY + branchH / 2} H${b2.x - 2}`, { accent: true });
  body += line(`M${at.add.cx},${branchY + branchH} V${midY - 13}`, { accent: true });
  body += text(at.add.cx + 8, midY - 30, "+δ on q, v", { size: 11, fill: C.accent });

  // Footer formula.
  body += text(24, H - 16, `y = W·x + β·G${sub("c")}·Q·x`, { family: MONO, size: 13, weight: 600, raw: true });
  body += text(206, H - 16, "c is the candidate that owns the row. Q is shared by every candidate; G is rebuilt from the seed, never stored.", {
    size: 12,
    fill: C.ink2,
  });

  return svg(W, H, "Where an Optimus perturbation enters a decoder layer", body);
}

// ---------------------------------------------------------------------------
// Diagram 2: the systems view. One model copy, many candidates per batch.
// ---------------------------------------------------------------------------

function systemDiagram() {
  const W = 960;
  const H = 412;
  let body = ARROW_DEFS;

  // Column headers.
  body += text(24, 34, "Search", { size: 12, weight: 600, fill: C.ink2 });
  body += text(276, 34, "vLLM: one model copy, one continuous batch", { size: 12, weight: 600, fill: C.ink2 });
  body += text(748, 34, "Selection", { size: 12, weight: 600, fill: C.ink2 });

  // Left column.
  body += node(24, 62, 200, 76, "Candidate panel", ["seed, sign and scale β", "P = 128 to 4,096"]);
  body += node(24, 178, 200, 76, "Activation bases Q", ["one per site, from an SVD", "of calibration activations"]);

  // vLLM container.
  body += rect(260, 48, 448, 330, { fill: C.panel, stroke: C.grid, rx: 12 });

  // Scheduler with the row strip.
  body += rect(280, 62, 408, 98, {});
  body += text(296, 84, "Scheduler", { weight: 600 });
  body += text(296, 101, "prompts × candidates packed into one batch", { size: 11, fill: C.ink2 });
  const ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
  ids.forEach((c, i) => {
    const sx = 296 + i * 22;
    body += `<rect x="${sx}" y="116" width="18" height="18" rx="3" fill="${C.candidates[c]}"/>`;
    if (i % 3 === 1) body += text(sx + 9, 150, `c${c + 1}`, { size: 10, fill: C.ink2, anchor: "middle", family: MONO });
  });
  body += text(568, 129, "row → candidate id", { size: 11, fill: C.ink2 });

  // Forward pass lanes.
  body += node(280, 196, 196, 72, "Lazy delta kernel", ["Q·x, then G per row,", "added in place"], { accent: true });
  body += node(492, 196, 196, 72, "Base linears", ["W·x, one GEMM", "shared by every row"]);
  body += node(280, 302, 408, 56, "KV cache", "kept per row; prefix cache off, because KV depends on the candidate");

  // Arrows inside and into vLLM.
  body += line(`M224,100 H278`);
  body += line(`M224,216 H278`, { accent: true });
  body += line(`M378,160 V194`);
  body += line(`M590,160 V194`);
  body += line(`M378,268 V300`);
  body += line(`M590,268 V300`);

  // Right column.
  body += node(748, 62, 188, 60, "Scores", "per candidate, per prompt");
  body += node(748, 158, 188, 60, "Top-K ensemble", "kept lazy: seeds, not weights");
  body += node(748, 254, 188, 60, "Confirmation", "Transformers or LightEval");
  body += line(`M708,92 H746`);
  body += text(727, 84, "tokens", { size: 10, fill: C.muted, anchor: "middle" });
  body += line(`M842,122 V156`);
  body += line(`M842,218 V252`);

  body += text(24, H - 12, "Every candidate shares the base weights and their GEMMs; only the rank-r delta differs per row, and it is computed, not loaded.", {
    size: 12,
    fill: C.ink2,
  });

  return svg(W, H, "Optimus systems view: candidates batched inside vLLM", body);
}

// ---------------------------------------------------------------------------
// Plots
// ---------------------------------------------------------------------------

function legend(x, y, items) {
  let out = "";
  let cursor = x;
  for (const it of items) {
    out += `<rect x="${cursor}" y="${y - 9}" width="10" height="10" rx="2" fill="${it.color}"/>`;
    out += text(cursor + 16, y, it.label, { size: 12, fill: C.ink2 });
    cursor += 16 + it.label.length * 6.6 + 22;
  }
  return out;
}

// Horizontal bar: square at the baseline, 4px rounded data end.
function hbar(x0, y, x1, h, color) {
  const r = Math.min(4, (x1 - x0) / 2);
  return `<path d="M${n(x0)},${n(y)} H${n(x1 - r)} Q${n(x1)},${n(y)} ${n(x1)},${n(y + r)} V${n(y + h - r)} Q${n(x1)},${n(y + h)} ${n(x1 - r)},${n(y + h)} H${n(x0)} Z" fill="${color}"/>`;
}

// Vertical column: square at the baseline, 4px rounded top.
function vbar(x, y0, y1, w, color) {
  const r = Math.min(4, (y0 - y1) / 2, w / 2);
  return `<path d="M${n(x)},${n(y0)} V${n(y1 + r)} Q${n(x)},${n(y1)} ${n(x + r)},${n(y1)} H${n(x + w - r)} Q${n(x + w)},${n(y1)} ${n(x + w)},${n(y1 + r)} V${n(y0)} Z" fill="${color}"/>`;
}

function hbarChart({ title, subtitle, rows, max, ticks, tickFmt, valueFmt, legendItems, notes = [] }) {
  const W = 720;
  const left = 268;
  const right = 640;
  const pitch = rows.some((r) => r.detail) ? 42 : 32;
  const barH = 18;
  const top = legendItems ? 104 : 80;
  const plotH = rows.length * pitch;
  const H = top + plotH + 30 + notes.length * 18 + 14;
  const x = (v) => left + (v / max) * (right - left);

  let body = "";
  body += text(24, 32, title, { size: 15, weight: 600 });
  body += text(24, 52, subtitle, { size: 12, fill: C.ink2 });
  if (legendItems) body += legend(24, 80, legendItems);

  for (const t of ticks) {
    const tx = n(x(t)) + 0.5;
    body += `<path d="M${tx},${top - 8} V${top + plotH}" stroke="${t === 0 ? C.rule : C.grid}" stroke-width="1"/>`;
    body += text(tx, top + plotH + 18, tickFmt(t), { size: 11, fill: C.muted, anchor: "middle", extra: ` font-variant-numeric="tabular-nums"` });
  }

  rows.forEach((r, i) => {
    const y = top + i * pitch + (pitch - barH) / 2 - 4;
    const labelY = r.detail ? y + 7 : y + barH / 2 + 4.5;
    body += text(left - 12, labelY, r.label, { size: 12.5, anchor: "end" });
    if (r.detail) body += text(left - 12, labelY + 15, r.detail, { size: 11, fill: C.muted, anchor: "end" });
    body += hbar(x(0) + 1, y, x(r.value), barH, r.color);
    body += text(x(r.value) + 8, y + barH / 2 + 4.5, valueFmt(r.value), { size: 12, weight: 600, fill: C.ink });
  });

  notes.forEach((note, i) => {
    body += text(24, top + plotH + 46 + i * 18, note, { size: 11, fill: C.muted });
  });
  return svg(W, H, title, body);
}

function timeBreakdownPlot() {
  const lazy = bridgeTiming.lazyDeltaSeconds;
  const rest = bridgeTiming.totalSeconds - lazy;
  const share = Math.round((100 * lazy) / bridgeTiming.totalSeconds);
  return hbarChart({
    title: "Where p128 replay time goes on the vLLM LoRA-kernel bridge",
    subtitle: `A100 80GB, population 128, rank 64 on q and v, ${bridgeTiming.totalSeconds.toFixed(1)} s total`,
    rows: bridgeTiming.parts.map((p) => ({ label: p.label, value: p.seconds, color: p.lazy ? C.accent : C.muted })),
    max: 50,
    ticks: [0, 10, 20, 30, 40, 50],
    tickFmt: (t) => `${t} s`,
    valueFmt: (v) => `${v.toFixed(1)} s`,
    legendItems: [
      { label: `perturbation path, ${lazy.toFixed(1)} s (${share}%)`, color: C.accent },
      { label: `everything else, ${rest.toFixed(1)} s`, color: C.muted },
    ],
    notes: ["Source: KERNEL.md, bridge p128 timing table."],
  });
}

function throughputPlot() {
  return hbarChart({
    title: "p128 replay throughput by perturbation backend",
    subtitle: "Candidates per second, higher is better. L40S 48GB, population 128, 8 prompts, rank 64 on q and v",
    rows: p128Throughput.map((r) => ({
      label: r.label,
      detail: r.detail,
      value: r.candSec,
      color: r.ours ? C.accent : C.muted,
    })),
    max: 16,
    ticks: [0, 4, 8, 12, 16],
    tickFmt: (t) => `${t}`,
    valueFmt: (v) => `${v.toFixed(1)} cand/s`,
    legendItems: [
      { label: "Optimus Triton kernels", color: C.accent },
      { label: "vLLM LoRA kernels, used as a bridge", color: C.muted },
    ],
    notes: [
      `Measured with a per-phase sync timer. With the host timer the in-place kernel runs at ${p128HostTimerCandSec.toFixed(1)} cand/s.`,
      "Source: KERNEL.md, L40S end-to-end tables.",
    ],
  });
}

function expandPlot() {
  const W = 720;
  const H = 392;
  const left = 64;
  const right = 696;
  const top = 112;
  const bottom = 306;
  const max = 3;
  const y = (v) => bottom - (v / max) * (bottom - top);
  const band = (right - left) / expandAB.length;
  const barW = 24;

  let body = "";
  body += text(24, 32, "Expand step: generating the random field in-kernel vs loading it", { size: 15, weight: 600 });
  body += text(24, 52, "Triton on L40S 48GB, 16 candidates; milliseconds, lower is better", { size: 12, fill: C.ink2 });
  body += legend(24, 80, [
    { label: "generate G in-kernel (counter RNG)", color: C.accent },
    { label: "load stored factors B", color: C.muted },
  ]);

  for (const t of [0, 1, 2, 3]) {
    const ty = n(y(t)) + 0.5;
    body += `<path d="M${left},${ty} H${right}" stroke="${t === 0 ? C.rule : C.grid}" stroke-width="1"/>`;
    body += text(left - 10, ty + 4, `${t}`, { size: 11, fill: C.muted, anchor: "end" });
  }

  expandAB.forEach((d, i) => {
    const cx = left + band * (i + 0.5);
    const xa = cx - barW - 1;
    const xb = cx + 1;
    body += vbar(xa, y(0) - 0.5, y(d.counterMs), barW, C.accent);
    body += vbar(xb, y(0) - 0.5, y(d.storedMs), barW, C.muted);
    const ratio = d.storedMs / d.counterMs;
    const peak = Math.min(y(d.counterMs), y(d.storedMs));
    body += text(cx, peak - 10, `${ratio.toFixed(1)}×`, { size: 12, weight: 600, anchor: "middle" });
    body += text(cx, bottom + 20, `${d.rows} rows, rank ${d.rank}`, { size: 11, fill: C.ink2, anchor: "middle" });
    body += text(cx, bottom + 35, `output ${d.out}`, { size: 11, fill: C.muted, anchor: "middle" });
  });

  body += text(24, H - 14, "Labels are stored ÷ in-kernel latency. Source: KERNEL.md, stateless-counter expand table.", {
    size: 11,
    fill: C.muted,
  });
  return svg(W, H, "Expand step latency: in-kernel random field vs stored factors", body);
}

// ---------------------------------------------------------------------------

const files = {
  "logo-light.svg": logo("light"),
  "logo-dark.svg": logo("dark"),
  "mark.svg": mark(),
  "block.svg": blockDiagram(),
  "system.svg": systemDiagram(),
  "time-breakdown.svg": timeBreakdownPlot(),
  "expand-kernel.svg": expandPlot(),
  "throughput.svg": throughputPlot(),
};

mkdirSync(OUT, { recursive: true });
for (const [name, content] of Object.entries(files)) {
  writeFileSync(join(OUT, name), content);
  console.log(`wrote docs/assets/${name}`);
}
