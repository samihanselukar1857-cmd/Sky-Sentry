import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.dirname(new URL(import.meta.url).pathname.replace(/^\/[A-Z]:/, p => p.slice(1)));
const EVAL = path.join(ROOT, 'evaluation');
const OUT = path.join(ROOT, 'output', 'visuals');
const MODEL_KEYS = ['logistic_regression', 'gaussian_naive_bayes'];

function parseCSV(text) {
  const rows = [];
  for (const line of text.trim().split(/\r?\n/)) {
    const row = []; let value = '', quoted = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (quoted && c === '"' && line[i + 1] === '"') { value += '"'; i++; }
      else if (c === '"') quoted = !quoted;
      else if (c === ',' && !quoted) { row.push(value); value = ''; }
      else value += c;
    }
    row.push(value); rows.push(row);
  }
  const [head, ...body] = rows;
  return body.map(row => Object.fromEntries(head.map((key, i) => [key, row[i]])));
}
function esc(s) { return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;'); }
function nice(s, digits = 2) { return Number(s).toFixed(digits); }

function scatterSVG(rows, model, threshold) {
  const W = 920, H = 610, L = 92, R = 145, T = 52, B = 82;
  const pw = W - L - R, ph = H - T - B;
  const xs = rows.map(r => Number(r.log_rms)), ys = rows.map(r => Number(r.zero_crossing_rate));
  let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const xpad = (xmax - xmin) * 0.06 || 0.1, ypad = (ymax - ymin) * 0.08 || 0.01;
  xmin -= xpad; xmax += xpad; ymin = Math.max(0, ymin - ypad); ymax += ypad;
  const X = v => L + (v - xmin) / (xmax - xmin) * pw;
  const Y = v => T + ph - (v - ymin) / (ymax - ymin) * ph;
  const palette = p => {
    const a = [47, 121, 183], b = [239, 125, 34];
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * p)).join(',')})`;
  };
  const grid = [];
  for (let i = 0; i <= 5; i++) {
    const xv = xmin + (xmax - xmin) * i / 5, yv = ymin + (ymax - ymin) * i / 5;
    grid.push(`<line x1="${X(xv)}" y1="${T}" x2="${X(xv)}" y2="${T + ph}" stroke="#e7ebf0"/><text x="${X(xv)}" y="${T + ph + 24}" text-anchor="middle">${nice(xv)}</text>`);
    grid.push(`<line x1="${L}" y1="${Y(yv)}" x2="${L + pw}" y2="${Y(yv)}" stroke="#e7ebf0"/><text x="${L - 12}" y="${Y(yv) + 4}" text-anchor="end">${nice(yv, 3)}</text>`);
  }
  const points = rows.map(r => {
    const cx = X(Number(r.log_rms)), cy = Y(Number(r.zero_crossing_rate)), p = Number(r[`prob_${model}`]);
    const fill = palette(Math.max(0, Math.min(1, p)));
    return Number(r.label) === 1
      ? `<path d="M${cx} ${cy - 3.6} L${cx + 3.6} ${cy} L${cx} ${cy + 3.6} L${cx - 3.6} ${cy} Z" fill="${fill}" fill-opacity=".78" stroke="#fff" stroke-width=".45"><title>Drone, P(drone)=${nice(p, 3)}</title></path>`
      : `<circle cx="${cx}" cy="${cy}" r="3.2" fill="${fill}" fill-opacity=".78" stroke="#fff" stroke-width=".45"><title>Background, P(drone)=${nice(p, 3)}</title></circle>`;
  }).join('');
  const barY = T + 75, barH = 300, barX = L + pw + 55;
  const gradient = `<defs><linearGradient id="prob" x1="0" x2="0" y1="1" y2="0"><stop offset="0%" stop-color="#2f79b7"/><stop offset="100%" stop-color="#ef7d22"/></linearGradient></defs>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(model)} probability scatter plot">
  <style>text{font:14px Arial,sans-serif;fill:#263445}.title{font-size:21px;font-weight:700}.small{font-size:12px}</style>
  <rect width="100%" height="100%" fill="white"/>${gradient}
  <text class="title" x="${L}" y="30">${esc(model.replaceAll('_', ' '))}: test-window probabilities</text>
  ${grid.join('')}<rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="#526171"/>${points}
  <text x="${L + pw / 2}" y="${H - 24}" text-anchor="middle">Log RMS amplitude</text>
  <text transform="translate(24 ${T + ph / 2}) rotate(-90)" text-anchor="middle">Zero-crossing rate</text>
  <text x="${barX - 4}" y="${barY - 15}" class="small">P(drone)</text><rect x="${barX}" y="${barY}" width="18" height="${barH}" fill="url(#prob)"/>
  <text x="${barX + 27}" y="${barY + 5}" class="small">1.0</text><text x="${barX + 27}" y="${barY + barH}" class="small">0.0</text>
  <circle cx="${barX + 8}" cy="${barY + barH + 35}" r="4" fill="#718096"/><text x="${barX + 20}" y="${barY + barH + 40}" class="small">Background</text>
  <path d="M${barX + 8} ${barY + barH + 53} l5 5 -5 5 -5 -5 Z" fill="#718096"/><text x="${barX + 20}" y="${barY + barH + 63}" class="small">Drone</text>
  <text x="${barX - 2}" y="${barY + barH + 88}" class="small">Threshold ${nice(threshold, 2)}</text>
  </svg>`;
}

function histogramSVG(rows, model, threshold) {
  const W = 920, H = 520, L = 86, R = 35, T = 60, B = 90, bins = 20;
  const pw = W - L - R, ph = H - T - B, cell = pw / bins;
  const hist = [Array(bins).fill(0), Array(bins).fill(0)], totals = [0, 0];
  for (const r of rows) {
    const label = Number(r.label), p = Math.max(0, Math.min(.999999, Number(r[`prob_${model}`])));
    hist[label][Math.floor(p * bins)]++; totals[label]++;
  }
  const shares = hist.map((h, c) => h.map(v => v / Math.max(1, totals[c])));
  const max = Math.max(...shares.flat()) * 1.12 || 1, Y = v => T + ph - v / max * ph;
  const yTicks = Array.from({ length: 5 }, (_, i) => max * i / 4);
  const grid = yTicks.map(v => `<line x1="${L}" y1="${Y(v)}" x2="${L + pw}" y2="${Y(v)}" stroke="#e7ebf0"/><text x="${L - 10}" y="${Y(v) + 4}" text-anchor="end">${nice(v * 100, 0)}%</text>`).join('');
  let bars = '';
  for (let i = 0; i < bins; i++) {
    const bw = cell * .38, x = L + i * cell;
    for (let c = 0; c <= 1; c++) {
      const h = shares[c][i] / max * ph, bx = x + (c ? cell * .53 : cell * .08);
      bars += `<rect x="${bx}" y="${T + ph - h}" width="${bw}" height="${h}" fill="${c ? '#ef7d22' : '#2f79b7'}" fill-opacity=".83"><title>${c ? 'Drone' : 'Background'}: ${(shares[c][i] * 100).toFixed(1)}% of class in probability bin ${i}</title></rect>`;
    }
  }
  const tx = L + threshold * pw;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(model)} probability distributions">
  <style>text{font:14px Arial,sans-serif;fill:#263445}.title{font-size:21px;font-weight:700}.small{font-size:12px}</style><rect width="100%" height="100%" fill="white"/>
  <text class="title" x="${L}" y="32">${esc(model.replaceAll('_', ' '))}: test probability by true class</text>${grid}${bars}
  <line x1="${tx}" y1="${T}" x2="${tx}" y2="${T + ph}" stroke="#202b38" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="${tx + 6}" y="${T + 18}" class="small">threshold ${nice(threshold, 2)}</text><rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="#526171"/>
  ${Array.from({ length: 6 }, (_, i) => { const x = L + i * pw / 5, val = i / 5; return `<text x="${x}" y="${T + ph + 24}" text-anchor="middle">${nice(val, 1)}</text>`; }).join('')}
  <text x="${L + pw / 2}" y="${H - 25}" text-anchor="middle">Predicted probability of drone</text><text transform="translate(23 ${T + ph / 2}) rotate(-90)" text-anchor="middle">Share of class windows</text>
  <rect x="${L + 12}" y="${T + 14}" width="13" height="13" fill="#2f79b7"/><text x="${L + 32}" y="${T + 26}" class="small">Background (n=${totals[0]})</text>
  <rect x="${L + 190}" y="${T + 14}" width="13" height="13" fill="#ef7d22"/><text x="${L + 210}" y="${T + 26}" class="small">Drone (n=${totals[1]})</text></svg>`;
}

function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const predictions = parseCSV(fs.readFileSync(path.join(EVAL, 'predictions.csv'), 'utf8')).filter(r => r.split === 'test');
  const manifest = parseCSV(fs.readFileSync(path.join(EVAL, 'labelled_manifest.csv'), 'utf8'));
  const metric = JSON.parse(fs.readFileSync(path.join(EVAL, 'metrics.json'), 'utf8'));
  const featureByFile = new Map(manifest.map(r => [r.file, r]));
  const rows = predictions.map(r => ({ ...r, ...featureByFile.get(r.file) }));
  if (!rows.length || rows.some(r => !r.log_rms || !r.zero_crossing_rate)) throw Error('Test rows or plotted features are missing.');
  const blocks = [];
  for (const model of MODEL_KEYS) {
    const threshold = metric.models[model].threshold;
    const scatterName = `${model}_probability_scatter.svg`;
    const distName = `${model}_probability_distribution.svg`;
    fs.writeFileSync(path.join(OUT, scatterName), scatterSVG(rows, model, threshold));
    fs.writeFileSync(path.join(OUT, distName), histogramSVG(rows, model, threshold));
    blocks.push(`<section><h2>${esc(model.replaceAll('_', ' '))}</h2><div class="pair"><figure><img src="${scatterName}" alt="Probability scatter plot"><figcaption>Each marker is a held-out test audio window. Marker shape shows its true label; color shows predicted drone probability.</figcaption></figure><figure><img src="${distName}" alt="Probability distributions"><figcaption>Class-normalized probability distributions; dashed line is the validation-selected threshold.</figcaption></figure></div></section>`);
  }
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SkySentry model probability visuals</title><style>
  body{font:16px/1.5 system-ui,Segoe UI,Arial,sans-serif;color:#263445;background:#f4f7fa;margin:0;padding:28px}main{max-width:1200px;margin:auto}h1{margin:0 0 8px}h2{text-transform:capitalize;margin:32px 0 10px}.note{background:#fff4df;border-left:4px solid #e09a29;padding:12px 16px}figure{margin:0;background:white;border:1px solid #dce3ea;border-radius:10px;padding:12px;min-width:0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#526171;padding:8px 4px 2px}.pair{display:grid;grid-template-columns:1.1fr 1fr;gap:16px}@media(max-width:800px){.pair{grid-template-columns:1fr}body{padding:14px}}</style></head><body><main><h1>SkySentry model probabilities</h1><p>Held-out test windows, n=${rows.length}. Probability color and distributions are shown for both saved baseline models.</p><p class="note">The scatter view projects the 13-feature model onto log RMS and zero-crossing rate for visualization. It is not a two-feature decision boundary; predictions use all saved model features.</p>${blocks.join('')}<p>Generated from evaluation/predictions.csv, labelled_manifest.csv, and metrics.json.</p></main></body></html>`;
  fs.writeFileSync(path.join(OUT, 'index.html'), html);
  console.log(`Created probability visuals in ${path.relative(ROOT, OUT)} (${rows.length} held-out test windows, two models).`);
}

main();
