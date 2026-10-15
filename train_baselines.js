// Reproducible acoustic baselines for Dataset/itu-aris-dataset.
// Uses only Node.js built-ins; no data is copied or modified.
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.join('Dataset', 'itu-aris-dataset');
const OUT = 'evaluation';
const SEED = 20261015;

function wavSamples(file) {
  const b = fs.readFileSync(file);
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') throw Error(`Not WAV: ${file}`);
  let fmt, data;
  for (let p = 12; p + 8 <= b.length;) {
    const id = b.toString('ascii', p, p + 4), n = b.readUInt32LE(p + 4), s = p + 8;
    if (id === 'fmt ') fmt = { format: b.readUInt16LE(s), channels: b.readUInt16LE(s + 2), rate: b.readUInt32LE(s + 4), bits: b.readUInt16LE(s + 14) };
    if (id === 'data') { data = [s, n]; break; }
    p = s + n + (n & 1);
  }
  if (!fmt || !data || fmt.format !== 1 || fmt.bits !== 16) throw Error(`Expected PCM16 WAV: ${file}`);
  const count = Math.floor(data[1] / (2 * fmt.channels));
  const x = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    let v = 0;
    for (let c = 0; c < fmt.channels; c++) v += b.readInt16LE(data[0] + (i * fmt.channels + c) * 2) / 32768;
    x[i] = v / fmt.channels;
  }
  return { x, rate: fmt.rate };
}

function features(file) {
  const { x, rate } = wavSamples(path.join(ROOT, file));
  const n = x.length;
  let sum = 0, sq = 0, abs = 0, crossings = 0, peak = 0;
  const scales = new Float64Array(8), scaleN = new Uint32Array(8);
  for (let i = 0; i < n; i++) {
    const v = x[i]; sum += v; sq += v * v; abs += Math.abs(v); peak = Math.max(peak, Math.abs(v));
    if (i && (x[i] >= 0) !== (x[i - 1] >= 0)) crossings++;
  }
  for (let k = 0; k < 8; k++) {
    const stride = 1 << k;
    for (let i = stride; i < n; i += stride) { const d = x[i] - x[i - stride]; scales[k] += d * d; scaleN[k]++; }
  }
  const rms = Math.sqrt(sq / n);
  return [rms, Math.log10(rms + 1e-8), abs / n, peak, crossings / Math.max(1, n - 1), ...Array.from(scales, (v,k) => Math.log10(v / Math.max(1, scaleN[k]) + 1e-14))];
}

function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z)))); }
function fitScaler(rows) {
  const d = rows[0].x.length, mean = Array(d).fill(0), sd = Array(d).fill(0);
  rows.forEach(r => r.x.forEach((v, j) => mean[j] += v / rows.length));
  rows.forEach(r => r.x.forEach((v, j) => sd[j] += (v - mean[j]) ** 2 / rows.length));
  sd.forEach((v, j) => sd[j] = Math.sqrt(v) || 1);
  return { mean, sd };
}
function transform(x, s) { return x.map((v, j) => (v - s.mean[j]) / s.sd[j]); }
function fitLogistic(rows, scaler) {
  const d = rows[0].x.length, w = Array(d + 1).fill(0), lr = 0.04, epochs = 350;
  const data = rows.map(r => ({ x: transform(r.x, scaler), y: r.y }));
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g = Array(d + 1).fill(0);
    for (const r of data) {
      let z = w[d]; for (let j = 0; j < d; j++) z += w[j] * r.x[j];
      const e = sigmoid(z) - r.y;
      for (let j = 0; j < d; j++) g[j] += e * r.x[j] / data.length;
      g[d] += e / data.length;
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (g[j] + 0.002 * w[j]);
    w[d] -= lr * g[d];
  }
  return { type: 'logistic_regression', scaler, weights: w };
}
function predict(model, x) {
  const z = transform(x, model.scaler); let v = model.weights[z.length];
  for (let j = 0; j < z.length; j++) v += z[j] * model.weights[j];
  return sigmoid(v);
}
function fitGaussianNB(rows, scaler) {
  const d = rows[0].x.length, stats = [];
  for (let c = 0; c <= 1; c++) {
    const a = rows.filter(r => r.y === c).map(r => transform(r.x, scaler));
    const mean = Array(d).fill(0), variance = Array(d).fill(0);
    a.forEach(x => x.forEach((v, j) => mean[j] += v / a.length));
    a.forEach(x => x.forEach((v, j) => variance[j] += (v - mean[j]) ** 2 / a.length));
    variance.forEach((v, j) => variance[j] = Math.max(v, 1e-6));
    stats.push({ mean, variance, prior: a.length / rows.length });
  }
  return { type: 'gaussian_naive_bayes', scaler, classes: stats };
}
function predictNB(model, x) {
  const z = transform(x, model.scaler), logp = model.classes.map(c => {
    let v = Math.log(c.prior);
    for (let j = 0; j < z.length; j++) v -= 0.5 * (Math.log(2 * Math.PI * c.variance[j]) + (z[j] - c.mean[j]) ** 2 / c.variance[j]);
    return v;
  });
  return sigmoid(logp[1] - logp[0]);
}
function bestThreshold(rows, scorer) {
  let best = { threshold: 0.5, f1: -1 };
  for (let t = 0.05; t <= 0.9501; t += 0.01) {
    let tp = 0, fp = 0, fn = 0;
    rows.forEach(r => { const p = scorer(r.x) >= t; if (p && r.y) tp++; else if (p) fp++; else if (r.y) fn++; });
    const f1 = 2 * tp / Math.max(1, 2 * tp + fp + fn);
    if (f1 > best.f1) best = { threshold: Number(t.toFixed(2)), f1 };
  }
  return best.threshold;
}
function evaluate(rows, scorer, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const r of rows) {
    const p = scorer(r.x) >= threshold;
    if (p && r.y) tp++; else if (p) fp++; else if (r.y) fn++; else tn++;
  }
  return { n: rows.length, threshold, confusion: { tp, fp, tn, fn }, accuracy: (tp + tn) / rows.length,
    balanced_accuracy: ((tp / Math.max(1, tp + fn)) + (tn / Math.max(1, tn + fp))) / 2,
    precision: tp / Math.max(1, tp + fp), recall: tp / Math.max(1, tp + fn), specificity: tn / Math.max(1, tn + fp),
    false_positive_rate: fp / Math.max(1, fp + tn), false_positives_per_hour_assuming_one_second_per_background_window: fp / (Math.max(1, tn + fp)) * 3600 };
}
function csvCell(v) { return `"${String(v).replaceAll('"', '""')}"`; }
function writeCSV(file, header, rows) { fs.writeFileSync(file, [header, ...rows].map(r => r.map(csvCell).join(',')).join('\n') + '\n'); }

function run() {
  fs.mkdirSync(OUT, { recursive: true });
  const split = JSON.parse(fs.readFileSync(path.join(ROOT, 'splits/split_interval_disjoint.json'), 'utf8')).assignment;
  const groups = JSON.parse(fs.readFileSync(path.join(ROOT, 'groups.json'), 'utf8'));
  const groupByFile = new Map(), startByFile = new Map();
  // Current released groups.json uses a numeric-keyed segment table.
  if (groups.segments) for (const rec of Object.values(groups.segments)) groupByFile.set(rec.file, String(rec.group));
  // Drone windows are 1 s with 50% overlap; recover their start times from the
  // annotated source interval and their order within that interval. Background
  // source times are not published, so leave them null rather than fabricate.
  for (const interval of groups.intervals || []) {
    const gid = `drone_interval_${String(interval.group_id).padStart(2, '0')}`;
    const files = entriesForGroup(groupByFile, gid);
    files.forEach((f, i) => startByFile.set(f, interval.start_s + i * (groups.segment_duration_s * (1 - groups.overlap))));
  }
  const entries = Object.entries(split).sort(([a], [b]) => a.localeCompare(b));
  if (!entries.length) throw Error('No split assignments found');
  const records = [];
  for (const [file, subset] of entries) {
    const label = file.includes('/drone/') ? 1 : file.includes('/background/') ? 0 : null;
    if (label === null) continue;
    const g = groupByFile.get(file);
    if (!g) throw Error(`Missing provenance group for ${file}; refusing to continue without leakage audit`);
    records.push({ file, subset, group: g, y: label, start: startByFile.get(file) ?? null, x: features(file) });
  }
  const tr = records.filter(r => r.subset === 'train'), va = records.filter(r => r.subset === 'val'), te = records.filter(r => r.subset === 'test');
  if (!tr.length || !va.length || !te.length) throw Error(`Expected train/val/test, got ${tr.length}/${va.length}/${te.length}`);
  // Guard against any source group crossing a split.
  const memberships = new Map();
  for (const r of records) { if (!memberships.has(r.group)) memberships.set(r.group, new Set()); memberships.get(r.group).add(r.subset); }
  const leaks = [...memberships].filter(([, s]) => s.size > 1);
  if (leaks.length) throw Error(`Leakage: ${leaks.length} provenance groups cross splits`);
  const header = ['file', 'label', 'class_name', 'split', 'group_id', 'segment_start_s', 'segment_end_s', 'segment_duration_s', 'sample_rate_hz', 'rf_pair_status', 'rms', 'log_rms', 'mean_abs', 'peak', 'zero_crossing_rate', ...Array.from({length: 8}, (_,i)=>`multiscale_diff_log_energy_${i+1}`)];
  const manifest = records.map(r => [r.file, r.y, r.y ? 'drone' : 'background', r.subset, r.group, r.start ?? '', r.start === null ? '' : r.start + 1, 1, 22050, 'no_paired_rf_recording_available', ...r.x]);
  writeCSV(path.join(OUT, 'labelled_manifest.csv'), header, manifest);
  const models = [fitLogistic(tr, fitScaler(tr)), fitGaussianNB(tr, fitScaler(tr))];
  const report = { dataset: 'ITU ARIS Lab. Acoustic Drone Dataset v1.0.0', protocol: 'provided split_interval_disjoint.json; threshold chosen on val only; final score on test', seed: SEED,
    rows: records.length, groups: memberships.size, feature_names: header.slice(10), splits: {}, models: {} };
  for (const s of ['train','val','test']) {
    const a = records.filter(r => r.subset === s);
    report.splits[s] = { n: a.length, drone: a.filter(r => r.y).length, background: a.filter(r => !r.y).length, groups: new Set(a.map(r=>r.group)).size };
  }
  for (const model of models) {
    const scorer = model.type === 'logistic_regression' ? x => predict(model,x) : x => predictNB(model,x);
    const threshold = bestThreshold(va, scorer);
    report.models[model.type] = { threshold, validation: evaluate(va, scorer, threshold), test: evaluate(te, scorer, threshold) };
    fs.writeFileSync(path.join(OUT, `${model.type}.json`), JSON.stringify({ ...model, threshold, feature_names: report.feature_names }, null, 2));
  }
  fs.writeFileSync(path.join(OUT, 'metrics.json'), JSON.stringify(report, null, 2));
  const scoreRows = records.map(r => [r.file, r.y, r.subset, r.group, ...models.map(m => m.type === 'logistic_regression' ? predict(m,r.x) : predictNB(m,r.x))]);
  writeCSV(path.join(OUT, 'predictions.csv'), ['file','label','split','group_id', ...models.map(m=>`prob_${m.type}`)], scoreRows);
  console.log(JSON.stringify(report, null, 2));
}

function entriesForGroup(groupByFile, group) {
  return [...groupByFile].filter(([, g]) => g === group).map(([f]) => f).sort((a, b) => a.localeCompare(b));
}

export { run, features, fitLogistic, fitGaussianNB, evaluate };
