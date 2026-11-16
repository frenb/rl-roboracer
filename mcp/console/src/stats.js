// The subset of dashboard/analysis.html's statistics the compare tool reports.

export const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

export function variance(a) {
  if (a.length < 2) return NaN;
  const m = mean(a);
  return a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1);
}

export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function lgamma(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  const tmp = x + 5.5 - (x + 0.5) * Math.log(x + 5.5);
  let ser = 1.000000000190015;
  for (const cj of c) ser += cj / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function betacf(a, b, x) {
  const EPS = 3e-12;
  const FPMIN = 1e-300;
  let c = 1;
  let d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** Regularized incomplete beta I_x(a, b). */
function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}

/** Two-sided p-value of Student's t with df degrees of freedom. */
export function tTwoSidedP(t, df) {
  if (!Number.isFinite(t) || df <= 0) return NaN;
  return betai(df / 2, 0.5, df / (df + t * t));
}

export function tCritical(df, alpha = 0.05) {
  let lo = 0;
  let hi = 30;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (tTwoSidedP(mid, df) > alpha) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

export function summary(values) {
  const s = values.filter(Number.isFinite);
  const n = s.length;
  if (!n) return { n: 0 };
  const sorted = s.slice().sort((a, b) => a - b);
  const m = mean(s);
  const sd = Math.sqrt(variance(s));
  const sem = sd / Math.sqrt(n);
  const t = n > 1 ? tCritical(n - 1) : NaN;
  return {
    n,
    mean: m,
    std: n > 1 ? sd : null,
    sem: n > 1 ? sem : null,
    median: quantile(sorted, 0.5),
    q25: quantile(sorted, 0.25),
    q75: quantile(sorted, 0.75),
    min: sorted[0],
    max: sorted[n - 1],
    ci95: n > 1 ? [m - t * sem, m + t * sem] : null,
  };
}

/** Welch's t-test and Hedges' g of a against b. */
export function compareSamples(a, b) {
  if (a.length < 2 || b.length < 2) return { note: 'needs at least 2 trials on each side' };
  const ma = mean(a); const mb = mean(b);
  const va = variance(a); const vb = variance(b);
  const se2 = va / a.length + vb / b.length;
  const t = (ma - mb) / Math.sqrt(se2);
  const df = se2 ** 2 / ((va / a.length) ** 2 / (a.length - 1) + (vb / b.length) ** 2 / (b.length - 1));
  const pooled = Math.sqrt(((a.length - 1) * va + (b.length - 1) * vb) / (a.length + b.length - 2));
  const d = (ma - mb) / pooled;
  const g = d * (1 - 3 / (4 * (a.length + b.length) - 9));
  return { mean_diff: ma - mb, welch_t: t, welch_df: df, p_value: tTwoSidedP(t, df), hedges_g: g };
}

export function round(obj, digits = 4) {
  return JSON.parse(JSON.stringify(obj, (_k, v) => (typeof v === 'number' && !Number.isInteger(v)
    ? Number(v.toFixed(digits)) : v)));
}
