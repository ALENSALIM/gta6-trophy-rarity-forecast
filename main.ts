// Leonida Rarity Forecast
//
// Forecasts every Grand Theft Auto VI PS5 trophy's Exophase earn rate at a
// case's target time, as five quantiles per trophy.
//
// Model: per-trophy saturating growth in logit space. Earn rates only rise,
// fastest just after launch and ever more slowly as the tracked player base
// saturates, so each trophy's logit rate is fitted as a straight line in
// log time since launch (log hours + 1) over the last day of reads and read
// at the target, with recent reads weighted most so a trophy that has just
// levelled off is fitted on its flat tail. The slope is shrunk toward zero when the fit cannot tell
// it from noise, and never allowed negative: a trophy never un-earns.
//
// Spreads are honest and data-driven. They combine a random-walk volatility
// (a half-and-half blend of the house's 0.5 logit units per sqrt-day prior
// and what the fit's own residuals imply, floored at 0.30 a day so a quiet
// stretch never collapses the band), the slope's own standard error times
// the extrapolation distance, the fit residual, and a small horizon-scaled
// floor. Stable trophies get tight bands, noisy or fast-moving ones wide
// ones, and longer horizons get wider bands than short ones.
//
// Deterministic: the only clock is the case file's own `at` and `target`.
// No network, no wall clock, no randomness, no environment beyond the two
// input/output directory variables.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Launch-day midnight UTC: the zero of log time. Every read of the page
// happens after this, so log time is always defined.
const LAUNCH_MS = Date.parse("2026-11-19T00:00:00Z");
const HOUR_MS = 3600_000;
const DAY_MS = 86_400_000;
const WINDOW_MS = 24 * HOUR_MS;

// Standard normal quantiles at the case's five levels.
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

// The scorer clamps rates to [0.01, 99.99] before the logit; we forecast
// inside the same box so no quantile is ever wasted outside it.
const LO = 0.01;
const HI = 99.99;

type Trophy = { id: number };
type Slot = { at?: string; slot?: string; rates: (number | null)[] };
type CaseFile = {
  id: string;
  at: string;
  target: string;
  horizon_minutes?: number;
  levels?: number[];
  trophies: Trophy[];
  history: Slot[];
};
type Row = { id: number; q: number[] };

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// Percent (0..100) -> logit. 100% has no logit, so it lands on 99.99.
function logitPct(p: number): number {
  const c = clamp(p, LO, HI) / 100;
  return Math.log(c / (1 - c));
}

function invLogitPct(l: number): number {
  const x = clamp(l, -30, 30);
  return (1 / (1 + Math.exp(-x))) * 100;
}

// Log hours since launch, shifted so launch itself maps to zero.
function logHours(ms: number): number {
  return Math.log(1 + Math.max(0, ms - LAUNCH_MS) / HOUR_MS);
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

type Fit = {
  n: number;
  xbar: number;
  ybar: number;
  slope: number;
  resid: number;
  seSlope: number;
};

// Ordinary least squares of y on x, in centered form so the intercept is
// the mean level and the slope is cleanly separated from it. Reads carry
// recency weights: a trophy that has just levelled off should be fitted
// on its flat tail, not on the climb that came before it.
function fitLine(xs: number[], ys: number[], ws: number[]): Fit {
  const n = xs.length;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  const xbar = sx / sw;
  const ybar = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xbar;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - ybar);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (ybar + slope * (xs[i] - xbar));
    rss += ws[i] * r * r;
  }
  // Effective sample size of the weights, so a heavily weighted fit does
  // not claim more certainty than its recent reads support.
  let sw2 = 0;
  for (let i = 0; i < n; i++) sw2 += ws[i] * ws[i];
  const nEff = (sw * sw) / Math.max(sw2, 1e-12);
  const resid = nEff > 2 ? Math.sqrt(rss / (nEff - 2)) : 0;
  const seSlope = sxx > 0 && nEff > 2 ? resid / Math.sqrt(sxx) : 0;
  return { n, xbar, ybar, slope, resid, seSlope };
}

// Forecast one trophy from its observed (time, rate) pairs.
function forecastTrophy(
  series: { t: number; rate: number }[],
  atMs: number,
  targetMs: number,
  horizonDays: number,
): Row["q"] {
  const last = series[series.length - 1];
  const lastLogit = logitPct(last.rate);

  // Fit the last day of reads: recent growth is what the target inherits,
  // and anything older (launch night's flat pre-release page) is another
  // regime. Fall back to the whole history when the window is thin.
  const windowStart = atMs - WINDOW_MS;
  let window = series.filter((s) => s.t > windowStart);
  if (window.length < 3) window = series;

  const xTarget = logHours(targetMs);
  let mu = lastLogit;
  let resid = 0;
  let seSlope = 0;
  let xbar = logHours(last.t);
  let n = window.length;

  if (window.length >= 2) {
    const xs = window.map((s) => logHours(s.t));
    const ys = window.map((s) => logitPct(s.rate));
    // Recent reads count most: a 6-hour half-life keeps the slope local
    // to how the trophy is moving now, not how it moved a day ago.
    const tLast = window[window.length - 1].t;
    const ws = window.map((s) => Math.pow(0.5, (tLast - s.t) / (6 * HOUR_MS)));
    const fit = fitLine(xs, ys, ws);
    n = fit.n;
    xbar = fit.xbar;
    resid = fit.resid;
    seSlope = fit.seSlope;
    // Shrink the slope toward zero in proportion to how surely it is
    // above noise, then forbid shrinkage from ever turning growth into
    // decline. A flat page fits a flat line and forecasts persistence.
    let b = fit.slope;
    if (b > 0) {
      const shrink = (b * b) / (b * b + 4 * seSlope * seSlope);
      b = clamp(b * shrink, 0, 3);
    } else {
      b = 0;
    }
    mu = fit.ybar + b * (xTarget - fit.xbar);
  }

  // Growth only, and no single trophy is allowed to run away: at most four
  // logit units above its last read, and never past 99% unless it is
  // already there (a trophy at 100% stays at 100%).
  const cap = Math.max(logitPct(99), lastLogit);
  mu = clamp(mu, lastLogit, cap);
  mu = Math.min(mu, lastLogit + 4);

  // Spread, in logit units, from four honest sources.
  const spanDays = Math.max(
    (window[window.length - 1].t - window[0].t) / DAY_MS,
    1 / 24,
  );
  const readsPerDay = Math.min(n / spanDays, 200);
  // What the residuals imply about day-to-day wiggle, if each read's
  // deviation were a fresh shock.
  const sigmaEmp = resid * Math.sqrt(readsPerDay);
  // Half the house's 0.5-a-day prior, half the data, never below 0.30 a
  // day: a perfectly smooth stretch is evidence of calm, not proof of it.
  const sigmaEff = Math.max(0.3, Math.sqrt(0.5 * sigmaEmp * sigmaEmp + 0.5 * 0.25));
  const sWalk = sigmaEff * Math.sqrt(horizonDays);
  // The slope's own error, stretched over the extrapolation distance.
  const sSlope = seSlope * Math.max(0, xTarget - xbar);
  // A floor that grows with the horizon, so even a rock-steady trophy
  // keeps some room for the future to surprise it.
  const sFloor = 0.15 * Math.sqrt(horizonDays / 4.52);
  let s = Math.sqrt(sWalk * sWalk + sSlope * sSlope + sFloor * sFloor + resid * resid);
  s = clamp(s, 0.03, 1.35);

  const qs = Z.map((z) => round4(clamp(invLogitPct(mu + z * s), LO, HI)));
  // Rounding can only tie neighbours, never invert them; enforce the
  // order explicitly so the file is valid by construction.
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
}

// The house persistence baseline, for comparison in testing: last rate,
// spread 0.5 logit units per sqrt-day.
export function houseBaseline(c: CaseFile): { trophies: Row[] } {
  const atMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max(0, (targetMs - atMs) / DAY_MS);
  const s = 0.5 * Math.sqrt(horizonDays);
  const rows: Row[] = c.trophies.map((t, i) => {
    const last = lastRate(c, i);
    const mu = logitPct(last);
    const qs = Z.map((z) => round4(clamp(invLogitPct(mu + z * s), LO, HI)));
    for (let k = 1; k < qs.length; k++) {
      if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    }
    return { id: t.id, q: qs };
  });
  return { trophies: rows };
}

function lastRate(c: CaseFile, i: number): number {
  for (let k = c.history.length - 1; k >= 0; k--) {
    const r = c.history[k].rates[i];
    if (r !== null && r !== undefined && Number.isFinite(r)) return r;
  }
  return 1;
}

export function forecastCase(c: CaseFile): { trophies: Row[] } {
  const atMs = Date.parse(c.at);
  const targetMs = Number.isFinite(Date.parse(c.target))
    ? Date.parse(c.target)
    : atMs + (c.horizon_minutes ?? 0) * 60_000;
  const horizonDays = Math.max(0, (targetMs - atMs) / DAY_MS);

  const rows: Row[] = c.trophies.map((t, i) => {
    const series: { t: number; rate: number }[] = [];
    for (const slot of c.history) {
      const r = slot.rates[i];
      if (r === null || r === undefined || !Number.isFinite(r)) continue;
      const ms = Date.parse(slot.at ?? slot.slot ?? "");
      if (!Number.isFinite(ms)) continue;
      series.push({ t: ms, rate: r });
    }
    series.sort((a, b) => a.t - b.t);
    if (series.length === 0) {
      // Never seen: a low rate with a wide band.
      const qs = Z.map((z) => round4(clamp(invLogitPct(logitPct(1) + z * 1), LO, HI)));
      return { id: t.id, q: qs };
    }
    return { id: t.id, q: forecastTrophy(series, atMs, targetMs, horizonDays) };
  });
  return { trophies: rows };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const c = JSON.parse(readFileSync(join(inDir, f), "utf8")) as CaseFile;
    const out = forecastCase(c);
    writeFileSync(join(outDir, f.replace(/\.in$/, ".out")), JSON.stringify(out));
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main();
}
