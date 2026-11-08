// Weakness tab: where on the track a job's policy drives badly. Scores
// cells exactly as dashboard/weakness.html does, then groups adjacent weak
// cells into hotspots and draws the track as text so the answer is "where",
// not a list of numbers.

import { z } from 'zod';
import { get, list } from '../api.js';
import { defineTool, resolveRow, objectIdTime } from '../util.js';

const LAYERS = ['weakness', 'crash', 'speed', 'risk', 'visits'];
// weakness.html slider defaults.
const DEFAULT_WEIGHTS = { crash: 0.5, slow: 0.3, risk: 0.2 };
const DEFAULT_MIN_VISITS = 3;
const ASCII_MAX = 64;
const THIN_EVIDENCE_VISITS = 10;
const SHADES = ['.', '-', '+', '*', '#'];

/** Min-max normalize a Map's values into [0,1], as weakness.html does. */
function normalize(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values.values()) {
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = (max - min) || 1;
  const out = new Map();
  for (const [k, v] of values) out.set(k, (v - min) / span);
  return out;
}

function median(xs) {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Per-cell metrics plus a 0-1 "badness" score for the chosen layer, higher
 * = weaker (for speed, slower; for visits, busier).
 */
function scoreCells(map, layer, minVisits, weights) {
  const { nx } = map;
  const cells = new Map();
  const crashRate = new Map();
  const speed = new Map();
  const risk = new Map();
  const visits = new Map();
  for (const c of map.cells) {
    const k = c.iy * nx + c.ix;
    cells.set(k, c);
    if (c.v < minVisits) continue;
    crashRate.set(k, c.v > 0 ? c.cr / c.v : 0);
    speed.set(k, c.sc > 0 ? c.ss / c.sc : 0);
    risk.set(k, c.v > 0 ? (c.hz || 0) / c.v : 0);
    visits.set(k, c.v);
  }
  const nCrash = normalize(crashRate);
  const nSpeed = normalize(speed);
  const nRisk = normalize(risk);
  const wsum = (weights.crash + weights.slow + weights.risk) || 1;

  const score = new Map();
  for (const k of crashRate.keys()) {
    let s;
    if (layer === 'crash') s = nCrash.get(k);
    else if (layer === 'speed') s = 1 - nSpeed.get(k);
    else if (layer === 'risk') s = nRisk.get(k);
    else if (layer === 'visits') s = null;
    else s = (weights.crash * nCrash.get(k) + weights.slow * (1 - nSpeed.get(k)) + weights.risk * nRisk.get(k)) / wsum;
    score.set(k, s);
  }
  if (layer === 'visits') for (const [k, v] of normalize(visits)) score.set(k, v);
  return { cells, score, crashRate, speed, risk };
}

function geometry(map) {
  const { extent, nx, ny, axisNames } = map;
  const cw = (extent.xmax - extent.xmin) / nx;
  const ch = (extent.ymax - extent.ymin) / ny;
  const center = (k) => [extent.xmin + ((k % nx) + 0.5) * cw, extent.ymin + (Math.floor(k / nx) + 0.5) * ch];
  // Where a point sits on the track's bounding box, in thirds per axis.
  const where = (x, y) => {
    const third = (f) => (f < 1 / 3 ? 'low' : f > 2 / 3 ? 'high' : 'mid');
    const fx = third((x - extent.xmin) / (extent.xmax - extent.xmin));
    const fy = third((y - extent.ymin) / (extent.ymax - extent.ymin));
    if (fx === 'mid' && fy === 'mid') return 'centre of the track area';
    if (fx === 'mid') return `${fy}-${axisNames[1]} side`;
    if (fy === 'mid') return `${fx}-${axisNames[0]} side`;
    return `${fx}-${axisNames[0]} / ${fy}-${axisNames[1]} corner`;
  };
  return { nx, ny, cw, ch, center, where, axisNames };
}

/** Connected (8-neighbour) groups of cells at or above the threshold. */
function hotspots(scored, geo, threshold, trackMedianSpeed) {
  const { score, cells, crashRate, speed } = scored;
  const { nx, ny, center, where, cw, ch, axisNames } = geo;
  const hot = new Set([...score].filter(([, s]) => s >= threshold).map(([k]) => k));
  const seen = new Set();
  const regions = [];
  for (const start of hot) {
    if (seen.has(start)) continue;
    const members = [];
    const stack = [start];
    seen.add(start);
    while (stack.length) {
      const k = stack.pop();
      members.push(k);
      const ix = k % nx;
      const iy = Math.floor(k / nx);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const jx = ix + dx;
          const jy = iy + dy;
          if (jx < 0 || jy < 0 || jx >= nx || jy >= ny) continue;
          const j = jy * nx + jx;
          if (hot.has(j) && !seen.has(j)) { seen.add(j); stack.push(j); }
        }
      }
    }
    let wsum = 0; let cx = 0; let cy = 0; let crashes = 0; let visits = 0; let scoreSum = 0;
    let peak = members[0];
    const xs = []; const ys = []; const speeds = [];
    for (const k of members) {
      const c = cells.get(k);
      const [x, y] = center(k);
      cx += x * c.v; cy += y * c.v; wsum += c.v;
      crashes += c.cr; visits += c.v; scoreSum += score.get(k);
      xs.push(x); ys.push(y); speeds.push(speed.get(k));
      if (score.get(k) > score.get(peak)) peak = k;
    }
    cx /= wsum; cy /= wsum;
    const [px, py] = center(peak);
    const regionSpeed = median(speeds);
    regions.push({
      where: where(cx, cy),
      centroid: { [axisNames[0]]: +cx.toFixed(2), [axisNames[1]]: +cy.toFixed(2) },
      bounds: {
        [axisNames[0]]: [+(Math.min(...xs) - cw / 2).toFixed(2), +(Math.max(...xs) + cw / 2).toFixed(2)],
        [axisNames[1]]: [+(Math.min(...ys) - ch / 2).toFixed(2), +(Math.max(...ys) + ch / 2).toFixed(2)],
      },
      cells: members.length,
      peak_score: +score.get(peak).toFixed(3),
      peak_at: { [axisNames[0]]: +px.toFixed(2), [axisNames[1]]: +py.toFixed(2) },
      visits,
      crashes,
      crash_rate: +(crashes / visits).toFixed(4),
      peak_cell_crash_rate: +crashRate.get(peak).toFixed(4),
      speed_vs_track_median: trackMedianSpeed ? +(regionSpeed / trackMedianSpeed).toFixed(2) : null,
      ...(visits < THIN_EVIDENCE_VISITS ? { evidence: `thin: ${visits} visits; may be noise` } : {}),
      _weight: scoreSum,
      _peak: peak,
    });
  }
  return regions.sort((a, b) => b._weight - a._weight);
}

/** The track as text, top row = high second axis. Hotspot peaks are numbered. */
function asciiMap(scored, geo, regions) {
  const { nx, ny } = geo;
  const f = Math.max(1, Math.ceil(Math.max(nx, ny) / ASCII_MAX));
  const W = Math.ceil(nx / f);
  const H = Math.ceil(ny / f);
  const grid = Array.from({ length: H }, () => new Array(W).fill(' '));
  const best = Array.from({ length: H }, () => new Array(W).fill(-1));
  for (const [k] of scored.cells) {
    const gx = Math.floor((k % nx) / f);
    const gy = Math.floor(Math.floor(k / nx) / f);
    const s = scored.score.has(k) ? scored.score.get(k) : 0;
    if (s > best[gy][gx]) best[gy][gx] = s;
  }
  for (let gy = 0; gy < H; gy++) {
    for (let gx = 0; gx < W; gx++) {
      if (best[gy][gx] < 0) continue;
      grid[gy][gx] = SHADES[Math.min(SHADES.length - 1, Math.floor(best[gy][gx] * SHADES.length))];
    }
  }
  regions.slice(0, 9).forEach((r, i) => {
    const gx = Math.floor((r._peak % nx) / f);
    const gy = Math.floor(Math.floor(r._peak / nx) / f);
    grid[gy][gx] = String(i + 1);
  });
  const rows = grid.reverse().map((r) => r.join('').replace(/\s+$/, ''));
  return [
    `+${'-'.repeat(W)}+  ^ ${geo.axisNames[1]}`,
    ...rows.map((r) => `|${r.padEnd(W)}|`),
    `+${'-'.repeat(W)}+  > ${geo.axisNames[0]}`,
    `legend: ' ' never driven, ${SHADES.map((s, i) => `'${s}' ${(i / SHADES.length).toFixed(1)}+`).join(', ')}, 1-9 = hotspot peaks`,
  ].join('\n');
}

async function pickJob(ref) {
  const jobs = await list('/get_jobs');
  if (ref) {
    try {
      return resolveRow(jobs, ref, 'job');
    } catch (e) {
      // Trajectory logs outlive a deleted job document.
      if (/^[0-9a-f]{24}$/i.test(ref)) return { _id: ref, status: 'job document not found (deleted?)' };
      throw e;
    }
  }
  const newest = jobs.slice().sort((a, b) => String(b._id).localeCompare(String(a._id)));
  const job = newest.find((j) => j.status === 'IN_PROGRESS') || newest.find((j) => j.job_type === 'TRAIN') || newest[0];
  if (!job) throw new Error('there are no jobs');
  return job;
}

async function analyse(args) {
  const job = await pickJob(args.job_id);
  const layer = args.layer || 'weakness';
  const minVisits = args.min_visits ?? DEFAULT_MIN_VISITS;
  const weights = { ...DEFAULT_WEIGHTS, ...(args.weights || {}) };
  const map = await get('/weakness_map', { job_id: String(job._id), bins: args.bins, limit: args.limit });
  const jobInfo = {
    _id: String(job._id), job_type: job.job_type, status: job.status, course_type: job.course_type,
    gym_name: job.gym_name, created_at: objectIdTime(job._id),
  };
  if (!map.cells || !map.cells.length) {
    return { job: jobInfo, episodes: map.episodes || 0, note: 'No trajectory data for this job yet (only finished episodes record a path).' };
  }
  const scored = scoreCells(map, layer, minVisits, weights);
  const geo = geometry(map);
  const trackMedianSpeed = median([...scored.speed.values()]);
  const regions = hotspots(scored, geo, args.threshold ?? 0.6, trackMedianSpeed);
  const crashEpisodes = map.cells.reduce((s, c) => s + c.cr, 0);
  const goalEpisodes = map.cells.reduce((s, c) => s + c.gl, 0);
  return { job: jobInfo, map, layer, minVisits, weights, scored, geo, regions, crashEpisodes, goalEpisodes, trackMedianSpeed };
}

// ---- difficulty by goal gate ------------------------------------------
//
// SimController starts every episode just past a uniformly random goal gate,
// so gates are the unit a start-position curriculum can choose from.

const DEFAULT_WEAK_SHARE = 0.25;
// Pseudo-attempts of the track-wide crash rate mixed into each gate's rate,
// so a gate seen twice cannot outrank one seen forty times.
const SHRINK_ATTEMPTS = 5;

/** 95% Wilson score interval for k successes in n trials. */
function wilson(k, n) {
  if (!n) return [0, 1];
  const z = 1.96;
  const p = k / n;
  const d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d;
  const h = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

async function gateStats(args) {
  const job = await pickJob(args.job_id);
  let data;
  try {
    data = await get('/weakness_by_gate', { job_id: String(job._id), limit: args.limit });
  } catch (e) {
    if (e.status === 404) {
      throw new Error('this dashboard has no /weakness_by_gate endpoint yet; restart the dashboard container from a '
        + 'checkout that includes it (dashboard/src/server.ts).');
    }
    throw e;
  }
  const gates = (data.gates || []).filter((g) => g.index !== null);
  const jobInfo = {
    _id: String(job._id), job_type: job.job_type, status: job.status, course_type: job.course_type,
    gym_name: job.gym_name, created_at: objectIdTime(job._id),
  };
  if (!gates.length) return { job: jobInfo, gates: [] };
  const gateCount = Math.max(...gates.map((g) => g.index));
  const attempts = gates.reduce((s, g) => s + g.attempts, 0);
  const crashes = gates.reduce((s, g) => s + g.crashes, 0);
  const trackRate = crashes / attempts;
  const medianSteps = median(gates.map((g) => g.mean_steps_success).filter(Number.isFinite));

  // Where each gate sits, from where cars reached (else crashed near) it.
  const pts = gates.map((g) => [g.reach_x ?? g.crash_x, g.reach_z ?? g.crash_z]).filter(([x, z]) => Number.isFinite(x) && Number.isFinite(z));
  const ext = {
    xmin: Math.min(...pts.map((p) => p[0])), xmax: Math.max(...pts.map((p) => p[0])),
    zmin: Math.min(...pts.map((p) => p[1])), zmax: Math.max(...pts.map((p) => p[1])),
  };
  const where = (x, z) => {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return null;
    const third = (f) => (f < 1 / 3 ? 'low' : f > 2 / 3 ? 'high' : 'mid');
    const fx = third((x - ext.xmin) / ((ext.xmax - ext.xmin) || 1));
    const fz = third((z - ext.zmin) / ((ext.zmax - ext.zmin) || 1));
    if (fx === 'mid' && fz === 'mid') return 'centre of the track area';
    if (fx === 'mid') return `${fz}-z side`;
    if (fz === 'mid') return `${fx}-x side`;
    return `${fx}-x / ${fz}-z corner`;
  };

  const rows = gates.map((g) => {
    const [lo, hi] = wilson(g.crashes, g.attempts);
    const x = g.reach_x ?? g.crash_x;
    const z = g.reach_z ?? g.crash_z;
    return {
      gate: g.gate,
      index: g.index,
      lap_fraction: +(g.index / gateCount).toFixed(3),
      where: where(x, z),
      position: Number.isFinite(x) ? { x: +x.toFixed(2), z: +z.toFixed(2) } : null,
      attempts: g.attempts,
      crashes: g.crashes,
      crash_rate: +(g.crashes / g.attempts).toFixed(3),
      crash_rate_ci95: [+lo.toFixed(3), +hi.toFixed(3)],
      smoothed_crash_rate: +((g.crashes + SHRINK_ATTEMPTS * trackRate) / (g.attempts + SHRINK_ATTEMPTS)).toFixed(3),
      steps_to_clear_vs_median: Number.isFinite(g.mean_steps_success) && medianSteps
        ? +(g.mean_steps_success / medianSteps).toFixed(2) : null,
      crash_position: Number.isFinite(g.crash_x) ? { x: +g.crash_x.toFixed(2), z: +g.crash_z.toFixed(2) } : null,
      _lo: lo,
    };
  });
  const seen = new Set(rows.map((r) => r.index));
  const missing = [];
  for (let i = 1; i <= gateCount; i++) if (!seen.has(i)) missing.push(i);
  return { job: jobInfo, gates: rows, gateCount, attempts, crashes, trackRate, missing };
}

const commonInput = {
  job_id: z.string().optional().describe('Job id or suffix. Default: the running job, else the newest TRAIN job.'),
  layer: z.enum(LAYERS).optional().describe('weakness (default) = weighted blend of crash rate, slowness and pre-crash risk; '
    + 'crash = crashes per visit; speed = slow is weak (the start straight always reads slow: episodes begin from rest); '
    + 'risk = share of visits within 30 steps of a crash; visits = traffic.'),
  weights: z.object({ crash: z.number().min(0).optional(), slow: z.number().min(0).optional(), risk: z.number().min(0).optional() })
    .optional().describe('Blend weights for the weakness layer. Default crash 0.5, slow 0.3, risk 0.2.'),
  bins: z.number().int().min(8).max(200).optional().describe('Grid cells per side. Default 48.'),
  limit: z.number().int().min(1).max(5000).optional().describe('Most recent episodes to use. Default 500.'),
  min_visits: z.number().int().min(1).max(30).optional().describe('Cells visited fewer times are not scored. Default 3.'),
};

export function register(server) {
  defineTool(server, 'get_weakness_map', {
    title: 'Where the car drives badly',
    description: 'Weakness tab: scores the track from a job\'s recent episodes (crashes, slow sections, approaches that end in '
      + 'a crash) and returns the weak hotspots, each with where it is on the track, its bounds in world coordinates, '
      + 'crash rate and speed relative to the rest of the track, plus a text map with the hotspots numbered.',
    readOnly: true,
    input: {
      ...commonInput,
      threshold: z.number().min(0).max(1).optional().describe('Score (0-1) a cell needs to count toward a hotspot. Default 0.6.'),
      top: z.number().int().min(1).max(30).optional().describe('Hotspots to return. Default 5.'),
      ascii: z.boolean().optional().describe('Include the text map. Default true.'),
      raw: z.boolean().optional().describe('Return the dashboard\'s raw grid instead.'),
    },
  }, async (args) => {
    if (args.raw) {
      const job = await pickJob(args.job_id);
      return get('/weakness_map', { job_id: String(job._id), bins: args.bins, limit: args.limit });
    }
    const a = await analyse(args);
    if (!a.regions) return a;
    const top = args.top ?? 5;
    const out = {
      job: a.job,
      layer: a.layer,
      weights: a.layer === 'weakness' ? a.weights : undefined,
      episodes: a.map.episodes,
      crashed_episodes: a.crashEpisodes,
      goal_episodes: a.goalEpisodes,
      ground_axes: a.geo.axisNames,
      extent: {
        [a.geo.axisNames[0]]: [+a.map.extent.xmin.toFixed(2), +a.map.extent.xmax.toFixed(2)],
        [a.geo.axisNames[1]]: [+a.map.extent.ymin.toFixed(2), +a.map.extent.ymax.toFixed(2)],
      },
      grid: `${a.map.nx}x${a.map.ny}, min ${a.minVisits} visits per scored cell`,
      track_median_step_distance: a.trackMedianSpeed && +a.trackMedianSpeed.toFixed(4),
      hotspots: a.regions.slice(0, top).map(({ _weight, _peak, ...r }, i) => ({ rank: i + 1, ...r })),
    };
    if (!a.regions.length) out.note = `No cell scored ${args.threshold ?? 0.6} or more; lower threshold to see milder weak spots.`;
    if (args.ascii !== false) out.map = asciiMap(a.scored, a.geo, a.regions);
    return out;
  });

  defineTool(server, 'get_weakness_at', {
    title: 'Weakness at a track position',
    description: 'Weakness tab hover: metrics for the grid cell containing a world position, and its neighbours, for a job.',
    readOnly: true,
    input: {
      ...commonInput,
      x: z.number().describe('Position on the first ground axis (see get_weakness_map ground_axes).'),
      y: z.number().describe('Position on the second ground axis.'),
      radius: z.number().int().min(0).max(5).optional().describe('Neighbouring cells to include on each side. Default 1.'),
    },
  }, async (args) => {
    const a = await analyse(args);
    if (!a.regions) return a;
    const { extent, nx, ny } = a.map;
    const ix = Math.floor((args.x - extent.xmin) / a.geo.cw);
    const iy = Math.floor((args.y - extent.ymin) / a.geo.ch);
    if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) {
      return {
        job: a.job,
        note: 'That position is outside the area the car has driven.',
        extent: { [a.geo.axisNames[0]]: [extent.xmin, extent.xmax], [a.geo.axisNames[1]]: [extent.ymin, extent.ymax] },
      };
    }
    const r = args.radius ?? 1;
    const cells = [];
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const jx = ix + dx; const jy = iy + dy;
        if (jx < 0 || jy < 0 || jx >= nx || jy >= ny) continue;
        const k = jy * nx + jx;
        const c = a.scored.cells.get(k);
        const [cx, cy] = a.geo.center(k);
        const row = { [a.geo.axisNames[0]]: +cx.toFixed(2), [a.geo.axisNames[1]]: +cy.toFixed(2), visits: c ? c.v : 0 };
        if (c) {
          Object.assign(row, {
            crashes: c.cr,
            goals: c.gl,
            crash_rate: +(a.scored.crashRate.get(k) ?? c.cr / c.v).toFixed(4),
            pre_crash_risk: +((c.hz || 0) / c.v).toFixed(4),
            speed_vs_track_median: a.trackMedianSpeed && c.sc ? +((c.ss / c.sc) / a.trackMedianSpeed).toFixed(2) : null,
            score: a.scored.score.has(k) ? +a.scored.score.get(k).toFixed(3) : `unscored (< ${a.minVisits} visits)`,
          });
        }
        if (dx === 0 && dy === 0) row.queried = true;
        cells.push(row);
      }
    }
    return { job: a.job, layer: a.layer, where: a.geo.where(args.x, args.y), cells };
  });

  const gateInput = {
    job_id: commonInput.job_id,
    limit: z.number().int().min(1).max(20000).optional().describe('Most recent gate-to-gate stretches to use. Default 2000.'),
    min_attempts: z.number().int().min(1).optional().describe('Leave out gates attempted fewer times. Default 3.'),
  };
  const GATE_NOTE = 'Gate numbers belong to one track layout; a curriculum stage that rebuilds the track renumbers them, '
    + 'so lap_fraction (gate / gate count) is the layout-independent position.';

  defineTool(server, 'get_difficult_sections', {
    title: 'Where the track is most difficult',
    description: 'Ranks the track\'s goal gates by how often the car crashes on the way to them (each gate-to-gate stretch '
      + 'is one attempt), ordered by the low end of the 95% confidence range so thinly-sampled gates do not dominate. '
      + 'Also groups adjacent hard gates into sections. Gates are where episodes start, so this is the input for '
      + 'start-position experiments (suggest_start_gates).',
    readOnly: true,
    input: { ...gateInput, top: z.number().int().min(1).max(100).optional().describe('Gates to return. Default 10.') },
  }, async (args) => {
    const s = await gateStats(args);
    if (!s.gates.length) return { job: s.job, note: 'No gate-to-gate driving logs for this job yet.' };
    const minAttempts = args.min_attempts ?? 3;
    const ranked = s.gates.filter((g) => g.attempts >= minAttempts).sort((a, b) => b._lo - a._lo);
    // Sections: runs of consecutive gates whose smoothed crash rate beats the track average.
    const hard = new Set(s.gates.filter((g) => g.attempts >= minAttempts && g.smoothed_crash_rate > s.trackRate).map((g) => g.index));
    const byIndex = new Map(s.gates.map((g) => [g.index, g]));
    const sections = [];
    for (let i = 1; i <= s.gateCount; i++) {
      if (!hard.has(i) || hard.has(i - 1)) continue;
      let j = i;
      while (hard.has(j + 1)) j++;
      const members = [];
      for (let k = i; k <= j; k++) members.push(byIndex.get(k));
      const att = members.reduce((t, g) => t + g.attempts, 0);
      const cr = members.reduce((t, g) => t + g.crashes, 0);
      sections.push({
        gates: i === j ? `Goal-${i}` : `Goal-${i} to Goal-${j}`,
        lap_fraction: [+(i / s.gateCount).toFixed(3), +(j / s.gateCount).toFixed(3)],
        where: members[0].where,
        attempts: att,
        crashes: cr,
        crash_rate: +(cr / att).toFixed(3),
        crash_rate_ci95: wilson(cr, att).map((v) => +v.toFixed(3)),
      });
    }
    sections.sort((a, b) => b.crash_rate_ci95[0] - a.crash_rate_ci95[0]);
    return {
      job: s.job,
      gate_count: s.gateCount,
      attempts: s.attempts,
      crashes: s.crashes,
      track_crash_rate: +s.trackRate.toFixed(3),
      ...(s.missing.length ? { gates_never_attempted: s.missing } : {}),
      hardest_sections: sections.slice(0, 5),
      hardest_gates: ranked.slice(0, args.top ?? 10).map(({ _lo, ...g }, i) => ({ rank: i + 1, ...g })),
      note: GATE_NOTE,
    };
  });

  defineTool(server, 'suggest_start_gates', {
    title: 'Where to place the car',
    description: 'Start-position distribution for weakness-focused training: a share of resets (default 25%) start a few '
      + 'gates before the hardest gates, so the car drives into its weak spots; the rest stay uniformly random so the rest '
      + 'of the track is not forgotten. Each gate\'s crash rate is shrunk toward the track average before weighting. '
      + 'Evaluation should keep uniform starts for a fair comparison. The trainer cannot apply this yet: start-gate control '
      + 'in Unity, the reset message and the trainer is a separate change.',
    readOnly: true,
    input: {
      ...gateInput,
      weak_share: z.number().min(0).max(1).optional().describe('Share of resets aimed at weak spots. Default 0.25.'),
      lead: z.number().int().min(0).max(10).optional().describe('Start this many gates before each hard gate, to give a run-up. Default 1.'),
      top_k: z.number().int().min(1).max(50).optional().describe('How many hard gates to target. Default 5.'),
    },
  }, async (args) => {
    const s = await gateStats(args);
    if (!s.gates.length) return { job: s.job, note: 'No gate-to-gate driving logs for this job yet.' };
    const weakShare = args.weak_share ?? DEFAULT_WEAK_SHARE;
    const lead = args.lead ?? 1;
    const minAttempts = args.min_attempts ?? 3;
    const N = s.gateCount;
    const targets = s.gates
      .filter((g) => g.attempts >= minAttempts && g.smoothed_crash_rate > s.trackRate)
      .sort((a, b) => b.smoothed_crash_rate - a.smoothed_crash_rate)
      .slice(0, args.top_k ?? 5);
    if (!targets.length) {
      return { job: s.job, note: 'No gate is reliably harder than the track average; keep uniform starts.', track_crash_rate: +s.trackRate.toFixed(3) };
    }
    const excess = targets.map((g) => g.smoothed_crash_rate - s.trackRate);
    const total = excess.reduce((a, b) => a + b, 0);
    const startOf = (idx) => ((idx - 1 - lead) % N + N) % N + 1;

    const prob = new Map();
    for (let i = 1; i <= N; i++) prob.set(i, (1 - weakShare) / N);
    const aimed = targets.map((g, k) => {
      const start = startOf(g.index);
      const p = weakShare * excess[k] / total;
      prob.set(start, prob.get(start) + p);
      return {
        hard_gate: g.gate,
        start_at: `Goal-${start}`,
        start_lap_fraction: +(start / N).toFixed(3),
        where: g.where,
        hard_gate_crash_rate: g.crash_rate,
        smoothed_crash_rate: g.smoothed_crash_rate,
        attempts: g.attempts,
        weak_share_of_this_target: +(excess[k] / total).toFixed(3),
      };
    });
    const uniform = (1 - weakShare) / N;
    const boosted = [...prob]
      .filter(([, p]) => p > uniform + 1e-9)
      .map(([i, p]) => ({ gate: `Goal-${i}`, lap_fraction: +(i / N).toFixed(3), probability: +p.toFixed(4) }))
      .sort((a, b) => b.probability - a.probability);
    return {
      job: s.job,
      gate_count: N,
      track_crash_rate: +s.trackRate.toFixed(3),
      weak_share: weakShare,
      lead_gates: lead,
      targets: aimed,
      start_distribution: boosted,
      uniform_probability_other_gates: +uniform.toFixed(4),
      // The shape a future experiment-design field would take; nothing reads it yet.
      proposed_experiment_fields: {
        start_gate_weak_share: weakShare,
        start_gate_weights_by_lap_fraction: aimed.reduce((m, t) => {
          m[t.start_lap_fraction] = +((m[t.start_lap_fraction] || 0) + t.weak_share_of_this_target).toFixed(3);
          return m;
        }, {}),
      },
      note: `${GATE_NOTE} Not yet applied by the trainer: Unity picks a uniformly random start gate on every reset.`,
    };
  });
}
