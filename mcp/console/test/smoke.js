// End-to-end check against the running dashboard: starts the server over
// stdio, calls every read tool, and runs the write tools on throwaway
// records named "mcp-smoke-*" (archived afterwards; jobs are created DONE so
// the trainer never runs them, then deleted). Leaves real records alone.

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const client = new Client({ name: 'smoke', version: '0' });
await client.connect(new StdioClientTransport({
  command: process.execPath,
  args: [path.join(here, '..', 'src', 'index.js')],
  env: { ...process.env },
}));

const results = [];
async function call(name, args = {}, { expectError = false, check } = {}) {
  const res = await client.callTool({ name, arguments: args });
  const text = res.content?.[0]?.text ?? '';
  let data = text;
  try { data = JSON.parse(text); } catch { /* plain text */ }
  let ok = expectError ? !!res.isError : !res.isError;
  let why = res.isError ? text.slice(0, 160) : '';
  if (ok && check) {
    try { check(data); } catch (e) { ok = false; why = e.message; }
  }
  results.push({ ok, name, why: why || (expectError ? 'refused as expected' : '') });
  return data;
}
const assert = (c, m) => { if (!c) throw new Error(m); };

const { tools } = await client.listTools();
console.log(`${tools.length} tools: ${tools.map((t) => t.name).join(', ')}\n`);

// ---- reads ----
const jobs = await call('list_jobs', { limit: 5 }, { check: (d) => assert(Array.isArray(d.jobs), 'no jobs array') });
const anyJob = jobs.jobs?.[0];
if (anyJob) {
  await call('get_job', { id: anyJob._id.slice(-8) }, { check: (d) => assert(d._id === anyJob._id, 'suffix lookup') });
  await call('get_weakness_map', { job_id: anyJob._id });
}
// Default job pick, then a job with trajectories: hotspots, text map, and a point lookup at the top hotspot.
await call('get_weakness_map', { top: 1, ascii: false });
const withPaths = jobs.jobs?.find((j) => j.job_type !== 'DEMO');
if (withPaths) {
  const wm = await call('get_weakness_map', { job_id: withPaths._id, layer: 'crash', threshold: 0.3 });
  const h = wm.hotspots?.[0];
  if (h) {
    const [ax, ay] = wm.ground_axes;
    await call('get_weakness_at', { job_id: withPaths._id, x: h.peak_at[ax], y: h.peak_at[ay] },
      { check: (d) => assert(d.cells.some((c) => c.queried && c.visits > 0), 'queried cell has visits') });
  }
}
await call('get_weakness_map', { job_id: 'zzzzzzzz' }, { expectError: true });
if (withPaths) {
  await call('get_difficult_sections', { job_id: withPaths._id, min_attempts: 1 },
    { check: (d) => assert(d.note || (d.gate_count > 0 && Array.isArray(d.hardest_gates)), 'gate ranking') });
  await call('suggest_start_gates', { job_id: withPaths._id, min_attempts: 1 }, {
    check: (d) => {
      if (!d.start_distribution) return;
      const boosted = d.start_distribution.reduce((s, g) => s + g.probability, 0);
      const total = boosted + d.uniform_probability_other_gates * (d.gate_count - d.start_distribution.length);
      assert(Math.abs(total - 1) < 0.01, `start probabilities sum to ${total}`);
    },
  });
}
await call('get_queue_state', {}, { check: (d) => assert(typeof d.paused === 'boolean', 'paused flag') });
await call('tail_trainer_log', { lines: 5, wait_ms: 1500 });
const models = await call('list_models', { robot: 'roboracer', limit: 5 }, { check: (d) => assert(Array.isArray(d.models), 'models') });
const anyModel = models.models?.[0];
if (anyModel) {
  await call('get_model', { id: anyModel._id });
  await call('get_model_reward_source', { id: anyModel._id });
  await call('compare_models', { models: [anyModel._id] }, { check: (d) => assert(d.models[0].stats, 'stats') });
  await call('eval_models', { models: [anyModel._id], num_trials: 3, dry_run: true },
    { check: (d) => assert(d.jobs[0].payload.job_type === 'EVAL' && d.jobs[0].payload.location === anyModel.location, 'eval payload') });
  await call('eval_models', { models: [anyModel._id], model_type: 'RandomPyPolicy', dry_run: true },
    { check: (d) => assert(!('location' in d.jobs[0].payload), 'scratch policy must drop location') });
  await call('eval_models', { models: [anyModel._id], model_type: 'FlyPyPolicy', course_type: 'donut', dry_run: true }, { expectError: true });
  await call('delete_model', { id: anyModel._id }, { expectError: true });
}
await call('get_leaderboard', { limit: 5 });
await call('get_env_specs');
await call('list_reward_designs');
await call('get_experiment_design_schema', { brief: true }, { check: (d) => assert(d.fields.length > 10, 'schema fields') });
const eds = await call('list_experiment_designs');
await call('list_gyms');
await call('list_demos');
await call('list_proposals', { limit: 3 });
await call('get_madscientist_activity', { limit: 3 });
await call('list_releases', { limit: 3 }, { check: (d) => assert(d.some((r) => r.tagName === 'v0.1'), 'v0.1 release') });
await call('list_gym_builds', {}, { check: (d) => assert(d.length > 0, 'builds') });
await call('get_unity_clients', {}, { check: (d) => assert(Array.isArray(d.unity_clients), 'unity_clients') });

// ---- microduck (reads and dry runs) ----
const duckTask = 'Mjlab-Velocity-Flat-MicroDuck';
await call('list_jobs', { robot: 'microduck', limit: 3 }, { check: (d) => assert(d.jobs.every((j) => j.robot === 'microduck'), 'duck jobs only') });
await call('list_gyms', { robot: 'microduck' }, { check: (d) => assert(d.length > 0 && d.every((g) => !g.reward_defaults), 'slim duck gyms') });
await call('get_gym', { id: duckTask }, { check: (d) => assert(Array.isArray(d.reward_defaults), 'reward_defaults') });
await call('get_reward_term_catalog', { brief: true }, { check: (d) => assert(d.count > 0, 'library functions') });
await call('get_experiment_design_schema', { robot: 'microduck', brief: true }, { check: (d) => assert(d.fields.length > 0, 'duck schema') });
await call('list_reward_designs', { robot: 'microduck' }, { check: (d) => assert(d.every((r) => r.robot === 'microduck'), 'duck designs only') });
await call('lint_reward_design', { robot: 'microduck', task: duckTask, code: 'TERMS = {"no_such_term": {"weight": 1.0}}\n' },
  { check: (d) => assert(d.ok === false, 'unknown duck term should fail') });
const duckModels = await call('list_models', { robot: 'microduck', limit: 5 }, { check: (d) => assert(d.models.every((m) => m.robot === 'microduck'), 'duck models only') });
await call('get_leaderboard', { robot: 'microduck' });
await call('create_job', { robot: 'microduck', job_type: 'TRAIN', gym: duckTask, max_iterations: 10, dry_run: true },
  { check: (d) => assert(d.job.robot === 'microduck' && d.job.task_id === duckTask && d.job.max_iterations === 10, 'duck TRAIN doc') });
await call('create_job', { job_type: 'TRAIN', gym: duckTask, course_type: 'donut', dry_run: true },
  { check: (d) => assert(d.job.robot === 'microduck' && d.notes.some((n) => n.startsWith('course_type')), 'duck gym implies robot; car arg noted') });
await call('create_job', { robot: 'microduck', job_type: 'DEMO', dry_run: true }, { expectError: true });
await call('create_job', { robot: 'microduck', job_type: 'TRAIN', gym: duckTask, reward_design: 'passthrough-course-default', dry_run: true }, { expectError: true });
const duckModel = duckModels && duckModels.models && duckModels.models[0];
if (duckModel) {
  await call('eval_models', { models: [duckModel._id], eval_steps: 100, dry_run: true },
    { check: (d) => assert(d.jobs[0].payload.job_type === 'EVAL' && d.jobs[0].payload.checkpoint === duckModel.location, 'duck EVAL doc') });
  await call('create_job', { robot: 'microduck', job_type: 'EXPORT', model: duckModel._id, dry_run: true },
    { check: (d) => assert(d.job.checkpoint === duckModel.location, 'duck EXPORT doc') });
}
await call('tail_trainer_log', { robot: 'microduck', lines: 5 });
const withMedia = await call('list_jobs', { robot: 'microduck', limit: 500 });
const mediaJob = (withMedia.jobs || []).find((j) => j.media && j.media.progress_video);
if (mediaJob) {
  const full = await call('get_job', { id: mediaJob._id }, { check: (d) => assert(d.media && d.media.progress_video.url, 'media urls') });
  const res = await fetch(full.media.progress_video.url, { headers: { Range: 'bytes=0-99' } });
  results.push({ ok: res.status === 206, name: 'GET /duck_media', why: `HTTP ${res.status}` });
  const bad = await fetch(full.media.progress_video.url.replace(/path=[^&]+/, 'path=' + encodeURIComponent('/microduck_rl/logs/../../etc/passwd')));
  results.push({ ok: bad.status === 400, name: 'GET /duck_media outside root', why: `HTTP ${bad.status}` });
}

// ---- job builder rules (dry runs) ----
const demoHungry = eds.find((d) => d._id !== 'experiment-default' && (d.fields.bc_pretrain_steps > 0 || d.fields.awac_lambda > 0 || d.fields.demo_prefill_count > 0));
if (demoHungry) {
  await call('create_job', { job_type: 'TRAIN', course_type: 'donut_camera', experiment_design: demoHungry._id, dry_run: true }, { expectError: true });
  await call('create_job', { job_type: 'TRAIN', course_type: 'donut_camera', experiment_design: demoHungry._id, allow_no_demos: true, dry_run: true },
    { check: (d) => assert(d.job.allow_no_demos === true, 'allow_no_demos stamped') });
}
await call('create_job', { job_type: 'TRAIN', num_trials: 3, location: '/x', dry_run: true },
  { check: (d) => assert(!('num_trials' in d.job) && !('location' in d.job) && d.notes.length >= 2, 'EVAL-only fields dropped with notes') });

// ---- writes on throwaway records ----
const job = await call('create_job', { job_type: 'EVAL', status: 'DONE', num_trials: 1, location: '/saved_models/mcp-smoke' },
  { check: (d) => assert(/^[0-9a-f]{24}$/.test(d.created), 'job id') });
if (job.created) {
  await call('update_jobs', { ids: [job.created], action: 'pause' }, { check: (d) => assert(d.skipped.length === 1, 'pause must skip a DONE job') });
  await call('update_jobs', { ids: [job.created], action: 'mark_done' }, { check: (d) => assert(d.updated.length === 1, 'mark_done') });
  await call('delete_jobs', { ids: [job.created] }, { expectError: true });
  await call('delete_jobs', { ids: [job.created], confirm: true }, { check: (d) => assert(d.deleted.length === 1, 'deleted') });
}

const code = 'def reward_standard(ctx):\n    return ctx.default()\n';
await call('lint_reward_design', { code: 'import os\ndef reward_standard(c):\n  pass' }, { check: (d) => assert(d.ok === false, 'lint should fail') });
const rd = await call('create_reward_design', { name: 'mcp-smoke-reward', code, author: 'mcp-smoke' });
if (rd.created) {
  await call('update_reward_design', { id: rd.created, description: 'updated by smoke test' });
  await call('get_reward_design', { id: rd.created }, { check: (d) => assert(d.version === 2, 'version bumped') });
  await call('set_reward_design_archived', { id: rd.created, archived: true });
}
await call('create_experiment_design', { name: 'mcp-smoke-bad', fields: { not_a_field: 1 } }, { expectError: true });
const ed = await call('create_experiment_design', { name: 'mcp-smoke-design', author: 'mcp-smoke', fields: { batch_size: 128 } });
if (ed.created) {
  await call('update_experiment_design', { id: ed.created, fields: { batch_size: 64 } });
  await call('get_experiment_design', { id: ed.created }, { check: (d) => assert(d.batch_size === 64, 'field updated') });
  await call('set_experiment_design_archived', { id: ed.created, archived: true });
}
const gym = await call('create_gym', { name: 'mcp-smoke-gym', file_path: '"C:\\nowhere\\robotaxi gym level 1.exe"' });
if (gym.created) {
  await call('update_gym', { id: gym.created, description: 'smoke' });
  await call('set_gym_archived', { id: gym.created, archived: true });
}
const demo = await call('create_demo', { tfrecords_path: '/tmp/mcp-smoke-demos', num_demos: 1, description: 'mcp-smoke-demo' });
if (demo.created) {
  await call('update_demo', { id: demo.created, num_demos: 2 });
  await call('set_demo_archived', { id: demo.created, archived: true });
}

// ---- release guards (no upload) ----
await call('add_gym_to_release', { build: 'real2sim2026.10.27-fences-Unity6.6-v3', tag: 'v0.1' }, { expectError: true });
await call('add_gym_to_release', { tag: 'no-such-release-mcp-smoke' }, { expectError: true });
await call('delete_release_asset', { tag: 'v0.1', asset: 'roboracer-demos-donut.zip' }, { expectError: true });

await client.close();
const failed = results.filter((r) => !r.ok);
for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${r.why ? '  - ' + r.why : ''}`);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length ? 1 : 0);
