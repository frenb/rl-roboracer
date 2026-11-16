// Settings a new training job shouldn't get by default: the expert demos, the
// iterations, the reward design and the experiment design. When the caller
// leaves one out (and doesn't list it in use_defaults), or the demos to match
// are ambiguous, create_job returns these questions, with options and a
// recommendation each, for the user to answer instead of queueing the job.

import { robotOf, isDuck } from './util.js';
import { resolveGym, COURSES_WITHOUT_DEFAULT_DEMOS, PINNED_EXPERIMENT_DESIGN, demoDependentFeatures } from './jobPayload.js';
import { summary } from './stats.js';
import { runsFor, pooled } from './scores.js';

// max_iterations is the microduck's num_iterations.
export const DEFAULTABLE = ['demo', 'num_iterations', 'max_iterations', 'reward_design', 'experiment_design'];
const DUCK_EXPERIMENT_DEFAULT = 'microduck-experiment-default';
const TRAINER_DEFAULT_ITERATIONS = 50000;
const ITERATION_CHOICES = [10000, 25000, 50000];
const MAX_DEMO_OPTIONS = 6;

const isCamera = (course) => String(course || '').startsWith('donut_camera');
const short = (id) => String(id).slice(-8);
const day = (iso) => String(iso || '').slice(0, 10);
const idOf = (row) => String(row._id);
const byNewest = (a, b) => String(b._id).localeCompare(String(a._id));

function hours(seconds) {
  if (seconds < 3600) return `about ${Math.max(1, Math.round(seconds / 60))} min`;
  return `about ${(seconds / 3600).toFixed(seconds < 36000 ? 1 : 0)} h`;
}

/** Seconds per training step of a finished job, start-up included, or null. */
function secondsPerStep(job) {
  const steps = Number(job.training_steps || job.duck_iteration);
  const took = (Date.parse(job.ended_at) - Date.parse(job.started_at)) / 1000;
  return steps > 0 && took > 0 ? took / steps : null;
}

function finishedTrain(jobs, match) {
  return jobs.filter((j) => j.job_type === 'TRAIN' && j.status === 'DONE' && secondsPerStep(j) && match(j)).sort(byNewest);
}

/** Options for one design question, best evidence first, de-duplicated by design. */
function designOptions(candidates, designs) {
  const byId = new Map(designs.map((d) => [idOf(d), d]));
  const seen = new Set();
  const options = [];
  for (const c of candidates) {
    const d = c.id && byId.get(String(c.id));
    if (!d || d.archived || seen.has(idOf(d))) continue;
    seen.add(idOf(d));
    options.push({ label: d.name || idOf(d), value: idOf(d), reason: c.reason });
  }
  return options;
}

function question(setting, prompt, options, extra = {}) {
  if (options.length) options[0].recommended = true;
  return { setting, prompt, options, ...extra };
}

/** The best leaderboard model on a course: { model, mean, n }, or null. */
function bestOnCourse(course, models, scores) {
  let best = null;
  for (const m of models) {
    if (isDuck(m) || m.course_type !== course) continue;
    const s = summary(pooled(runsFor(m, scores), 'score'));
    if (s.n && (!best || s.mean > best.mean)) best = { model: m, mean: s.mean, n: s.n };
  }
  return best;
}

function designKeys(d) {
  const keys = [
    ['bc_pretrain_steps', (v) => `BC ${v} steps`],
    ['critic_warmup_steps', (v) => `critic warm-up ${v}`],
    ['awac_lambda', (v) => `AWAC lambda ${v}`],
    ['demo_prefill_count', (v) => `${v} demo steps kept`],
    ['initial_alpha', (v) => `alpha ${v}`],
    ['num_iterations', (v) => `${v} iterations`],
  ];
  const parts = keys.filter(([k]) => d[k] !== undefined && d[k] !== null).map(([k, f]) => f(d[k]));
  return parts.length ? ` (${parts.join(', ')})` : '';
}

function carQuestions(o, data, notes, resolved) {
  const questions = [];
  const accepted = new Set(o.use_defaults || []);
  const gym = o.gym ? resolveGym(data.gyms, o.gym) : null;
  const course = o.course_type || (gym && gym.default_course_type) || 'donut';
  const sameSetup = (j) => !isDuck(j) && (j.course_type || 'donut') === course && (!gym || String(j.gym_id) === idOf(gym));
  const recent = data.jobs.filter((j) => j.job_type === 'TRAIN' && sameSetup(j)).sort(byNewest)[0];
  const best = bestOnCourse(course, data.models, data.scores);
  const bestWhy = best && `used by the best ${course} model on the leaderboard (mean score ${best.mean.toFixed(1)} over ${best.n} trials)`;
  const recentWhy = recent && `used by the latest ${course} job on this gym, ${short(recent._id)} (${day(recent.started_at || recent.created_at)})`;

  // ---- reward design
  if (!o.reward_design && !accepted.has('reward_design')) {
    const options = designOptions([
      { id: best && best.model.reward_design_id, reason: bestWhy },
      { id: recent && recent.reward_design_id, reason: recentWhy },
      { id: gym && gym.default_reward_design_id, reason: "the gym's default" },
    ], data.rewardDesigns.filter((d) => robotOf(d) === 'roboracer'));
    options.push({ label: gym && gym.default_reward_design_id ? "The gym's default" : `The trainer's built-in ${course} reward`,
      use_default: true, reason: 'what the job gets with no reward design' });
    questions.push(question('reward_design', 'Which reward design?', options));
  }

  // ---- experiment design
  let design = null;
  if (o.experiment_design) {
    design = data.experimentDesigns.find((d) => idOf(d) === String(o.experiment_design) || d.name === o.experiment_design
      || idOf(d).endsWith(String(o.experiment_design))) || null;
  } else if (!accepted.has('experiment_design')) {
    const designs = data.experimentDesigns.filter((d) => robotOf(d) === 'roboracer');
    const options = designOptions([
      { id: best && best.model.experiment_design_id, reason: bestWhy },
      { id: recent && recent.experiment_design_id, reason: recentWhy },
      { id: gym && gym.default_experiment_design_id, reason: "the gym's default" },
      { id: PINNED_EXPERIMENT_DESIGN, reason: "the trainer's defaults, the baseline every other design is compared against" },
    ], designs);
    const byId = new Map(designs.map((d) => [idOf(d), d]));
    for (const opt of options) {
      opt.label += designKeys(byId.get(opt.value));
      if (isCamera(course) !== /camera/i.test(opt.label) && opt.value !== PINNED_EXPERIMENT_DESIGN) {
        opt.reason += isCamera(course) ? '; not made for camera courses' : '; made for camera courses';
      }
    }
    questions.push(question('experiment_design', 'Which experiment design?', options));
  }

  // ---- iterations
  if (o.num_iterations === undefined && !accepted.has('num_iterations')) {
    const pace = finishedTrain(data.jobs, sameSetup)[0] || finishedTrain(data.jobs, (j) => !isDuck(j) && j.course_type === course)[0];
    const rate = pace && secondsPerStep(pace);
    const recommended = (recent && Number(recent.num_iterations)) || TRAINER_DEFAULT_ITERATIONS;
    const sizes = [recommended, ...ITERATION_CHOICES.filter((n) => n !== recommended)];
    const options = sizes.map((n) => ({
      label: `${n.toLocaleString('en-US')} iterations${rate ? `, ${hours(n * rate)}` : ''}`,
      value: n,
      reason: n === recommended && recent ? `what ${short(recent._id)} ran`
        : n === TRAINER_DEFAULT_ITERATIONS ? "the trainer's default" : 'a shorter run',
    }));
    questions.push(question('num_iterations', 'How many training iterations?', options, rate ? {
      note: `Times assume job ${short(pace._id)}'s pace, ${rate.toFixed(2)} s per step including start-up; `
        + 'start-up is longer on camera courses and with BC pretraining.',
    } : { note: 'No finished job on this course to estimate run times from.' }));
  }

  // ---- expert demos
  const usesDemos = design ? demoDependentFeatures(design).length > 0 || String(design._id) === PINNED_EXPERIMENT_DESIGN : true;
  if (!usesDemos) return questions;
  const demos = data.demos.filter((d) => !d.archived && Number(d.num_steps) > 0);
  if (o.demo) {
    const demo = demos.find((d) => idOf(d) === String(o.demo) || idOf(d).endsWith(String(o.demo)));
    if (demo && demo.course && isCamera(demo.course) !== isCamera(course)) {
      throw new Error(`demo ${short(demo._id)} is ${demo.course} (${isCamera(demo.course) ? 'camera' : 'flat'} observations) `
        + `but the job's course ${course} has ${isCamera(course) ? 'camera' : 'flat'} observations; pick a matching demo (list_demos)`);
    }
    if (demo && gym && demo.gym_name !== gym.name) {
      notes.push(`demo ${short(demo._id)} was recorded on ${demo.gym_name || 'an unknown gym'}, the job runs on ${gym.name}`);
    }
    return questions;
  }
  if (accepted.has('demo')) return questions;
  const matching = demos.filter((d) => d.course === course).sort(byNewest);
  const sameGym = gym ? matching.filter((d) => d.gym_name === gym.name) : [];
  if (sameGym.length === 1) {
    resolved.demo = idOf(sameGym[0]);
    notes.push(`demo ${short(sameGym[0]._id)} picked: the only ${course} demo set recorded on ${gym.name}`);
    return questions;
  }
  const ranked = [...sameGym, ...matching.filter((d) => !sameGym.includes(d))].slice(0, MAX_DEMO_OPTIONS);
  const options = ranked.map((d) => ({
    label: `${d.description || short(d._id)}: ${d.num_demos} demos, ${Number(d.num_steps).toLocaleString('en-US')} steps`,
    value: idOf(d),
    reason: `${d.gym_name ? `recorded on ${d.gym_name}` : 'recorded on an unknown gym'}, ${day(d.collected_at || d.created_at)}`
      + (gym && d.gym_name === gym.name ? ', the job\'s gym' : ''),
  }));
  if (!COURSES_WITHOUT_DEFAULT_DEMOS.has(course)) {
    options.push({ label: `The course's default ${course} demos`, use_default: true, reason: 'what the job gets with no demo' });
  }
  if (!options.length) return questions;
  questions.push(question('demo', `Which expert demos? ${course} needs ${isCamera(course) ? 'camera' : 'flat'} demos.`, options,
    gym && !sameGym.length ? {
      note: `No ${course} demo set was recorded on ${gym.name}: a clone of another gym's demos learns that gym's handling. `
        + 'Recording a set there first (a DEMO job) is the alternative.',
    } : {}));
  return questions;
}

function duckQuestions(o, data) {
  const questions = [];
  const accepted = new Set(o.use_defaults || []);
  const gym = resolveGym(data.gyms, o.gym);
  const task = String(gym.task_id || idOf(gym).replace(/^microduck:/, ''));
  const sameTask = (j) => isDuck(j) && j.task_id === task;
  const recent = data.jobs.filter((j) => j.job_type === 'TRAIN' && sameTask(j)).sort(byNewest)[0];
  const recentWhy = recent && `used by the latest job on this task, ${short(recent._id)} (${day(recent.started_at || recent.created_at)})`;

  for (const [setting, list, fallbackId] of [
    ['reward_design', data.rewardDesigns, null],
    ['experiment_design', data.experimentDesigns, DUCK_EXPERIMENT_DEFAULT],
  ]) {
    if (o[setting] || accepted.has(setting)) continue;
    const options = designOptions([
      { id: recent && recent[`${setting}_id`], reason: recentWhy },
      { id: fallbackId, reason: 'sets nothing: every value comes from the task' },
    ], list.filter((d) => robotOf(d) === 'microduck'));
    if (!options.some((opt) => opt.value === fallbackId)) {
      options.push({ label: 'The task defaults', use_default: true, reason: 'every value comes from the mjlab task' });
    }
    questions.push(question(setting, `Which ${setting.replace('_', ' ')}?`, options));
  }

  if (o.max_iterations === undefined && !accepted.has('max_iterations')) {
    const pace = finishedTrain(data.jobs, sameTask)[0];
    const rate = pace && secondsPerStep(pace);
    const recommended = (recent && Number(recent.max_iterations)) || 1000;
    const sizes = [recommended, ...[1000, 2000, 5000].filter((n) => n !== recommended)];
    questions.push(question('max_iterations', 'How many PPO iterations?', sizes.map((n) => ({
      label: `${n.toLocaleString('en-US')} iterations${rate ? `, ${hours(n * rate)}` : ''}`,
      value: n,
      reason: n === recommended && recent ? `what ${short(recent._id)} ran` : n === 1000 ? "the worker's default" : 'a longer run',
    })), rate ? { note: `Times assume job ${short(pace._id)}'s pace, ${rate.toFixed(2)} s per iteration.` } : {}));
  }
  return questions;
}

/**
 * Questions for the settings `o` (create_job arguments) leaves open, and any
 * setting worked out unambiguously (`resolved`, merged into the arguments).
 * Only TRAIN jobs (and the car's BC_TRAINING_ONLY) are asked about.
 */
export function planJob(o, data) {
  const notes = [];
  const resolved = {};
  const duck = o.robot ? o.robot === 'microduck' : Boolean(o.gym && isDuck(resolveGym(data.gyms, o.gym)));
  let questions = [];
  if (duck && o.job_type === 'TRAIN' && o.gym) questions = duckQuestions(o, data);
  if (!duck && (o.job_type === 'TRAIN' || o.job_type === 'BC_TRAINING_ONLY')) {
    questions = carQuestions(o, data, notes, resolved);
    if (o.job_type === 'BC_TRAINING_ONLY') questions = questions.filter((q) => q.setting === 'demo');
  }
  return { questions, resolved, notes };
}

export async function loadPlanData(lists) {
  const [gyms, demos, jobs, models, scores, rewardDesigns, experimentDesigns] = await Promise.all([
    lists.gyms(), lists.demos(), lists.jobs(), lists.models(), lists.scores(), lists.rewardDesigns(), lists.experimentDesigns(),
  ]);
  return { gyms, demos, jobs, models, scores, rewardDesigns, experimentDesigns };
}
