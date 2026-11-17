// The /add_job document, built the way dashboard/components.js
// buildJobPayload does for the New Job and Eval dialogs. /add_job inserts
// the body verbatim, so the conventions live client-side:
//   * omit rather than zero: do_job applies its own per-type default for a
//     missing field but treats any present value, including 0, as deliberate;
//   * the "trainer default" course is expressed by leaving course_type out.

import { list, post } from './api.js';
import { resolveRow, robotOf, isMjlab } from './util.js';

export const JOB_TYPES = ['TRAIN', 'DEMO', 'EVAL', 'BC_TRAINING_ONLY'];
// rl_agent/duck/worker.py do_job.
export const DUCK_JOB_TYPES = ['TRAIN', 'EVAL', 'EXPORT', 'VIDEO'];
export const ALL_JOB_TYPES = [...new Set([...JOB_TYPES, ...DUCK_JOB_TYPES])];
export const MODEL_TYPES = ['SacAgent', 'GreedyPolicy', 'RandomPyPolicy', 'FlyPyPolicy'];
export const JOB_STATUSES = ['NOT_STARTED', 'IN_PROGRESS', 'DONE'];
export const COURSES = [
  'donut', 'donut_no_hint', 'donut_camera', 'donut_camera_no_rays', 'fly_donut', 'fly_donut_flow',
];

// Must match COURSE_DEFAULT_DEMO_JOB_IDS (None entries) in rl_agent/robotaxi.py.
export const COURSES_WITHOUT_DEFAULT_DEMOS = new Set([
  'donut_camera', 'donut_camera_no_rays', 'fly_donut', 'fly_donut_flow',
]);
export const PINNED_EXPERIMENT_DESIGN = 'experiment-default';

// Rebuilt from scratch rather than loaded, so do_job refuses a location.
export const SCRATCH_BUILT_POLICIES = new Set(['RandomPyPolicy', 'FlyPyPolicy']);
// FlyPyPolicy encodes the 31-D ray vector itself.
export const FLY_POLICY_COURSE = 'donut_no_hint';

/** Mirrors _demo_dependent_design_features in robotaxi.py. */
export function demoDependentFeatures(design) {
  if (!design || String(design._id) === PINNED_EXPERIMENT_DESIGN) return [];
  return [
    ['bc_pretrain_steps', 'BC pretrain'],
    ['critic_warmup_steps', 'critic warm-up'],
    ['awac_lambda', 'AWAC'],
    ['demo_prefill_count', 'demo prefill'],
  ].filter(([k]) => Number(design[k]) > 0).map(([, label]) => label);
}

export function demoLabel(r) {
  const desc = r.description || r.tfrecords_path || '(no description)';
  const n = r.num_demos == null ? '?' : r.num_demos;
  return `${desc} \u2014 ${n} demos${r.course ? ', ' + r.course : ''}`;
}

/** Lazily-fetched pick lists, one fetch per tool call. */
export function pickLists() {
  const cache = {};
  const load = (k, path) => (cache[k] ||= list(path));
  return {
    gyms: () => load('gyms', '/get_gyms'),
    demos: () => load('demos', '/get_demos'),
    rewardDesigns: () => load('rd', '/get_reward_designs'),
    experimentDesigns: () => load('ed', '/get_experiment_designs'),
    jobs: () => load('jobs', '/get_jobs'),
    models: () => load('models', '/get_models'),
    scores: () => load('scores', '/leaderboard_scores'),
    // /get_models carries every car model; mjlab robots use /get_robot_models.
    robotModels: (robot = 'microduck') => (cache[`rm:${robot}`] ||= list(`/get_robot_models?robot=${encodeURIComponent(robot)}`)),
    duckModels: () => (cache['rm:microduck'] ||= list('/get_robot_models?robot=microduck')),
  };
}

function activeRow(rows, ref, kind) {
  const row = resolveRow(rows, ref, kind);
  if (row.archived) throw new Error(`${kind} "${row.name || row._id}" is archived; unarchive it first`);
  return row;
}

/** A gym by id, name, or (duck) bare mjlab task id such as Mjlab-Velocity-Flat-MicroDuck-Rollers. */
export function resolveGym(rows, ref) {
  const want = String(ref || '').trim();
  const byTask = rows.filter((g) => g.task_id && String(g.task_id).toLowerCase() === want.toLowerCase());
  if (byTask.length === 1) return byTask[0];
  return resolveRow(rows, want, 'gym');
}

/** A design usable by `robot`: car designs carry no robot field. */
function robotRow(rows, ref, kind, robot) {
  const row = activeRow(rows, ref, kind);
  if (robotOf(row) !== robot) {
    throw new Error(`${kind} "${row.name || row._id}" is a ${robotOf(row)} design; pick one made for the ${robot}`);
  }
  return row;
}

/** Whether these create_job arguments describe an mjlab robot job (e.g. microduck, openbeni). */
export async function wantsDuck(o, lists) {
  if (o.robot) return isMjlab(o.robot);
  if (o.gym) return isMjlab(resolveGym(await lists.gyms(), o.gym));
  return false;
}

// Car-only create_job arguments, reported when a duck job is given them.
const CAR_ONLY_ARGS = [
  'model_type', 'robot_type', 'course_type', 'demo', 'demo_job_id', 'allow_no_demos', 'num_iterations',
  'training_steps', 'skip_first_eval', 'corner_radius_val', 'curvature_difficulty_val', 'demo_step_budget',
  'num_trials', 'num_eval_episodes', 'location', 'eval_all_stages', 'eval_stage_start', 'eval_stage_end',
  'headless',
];
// Duck-only arguments, reported when a car job is given them.
const DUCK_ONLY_ARGS = [
  'name', 'init_from', 'model', 'num_envs', 'max_iterations', 'save_interval', 'seed',
  'eval_num_envs', 'eval_steps',
];

function positiveInt(v, label) {
  if (v === undefined || v === null || v === '') return undefined;
  const n = Number(v);
  if (!Number.isInteger(n) || n <= 0) throw new Error(`${label} must be a positive whole number`);
  return n;
}

/**
 * One MicroDuck job document, as dashboard/jobs.html buildDuckPayload builds
 * it. /add_job then stamps robot, task_id and gym_name from the gym.
 */
export async function buildDuckJob(o, lists = pickLists()) {
  const notes = [];
  let robot = o.robot ? String(o.robot) : '';
  let resolvedGym = null;
  if (!robot && o.gym) {
    resolvedGym = resolveGym(await lists.gyms(), o.gym);
    robot = robotOf(resolvedGym);
  }
  if (!robot) robot = 'microduck';
  const modelsFn = () => (lists.robotModels ? lists.robotModels(robot) : lists.duckModels());

  const jobType = o.job_type;
  if (!DUCK_JOB_TYPES.includes(jobType)) {
    throw new Error(`${robot} jobs are ${DUCK_JOB_TYPES.join(', ')}; ${jobType} is a roboracer job type`);
  }
  for (const k of CAR_ONLY_ARGS) {
    if (o[k] !== undefined && o[k] !== false && o[k] !== '') notes.push(`${k} ignored: roboracer only`);
  }
  const payload = { robot, job_type: jobType, status: o.status || 'NOT_STARTED', create_date: new Date().toISOString() };
  if (o.name) payload.name = String(o.name);

  if (jobType === 'TRAIN') {
    if (!o.gym) throw new Error(`gym is required for a ${robot} TRAIN job: a ${robot} gym id, name or mjlab task id (list_gyms robot: ${robot})`);
    const gym = resolvedGym || resolveGym(await lists.gyms(), o.gym);
    if (gym.archived) throw new Error(`gym "${gym.name || gym._id}" is archived; unarchive it first`);
    if (robotOf(gym) !== robot) throw new Error(`gym "${gym.name || gym._id}" is a ${robotOf(gym)} gym`);
    payload.gym_id = String(gym._id);
    const prefix = `${robot}:`;
    const rawId = String(gym.task_id || gym._id);
    payload.task_id = rawId.startsWith(prefix) ? rawId.slice(prefix.length) : rawId;
    if (gym.name) payload.gym_name = gym.name;
    if (o.reward_design) {
      const rd = robotRow(await lists.rewardDesigns(), o.reward_design, 'reward design', robot);
      payload.reward_design_id = String(rd._id);
      if (rd.name) payload.reward_design_name = rd.name;
    }
    if (o.experiment_design) {
      const ed = robotRow(await lists.experimentDesigns(), o.experiment_design, 'experiment design', robot);
      payload.experiment_design_id = String(ed._id);
      if (ed.name) payload.experiment_design_name = ed.name;
    }
    // The form preselects the gym's default warm start; init_from "none" opts out.
    if (o.init_from && String(o.init_from).toLowerCase() !== 'none') {
      const m = resolveRow(await modelsFn(), o.init_from, `${robot} model`);
      payload.init_checkpoint = m.location;
    } else if (o.init_from === undefined && gym.default_init_checkpoint) {
      payload.init_checkpoint = gym.default_init_checkpoint;
      notes.push(`init_checkpoint defaulted from gym: ${gym.default_init_checkpoint} (init_from: "none" trains from scratch)`);
    }
    for (const [k, label] of [['num_envs', 'num_envs'], ['max_iterations', 'max_iterations'], ['save_interval', 'save_interval']]) {
      const n = positiveInt(o[k], label);
      if (n !== undefined) payload[k] = n;
    }
    if (o.seed !== undefined && o.seed !== null && o.seed !== '') payload.seed = Number(o.seed);
    payload.video = o.video !== false;
    payload.progress_video = o.progress_video !== false;
    for (const k of ['model', 'eval_num_envs', 'eval_steps']) {
      if (o[k] !== undefined) notes.push(`${k} ignored: not used by TRAIN jobs`);
    }
    return { payload, notes };
  }

  if (!o.model) throw new Error(`model is required for a ${robot} ${jobType} job (list_models robot: ${robot})`);
  const model = resolveRow(await modelsFn(), o.model, `${robot} model`);
  payload.task_id = model.task_id;
  payload.source_job_id = model.job_id;
  if (jobType !== 'VIDEO') payload.checkpoint = model.location;
  if (jobType === 'EVAL') {
    for (const k of ['eval_num_envs', 'eval_steps']) {
      const n = positiveInt(o[k], k);
      if (n !== undefined) payload[k] = n;
    }
  }
  for (const k of ['gym', 'reward_design', 'experiment_design', 'init_from', 'num_envs', 'max_iterations', 'save_interval', 'seed']) {
    if (o[k] !== undefined) notes.push(`${k} ignored: ${jobType} jobs use the model's task and settings`);
  }
  return { payload, notes };
}

const optionalNumber = (v) => (v === undefined || v === null || v === '' || !Number.isFinite(Number(v)) ? undefined : Number(v));

/**
 * Build one job document. `o` uses tool-argument names; gym / demo /
 * reward_design / experiment_design are references (id, id suffix or
 * name) resolved against the live collections. Returns
 * { payload, notes } where notes explain defaults applied or inputs
 * dropped, the way the dialog would have silently done.
 */
export async function buildJob(o, lists = pickLists()) {
  if (await wantsDuck(o, lists)) return buildDuckJob(o, lists);
  const notes = [];
  const jobType = o.job_type;
  const isEval = jobType === 'EVAL';
  if (!JOB_TYPES.includes(jobType)) {
    throw new Error(`roboracer jobs are ${JOB_TYPES.join(', ')}; ${jobType} is a microduck job type (pass robot: "microduck")`);
  }
  for (const k of DUCK_ONLY_ARGS) {
    if (o[k] !== undefined) notes.push(`${k} ignored: microduck only`);
  }

  let course = o.course_type || '';
  let rewardRef = o.reward_design;
  let experimentRef = o.experiment_design;

  let gym = null;
  if (o.gym) {
    gym = activeRow(await lists.gyms(), o.gym, 'gym');
    // The dialog preselects the gym's validated course/designs unless set.
    const defaults = o.no_gym_defaults ? [] : [
      ['course_type', gym.default_course_type, () => !course, (v) => { course = v; }],
      ['reward_design', gym.default_reward_design_id, () => !rewardRef, (v) => { rewardRef = v; }],
      ['experiment_design', gym.default_experiment_design_id, () => !experimentRef, (v) => { experimentRef = v; }],
    ];
    for (const [name, value, unset, apply] of defaults) {
      if (value && unset()) { apply(String(value)); notes.push(`${name} defaulted from gym: ${value}`); }
    }
  }

  const payload = {
    status: o.status || 'NOT_STARTED',
    job_type: jobType,
    model_type: o.model_type || 'SacAgent',
    robot_type: o.robot_type || 'robotaxi',
    demo_job_id: String(o.demo_job_id || ''),
    pass_through_actions: false,
  };

  const typeOnly = {
    demo_step_budget: jobType === 'DEMO',
    num_trials: isEval,
    num_eval_episodes: isEval,
    location: isEval,
    eval_all_stages: isEval,
    eval_stage_start: isEval,
    eval_stage_end: isEval,
    skip_first_eval: jobType === 'TRAIN',
    video: jobType === 'TRAIN',
    progress_video: jobType === 'TRAIN',
  };
  for (const [k, ok] of Object.entries(typeOnly)) {
    if (!ok && o[k] !== undefined && o[k] !== false && o[k] !== '') notes.push(`${k} ignored: not used by ${jobType} jobs`);
  }

  const numbers = {
    num_iterations: o.num_iterations,
    training_steps: o.training_steps,
    demo_step_budget: typeOnly.demo_step_budget ? o.demo_step_budget : undefined,
    num_trials: isEval ? o.num_trials : undefined,
    num_eval_episodes: isEval ? o.num_eval_episodes : undefined,
    corner_radius_val: o.corner_radius_val,
    curvature_difficulty_val: o.curvature_difficulty_val,
  };
  for (const [k, v] of Object.entries(numbers)) {
    const n = optionalNumber(v);
    if (n !== undefined) payload[k] = n;
  }
  if (o.skip_first_eval && jobType === 'TRAIN') payload.skip_first_eval = true;
  // Training clips (rl_agent/car_clips.py), on unless turned off, as in the New Job dialog.
  if (jobType === 'TRAIN') {
    payload.video = o.video !== false;
    payload.progress_video = o.progress_video !== false;
  }
  // The Unity supervisors relaunch on the gym switch at job start, so a job
  // without a gym can't change how its clients run.
  if (o.headless) {
    if (!gym) throw new Error('headless needs a gym: the clients are relaunched headless when the job switches to its gym');
    payload.headless = true;
  }
  if (course) payload.course_type = course;
  if (isEval && o.location) payload.location = String(o.location);

  if (gym) {
    payload.gym_id = String(gym._id);
    payload.gym_name = gym.name || '';
    payload.gym_file_path = gym.file_path || '';
  }

  if (o.demo) {
    if (jobType === 'TRAIN' || jobType === 'BC_TRAINING_ONLY') {
      const demo = activeRow(await lists.demos(), o.demo, 'demo');
      if (!demo.tfrecords_path) throw new Error(`demo ${demo._id} has no tfrecords_path`);
      payload.demo_id = String(demo._id);
      payload.demo_label = demoLabel(demo);
      payload.demo_tfrecords_paths = [demo.tfrecords_path];
      payload.skip_default_demo = true;
      if (demo.source_job_id) {
        payload.demo_job_ids = [String(demo.source_job_id)];
        if (!payload.demo_job_id) payload.demo_job_id = String(demo.source_job_id);
      }
    } else {
      notes.push(`demo ignored: only TRAIN and BC_TRAINING_ONLY jobs read demos`);
    }
  }

  if (rewardRef) {
    const rd = robotRow(await lists.rewardDesigns(), rewardRef, 'reward design', 'roboracer');
    payload.reward_design_id = String(rd._id);
    if (rd.name) payload.reward_design_name = rd.name;
  }
  let design = null;
  if (experimentRef) {
    design = robotRow(await lists.experimentDesigns(), experimentRef, 'experiment design', 'roboracer');
    payload.experiment_design_id = String(design._id);
    if (design.name) payload.experiment_design_name = design.name;
  }

  if (isEval && o.eval_all_stages) {
    payload.eval_all_stages = true;
    const s = optionalNumber(o.eval_stage_start);
    const e = optionalNumber(o.eval_stage_end);
    if (s !== undefined) payload.eval_stage_start = s;
    if (e !== undefined) payload.eval_stage_end = e;
  }

  // Same guard as the New Job dialog's confirm and do_job's refusal.
  if (jobType === 'TRAIN' && !payload.demo_tfrecords_paths && COURSES_WITHOUT_DEFAULT_DEMOS.has(course)) {
    const features = demoDependentFeatures(design);
    if (features.length) {
      if (!o.allow_no_demos) {
        throw new Error(`experiment design "${design.name}" uses ${features.join(', ')}, but no demo is `
          + `picked and ${course} has no default demos, so all of these would be skipped and the job `
          + `would train plain SAC from scratch. Pass demo (see list_demos), or allow_no_demos: true `
          + `to train from scratch anyway.`);
      }
      payload.allow_no_demos = true;
      notes.push(`allow_no_demos set: ${features.join(', ')} will be skipped`);
    }
  }
  return { payload, notes };
}

export async function insertJob(payload) {
  const res = await post('/add_job', payload);
  return String(res.insertedId || '');
}
