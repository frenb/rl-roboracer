// The /add_job document, built the way dashboard/components.js
// buildJobPayload does for the New Job and Eval dialogs. /add_job inserts
// the body verbatim, so the conventions live client-side:
//   * omit rather than zero: do_job applies its own per-type default for a
//     missing field but treats any present value, including 0, as deliberate;
//   * the "trainer default" course is expressed by leaving course_type out.

import { list, post } from './api.js';
import { resolveRow } from './util.js';

export const JOB_TYPES = ['TRAIN', 'DEMO', 'EVAL', 'BC_TRAINING_ONLY'];
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
  };
}

function activeRow(rows, ref, kind) {
  const row = resolveRow(rows, ref, kind);
  if (row.archived) throw new Error(`${kind} "${row.name || row._id}" is archived; unarchive it first`);
  return row;
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
  const notes = [];
  const jobType = o.job_type;
  const isEval = jobType === 'EVAL';

  let course = o.course_type || '';
  let rewardRef = o.reward_design;
  let experimentRef = o.experiment_design;

  let gym = null;
  if (o.gym) {
    gym = activeRow(await lists.gyms(), o.gym, 'gym');
    if (gym.robot && gym.robot !== 'roboracer') {
      throw new Error(`gym "${gym.name || gym._id}" is a ${gym.robot} gym; this tool builds roboracer jobs only`);
    }
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
    const rd = activeRow(await lists.rewardDesigns(), rewardRef, 'reward design');
    payload.reward_design_id = String(rd._id);
    if (rd.name) payload.reward_design_name = rd.name;
  }
  let design = null;
  if (experimentRef) {
    design = activeRow(await lists.experimentDesigns(), experimentRef, 'experiment design');
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
