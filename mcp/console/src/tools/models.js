import { z } from 'zod';
import { get, list, post, BASE_URL } from '../api.js';
import { defineTool, confirmArg, requireConfirm, resolveRow, compact, robotArg, robotOf, isDuck } from '../util.js';
import { buildJob, buildDuckJob, insertJob, pickLists, MODEL_TYPES, COURSES, SCRATCH_BUILT_POLICIES, FLY_POLICY_COURSE } from '../jobPayload.js';
import { summary, compareSamples, round } from '../stats.js';
import { METRICS, runsFor, pooled } from '../scores.js';
import { jobMedia } from './jobs.js';

const MODEL_ROW_KEYS = [
  'create_date', 'location', 'model_type', 'course_type', 'training_iterations', 'avg_return',
  'reward_design_name', 'experiment_design_name', 'job_id', 'is_global_best', 'notes',
];
// rl_agent/duck/worker.py upsert_model: one model per TRAIN job, at its latest checkpoint.
const DUCK_MODEL_KEYS = [
  'create_date', 'updated_at', 'location', 'onnx_location', 'model_type', 'task_id', 'training_iterations',
  'avg_return', 'reward_design_name', 'experiment_design_name', 'job_id', 'obs_dim', 'act_dim',
];
// A duck EVAL writes one aggregate (rl_agent/duck evaluate script) rather than per-trial arrays.
const DUCK_METRICS = {
  mean_return: { better: 'higher' },
  fall_rate: { better: 'lower' },
  mean_episode_length: { better: 'higher' },
  mean_reward_per_step: { better: 'higher' },
};

function duckEval(m) {
  const r = m.eval_results;
  if (!r) return { runs: 0 };
  return round({ runs: 1, ...compact(r, ['episodes', 'falls', 'fall_rate', 'mean_return', 'mean_episode_length', 'mean_reward_per_step', 'num_envs', 'steps']) });
}

function evalSummary(runs) {
  if (!runs.length) return { runs: 0 };
  const s = summary(pooled(runs, 'score'));
  const last = runs.map((r) => String(r.create_date || '')).sort().pop();
  return round({ runs: runs.length, trials: s.n, mean_score: s.mean, ci95: s.ci95, last_eval: last });
}

function modelRow(m, scores) {
  if (isDuck(m)) return { _id: String(m._id), robot: 'microduck', ...compact(m, DUCK_MODEL_KEYS), eval: duckEval(m) };
  const row = { _id: String(m._id), robot: 'roboracer', ...compact(m, MODEL_ROW_KEYS) };
  if (scores) row.eval = evalSummary(runsFor(m, scores));
  return row;
}

/** Sort key for "best first": car mean eval score, duck mean eval return. */
const bestKey = (row) => (row.robot === 'microduck' ? row.eval.mean_return : row.eval.mean_score) ?? -Infinity;

async function modelsAndScores() {
  const [models, scores] = await Promise.all([list('/get_models'), list('/leaderboard_scores')]);
  return { models, scores };
}

export function register(server) {
  defineTool(server, 'list_models', {
    title: 'List models',
    description: 'Models tab: saved models, newest first (or best eval first), each with a summary of its evals. roboracer '
      + 'models carry leaderboard eval trials; microduck models (one per TRAIN job, at its latest checkpoint) carry the '
      + 'result of their latest EVAL job (fall rate, mean return).',
    readOnly: true,
    input: {
      robot: robotArg.optional().describe('Only this robot\'s models.'),
      course_type: z.string().optional().describe('roboracer course.'),
      task_id: z.string().optional().describe('microduck mjlab task id (substring match).'),
      model_type: z.string().optional(),
      job_id: z.string().optional().describe('Models written by this training job.'),
      sort: z.enum(['newest', 'best_score']).optional().describe('Default newest. best_score: car mean eval score, duck mean eval return.'),
      limit: z.number().int().min(1).max(500).optional().describe('Default 50.'),
    },
  }, async ({ robot, course_type, task_id, model_type, job_id, sort = 'newest', limit = 50 }) => {
    const { models, scores } = await modelsAndScores();
    let rows = models
      .filter((m) => !robot || robotOf(m) === robot)
      .filter((m) => !course_type || (!isDuck(m) && m.course_type === course_type))
      .filter((m) => !task_id || String(m.task_id || '').toLowerCase().includes(task_id.toLowerCase()))
      .filter((m) => !model_type || m.model_type === model_type)
      .filter((m) => !job_id || String(m.job_id || '').endsWith(job_id))
      .map((m) => modelRow(m, scores));
    const newest = (r) => String(r.updated_at || r.create_date || '');
    rows = sort === 'best_score'
      ? rows.sort((a, b) => bestKey(b) - bestKey(a))
      : rows.sort((a, b) => newest(b).localeCompare(newest(a)));
    return { total_models: models.length, matched: rows.length, models: rows.slice(0, limit) };
  });

  defineTool(server, 'get_model', {
    title: 'Get model',
    description: 'Full model document plus every eval recorded for it: leaderboard runs for roboracer, EVAL jobs for microduck.',
    readOnly: true,
    input: { id: z.string() },
  }, async ({ id }) => {
    const { models, scores } = await modelsAndScores();
    const model = resolveRow(models, id, 'model');
    if (isDuck(model)) {
      const jobs = await list('/get_jobs');
      const evals = jobs
        .filter((j) => isDuck(j) && j.job_type === 'EVAL' && j.checkpoint === model.location)
        .map((j) => ({ job: String(j._id), status: j.status, ended_at: j.ended_at || null, eval_results: j.eval_results || null }));
      const trainJob = jobs.find((j) => String(j._id) === String(model.job_id));
      const media = trainJob ? jobMedia(trainJob) : null;
      return { model, eval_summary: duckEval(model), eval_jobs: evals, ...(media ? { media } : {}) };
    }
    const runs = runsFor(model, scores);
    // A car model with its own clip: its training job's videos.
    const trainJob = model.clip && model.job_id
      ? (await list('/get_jobs')).find((j) => String(j._id) === String(model.job_id)) : null;
    const media = trainJob ? jobMedia(trainJob) : null;
    return { model, eval_summary: evalSummary(runs), eval_runs: runs, ...(media ? { media } : {}) };
  });

  defineTool(server, 'get_model_reward_source', {
    title: 'Model reward-design source',
    description: 'Models tab "View source": the reward-design code the model was trained with.',
    readOnly: true,
    input: { id: z.string() },
  }, async ({ id }) => {
    const model = resolveRow(await list('/get_models'), id, 'model');
    return get(`/get_model_source/${encodeURIComponent(String(model._id))}`);
  });

  defineTool(server, 'delete_model', {
    title: 'Delete model',
    description: 'Models tab "Delete": removes the saved-model files under /saved_models, its leaderboard scores, and the model '
      + 'document. A microduck model\'s checkpoints live in duck-trainer and are kept; only the document goes. Requires confirm: true.',
    destructive: true,
    input: { id: z.string(), confirm: confirmArg },
  }, async ({ id, confirm }) => {
    const model = resolveRow(await list('/get_models'), id, 'model');
    requireConfirm(confirm, `Deleting model ${model._id} (${model.location || 'no files'})`);
    const res = await post('/delete_model', { _id: String(model._id), location: model.location || '' });
    const out = { deleted: modelRow(model), result: res };
    if (isDuck(model)) out.note = `checkpoint files kept in duck-trainer: ${model.location}`;
    return out;
  });

  defineTool(server, 'eval_models', {
    title: 'Evaluate models',
    description: 'Models tab "Eval selected": queue one EVAL job per model, for either robot. roboracer: course, model type and '
      + `designs follow each model unless overridden; RandomPyPolicy / FlyPyPolicy are rebuilt from scratch, so their jobs carry `
      + `no location; FlyPyPolicy only runs on ${FLY_POLICY_COURSE}. microduck: runs the model's checkpoint on its task under `
      + 'training conditions (random commands and pushes) and records falls, return and episode length on the job and model.',
    input: {
      models: z.array(z.string()).min(1).describe('Model ids or id suffixes.'),
      num_trials: z.number().int().min(1).max(100).optional().describe('Trials per model. Default 10 (3 smoke, 10 standard, 30 tight CI, 50 publication).'),
      num_eval_episodes: z.number().int().min(1).optional().describe('Episodes per trial. Trainer default 5.'),
      gym: z.string().optional(),
      course_type: z.enum(COURSES).optional().describe('Override every model\'s course.'),
      model_type: z.enum(MODEL_TYPES).optional().describe('Override every model\'s type (e.g. a RandomPyPolicy baseline).'),
      reward_design: z.string().optional(),
      experiment_design: z.string().optional(),
      corner_radius_val: z.number().optional(),
      curvature_difficulty_val: z.number().optional(),
      eval_all_stages: z.boolean().optional(),
      eval_stage_start: z.number().optional(),
      eval_stage_end: z.number().optional(),
      robot_type: z.string().optional(),
      eval_num_envs: z.number().int().min(1).optional().describe('microduck: robots in parallel (default 512).'),
      eval_steps: z.number().int().min(1).optional().describe('microduck: control steps at 50 Hz (default 1000 = 20 s).'),
      dry_run: z.boolean().optional(),
    },
  }, async (args) => {
    const models = await list('/get_models');
    const lists = pickLists();
    const jobs = [];
    const carRefs = [];
    for (const ref of args.models) {
      const m = resolveRow(models, ref, 'model');
      if (!isDuck(m)) { carRefs.push(ref); continue; }
      const { payload, notes } = await buildDuckJob({
        job_type: 'EVAL', model: String(m._id), eval_num_envs: args.eval_num_envs, eval_steps: args.eval_steps,
      }, lists);
      const carOnly = ['num_trials', 'num_eval_episodes', 'gym', 'course_type', 'model_type', 'reward_design',
        'experiment_design', 'corner_radius_val', 'curvature_difficulty_val', 'eval_all_stages'].filter((k) => args[k] !== undefined);
      if (carOnly.length) notes.push(`${carOnly.join(', ')} ignored: roboracer only`);
      jobs.push({ model: String(m._id), payload, notes });
    }
    const targets = carRefs.map((ref) => {
      const m = resolveRow(models, ref, 'model');
      const modelType = args.model_type || m.model_type || 'SacAgent';
      return { m, modelType, courseType: args.course_type || m.course_type || '', fromScratch: SCRATCH_BUILT_POLICIES.has(modelType) };
    });
    const wrong = targets.filter((t) => t.modelType === 'FlyPyPolicy' && t.courseType !== FLY_POLICY_COURSE);
    if (wrong.length) {
      throw new Error(`FlyPyPolicy can only be evaluated on ${FLY_POLICY_COURSE}; ${wrong.length} of ${targets.length} `
        + `would run on ${[...new Set(wrong.map((t) => t.courseType || '(unset)'))].join(', ')}. Set course_type.`);
    }
    for (const t of targets) {
      const { payload, notes } = await buildJob({
        robot: 'roboracer',
        job_type: 'EVAL',
        status: 'NOT_STARTED',
        location: t.fromScratch ? '' : t.m.location,
        robot_type: args.robot_type,
        model_type: t.modelType,
        course_type: t.courseType,
        num_trials: args.num_trials ?? 10,
        num_eval_episodes: args.num_eval_episodes,
        corner_radius_val: args.corner_radius_val,
        curvature_difficulty_val: args.curvature_difficulty_val,
        gym: args.gym,
        no_gym_defaults: true,
        reward_design: args.reward_design,
        experiment_design: args.experiment_design,
        eval_all_stages: args.eval_all_stages,
        eval_stage_start: args.eval_stage_start,
        eval_stage_end: args.eval_stage_end,
      }, lists);
      // Designs default to the ones the model was trained with, even if since archived.
      if (!args.reward_design && t.m.reward_design_id) {
        payload.reward_design_id = String(t.m.reward_design_id);
        payload.reward_design_name = String(t.m.reward_design_name || '');
      }
      if (!args.experiment_design && t.m.experiment_design_id) {
        payload.experiment_design_id = String(t.m.experiment_design_id);
        payload.experiment_design_name = String(t.m.experiment_design_name || '');
      }
      if (t.fromScratch) notes.push(`${t.modelType} is rebuilt from scratch: queued without a location`);
      jobs.push({ model: String(t.m._id), payload, notes });
    }
    if (args.dry_run) return { dry_run: true, jobs };
    for (const j of jobs) j.created = await insertJob(j.payload);
    return { queued: jobs.length, jobs };
  });

  defineTool(server, 'compare_in_tensorboard', {
    title: 'Compare runs in TensorBoard',
    description: 'Models / Analysis "Compare": point TensorBoard\'s compare/ runs at these jobs\' archived data. Replaces the '
      + 'current comparison set. Accepts job ids or model ids (a model maps to the job that trained it).',
    input: {
      runs: z.array(z.object({
        id: z.string().describe('Job id or model id.'),
        label: z.string().regex(/^[a-zA-Z0-9_-]{1,32}$/).optional().describe('Run name prefix shown in TensorBoard, e.g. "base".'),
      })).min(1),
    },
  }, async ({ runs }) => {
    const [jobs, models] = await Promise.all([list('/get_jobs'), list('/get_models')]);
    const entries = runs.map(({ id, label }) => {
      let job = null;
      try { job = resolveRow(jobs, id, 'job'); } catch {
        const m = resolveRow(models, id, 'job or model');
        if (!m.job_id) throw new Error(`model ${m._id} has no job_id`);
        job = jobs.find((j) => String(j._id) === String(m.job_id)) || { _id: m.job_id, robot: m.robot };
      }
      if (isDuck(job)) {
        throw new Error(`job ${job._id} is a microduck run: its TensorBoard events are in duck-trainer under `
          + `${(job.duck_run_dirs || []).join(', ') || '/microduck_rl/logs/rsl_rl'}, which the car TensorBoard does not read. `
          + 'Use compare_models or list_jobs (mean_reward) for duck runs.');
      }
      const jobId = String(job._id);
      return label ? { jobId, label } : { jobId };
    });
    const res = await post('/set_tb_compare_jobs', { jobs: entries });
    const host = new URL(BASE_URL).hostname;
    return { ...res, tensorboard: `http://${host}:6006/`, note: 'TensorBoard rescans within ~5 s. Jobs still training appear under current/, not compare/.' };
  });

  defineTool(server, 'get_leaderboard', {
    title: 'Leaderboard',
    description: 'Leaderboard tab: roboracer models ranked by mean eval score pooled over all their eval trials. With robot: '
      + 'microduck, duck models that have an EVAL result ranked by metric (default mean_return), optionally per task.',
    readOnly: true,
    input: {
      robot: robotArg.optional().describe('Default roboracer.'),
      course_type: z.string().optional().describe('roboracer course.'),
      task_id: z.string().optional().describe('microduck mjlab task id (substring match).'),
      metric: z.enum(Object.keys(DUCK_METRICS)).optional().describe('microduck ranking metric. Default mean_return.'),
      min_trials: z.number().int().min(1).optional().describe('roboracer: only models with at least this many eval trials. Default 1.'),
      limit: z.number().int().min(1).max(200).optional().describe('Default 20.'),
    },
  }, async ({ robot = 'roboracer', course_type, task_id, metric = 'mean_return', min_trials = 1, limit = 20 }) => {
    const { models, scores } = await modelsAndScores();
    if (robot === 'microduck') {
      const sign = DUCK_METRICS[metric].better === 'higher' ? -1 : 1;
      const ranked = models
        .filter((m) => isDuck(m) && m.eval_results && Number.isFinite(Number(m.eval_results[metric])))
        .filter((m) => !task_id || String(m.task_id || '').toLowerCase().includes(task_id.toLowerCase()))
        .sort((a, b) => sign * (Number(a.eval_results[metric]) - Number(b.eval_results[metric])))
        .slice(0, limit)
        .map((m, i) => round({
          rank: i + 1,
          model: String(m._id),
          ...compact(m, ['task_id', 'training_iterations', 'reward_design_name', 'experiment_design_name', 'job_id']),
          eval: duckEval(m),
        }));
      return { robot, metric, better: DUCK_METRICS[metric].better, leaderboard: ranked };
    }
    const ranked = models
      .filter((m) => !isDuck(m))
      .filter((m) => !course_type || m.course_type === course_type)
      .map((m) => ({ m, runs: runsFor(m, scores) }))
      .map(({ m, runs }) => ({ m, runs, s: summary(pooled(runs, 'score')), goals: summary(pooled(runs, 'goals_per_episode')) }))
      .filter((x) => x.s.n >= min_trials)
      .sort((a, b) => b.s.mean - a.s.mean)
      .slice(0, limit)
      .map((x, i) => round({
        rank: i + 1,
        model: String(x.m._id),
        ...compact(x.m, ['location', 'model_type', 'course_type', 'experiment_design_name', 'reward_design_name', 'create_date']),
        eval_runs: x.runs.length,
        trials: x.s.n,
        mean_score: x.s.mean,
        ci95: x.s.ci95,
        median_score: x.s.median,
        best_trial: x.s.max,
        mean_goals_per_episode: x.goals.n ? x.goals.mean : null,
      }));
    return { leaderboard: ranked };
  });

  defineTool(server, 'compare_models', {
    title: 'Compare models (Analysis)',
    description: 'Analysis tab: per-model statistics over pooled eval trials, and each model against the first one (baseline) '
      + 'with Welch\'s t-test and Hedges\' g. microduck models (all the same robot) are compared on their EVAL aggregates '
      + 'with differences from the baseline; a duck EVAL keeps no per-trial samples, so there is no significance test.',
    readOnly: true,
    input: {
      models: z.array(z.string()).min(1).describe('Model ids; the first is the baseline.'),
      metric: z.enum([...Object.keys(METRICS), ...Object.keys(DUCK_METRICS)]).optional()
        .describe(`roboracer: ${Object.keys(METRICS).join(', ')} (default score). microduck: ${Object.keys(DUCK_METRICS).join(', ')} (default mean_return).`),
    },
  }, async ({ models: refs, metric }) => {
    const { models, scores } = await modelsAndScores();
    const picked = refs.map((ref) => resolveRow(models, ref, 'model'));
    const ducks = picked.filter(isDuck).length;
    if (ducks && ducks !== picked.length) throw new Error('compare models of one robot at a time');
    if (ducks) {
      const key = metric || 'mean_return';
      if (!DUCK_METRICS[key]) throw new Error(`microduck metrics: ${Object.keys(DUCK_METRICS).join(', ')}`);
      const value = (m) => (m.eval_results && Number.isFinite(Number(m.eval_results[key])) ? Number(m.eval_results[key]) : null);
      const base = value(picked[0]);
      return round({
        robot: 'microduck',
        metric: key,
        better: DUCK_METRICS[key].better,
        baseline: String(picked[0]._id),
        models: picked.map((m, i) => ({
          model: String(m._id),
          ...compact(m, ['task_id', 'training_iterations', 'reward_design_name', 'experiment_design_name']),
          eval: duckEval(m),
          vs_baseline: i === 0 || base === null || value(m) === null ? null
            : { diff: value(m) - base, ratio: base ? value(m) / base : null },
        })),
        note: picked.some((m) => !m.eval_results) ? 'Models without eval results need an EVAL job first (eval_models).' : undefined,
      });
    }
    metric = metric || 'score';
    if (!METRICS[metric]) throw new Error(`roboracer metrics: ${Object.keys(METRICS).join(', ')}`);
    const bundles = refs.map((ref) => {
      const m = resolveRow(models, ref, 'model');
      const runs = runsFor(m, scores);
      return { m, runs, values: pooled(runs, metric) };
    });
    const base = bundles[0];
    return round({
      metric,
      baseline: String(base.m._id),
      models: bundles.map((b, i) => ({
        model: String(b.m._id),
        ...compact(b.m, ['location', 'course_type', 'experiment_design_name']),
        eval_runs: b.runs.length,
        stats: summary(b.values),
        vs_baseline: i === 0 ? null : compareSamples(b.values, base.values),
      })),
    });
  });

  defineTool(server, 'get_env_specs', {
    title: 'Environment specs',
    description: 'Live observation/action spec per robot type, which the Models tab "Compat" column compares models against.',
    readOnly: true,
  }, async () => list('/get_env_specs'));
}
