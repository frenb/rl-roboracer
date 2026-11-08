import { z } from 'zod';
import { get, list, post, BASE_URL } from '../api.js';
import { defineTool, confirmArg, requireConfirm, resolveRow, compact } from '../util.js';
import { buildJob, insertJob, pickLists, MODEL_TYPES, COURSES, SCRATCH_BUILT_POLICIES, FLY_POLICY_COURSE } from '../jobPayload.js';
import { summary, compareSamples, round } from '../stats.js';

const MODEL_ROW_KEYS = [
  'create_date', 'location', 'model_type', 'course_type', 'training_iterations', 'avg_return',
  'reward_design_name', 'experiment_design_name', 'job_id', 'is_global_best', 'notes',
];

// Per-trial arrays on a leaderboard_scores row, by metric name.
const METRICS = {
  score: 'scores',
  goals_per_episode: 'avg_goals_per_episode',
  speed: 'avg_speeds',
  episode_length: 'episode_lengths',
};

/** Eval runs for a model: linked by model_id, or by path for rows older than that link. */
function runsFor(model, scores) {
  const id = String(model._id);
  return scores.filter((s) => String(s.model_id || '') === id || (!s.model_id && model.location && s.path === model.location));
}

function pooled(runs, metric) {
  const key = METRICS[metric];
  return runs.flatMap((r) => (Array.isArray(r[key]) ? r[key] : [])).map(Number).filter(Number.isFinite);
}

function evalSummary(runs) {
  if (!runs.length) return { runs: 0 };
  const s = summary(pooled(runs, 'score'));
  const last = runs.map((r) => String(r.create_date || '')).sort().pop();
  return round({ runs: runs.length, trials: s.n, mean_score: s.mean, ci95: s.ci95, last_eval: last });
}

function modelRow(m, scores) {
  const row = { _id: String(m._id), ...compact(m, MODEL_ROW_KEYS) };
  if (scores) row.eval = evalSummary(runsFor(m, scores));
  return row;
}

async function modelsAndScores() {
  const [models, scores] = await Promise.all([list('/get_models'), list('/leaderboard_scores')]);
  return { models, scores };
}

export function register(server) {
  defineTool(server, 'list_models', {
    title: 'List models',
    description: 'Models tab: saved models, newest first (or best eval score first), each with a summary of its leaderboard evals.',
    readOnly: true,
    input: {
      course_type: z.string().optional(),
      model_type: z.string().optional(),
      job_id: z.string().optional().describe('Models written by this training job.'),
      sort: z.enum(['newest', 'best_score']).optional().describe('Default newest.'),
      limit: z.number().int().min(1).max(500).optional().describe('Default 50.'),
    },
  }, async ({ course_type, model_type, job_id, sort = 'newest', limit = 50 }) => {
    const { models, scores } = await modelsAndScores();
    let rows = models
      .filter((m) => !course_type || m.course_type === course_type)
      .filter((m) => !model_type || m.model_type === model_type)
      .filter((m) => !job_id || String(m.job_id || '').endsWith(job_id))
      .map((m) => modelRow(m, scores));
    rows = sort === 'best_score'
      ? rows.sort((a, b) => (b.eval.mean_score ?? -Infinity) - (a.eval.mean_score ?? -Infinity))
      : rows.sort((a, b) => String(b.create_date || '').localeCompare(String(a.create_date || '')));
    return { total_models: models.length, matched: rows.length, models: rows.slice(0, limit) };
  });

  defineTool(server, 'get_model', {
    title: 'Get model',
    description: 'Full model document plus every leaderboard eval run recorded for it.',
    readOnly: true,
    input: { id: z.string() },
  }, async ({ id }) => {
    const { models, scores } = await modelsAndScores();
    const model = resolveRow(models, id, 'model');
    const runs = runsFor(model, scores);
    return { model, eval_summary: evalSummary(runs), eval_runs: runs };
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
      + 'document. Requires confirm: true.',
    destructive: true,
    input: { id: z.string(), confirm: confirmArg },
  }, async ({ id, confirm }) => {
    const model = resolveRow(await list('/get_models'), id, 'model');
    requireConfirm(confirm, `Deleting model ${model._id} (${model.location || 'no files'})`);
    const res = await post('/delete_model', { _id: String(model._id), location: model.location || '' });
    return { deleted: modelRow(model), result: res };
  });

  defineTool(server, 'eval_models', {
    title: 'Evaluate models',
    description: 'Models tab "Eval selected": queue one EVAL job per model. Course, model type and designs follow each model unless '
      + `overridden. RandomPyPolicy / FlyPyPolicy are rebuilt from scratch, so their jobs carry no location; FlyPyPolicy only `
      + `runs on ${FLY_POLICY_COURSE}.`,
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
      dry_run: z.boolean().optional(),
    },
  }, async (args) => {
    const models = await list('/get_models');
    const targets = args.models.map((ref) => {
      const m = resolveRow(models, ref, 'model');
      const modelType = args.model_type || m.model_type || 'SacAgent';
      return { m, modelType, courseType: args.course_type || m.course_type || '', fromScratch: SCRATCH_BUILT_POLICIES.has(modelType) };
    });
    const wrong = targets.filter((t) => t.modelType === 'FlyPyPolicy' && t.courseType !== FLY_POLICY_COURSE);
    if (wrong.length) {
      throw new Error(`FlyPyPolicy can only be evaluated on ${FLY_POLICY_COURSE}; ${wrong.length} of ${targets.length} `
        + `would run on ${[...new Set(wrong.map((t) => t.courseType || '(unset)'))].join(', ')}. Set course_type.`);
    }
    const lists = pickLists();
    const jobs = [];
    for (const t of targets) {
      const { payload, notes } = await buildJob({
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
      let jobId;
      try { jobId = String(resolveRow(jobs, id, 'job')._id); } catch {
        const m = resolveRow(models, id, 'job or model');
        if (!m.job_id) throw new Error(`model ${m._id} has no job_id`);
        jobId = String(m.job_id);
      }
      return label ? { jobId, label } : { jobId };
    });
    const res = await post('/set_tb_compare_jobs', { jobs: entries });
    const host = new URL(BASE_URL).hostname;
    return { ...res, tensorboard: `http://${host}:6006/`, note: 'TensorBoard rescans within ~5 s. Jobs still training appear under current/, not compare/.' };
  });

  defineTool(server, 'get_leaderboard', {
    title: 'Leaderboard',
    description: 'Leaderboard tab: models ranked by mean eval score pooled over all their eval trials.',
    readOnly: true,
    input: {
      course_type: z.string().optional(),
      min_trials: z.number().int().min(1).optional().describe('Only models with at least this many eval trials. Default 1.'),
      limit: z.number().int().min(1).max(200).optional().describe('Default 20.'),
    },
  }, async ({ course_type, min_trials = 1, limit = 20 }) => {
    const { models, scores } = await modelsAndScores();
    const ranked = models
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
      + 'with Welch\'s t-test and Hedges\' g.',
    readOnly: true,
    input: {
      models: z.array(z.string()).min(1).describe('Model ids; the first is the baseline.'),
      metric: z.enum(Object.keys(METRICS)).optional().describe('Default score.'),
    },
  }, async ({ models: refs, metric = 'score' }) => {
    const { models, scores } = await modelsAndScores();
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
