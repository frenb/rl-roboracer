import { z } from 'zod';
import path from 'node:path';
import { get, list, post, LOG_WS_URL, BASE_URL } from '../api.js';
import { defineTool, confirmArg, requireConfirm, objectIdTime, resolveRow, compact, truncate, robotArg, robotOf, isMjlab } from '../util.js';
import { buildJob, insertJob, pickLists, ALL_JOB_TYPES, MODEL_TYPES, JOB_STATUSES, COURSES } from '../jobPayload.js';
import { planJob, loadPlanData, DEFAULTABLE } from '../jobDecisions.js';
import { composeExec, composeLogs, REPO_ROOT } from '../docker.js';

const JOB_ROW_KEYS = [
  'status', 'job_type', 'model_type', 'course_type', 'gym_name', 'reward_design_name',
  'experiment_design_name', 'demo_label', 'percent_complete', 'training_steps', 'num_iterations',
  'num_trials', 'location', 'started_at', 'ended_at', 'paused_at_step', 'proposal_id',
];
// What rl_agent/duck/worker.py writes in place of the car fields.
const DUCK_ROW_KEYS = [
  'status', 'job_type', 'name', 'task_id', 'gym_name', 'reward_design_name', 'experiment_design_name',
  'percent_complete', 'max_iterations', 'duck_iteration', 'mean_reward', 'mean_episode_length',
  'duck_checkpoint', 'checkpoint', 'onnx_location', 'progress_video', 'started_at', 'ended_at', 'paused_at_step',
];
const DUCK_EVAL_KEYS = ['episodes', 'falls', 'fall_rate', 'mean_return', 'mean_episode_length', 'num_envs', 'steps'];

// Trainer paths of training videos (rl_agent/media.py) and where they are on
// this host; the dashboard's /media route serves the same two roots.
const SAVED_MODELS_HOST = path.join(REPO_ROOT, '..', 'saved_models');
const MEDIA_ROOTS = [
  ['/microduck_rl/logs/', path.join(SAVED_MODELS_HOST, 'microduck')],
  ['/saved_models/', SAVED_MODELS_HOST],
];

/** A trainer media path as a dashboard URL and the file on this host. */
function mediaFile(p) {
  const root = typeof p === 'string' && MEDIA_ROOTS.find(([prefix]) => p.startsWith(prefix));
  if (!root) return null;
  return {
    url: `${BASE_URL}/media?path=${encodeURIComponent(p)}`,
    host_path: path.join(root[1], ...p.slice(root[0].length).split('/')),
  };
}

/** Progression video, its GIF and the checkpoint clips (each with its GIF) of a training job, or null.
 * A car clip's iteration is the train step of the eval it recorded. */
export function jobMedia(j) {
  const media = {};
  const video = mediaFile(j.progress_video);
  const gif = mediaFile(j.progress_gif);
  if (video) media.progress_video = video;
  if (gif) media.progress_gif = gif;
  const clips = (Array.isArray(j.videos) ? j.videos : [])
    .map((c) => {
      const clip = { iteration: c && c.iteration, ...mediaFile(c && c.path) };
      const clipGif = mediaFile(c && c.gif);
      if (clipGif) clip.gif = clipGif;
      return clip;
    })
    .filter((c) => c.url);
  if (clips.length) media.training_clips = clips;
  return Object.keys(media).length ? media : null;
}

export function jobRow(j) {
  const duck = isDuck(j);
  const row = { _id: String(j._id), created_at: objectIdTime(j._id), robot: robotOf(j), ...compact(j, duck ? DUCK_ROW_KEYS : JOB_ROW_KEYS) };
  if (typeof row.percent_complete === 'number') row.percent_complete = Math.round(row.percent_complete * 10) / 10;
  if (duck && j.eval_results) row.eval_results = compact(j.eval_results, DUCK_EVAL_KEYS);
  const media = jobMedia(j);
  if (media) row.media = { ...media, training_clips: media.training_clips && media.training_clips.length };
  if (j.eval_error) row.eval_error = truncate(j.eval_error, 300);
  if (j.curriculum_run_id) {
    row.curriculum = compact(j, ['curriculum_run_id', 'curriculum_name', 'curriculum_stage_id', 'curriculum_attempt', 'curriculum_role']);
  }
  return row;
}

/** Fold each curriculum run's jobs under one entry, placed where its newest job falls. */
function groupByCurriculum(rows) {
  const out = [];
  const runs = new Map();
  for (const row of rows) {
    const runId = row.curriculum && row.curriculum.curriculum_run_id;
    if (!runId) { out.push(row); continue; }
    let group = runs.get(runId);
    if (!group) {
      group = { curriculum_run_id: runId, curriculum_name: row.curriculum.curriculum_name, robot: row.robot, jobs: [] };
      runs.set(runId, group);
      out.push(group);
    }
    group.jobs.push(row);
  }
  return out;
}

const byNewest = (a, b) => String(b._id).localeCompare(String(a._id));

// Per-action eligibility, as the Jobs tab buttons apply it.
// The duck worker honours the same statuses (it pauses at its next checkpoint).
const STATUS_ACTIONS = {
  restart: { to: 'NOT_STARTED', from: null, why: 're-queue from the start' },
  pause: { to: 'PAUSE_REQUESTED', from: ['IN_PROGRESS'], why: 'the car trainer checkpoints within ~1 iteration, the duck worker at its last checkpoint, then PAUSED' },
  resume: { to: 'NOT_STARTED', from: ['PAUSED'], why: 'resumes from the checkpoint (car: skipping BC pretrain)' },
  mark_done: { to: 'DONE', from: null, why: 'retire without running; the workers skip it' },
  cancel: { to: 'CANCELLED', from: ['NOT_STARTED', 'IN_PROGRESS', 'PAUSE_REQUESTED'], why: 'the workers skip it; restart re-queues' },
};

const jobInput = {
  job_type: z.enum(ALL_JOB_TYPES).describe('roboracer: TRAIN, DEMO, EVAL, BC_TRAINING_ONLY. microduck: TRAIN, EVAL, EXPORT (checkpoint to ONNX), VIDEO (progression video of a TRAIN job\'s checkpoints).'),
  robot: robotArg.optional().describe('Default roboracer, or microduck when gym is a microduck gym.'),
  model_type: z.enum(MODEL_TYPES).optional().describe('Default SacAgent.'),
  robot_type: z.string().optional().describe('Default robotaxi.'),
  course_type: z.enum(COURSES).optional().describe("Omit for the trainer's default course (donut), or to take the gym's default."),
  status: z.enum(JOB_STATUSES).optional().describe('Default NOT_STARTED (queued).'),
  gym: z.string().optional().describe('Gym id, id suffix or name (list_gyms); for the microduck also a bare mjlab task id. '
    + 'roboracer: preselects the gym\'s default course and designs when those are omitted. microduck TRAIN: required (the task).'),
  reward_design: z.string().optional().describe('Reward design id or name (list_reward_designs), made for the job\'s robot.'),
  experiment_design: z.string().optional().describe('Experiment design id or name (list_experiment_designs), made for the job\'s robot.'),
  demo: z.string().optional().describe('TRAIN / BC_TRAINING_ONLY: demo id (list_demos). Replaces the course-default expert demos.'),
  demo_job_id: z.string().optional(),
  allow_no_demos: z.boolean().optional().describe('TRAIN: run even though the experiment design needs demos the course lacks (trains plain SAC).'),
  num_iterations: z.number().optional().describe('Omit for the trainer default (50000).'),
  training_steps: z.number().optional(),
  skip_first_eval: z.boolean().optional().describe('TRAIN only.'),
  corner_radius_val: z.number().optional(),
  curvature_difficulty_val: z.number().optional(),
  demo_step_budget: z.number().optional().describe('DEMO only.'),
  num_trials: z.number().optional().describe('EVAL only.'),
  num_eval_episodes: z.number().optional().describe('EVAL only: episodes per trial (trainer default 5).'),
  location: z.string().optional().describe('EVAL only: saved-model path, e.g. /saved_models/robotaxi/SacAgent/8.'),
  eval_all_stages: z.boolean().optional().describe("EVAL only: sweep the experiment design's curriculum stages."),
  eval_stage_start: z.number().optional(),
  eval_stage_end: z.number().optional(),
  // ---- microduck ----
  name: z.string().optional().describe('microduck: job name shown on the Jobs tab.'),
  init_from: z.string().optional().describe('microduck TRAIN: warm-start from this model (list_models robot: microduck); '
    + '"none" trains from scratch. Omitted: the gym\'s default_init_checkpoint, if any.'),
  model: z.string().optional().describe('microduck EVAL / EXPORT / VIDEO: the model (checkpoint) to use.'),
  num_envs: z.number().int().optional().describe('microduck TRAIN: robots simulated in parallel. Worker default 2048.'),
  max_iterations: z.number().int().optional().describe('microduck TRAIN: PPO iterations. Worker default 1000.'),
  save_interval: z.number().int().optional().describe('microduck TRAIN: iterations between checkpoints (default 50; also the most a pause loses).'),
  seed: z.number().int().optional().describe('microduck TRAIN.'),
  video: z.boolean().optional().describe('TRAIN, both robots: an MP4 + GIF clip of each checkpoint; training does not wait for it. '
    + 'microduck: rendered in mjlab beside training (mjlab and Unity tasks alike). roboracer: one per eval, its first episode (up to 2 min) '
    + 'from the eval client\'s follow camera, also linked from the best model it saves (gyms built before training clips send none). Default true.'),
  progress_video: z.boolean().optional().describe('TRAIN, both robots: progression video stitched from evenly spaced checkpoint clips when training finishes. Default true.'),
  headless: z.boolean().optional().describe('roboracer, needs gym: run the Unity car clients without windows (-batchmode) for this job; they still render on the GPU, '
    + 'so camera courses and clips work. The clients are relaunched when the job starts, and again windowed by the next job without it. Default false.'),
  eval_num_envs: z.number().int().optional().describe('microduck EVAL: robots in parallel (default 512).'),
  eval_steps: z.number().int().optional().describe('microduck EVAL: control steps at 50 Hz (default 1000 = 20 s).'),
};

/** Last messages from the log viewer's WebSocket (tail -F -n 100 of /tmp/trainer.log). */
function readTrainerLog(waitMs) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let ws;
    try { ws = new WebSocket(LOG_WS_URL); } catch (e) { reject(e); return; }
    const done = () => { try { ws.close(); } catch { /* closing */ } resolve(chunks.join('')); };
    const timer = setTimeout(done, waitMs);
    ws.onmessage = (ev) => chunks.push(String(ev.data));
    ws.onerror = () => { clearTimeout(timer); reject(new Error(`log stream unreachable at ${LOG_WS_URL}`)); };
  });
}

export function register(server) {
  defineTool(server, 'list_jobs', {
    title: 'List jobs',
    description: 'Jobs tab: the job queue, newest first, as compact rows. Use get_job for the full document.',
    readOnly: true,
    input: {
      status: z.array(z.string()).optional().describe('e.g. ["IN_PROGRESS","NOT_STARTED"]. Statuses: NOT_STARTED, IN_PROGRESS, PAUSE_REQUESTED, PAUSED, DONE, FAILED, CANCELLED.'),
      robot: robotArg.optional().describe('Only this robot\'s jobs.'),
      job_type: z.enum(ALL_JOB_TYPES).optional(),
      course_type: z.string().optional().describe('roboracer course.'),
      task_id: z.string().optional().describe('microduck mjlab task id (substring match).'),
      curriculum_run: z.string().optional().describe('Only the jobs of this curriculum run (full id, see list_curriculum_runs).'),
      group_curricula: z.boolean().optional().describe('Fold each curriculum run\'s jobs into one entry with a jobs list, '
        + 'as the Jobs tab does. limit then counts entries.'),
      limit: z.number().int().min(1).max(500).optional().describe('Default 50.'),
    },
  }, async ({ status, robot, job_type, course_type, task_id, curriculum_run, group_curricula, limit = 50 }) => {
    const all = await list('/get_jobs');
    const rows = all
      .filter((j) => !robot || robotOf(j) === robot)
      .filter((j) => !status || status.includes(j.status))
      .filter((j) => !job_type || j.job_type === job_type)
      .filter((j) => !course_type || (!isDuck(j) && j.course_type === course_type))
      .filter((j) => !task_id || String(j.task_id || '').toLowerCase().includes(task_id.toLowerCase()))
      .filter((j) => !curriculum_run || String(j.curriculum_run_id || '') === curriculum_run)
      .sort(byNewest);
    const counts = {};
    const robots = {};
    for (const j of all) {
      counts[j.status] = (counts[j.status] || 0) + 1;
      robots[robotOf(j)] = (robots[robotOf(j)] || 0) + 1;
    }
    const mapped = rows.map(jobRow);
    const entries = group_curricula ? groupByCurriculum(mapped) : mapped;
    return { total_jobs: all.length, status_counts: counts, robot_counts: robots, matched: rows.length, jobs: entries.slice(0, limit) };
  });

  defineTool(server, 'get_job', {
    title: 'Get job',
    description: 'Full job document by id (or the last 8 characters of it). Training jobs with videos also get media: '
      + 'dashboard URLs and host file paths for the progression video, its GIF and the training clips '
      + '(a roboracer clip\'s iteration is the train step of the model it saved).',
    readOnly: true,
    input: { id: z.string() },
  }, async ({ id }) => {
    const job = resolveRow(await list('/get_jobs'), id, 'job');
    const media = jobMedia(job);
    return media ? { ...job, media } : job;
  });

  defineTool(server, 'create_job', {
    title: 'Create job',
    description: 'Jobs tab "New job", for either robot. roboracer (car, Unity + SAC): TRAIN, DEMO, EVAL or BC_TRAINING_ONLY; '
      + 'refuses a TRAIN job whose experiment design needs demos the course lacks, unless demo or allow_no_demos is given. '
      + 'microduck (mjlab + PPO): TRAIN on a gym (task), or EVAL / EXPORT / VIDEO of a model. Builds the same document as '
      + 'the dialog (blank fields omitted so the worker applies its defaults); arguments for the other robot are dropped '
      + 'with a note. Use dry_run to preview the document. '
      + 'TRAIN jobs (and BC_TRAINING_ONLY, for demos) are not queued while the expert demos, iterations, reward design or '
      + 'experiment design are left open: the reply is needs_decision with a question per open setting, each with options '
      + '(recommended first, with reasons from the leaderboard, recent jobs and gym defaults). Put those questions to the '
      + 'user, then call again with the chosen values, or name the setting in use_defaults when they chose its default. '
      + 'Demos are matched to the course and gym: the only set recorded on the job\'s gym is picked by itself, several '
      + 'or none are asked about, and a camera/flat mismatch is refused.',
    input: {
      ...jobInput,
      use_defaults: z.array(z.enum(DEFAULTABLE)).optional().describe('Settings the user chose to leave at their default '
        + '(gym, course or trainer default) rather than be asked about. Only list a setting the user chose this way.'),
      dry_run: z.boolean().optional().describe('Return the document, and any open questions, without queueing it.'),
    },
  }, async (args) => {
    const lists = pickLists();
    const plan = planJob(args, await loadPlanData(lists));
    const merged = { ...args, ...plan.resolved };
    const { payload, notes } = await buildJob(merged, lists);
    notes.unshift(...plan.notes);
    if (args.dry_run) return { dry_run: true, job: payload, notes, questions: plan.questions };
    if (plan.questions.length) {
      return {
        needs_decision: true,
        queued: false,
        ask_user: 'Nothing was queued. Ask the user each question below, showing its options with the recommended one '
          + 'first and its reason; don\'t answer them yourself unless the user already said so. Then call create_job '
          + 'again with the same arguments plus each choice: the option\'s value as that setting, or the setting '
          + 'added to use_defaults for an option marked use_default.',
        questions: plan.questions,
        notes,
      };
    }
    const id = await insertJob(payload);
    return { created: id, job: payload, notes };
  });

  defineTool(server, 'update_jobs', {
    title: 'Restart / pause / resume / mark done / cancel jobs',
    description: 'Jobs tab status buttons, for one or more jobs. Each action only touches eligible jobs, as the buttons do: '
      + 'pause = IN_PROGRESS only; resume = PAUSED only; cancel = NOT_STARTED, IN_PROGRESS or PAUSE_REQUESTED; '
      + 'restart and mark_done apply to any job. Ineligible jobs are reported as skipped.',
    input: {
      ids: z.array(z.string()).min(1).describe('Job ids or id suffixes.'),
      action: z.enum(Object.keys(STATUS_ACTIONS)),
    },
  }, async ({ ids, action }) => {
    const spec = STATUS_ACTIONS[action];
    const jobs = await list('/get_jobs');
    const updated = [];
    const skipped = [];
    for (const ref of ids) {
      let job;
      try { job = resolveRow(jobs, ref, 'job'); } catch (e) { skipped.push({ id: ref, reason: e.message }); continue; }
      if (spec.from && !spec.from.includes(job.status)) {
        skipped.push({ id: String(job._id), status: job.status, reason: `${action} applies to ${spec.from.join(' / ')} jobs` });
        continue;
      }
      await post('/update_job_status', { _id: String(job._id), status: spec.to });
      updated.push({ id: String(job._id), from: job.status, to: spec.to });
    }
    return { action, effect: spec.why, updated, skipped };
  });

  defineTool(server, 'delete_jobs', {
    title: 'Delete jobs',
    description: 'Jobs tab "Delete": remove job documents from MongoDB. Model checkpoints the jobs wrote are kept. Requires confirm: true.',
    destructive: true,
    input: { ids: z.array(z.string()).min(1), confirm: confirmArg },
  }, async ({ ids, confirm }) => {
    requireConfirm(confirm, `Deleting ${ids.length} job(s)`);
    const jobs = await list('/get_jobs');
    const deleted = [];
    const failed = [];
    for (const ref of ids) {
      try {
        const job = resolveRow(jobs, ref, 'job');
        await post('/delete_job', { _id: String(job._id) });
        deleted.push(jobRow(job));
      } catch (e) {
        failed.push({ id: ref, error: e.message });
      }
    }
    return { deleted, failed };
  });

  defineTool(server, 'get_queue_state', {
    title: 'Get queue state',
    description: 'Whether the job queue is paused (Jobs tab Pause/Resume queue toggle).',
    readOnly: true,
  }, async () => get('/get_queue_state'));

  defineTool(server, 'set_queue_paused', {
    title: 'Pause or resume the queue',
    description: 'Jobs tab queue toggle. Pausing stops the trainer picking up new jobs and pause-requests running ones '
      + '(they checkpoint). Resuming re-queues only the jobs the queue pause stopped; jobs paused by hand stay paused.',
    input: { paused: z.boolean() },
  }, async ({ paused }) => post('/set_queue_state', { paused }));

  defineTool(server, 'tail_trainer_log', {
    title: 'Tail trainer log',
    description: 'Logs tab: recent trainer output. roboracer: the car trainer\'s /tmp/trainer.log (the stream starts with the '
      + 'last 100 lines, then adds whatever is written during wait_ms). microduck: a duck job\'s own log in duck-trainer '
      + '(the given job, else the running one), or the duck worker\'s output when no duck job is running.',
    readOnly: true,
    input: {
      robot: robotArg.optional().describe('Default roboracer, or the job\'s robot when job is given.'),
      job: z.string().optional().describe('microduck: job id or suffix whose log to read.'),
      lines: z.number().int().min(1).max(1000).optional().describe('Lines to return from the end. Default 100.'),
      wait_ms: z.number().int().min(500).max(60000).optional().describe('roboracer: how long to listen. Default 2000.'),
      grep: z.string().optional().describe('Only lines matching this regular expression (case-insensitive).'),
    },
  }, async ({ robot, job, lines = 100, wait_ms = 2000, grep }) => {
    let text;
    let source = null;
    let target = null;
    const serviceFor = (r) => (r === 'microduck' ? 'duck-trainer' : `${r}-trainer`);
    if (job || (robot && isMjlab(robot))) {
      const jobs = await list('/get_jobs');
      target = job
        ? resolveRow(jobs, job, 'job')
        : jobs.filter((j) => robotOf(j) === robot && j.status === 'IN_PROGRESS').sort(byNewest)[0];
      if (target && !isMjlab(target)) {
        if (robot && isMjlab(robot)) throw new Error(`job ${target._id} is a roboracer job`);
        target = null;
      }
    }
    if (target) {
      if (!target.log_path) throw new Error(`job ${target._id} has no log yet (status ${target.status})`);
      const n = grep ? 5000 : lines;
      const svc = serviceFor(robotOf(target));
      text = await composeExec(['-T', svc, 'tail', '-n', String(n), target.log_path], { timeoutMs: 30000 });
      source = `${target.log_path} (job ${target._id}, ${target.status})`;
    } else if (robot && isMjlab(robot)) {
      const svc = serviceFor(robot);
      text = await composeLogs(svc, grep ? 5000 : lines, { timeoutMs: 30000 });
      source = `${svc} worker output (no ${robot} job is running)`;
    } else {
      text = await readTrainerLog(wait_ms);
    }
    let out = text.split(/\r?\n/);
    if (out.length && out[out.length - 1] === '') out.pop();
    if (grep) {
      const re = new RegExp(grep, 'i');
      out = out.filter((l) => re.test(l));
    }
    const body = out.slice(-lines).join('\n') || '(no log lines)';
    return source ? `[${source}]\n${body}` : body;
  });
}
