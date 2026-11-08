import { z } from 'zod';
import { get, list, post, LOG_WS_URL } from '../api.js';
import { defineTool, confirmArg, requireConfirm, objectIdTime, resolveRow, compact, truncate } from '../util.js';
import { buildJob, insertJob, JOB_TYPES, MODEL_TYPES, JOB_STATUSES, COURSES } from '../jobPayload.js';

const JOB_ROW_KEYS = [
  'status', 'job_type', 'model_type', 'course_type', 'gym_name', 'reward_design_name',
  'experiment_design_name', 'demo_label', 'percent_complete', 'training_steps', 'num_iterations',
  'num_trials', 'location', 'started_at', 'ended_at', 'paused_at_step', 'proposal_id',
];

export function jobRow(j) {
  const row = { _id: String(j._id), created_at: objectIdTime(j._id), ...compact(j, JOB_ROW_KEYS) };
  if (typeof row.percent_complete === 'number') row.percent_complete = Math.round(row.percent_complete * 10) / 10;
  if (j.eval_error) row.eval_error = truncate(j.eval_error, 300);
  return row;
}

const byNewest = (a, b) => String(b._id).localeCompare(String(a._id));

// Per-action eligibility, as the Jobs tab buttons apply it.
const STATUS_ACTIONS = {
  restart: { to: 'NOT_STARTED', from: null, why: 're-queue from the start' },
  pause: { to: 'PAUSE_REQUESTED', from: ['IN_PROGRESS'], why: 'trainer checkpoints within ~1 iteration and sets PAUSED' },
  resume: { to: 'NOT_STARTED', from: ['PAUSED'], why: 'trainer resumes from the checkpoint, skipping BC pretrain' },
  mark_done: { to: 'DONE', from: null, why: 'retire without running; trainer skips it' },
  cancel: { to: 'CANCELLED', from: ['NOT_STARTED', 'IN_PROGRESS', 'PAUSE_REQUESTED'], why: 'trainer skips it; restart re-queues' },
};

const jobInput = {
  job_type: z.enum(JOB_TYPES),
  model_type: z.enum(MODEL_TYPES).optional().describe('Default SacAgent.'),
  robot_type: z.string().optional().describe('Default robotaxi.'),
  course_type: z.enum(COURSES).optional().describe("Omit for the trainer's default course (donut), or to take the gym's default."),
  status: z.enum(JOB_STATUSES).optional().describe('Default NOT_STARTED (queued).'),
  gym: z.string().optional().describe('Gym id, id suffix or name (list_gyms). Also preselects the gym\'s default course and designs when those are omitted.'),
  reward_design: z.string().optional().describe('Reward design id or name (list_reward_designs).'),
  experiment_design: z.string().optional().describe('Experiment design id or name (list_experiment_designs).'),
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
      job_type: z.enum(JOB_TYPES).optional(),
      course_type: z.string().optional(),
      limit: z.number().int().min(1).max(500).optional().describe('Default 50.'),
    },
  }, async ({ status, job_type, course_type, limit = 50 }) => {
    const all = await list('/get_jobs');
    const rows = all
      .filter((j) => !status || status.includes(j.status))
      .filter((j) => !job_type || j.job_type === job_type)
      .filter((j) => !course_type || j.course_type === course_type)
      .sort(byNewest);
    const counts = {};
    for (const j of all) counts[j.status] = (counts[j.status] || 0) + 1;
    return { total_jobs: all.length, status_counts: counts, matched: rows.length, jobs: rows.slice(0, limit).map(jobRow) };
  });

  defineTool(server, 'get_job', {
    title: 'Get job',
    description: 'Full job document by id (or the last 8 characters of it).',
    readOnly: true,
    input: { id: z.string() },
  }, async ({ id }) => resolveRow(await list('/get_jobs'), id, 'job'));

  defineTool(server, 'create_job', {
    title: 'Create job',
    description: 'Jobs tab "New job": queue a TRAIN, DEMO, EVAL or BC_TRAINING_ONLY job. Builds the same document as the dialog '
      + '(blank fields omitted so the trainer applies its defaults) and refuses a TRAIN job whose experiment design needs '
      + 'demos the course lacks, unless demo or allow_no_demos is given. Use dry_run to preview the document.',
    input: { ...jobInput, dry_run: z.boolean().optional().describe('Return the document without queueing it.') },
  }, async (args) => {
    const { payload, notes } = await buildJob(args);
    if (args.dry_run) return { dry_run: true, job: payload, notes };
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
    description: 'Logs tab: recent trainer output (/tmp/trainer.log). The stream starts with the last 100 lines, then adds '
      + 'whatever is written during wait_ms.',
    readOnly: true,
    input: {
      lines: z.number().int().min(1).max(1000).optional().describe('Lines to return from the end. Default 100.'),
      wait_ms: z.number().int().min(500).max(60000).optional().describe('How long to listen. Default 2000.'),
      grep: z.string().optional().describe('Only lines matching this regular expression (case-insensitive).'),
    },
  }, async ({ lines = 100, wait_ms = 2000, grep }) => {
    let out = (await readTrainerLog(wait_ms)).split(/\r?\n/);
    if (out.length && out[out.length - 1] === '') out.pop();
    if (grep) {
      const re = new RegExp(grep, 'i');
      out = out.filter((l) => re.test(l));
    }
    return out.slice(-lines).join('\n') || '(no log lines)';
  });
}
