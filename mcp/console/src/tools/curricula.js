import { z } from 'zod';
import { get, list, post } from '../api.js';
import { defineTool, confirmArg, requireConfirm, resolveRow, compact, robotArg, robotOf } from '../util.js';

const RUN_KEYS = ['curriculum_name', 'curriculum_version', 'robot', 'status', 'stage_index', 'attempt', 'error',
  'created_at', 'ended_at'];

const gateSchema = z.object({
  conditions: z.array(z.object({
    metric: z.string().describe('A gate metric of the robot (get_curriculum_capabilities), e.g. fall_rate, mean_return.'),
    op: z.enum(['>=', '>', '<=', '<', '==']),
    value: z.number(),
  })).min(1),
  consecutive: z.number().int().min(1).optional().describe('Evaluations in a row that must pass. Above 1 needs a trainer with early stop.'),
}).nullable();

const stageSchema = z.object({
  id: z.string().optional().describe('Stable stage id; generated from the name when omitted.'),
  name: z.string().optional(),
  gym: z.string().optional().describe('Gym id (list_gyms), e.g. openbeni:Beni-Velocity-Flat. Required on the first stage, inherited after.'),
  reward_design_id: z.string().nullable().optional().describe('Inherited when omitted; null means the task\'s default reward.'),
  experiment_design_id: z.string().nullable().optional().describe('Inherited when omitted; null means the trainer defaults.'),
  budget: z.number().int().positive().optional().describe('Training budget in the robot\'s unit (mjlab robots: PPO iterations). Required on the first stage, inherited after.'),
  settings: z.record(z.any()).optional().describe('Robot stage settings (get_curriculum_capabilities), merged over the previous stage\'s.'),
  init: z.string().optional().describe('"previous" (default after the first stage), "none" (default for the first), or "model:<model id>".'),
  carry: z.array(z.string()).optional().describe('Training state taken from init: weights, optimizer, entropy, normalizer, replay, jsrl_guide. '
    + 'Omit for the robot\'s default; [] starts from scratch.'),
  gate: gateSchema.optional().describe('Metrics the stage must reach on its gate evaluation to pass. Omit: the stage passes when training ends.'),
  on_fail: z.object({
    retries: z.number().int().min(0).max(10).optional(),
    extra_budget: z.number().int().positive().optional().describe('Budget of each retry, continuing from the failed attempt. Default: the stage budget.'),
  }).optional(),
});

function runRow(r) {
  const stages = Array.isArray(r.stages) ? r.stages : [];
  const stage = stages[r.stage_index];
  return {
    _id: String(r._id),
    ...compact(r, RUN_KEYS),
    curriculum_id: r.curriculum_id,
    stage: stage ? `${r.stage_index + 1}/${stages.length} ${stage.name}` : undefined,
    pending: r.pending ? { role: r.pending.role, job_id: r.pending.job_id } : null,
    results: (r.results || []).map((x) => compact(x, ['stage_name', 'attempt', 'passed', 'train_job_id', 'eval_job_id', 'checkpoint', 'metrics'])),
  };
}

function curriculumRow(c) {
  const stages = c.resolved_stages || c.stages || [];
  return {
    _id: String(c._id),
    ...compact(c, ['name', 'robot', 'description', 'version', 'archived', 'updated_at']),
    stages: stages.map((s) => compact(s, ['id', 'name', 'gym', 'budget', 'init', 'carry', 'gate'])),
  };
}

export function register(server) {
  defineTool(server, 'get_curriculum_capabilities', {
    title: 'Curriculum capabilities',
    description: 'Per robot: whether curricula are enabled, the training state a stage can carry from the one before '
      + '(and whether it is all-or-nothing), the budget unit, stage settings and gate metrics.',
    readOnly: true,
  }, async () => get('/get_curriculum_capabilities'));

  defineTool(server, 'list_curricula', {
    title: 'List curricula',
    description: 'Curricula (ordered training stages for one robot), with their resolved stages.',
    readOnly: true,
    input: {
      robot: robotArg.optional(),
      include_archived: z.boolean().optional(),
    },
  }, async ({ robot, include_archived }) => (await list('/get_curricula'))
    .filter((c) => !robot || robotOf(c) === robot)
    .filter((c) => include_archived || !c.archived)
    .map(curriculumRow));

  defineTool(server, 'get_curriculum', {
    title: 'Get curriculum',
    description: 'One curriculum by id, id suffix or name: the stages as authored and as resolved (inheritance and default carry applied).',
    readOnly: true,
    input: { id: z.string() },
  }, async ({ id }) => resolveRow(await list('/get_curricula'), id, 'curriculum'));

  defineTool(server, 'create_curriculum', {
    title: 'Create curriculum',
    description: 'Create a curriculum: ordered stages that each run as one TRAIN job, the next starting from the previous '
      + 'stage\'s checkpoint. A stage leaves out what it inherits (gym, designs, budget, settings). A gated stage is '
      + 'evaluated after training and must meet its gate to pass. Validated before saving; dry_run only validates and '
      + 'returns the resolved stages.',
    input: {
      robot: robotArg,
      name: z.string(),
      description: z.string().optional(),
      stages: z.array(stageSchema).min(1),
      dry_run: z.boolean().optional(),
    },
  }, async ({ dry_run, ...doc }) => {
    if (dry_run) return { dry_run: true, ...(await post('/validate_curriculum', doc)) };
    const out = await post('/create_curriculum', doc);
    return { created: String(out.insertedId), curriculum: curriculumRow(out.curriculum) };
  });

  defineTool(server, 'update_curriculum', {
    title: 'Update curriculum',
    description: 'Edit a curriculum\'s name, description or stages (stages replace the whole list). Bumps its version; '
      + 'runs already started keep the stages they started with.',
    input: {
      id: z.string(),
      name: z.string().optional(),
      description: z.string().optional(),
      stages: z.array(stageSchema).min(1).optional(),
    },
  }, async ({ id, ...changes }) => {
    const cur = resolveRow(await list('/get_curricula'), id, 'curriculum');
    const out = await post('/update_curriculum', { _id: String(cur._id), ...changes });
    return { updated: String(cur._id), curriculum: curriculumRow(out.curriculum) };
  });

  defineTool(server, 'set_curriculum_archived', {
    title: 'Archive / unarchive curriculum',
    description: 'Archived curricula are hidden and can\'t be started; their runs and jobs are kept.',
    input: { id: z.string(), archived: z.boolean() },
  }, async ({ id, archived }) => {
    const cur = resolveRow(await list('/get_curricula'), id, 'curriculum');
    await post('/set_curriculum_archived', { _id: String(cur._id), archived });
    return { id: String(cur._id), archived };
  });

  defineTool(server, 'start_curriculum_run', {
    title: 'Start curriculum run',
    description: 'Start a run of a curriculum. The curriculum-runner service queues the first stage\'s TRAIN job and '
      + 'chains the rest; each stage is a normal job on the shared queue (list_jobs curriculum_run).',
    input: {
      curriculum: z.string().describe('Curriculum id, id suffix or name.'),
      start_stage: z.union([z.string(), z.number().int()]).optional().describe('Stage id or 0-based index. A later '
        + 'stage can only be the start if its init is a model or none.'),
      priority: z.enum(['front', 'normal']).optional().describe('front (default): each stage job goes to the head of '
        + 'the queue so stages run back to back. normal: behind the jobs already queued.'),
    },
  }, async ({ curriculum, start_stage, priority }) => {
    const cur = resolveRow(await list('/get_curricula'), curriculum, 'curriculum');
    const out = await post('/start_curriculum_run', { curriculum_id: String(cur._id), start_stage, priority });
    return { started: String(out.insertedId), run: runRow({ ...out.run, _id: out.insertedId }) };
  });

  defineTool(server, 'list_curriculum_runs', {
    title: 'List curriculum runs',
    description: 'Curriculum runs, newest first: status, current stage, the pending job and each finished attempt\'s gate metrics.',
    readOnly: true,
    input: {
      robot: robotArg.optional(),
      status: z.array(z.string()).optional().describe('RUNNING, CANCEL_REQUESTED, PASSED, FAILED_GATE, FAILED, CANCELLED.'),
      curriculum: z.string().optional().describe('Only runs of this curriculum (id, suffix or name).'),
    },
  }, async ({ robot, status, curriculum }) => {
    let curId = null;
    if (curriculum) curId = String(resolveRow(await list('/get_curricula'), curriculum, 'curriculum')._id);
    return (await list('/get_curriculum_runs'))
      .filter((r) => !robot || robotOf(r) === robot)
      .filter((r) => !status || status.includes(r.status))
      .filter((r) => !curId || r.curriculum_id === curId)
      .sort((a, b) => String(b._id).localeCompare(String(a._id)))
      .map(runRow);
  });

  defineTool(server, 'get_curriculum_run', {
    title: 'Get curriculum run',
    description: 'Full curriculum run document: the stage snapshot it runs, pending job, and per-attempt results with gate details.',
    readOnly: true,
    input: { id: z.string() },
  }, async ({ id }) => resolveRow(await list('/get_curriculum_runs'), id, 'curriculum run'));

  defineTool(server, 'update_curriculum_run', {
    title: 'Cancel / retry curriculum run',
    description: 'cancel: stop a running run and cancel its pending job. retry: after FAILED_GATE, one more attempt '
      + 'continuing from the last attempt\'s checkpoint; after FAILED or CANCELLED, the same stage again as new jobs.',
    input: { id: z.string(), action: z.enum(['cancel', 'retry']) },
  }, async ({ id, action }) => {
    const run = resolveRow(await list('/get_curriculum_runs'), id, 'curriculum run');
    return post(`/${action}_curriculum_run`, { _id: String(run._id) });
  });

  defineTool(server, 'delete_curriculum_run', {
    title: 'Delete curriculum run',
    description: 'Delete a finished curriculum run. Its stage jobs are kept (as plain jobs) unless delete_jobs is true; '
      + 'models are always kept. Requires confirm: true.',
    destructive: true,
    input: { id: z.string(), delete_jobs: z.boolean().optional(), confirm: confirmArg },
  }, async ({ id, delete_jobs, confirm }) => {
    requireConfirm(confirm, 'Deleting a curriculum run');
    const run = resolveRow(await list('/get_curriculum_runs'), id, 'curriculum run');
    return post('/delete_curriculum_run', { _id: String(run._id), delete_jobs: !!delete_jobs });
  });
}
