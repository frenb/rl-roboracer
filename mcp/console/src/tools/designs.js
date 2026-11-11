import { z } from 'zod';
import { get, list, post } from '../api.js';
import { defineTool, resolveRow, compact, defined, truncate, robotArg, robotOf } from '../util.js';

const META_KEYS = new Set(['_id', 'name', 'description', 'author', 'archived', 'version', 'created_at', 'updated_at', 'robot']);
const archivedFilter = (include) => (r) => include || !r.archived;
const robotFilter = (robot) => (r) => !robot || robotOf(r) === robot;
const fieldsArg = z.record(z.string(), z.any())
  .describe('Field overrides by name (see get_experiment_design_schema for the design\'s robot). A field left out uses the default.');
const taskArg = z.string().optional()
  .describe('microduck: mjlab task id to check the design against (its default reward terms). Without it only the library is checked.');

async function schemaFields(robot = 'roboracer') {
  const { fields, source } = await get('/get_experiment_design_schema', { robot });
  return { fields: fields.filter((f) => f.kind === 'field'), all: fields, source };
}

// Car designs define reward_standard / reward_success / reward_failure; duck
// designs edit the task's reward terms through a top-level TERMS dict.
const lintDesign = (code, robot, task) => post('/lint_reward_design', defined({ code, robot, task_id: task }));

export function register(server) {
  // ---- reward designs ------------------------------------------------

  defineTool(server, 'list_reward_designs', {
    title: 'List reward designs',
    description: 'Reward Designs tab: designs without their code (get_reward_design has it). Each belongs to one robot: '
      + 'roboracer designs are Python reward functions, microduck designs edit the task\'s reward terms (TERMS dict).',
    readOnly: true,
    input: { robot: robotArg.optional().describe('Only this robot\'s designs.'), include_archived: z.boolean().optional() },
  }, async ({ robot, include_archived }) => (await list('/get_reward_designs'))
    .filter(archivedFilter(include_archived))
    .filter(robotFilter(robot))
    .map((r) => ({ _id: String(r._id), robot: robotOf(r), ...compact(r, ['name', 'version', 'archived', 'author', 'updated_at', 'spec_status']), description: truncate(r.description, 200) })));

  defineTool(server, 'get_reward_design', {
    title: 'Get reward design',
    description: 'One reward design including its Python code.',
    readOnly: true,
    input: { id: z.string().describe('Id or name.') },
  }, async ({ id }) => resolveRow(await list('/get_reward_designs'), id, 'reward design'));

  defineTool(server, 'lint_reward_design', {
    title: 'Lint reward design code',
    description: 'Reward Designs tab "Lint". roboracer: checks the code defines reward_standard / reward_success / '
      + 'reward_failure and avoids restricted imports; a pass does not guarantee it loads, the trainer compiles it at job '
      + 'start. microduck: runs the duck worker\'s own check of the TERMS dict and term_<name>() functions against the '
      + 'reward-function library and, with task, the task\'s default terms. Code with a top-level TERMS is linted as microduck.',
    readOnly: true,
    input: { code: z.string(), robot: robotArg.optional(), task: taskArg },
  }, async ({ code, robot, task }) => lintDesign(code, robot, task));

  defineTool(server, 'create_reward_design', {
    title: 'Create reward design',
    description: 'Reward Designs tab "New": save a reward design for one robot (microduck when robot is set so or the code has '
      + 'a top-level TERMS dict). Lints first and refuses code that fails, unless skip_lint.',
    input: {
      name: z.string(),
      code: z.string(),
      robot: robotArg.optional(),
      task: taskArg,
      description: z.string().optional(),
      author: z.string().optional(),
      skip_lint: z.boolean().optional(),
    },
  }, async ({ name, code, robot, task, description, author, skip_lint }) => {
    if (!skip_lint) {
      const lint = await lintDesign(code, robot, task);
      if (!lint.ok) throw new Error(`lint failed: ${lint.error}`);
    }
    const res = await post('/add_reward_design', defined({ name, code, description, author, robot }));
    return { created: String(res.insertedId), name };
  });

  defineTool(server, 'update_reward_design', {
    title: 'Update reward design',
    description: 'Reward Designs tab edit: change name, description or code (bumps version). Models keep the code they trained '
      + 'with. New code is linted first unless skip_lint.',
    input: {
      id: z.string(),
      name: z.string().optional(),
      description: z.string().optional(),
      code: z.string().optional(),
      task: taskArg,
      skip_lint: z.boolean().optional(),
    },
  }, async ({ id, name, description, code, task, skip_lint }) => {
    const row = resolveRow(await list('/get_reward_designs'), id, 'reward design');
    if (code !== undefined && !skip_lint) {
      const lint = await lintDesign(code, robotOf(row), task);
      if (!lint.ok) throw new Error(`lint failed: ${lint.error}`);
    }
    const res = await post('/update_reward_design', defined({ _id: String(row._id), name, description, code }));
    return { updated: String(row._id), matched: res.matchedCount, version: (row.version || 0) + 1 };
  });

  defineTool(server, 'set_reward_design_archived', {
    title: 'Archive / unarchive reward design',
    description: 'Archived designs are hidden from the New Job picker; models that used them still resolve them.',
    input: { id: z.string(), archived: z.boolean() },
  }, async ({ id, archived }) => {
    const row = resolveRow(await list('/get_reward_designs'), id, 'reward design');
    await post(archived ? '/archive_reward_design' : '/update_reward_design', { _id: String(row._id), archived });
    return { id: String(row._id), name: row.name, archived };
  });

  defineTool(server, 'get_reward_term_catalog', {
    title: 'MicroDuck reward-term library',
    description: 'The reward functions a microduck design can add as a term ("func": name), with their parameters. A task\'s '
      + 'default terms (what TERMS edits) are on its gym: get_gym.',
    readOnly: true,
    input: { brief: z.boolean().optional().describe('Names and parameters only, docs cut to 120 characters.') },
  }, async ({ brief }) => {
    const doc = await get('/get_reward_term_catalog', { robot: 'microduck' });
    const fns = doc.functions || [];
    return { robot: 'microduck', count: fns.length, functions: brief ? fns.map((f) => ({ ...f, doc: truncate(f.doc, 120) })) : fns };
  });

  // ---- experiment designs ----------------------------------------------

  defineTool(server, 'get_experiment_design_schema', {
    title: 'Experiment design schema',
    description: 'Every experiment-design field for a robot, with type, default, range and meaning. roboracer: SAC, demos, '
      + 'curriculum and network settings; source is "trainer" when the running trainer published it, "fallback" for the '
      + 'dashboard\'s built-in copy. microduck: PPO, network and environment settings the duck worker publishes.',
    readOnly: true,
    input: {
      robot: robotArg.optional().describe('Default roboracer.'),
      brief: z.boolean().optional().describe('Truncate each field\'s doc to 160 characters.'),
    },
  }, async ({ robot = 'roboracer', brief }) => {
    const { all, source } = await schemaFields(robot);
    return {
      robot,
      source,
      fields: all.map((f) => (f.kind === 'section'
        ? { section: f.label }
        : { ...compact(f, ['name', 'type', 'default', 'min', 'max', 'paper_ref']), doc: brief ? truncate(f.doc, 160) : f.doc })),
    };
  });

  defineTool(server, 'list_experiment_designs', {
    title: 'List experiment designs',
    description: 'Experiment Designs tab: each design with the fields it sets (everything else is the trainer\'s or, for the '
      + 'microduck, the task\'s default). Each design belongs to one robot.',
    readOnly: true,
    input: { robot: robotArg.optional().describe('Only this robot\'s designs.'), include_archived: z.boolean().optional() },
  }, async ({ robot, include_archived }) => (await list('/get_experiment_designs'))
    .filter(archivedFilter(include_archived))
    .filter(robotFilter(robot))
    .map((r) => ({
      _id: String(r._id),
      robot: robotOf(r),
      ...compact(r, ['name', 'version', 'archived', 'author']),
      description: truncate(r.description, 200),
      fields: Object.fromEntries(Object.entries(r).filter(([k, v]) => !META_KEYS.has(k) && v !== null)),
    })));

  defineTool(server, 'get_experiment_design', {
    title: 'Get experiment design',
    description: 'One experiment design document.',
    readOnly: true,
    input: { id: z.string().describe('Id or name.') },
  }, async ({ id }) => resolveRow(await list('/get_experiment_designs'), id, 'experiment design'));

  defineTool(server, 'create_experiment_design', {
    title: 'Create experiment design',
    description: 'Experiment Designs tab "New", for one robot (fields come from get_experiment_design_schema for that robot). '
      + 'Unknown field names are rejected rather than silently dropped.',
    input: {
      name: z.string(),
      robot: robotArg.optional().describe('Default roboracer.'),
      description: z.string().optional(),
      author: z.string().optional(),
      fields: fieldsArg.optional(),
    },
  }, async ({ name, robot, description, author, fields }) => {
    const res = await post('/add_experiment_design', defined({ name, robot, description, author, fields: fields || {} }));
    return { created: String(res.insertedId), name };
  });

  defineTool(server, 'update_experiment_design', {
    title: 'Update experiment design',
    description: 'Experiment Designs tab edit: rename, redescribe, or set fields (bumps version), checked against the design\'s '
      + 'own robot\'s schema. Set a field to null to return it to the default.',
    input: { id: z.string(), name: z.string().optional(), description: z.string().optional(), fields: fieldsArg.optional() },
  }, async ({ id, name, description, fields }) => {
    const row = resolveRow(await list('/get_experiment_designs'), id, 'experiment design');
    const res = await post('/update_experiment_design', defined({ _id: String(row._id), name, description, fields }));
    return { updated: String(row._id), matched: res.matchedCount, version: (row.version || 0) + 1 };
  });

  defineTool(server, 'set_experiment_design_archived', {
    title: 'Archive / unarchive experiment design',
    description: 'Archived designs are hidden from the New Job picker; models that used them still resolve them.',
    input: { id: z.string(), archived: z.boolean() },
  }, async ({ id, archived }) => {
    const row = resolveRow(await list('/get_experiment_designs'), id, 'experiment design');
    await post(archived ? '/archive_experiment_design' : '/update_experiment_design', { _id: String(row._id), archived });
    return { id: String(row._id), name: row.name, archived };
  });
}
