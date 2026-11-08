import { z } from 'zod';
import { get, list, post } from '../api.js';
import { defineTool, resolveRow, compact, defined, truncate } from '../util.js';

const META_KEYS = new Set(['_id', 'name', 'description', 'author', 'archived', 'version', 'created_at', 'updated_at']);
const archivedFilter = (include) => (r) => include || !r.archived;
const fieldsArg = z.record(z.string(), z.any())
  .describe('Field overrides by name (see get_experiment_design_schema). A field left out uses the trainer default.');

async function schemaFields() {
  const { fields, source } = await get('/get_experiment_design_schema');
  return { fields: fields.filter((f) => f.kind === 'field'), all: fields, source };
}

export function register(server) {
  // ---- reward designs ------------------------------------------------

  defineTool(server, 'list_reward_designs', {
    title: 'List reward designs',
    description: 'Reward Designs tab: designs without their code (get_reward_design has it).',
    readOnly: true,
    input: { include_archived: z.boolean().optional() },
  }, async ({ include_archived }) => (await list('/get_reward_designs'))
    .filter(archivedFilter(include_archived))
    .map((r) => ({ _id: String(r._id), ...compact(r, ['name', 'version', 'archived', 'author', 'updated_at']), description: truncate(r.description, 200) })));

  defineTool(server, 'get_reward_design', {
    title: 'Get reward design',
    description: 'One reward design including its Python code.',
    readOnly: true,
    input: { id: z.string().describe('Id or name.') },
  }, async ({ id }) => resolveRow(await list('/get_reward_designs'), id, 'reward design'));

  defineTool(server, 'lint_reward_design', {
    title: 'Lint reward design code',
    description: 'Reward Designs tab "Lint": checks the code defines reward_standard / reward_success / reward_failure and avoids '
      + 'restricted imports. A pass does not guarantee it loads; the trainer compiles it at job start.',
    readOnly: true,
    input: { code: z.string() },
  }, async ({ code }) => post('/lint_reward_design', { code }));

  defineTool(server, 'create_reward_design', {
    title: 'Create reward design',
    description: 'Reward Designs tab "New": save a reward-design module. Lints first and refuses code that fails, unless skip_lint.',
    input: {
      name: z.string(),
      code: z.string(),
      description: z.string().optional(),
      author: z.string().optional(),
      skip_lint: z.boolean().optional(),
    },
  }, async ({ name, code, description, author, skip_lint }) => {
    if (!skip_lint) {
      const lint = await post('/lint_reward_design', { code });
      if (!lint.ok) throw new Error(`lint failed: ${lint.error}`);
    }
    const res = await post('/add_reward_design', defined({ name, code, description, author }));
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
      skip_lint: z.boolean().optional(),
    },
  }, async ({ id, name, description, code, skip_lint }) => {
    const row = resolveRow(await list('/get_reward_designs'), id, 'reward design');
    if (code !== undefined && !skip_lint) {
      const lint = await post('/lint_reward_design', { code });
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

  // ---- experiment designs ----------------------------------------------

  defineTool(server, 'get_experiment_design_schema', {
    title: 'Experiment design schema',
    description: 'Every experiment-design field with type, default, range and meaning. source is "trainer" when the running '
      + 'trainer published it, "fallback" when the dashboard used its built-in (possibly older) copy.',
    readOnly: true,
    input: { brief: z.boolean().optional().describe('Truncate each field\'s doc to 160 characters.') },
  }, async ({ brief }) => {
    const { all, source } = await schemaFields();
    return {
      source,
      fields: all.map((f) => (f.kind === 'section'
        ? { section: f.label }
        : { ...compact(f, ['name', 'type', 'default', 'min', 'max', 'paper_ref']), doc: brief ? truncate(f.doc, 160) : f.doc })),
    };
  });

  defineTool(server, 'list_experiment_designs', {
    title: 'List experiment designs',
    description: 'Experiment Designs tab: each design with the fields it sets (everything else is a trainer default).',
    readOnly: true,
    input: { include_archived: z.boolean().optional() },
  }, async ({ include_archived }) => (await list('/get_experiment_designs'))
    .filter(archivedFilter(include_archived))
    .map((r) => ({
      _id: String(r._id),
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
    description: 'Experiment Designs tab "New". Unknown field names are rejected rather than silently dropped.',
    input: { name: z.string(), description: z.string().optional(), author: z.string().optional(), fields: fieldsArg.optional() },
  }, async ({ name, description, author, fields }) => {
    const res = await post('/add_experiment_design', defined({ name, description, author, fields: fields || {} }));
    return { created: String(res.insertedId), name };
  });

  defineTool(server, 'update_experiment_design', {
    title: 'Update experiment design',
    description: 'Experiment Designs tab edit: rename, redescribe, or set fields (bumps version). Set a field to null to '
      + 'return it to the trainer default.',
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
