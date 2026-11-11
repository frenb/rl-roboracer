import { z } from 'zod';
import { list, post } from '../api.js';
import { defineTool, resolveRow, defined, objectIdTime, robotArg, robotOf, isDuck } from '../util.js';
import { demoLabel, resolveGym } from '../jobPayload.js';

// A duck gym carries its task's full reward-term table; get_gym returns it.
const DUCK_HEAVY_KEYS = new Set(['reward_terms', 'reward_defaults']);

function gymRow(g) {
  if (!isDuck(g)) return { ...g, robot: 'roboracer' };
  const row = Object.fromEntries(Object.entries(g).filter(([k]) => !DUCK_HEAVY_KEYS.has(k)));
  row.reward_term_count = Array.isArray(g.reward_terms) ? g.reward_terms.length : 0;
  return row;
}

export function register(server) {
  // ---- gyms ------------------------------------------------------------

  defineTool(server, 'list_gyms', {
    title: 'List gyms',
    description: 'Gyms tab: where jobs run. roboracer gyms are Unity builds with a default course and designs; microduck gyms '
      + 'are the mjlab tasks the duck worker registers (task_id, skill, terrain, rollers, backlash). Duck rows leave out '
      + 'the reward-term table; get_gym has it.',
    readOnly: true,
    input: { robot: robotArg.optional().describe('Only this robot\'s gyms.'), include_archived: z.boolean().optional() },
  }, async ({ robot, include_archived }) => (await list('/get_gyms'))
    .filter((g) => include_archived || !g.archived)
    .filter((g) => !robot || robotOf(g) === robot)
    .map(gymRow));

  defineTool(server, 'get_gym', {
    title: 'Get gym',
    description: 'One gym in full. For a microduck task this includes reward_defaults: every default reward term with its '
      + 'function, weight and parameters, which is what a microduck reward design\'s TERMS edits.',
    readOnly: true,
    input: { id: z.string().describe('Gym id, id suffix or name; for the microduck also a bare mjlab task id.') },
  }, async ({ id }) => resolveGym(await list('/get_gyms'), id));

  defineTool(server, 'create_gym', {
    title: 'Register gym',
    description: 'Gyms tab "New": register a roboracer Unity build. file_path is the build\'s .exe on the host running the Unity '
      + 'clients. microduck gyms are registered automatically by the duck worker from its mjlab tasks.',
    input: { name: z.string(), file_path: z.string(), description: z.string().optional() },
  }, async ({ name, file_path, description }) => {
    const res = await post('/add_gym', defined({ name, file_path, description }));
    return { created: String(res.insertedId), name, file_path };
  });

  defineTool(server, 'update_gym', {
    title: 'Update gym',
    description: 'Gyms tab edit: rename, change the build path, or redescribe. microduck gyms only take a new description.',
    input: { id: z.string(), name: z.string().optional(), file_path: z.string().optional(), description: z.string().optional() },
  }, async ({ id, name, file_path, description }) => {
    const row = resolveGym(await list('/get_gyms'), id);
    const res = await post('/update_gym', defined({ _id: String(row._id), name, file_path, description }));
    return { updated: String(row._id), matched: res.matchedCount };
  });

  defineTool(server, 'set_gym_archived', {
    title: 'Archive / unarchive gym',
    description: 'Archived gyms are hidden from the job pickers.',
    input: { id: z.string(), archived: z.boolean() },
  }, async ({ id, archived }) => {
    const row = resolveGym(await list('/get_gyms'), id);
    await post(archived ? '/archive_gym' : '/update_gym', { _id: String(row._id), archived });
    return { id: String(row._id), name: row.name, archived };
  });

  // ---- demos -----------------------------------------------------------

  defineTool(server, 'list_demos', {
    title: 'List demos',
    description: 'Demos tab: roboracer expert-demonstration TFRecords sets, newest first. Pass one as create_job\'s demo. '
      + '(The microduck trains with PPO and uses no demos.)',
    readOnly: true,
    input: { course: z.string().optional(), include_archived: z.boolean().optional() },
  }, async ({ course, include_archived }) => (await list('/get_demos'))
    .filter((d) => include_archived || !d.archived)
    .filter((d) => !course || d.course === course)
    .sort((a, b) => String(b.collected_at || b.created_at || objectIdTime(b._id) || '')
      .localeCompare(String(a.collected_at || a.created_at || objectIdTime(a._id) || '')))
    .map((d) => ({ ...d, label: demoLabel(d) })));

  defineTool(server, 'create_demo', {
    title: 'Register demo',
    description: 'Demos tab "New": register a TFRecords directory by hand (DEMO jobs register theirs automatically).',
    input: {
      tfrecords_path: z.string(),
      num_demos: z.number().int().min(0),
      course: z.string().optional(),
      description: z.string().optional(),
    },
  }, async (args) => {
    const res = await post('/add_demo', defined(args));
    return { created: String(res.insertedId) };
  });

  defineTool(server, 'update_demo', {
    title: 'Update demo',
    description: 'Demos tab edit.',
    input: {
      id: z.string(),
      tfrecords_path: z.string().optional(),
      num_demos: z.number().int().min(0).optional(),
      course: z.string().optional(),
      description: z.string().optional(),
    },
  }, async ({ id, ...rest }) => {
    const row = resolveRow(await list('/get_demos'), id, 'demo');
    const res = await post('/update_demo', defined({ _id: String(row._id), ...rest }));
    return { updated: String(row._id), matched: res.matchedCount };
  });

  defineTool(server, 'set_demo_archived', {
    title: 'Archive / unarchive demo',
    description: 'Archived demos are hidden from the New Job picker.',
    input: { id: z.string(), archived: z.boolean() },
  }, async ({ id, archived }) => {
    const row = resolveRow(await list('/get_demos'), id, 'demo');
    await post('/archive_demo', { _id: String(row._id), archived });
    return { id: String(row._id), archived };
  });
}
