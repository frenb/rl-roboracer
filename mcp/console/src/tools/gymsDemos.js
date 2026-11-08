import { z } from 'zod';
import { list, post } from '../api.js';
import { defineTool, resolveRow, defined, objectIdTime } from '../util.js';
import { demoLabel } from '../jobPayload.js';

export function register(server) {
  // ---- gyms ------------------------------------------------------------

  defineTool(server, 'list_gyms', {
    title: 'List gyms',
    description: 'Gyms tab: registered Unity builds jobs can run in, with their default course and designs.',
    readOnly: true,
    input: { include_archived: z.boolean().optional() },
  }, async ({ include_archived }) => (await list('/get_gyms')).filter((g) => include_archived || !g.archived));

  defineTool(server, 'create_gym', {
    title: 'Register gym',
    description: 'Gyms tab "New": register a Unity build. file_path is the build\'s .exe on the host running the Unity clients.',
    input: { name: z.string(), file_path: z.string(), description: z.string().optional() },
  }, async ({ name, file_path, description }) => {
    const res = await post('/add_gym', defined({ name, file_path, description }));
    return { created: String(res.insertedId), name, file_path };
  });

  defineTool(server, 'update_gym', {
    title: 'Update gym',
    description: 'Gyms tab edit: rename, change the build path, or redescribe.',
    input: { id: z.string(), name: z.string().optional(), file_path: z.string().optional(), description: z.string().optional() },
  }, async ({ id, name, file_path, description }) => {
    const row = resolveRow(await list('/get_gyms'), id, 'gym');
    const res = await post('/update_gym', defined({ _id: String(row._id), name, file_path, description }));
    return { updated: String(row._id), matched: res.matchedCount };
  });

  defineTool(server, 'set_gym_archived', {
    title: 'Archive / unarchive gym',
    description: 'Archived gyms are hidden from the job pickers.',
    input: { id: z.string(), archived: z.boolean() },
  }, async ({ id, archived }) => {
    const row = resolveRow(await list('/get_gyms'), id, 'gym');
    await post(archived ? '/archive_gym' : '/update_gym', { _id: String(row._id), archived });
    return { id: String(row._id), name: row.name, archived };
  });

  // ---- demos -----------------------------------------------------------

  defineTool(server, 'list_demos', {
    title: 'List demos',
    description: 'Demos tab: expert-demonstration TFRecords sets, newest first. Pass one as create_job\'s demo.',
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
