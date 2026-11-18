import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROBOTS_DIR = path.resolve(HERE, '../../../robots');

const DEFAULT_META = {
  roboracer: { id: 'roboracer', engine: 'tf_agents_unity' },
  microduck: { id: 'microduck', engine: 'mjlab_rsl_rl' },
  openbeni: { id: 'openbeni', engine: 'mjlab_rsl_rl' },
};

function loadRobotMeta() {
  const out = {};
  if (fs.existsSync(REPO_ROBOTS_DIR)) {
    for (const entry of fs.readdirSync(REPO_ROBOTS_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const yamlPath = path.join(REPO_ROBOTS_DIR, entry.name, 'robot.yaml');
      if (!fs.existsSync(yamlPath)) continue;
      const text = fs.readFileSync(yamlPath, 'utf8');
      const idM = /^id:\s*["']?([^"'\r\n#]+)["']?/m.exec(text);
      const engM = /^engine:\s*["']?([^"'\r\n#]+)["']?/m.exec(text);
      if (idM) {
        const id = idM[1].trim();
        out[id] = { id, engine: engM ? engM[1].trim() : (id === 'roboracer' ? 'tf_agents_unity' : 'mjlab_rsl_rl') };
      }
    }
  }
  return Object.keys(out).length ? out : DEFAULT_META;
}

export const ROBOT_META = loadRobotMeta();

/**
 * Register a tool whose handler returns a plain value. The value is sent
 * back as pretty JSON; a thrown error becomes an isError result carrying
 * its message, so the dashboard's own validation errors reach the caller.
 */
export function defineTool(server, name, { title, description, input = {}, readOnly = false, destructive = false }, handler) {
  server.registerTool(name, {
    title,
    description,
    inputSchema: input,
    annotations: { readOnlyHint: readOnly, destructiveHint: destructive, openWorldHint: false },
  }, async (args) => {
    try {
      const out = await handler(args || {});
      const text = typeof out === 'string' ? out : JSON.stringify(out, null, 2);
      return { content: [{ type: 'text', text }] };
    } catch (e) {
      return { isError: true, content: [{ type: 'text', text: `Error: ${e.message || e}` }] };
    }
  });
}

export const confirmArg = z.boolean().optional()
  .describe('Must be true. Guards an action that cannot be undone.');

export function requireConfirm(confirm, what) {
  if (confirm !== true) {
    throw new Error(`${what} cannot be undone; call again with confirm: true to proceed.`);
  }
}

/** Creation time encoded in a 24-hex Mongo ObjectId, as ISO, else null. */
export function objectIdTime(id) {
  const s = String(id || '');
  if (!/^[0-9a-f]{24}$/i.test(s)) return null;
  return new Date(parseInt(s.slice(0, 8), 16) * 1000).toISOString();
}

/**
 * Find one row by full _id, a unique _id suffix (the dashboard shows the
 * last 8 characters), or an exact case-insensitive name.
 */
export function resolveRow(rows, ref, kind) {
  const want = String(ref || '').trim();
  if (!want) throw new Error(`${kind} id is required`);
  const exact = rows.find((r) => String(r._id) === want);
  if (exact) return exact;
  if (want.length >= 6) {
    const bySuffix = rows.filter((r) => String(r._id).endsWith(want));
    if (bySuffix.length === 1) return bySuffix[0];
    if (bySuffix.length > 1) {
      throw new Error(`${kind} "${want}" is ambiguous: matches ${bySuffix.map((r) => r._id).join(', ')}`);
    }
  }
  const lower = want.toLowerCase();
  const byName = rows.filter((r) => typeof r.name === 'string' && r.name.toLowerCase() === lower);
  if (byName.length === 1) return byName[0];
  if (byName.length > 1) {
    throw new Error(`${kind} name "${want}" is ambiguous: ids ${byName.map((r) => r._id).join(', ')}`);
  }
  throw new Error(`no ${kind} matches "${want}"`);
}

/** Copy only the keys that are present (not undefined / null / ''). */
export function compact(obj, keys) {
  const out = {};
  for (const k of keys) {
    const v = obj[k];
    if (v !== undefined && v !== null && v !== '') out[k] = v;
  }
  return out;
}

/** Drop undefined values so an omitted tool argument is not sent. */
export function defined(obj) {
  return Object.fromEntries(Object.entries(obj).filter(([, v]) => v !== undefined));
}

const _order = { roboracer: 0, microduck: 1 };
export const ROBOTS = Object.keys(ROBOT_META).sort((a, b) => (_order[a] ?? 10) - (_order[b] ?? 10) || a.localeCompare(b));

export const robotArg = z.enum(ROBOTS);

/** Car rows predate the robot field, so a missing one means roboracer. */
export const robotOf = (row) => (typeof row === 'string' ? (row || 'roboracer') : ((row && row.robot) || 'roboracer'));

export const isMjlab = (rowOrId) => {
  const r = robotOf(rowOrId);
  return ROBOT_META[r] ? ROBOT_META[r].engine === 'mjlab_rsl_rl' : r !== 'roboracer';
};

export const isDuck = isMjlab;

export function truncate(s, n) {
  const str = String(s ?? '');
  return str.length > n ? str.slice(0, n) + `… [${str.length - n} more chars]` : str;
}
