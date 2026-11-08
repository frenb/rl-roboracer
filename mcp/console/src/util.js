import { z } from 'zod';

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

export function truncate(s, n) {
  const str = String(s ?? '');
  return str.length > n ? str.slice(0, n) + `… [${str.length - n} more chars]` : str;
}
