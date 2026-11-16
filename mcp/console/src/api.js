// HTTP client for the dashboard API (dashboard/src/server.ts).

export const BASE_URL = (process.env.ROBORACER_DASHBOARD_URL || 'http://localhost').replace(/\/+$/, '');
export const LOG_WS_URL = process.env.ROBORACER_LOG_WS_URL
  || BASE_URL.replace(/^http/, 'ws').replace(/:\d+$/, '') + ':8080';

const TIMEOUT_MS = 60000;

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request(method, path, { query, body } = {}) {
  const url = new URL(BASE_URL + path);
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  }
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (e) {
    throw new ApiError(`dashboard unreachable at ${BASE_URL} (${e.cause?.code || e.message}). `
      + 'Is the Docker stack running? (scripts\\Start-Stack.ps1)', 0, null);
  }
  const text = await res.text();
  let data = text;
  // The change-gated list endpoints answer this literal string instead of JSON.
  if (text && text !== 'NO_CHANGES') {
    try { data = JSON.parse(text); } catch { /* HTML or plain text */ }
  }
  if (!res.ok) {
    const msg = (data && typeof data === 'object' && data.error) ? data.error : String(text).slice(0, 500);
    throw new ApiError(`${method} ${path} failed (HTTP ${res.status}): ${msg}`, res.status, data);
  }
  return data;
}

export const get = (path, query) => request('GET', path, { query });
export const post = (path, body) => request('POST', path, { body: body || {} });

/** A change-gated list endpoint (get_jobs, get_models, ...), always fresh. */
export async function list(path) {
  const data = await get(path, { force: 'true' });
  return Array.isArray(data) ? data : [];
}
