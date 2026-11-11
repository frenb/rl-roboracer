// Commands on this host: docker compose for the stack's containers, and
// PowerShell for the scripts/ that manage the Unity clients.

import { execFile } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
export const SCRIPTS = path.join(REPO_ROOT, 'scripts');
const COMPOSE = ['compose', '-f', 'docker-compose.yml', '-f', 'compose/scale.yml', '-f', 'compose/duck.yml'];

export function run(cmd, args, { timeoutMs = 60000, okCodes = [0] } = {}) {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { cwd: REPO_ROOT, maxBuffer: 16 * 1024 * 1024, timeout: timeoutMs, windowsHide: true }, (err, stdout, stderr) => {
      if (err && !okCodes.includes(err.code)) {
        const msg = (stderr || stdout || err.message || '').trim().split('\n').slice(-15).join('\n');
        reject(new Error(err.code === 'ENOENT' ? `${cmd} is not installed or not on PATH` : `${cmd} ${args.slice(0, 6).join(' ')}: ${msg}`));
        return;
      }
      resolve(String(stdout));
    });
  });
}

export const powershell = (args, opts) => run('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', ...args], opts);

/** `docker compose exec <args>` against the stack (base, scale and duck overlays). */
export const composeExec = (args, opts) => run('docker', [...COMPOSE, 'exec', ...args], opts);

export const composeLogs = (service, lines, opts) => run('docker', [...COMPOSE, 'logs', '--no-color', '--tail', String(lines), service], opts);
