// Unity car clients on this host and the car trainer that drives them.
// The trainer (robotaxi.py in sim-controller) runs one env per client, so
// its --num-envs must equal the number of clients or its first reset hangs.

import { z } from 'zod';
import path from 'node:path';
import { list } from '../api.js';
import { defineTool, isDuck } from '../util.js';
import { SCRIPTS, powershell, composeExec as sim } from '../docker.js';

const UNITY_PROCESS = 'robotaxi gym level 1';
// ros-server plus ros-server-1..3 from compose/scale.yml.
const MAX_CLIENTS = 4;
const trainerCmd = (n) => `cd /python_ws/src && python -u robotaxi.py --num-envs ${n} 2>&1 | tee /tmp/trainer.log`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function unityClients() {
  const out = await powershell(['-Command',
    `ConvertTo-Json -Compress -InputObject @(Get-Process -Name '${UNITY_PROCESS}' -ErrorAction SilentlyContinue | `
    + "Select-Object Id, Path, Responding, @{n='Started';e={$_.StartTime.ToString('o')}})"]);
  return (JSON.parse(out.trim() || '[]') || []).map((p) => {
    const m = /instances[\\/](\d+)[\\/]/i.exec(p.Path || '');
    return { pid: p.Id, instance: m ? Number(m[1]) : null, path: p.Path || null, started: p.Started || null, responding: p.Responding };
  }).sort((a, b) => (a.instance ?? 99) - (b.instance ?? 99));
}

/**
 * Top-level robotaxi.py processes in sim-controller (not the bash/tee wrappers
 * around them). A running job forks one env worker per client with the same
 * command line; those are children of a trainer and are counted, not listed.
 */
async function trainers() {
  const out = await sim(['-T', 'sim-controller', 'ps', '-eo', 'pid,ppid,etimes,args']);
  const procs = out.split('\n').map((l) => /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S*python\S*\s.*robotaxi\.py.*)$/.exec(l)).filter(Boolean).map((m) => {
    const n = /--num-envs[ =](\d+)/.exec(m[4]);
    return { pid: Number(m[1]), ppid: Number(m[2]), uptime_min: Math.round(Number(m[3]) / 60), num_envs: n ? Number(n[1]) : 1, cmd: m[4].trim() };
  });
  const pids = new Set(procs.map((p) => p.pid));
  return procs.filter((p) => !pids.has(p.ppid)).map(({ ppid, ...p }) => ({
    ...p,
    env_workers: procs.filter((c) => c.ppid === p.pid).length,
  }));
}

async function activeCarJobs() {
  return (await list('/get_jobs'))
    .filter((j) => j.status === 'IN_PROGRESS' && !isDuck(j))
    .map((j) => ({ id: String(j._id), name: j.name || null, course_type: j.course_type || null }));
}

async function status() {
  const [clients, procs, jobs] = await Promise.all([unityClients(), trainers(), activeCarJobs()]);
  const trainer = procs[0] || null;
  const notes = [];
  if (procs.length > 1) notes.push(`${procs.length} car trainers are running; only one should be.`);
  if (!trainer) notes.push('No car trainer is running, so queued car jobs will not start.');
  else if (trainer.num_envs !== clients.length) {
    notes.push(`Trainer expects ${trainer.num_envs} client(s) but ${clients.length} are running; its next car job will hang on reset.`);
  }
  if (clients.some((c) => c.responding === false)) notes.push('A Unity client is not responding.');
  return {
    unity_clients: clients,
    client_count: clients.length,
    trainer,
    ...(procs.length > 1 ? { extra_trainers: procs.slice(1) } : {}),
    in_sync: !!trainer && procs.length === 1 && trainer.num_envs === clients.length,
    active_car_jobs: jobs,
    notes,
  };
}

async function stopTrainer() {
  await sim(['-T', 'sim-controller', 'pkill', '-f', 'robotaxi[.]py'], { okCodes: [0, 1] });
  for (let i = 0; i < 20 && (await trainers()).length; i++) await sleep(1000);
  const left = await trainers();
  if (left.length) {
    await sim(['-T', 'sim-controller', 'pkill', '-9', '-f', 'robotaxi[.]py'], { okCodes: [0, 1] });
    await sleep(2000);
  }
}

const tail = (s, n = 12) => String(s).replace(/\r/g, '').trim().split('\n').slice(-n);

export function register(server) {
  defineTool(server, 'get_unity_clients', {
    title: 'Unity clients and car trainer',
    description: `Unity car clients ("${UNITY_PROCESS}") running on this host, the car trainer (robotaxi.py in sim-controller) `
      + 'and its --num-envs, whether the two match, and any IN_PROGRESS car job. A mismatch makes the next car job hang '
      + 'on its first reset.',
    readOnly: true,
  }, status);

  defineTool(server, 'set_unity_clients', {
    title: 'Set Unity client count',
    description: `Restart the Unity car clients with a new count (1-${MAX_CLIENTS}) and relaunch the car trainer with a matching `
      + '--num-envs. Stops every running client and its supervisor (scripts/Stop-Clients.ps1), starts the new set '
      + '(scripts/Start-Clients.ps1, one client every stagger_seconds), then restarts robotaxi.py. count 0 stops the '
      + 'clients and the trainer. Docker and duck training are left alone. Refuses while a car job is IN_PROGRESS: '
      + 'pause it first (update_jobs action pause). Takes about stagger_seconds x count plus 30 s.',
    destructive: true,
    input: {
      count: z.number().int().min(0).max(MAX_CLIENTS).describe('Number of Unity clients; the trainer gets the same --num-envs.'),
      stagger_seconds: z.number().min(0).max(60).optional()
        .describe('Seconds between client launches. Default 15; shorter risks clients wedging during GPU init.'),
      restart_trainer: z.boolean().optional().describe('Relaunch robotaxi.py with --num-envs = count. Default true.'),
    },
  }, async ({ count, stagger_seconds, restart_trainer = true }) => {
    const jobs = await activeCarJobs();
    if (jobs.length) {
      throw new Error(`car job ${jobs.map((j) => j.id).join(', ')} is IN_PROGRESS and drives these clients; `
        + 'pause it (update_jobs action pause) or wait for it to finish');
    }
    const before = await status();
    const steps = [];

    if (restart_trainer || count === 0) {
      await stopTrainer();
      steps.push(before.trainer ? `stopped trainer (was --num-envs ${before.trainer.num_envs})` : 'no trainer was running');
    }

    let script;
    if (count === 0) {
      script = await powershell(['-File', path.join(SCRIPTS, 'Stop-Clients.ps1')], { timeoutMs: 120000 });
      steps.push('stopped all Unity clients');
    } else {
      const stagger = stagger_seconds ?? 15;
      script = await powershell(['-File', path.join(SCRIPTS, 'Start-Clients.ps1'), '-N', String(count), '-StaggerSeconds', String(stagger)],
        { timeoutMs: (180 + stagger * count) * 1000 });
      steps.push(`started ${count} Unity client(s)`);
      if (restart_trainer) {
        await sim(['-d', 'sim-controller', 'bash', '-c', trainerCmd(count)]);
        for (let i = 0; i < 15 && !(await trainers()).length; i++) await sleep(1000);
        steps.push(`launched trainer with --num-envs ${count}`);
      }
    }

    await sleep(3000);
    const after = await status();
    if (count > 0 && after.client_count < count) {
      after.notes.push(`Only ${after.client_count} of ${count} clients are up yet; supervisors relaunch slow starters, so check again with get_unity_clients.`);
    }
    return {
      steps,
      before: { client_count: before.client_count, trainer_num_envs: before.trainer && before.trainer.num_envs },
      after,
      script_output: tail(script),
      note: 'NUM_ENVS in .env still sets --num-envs for the trainer started when sim-controller is recreated.',
    };
  });
}
