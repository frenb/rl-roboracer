// Gym builds as GitHub release assets, the way scripts/install/Install.ps1
// downloads them: roboracer-gym-<Folder>.zip plus "<zip>.sha256", each zip
// holding one <Folder>/ directory with the player .exe at its top.

import { z } from 'zod';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import yazl from 'yazl';
import { list, post } from '../api.js';
import { defineTool, confirmArg, requireConfirm } from '../util.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const GITHUB_REPO = process.env.ROBORACER_GITHUB_REPO || 'frenb/rl-roboracer';
const UNITY_BINARY_DIR = process.env.ROBORACER_UNITY_BINARY_DIR || path.join(REPO_ROOT, '..', 'UnityBinary');
const UNITY_BUILDS_DIR = path.join(REPO_ROOT, 'unity', 'Builds');
const RELEASE_DIR = path.join(UNITY_BUILDS_DIR, 'release');
const GYM_EXE = 'robotaxi gym level 1.exe';
const RESERVED_BUILD_NAMES = new Set(['latest', 'instances', 'release']);
const INSTALLER_GYM_LISTS = 'scripts/install/Install.ps1 ($Gyms), seed/seed.json (gyms) and install.md (gym download step)';

const assetName = (build) => `roboracer-gym-${build}.zip`;
const downloadUrl = (tag, name) => `https://github.com/${GITHUB_REPO}/releases/download/${encodeURIComponent(tag)}/${encodeURIComponent(name)}`;

function gh(args, { timeoutMs = 120000 } = {}) {
  return new Promise((resolve, reject) => {
    execFile('gh', args, { maxBuffer: 64 * 1024 * 1024, timeout: timeoutMs, windowsHide: true }, (err, stdout, stderr) => {
      if (err) {
        const msg = (stderr || err.message || '').trim();
        reject(new Error(err.code === 'ENOENT' ? 'GitHub CLI (gh) is not installed or not on PATH' : `gh ${args[0]} ${args[1] || ''}: ${msg}`));
        return;
      }
      resolve(stdout);
    });
  });
}

async function releaseView(tag) {
  try {
    return JSON.parse(await gh(['release', 'view', tag, '-R', GITHUB_REPO, '--json',
      'tagName,name,url,isDraft,isPrerelease,publishedAt,assets']));
  } catch (e) {
    if (/not found|release not found/i.test(e.message)) return null;
    throw e;
  }
}

function dirSize(dir) {
  let total = 0;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    total += e.isDirectory() ? dirSize(p) : fs.statSync(p).size;
  }
  return total;
}

/** Gym build folders (those with the player .exe at their top level). */
function gymBuilds() {
  const out = [];
  const scan = (root, source) => {
    if (!fs.existsSync(root)) return;
    for (const e of fs.readdirSync(root, { withFileTypes: true })) {
      if (!e.isDirectory() || (source === 'unity/Builds' && RESERVED_BUILD_NAMES.has(e.name))) continue;
      const dir = path.join(root, e.name);
      const exe = path.join(dir, GYM_EXE);
      if (!fs.existsSync(exe)) continue;
      out.push({ name: e.name, source, path: dir, exe, built_at: fs.statSync(exe).mtime.toISOString() });
    }
  };
  scan(UNITY_BINARY_DIR, 'UnityBinary');
  scan(UNITY_BUILDS_DIR, 'unity/Builds');
  return out.sort((a, b) => b.built_at.localeCompare(a.built_at));
}

function resolveBuild(ref) {
  if (ref && fs.existsSync(path.join(ref, GYM_EXE))) {
    const dir = path.resolve(ref);
    return { name: path.basename(dir), path: dir, exe: path.join(dir, GYM_EXE) };
  }
  const builds = gymBuilds();
  if (!builds.length) throw new Error(`no gym builds found under ${UNITY_BINARY_DIR} or ${UNITY_BUILDS_DIR}`);
  if (!ref) return builds[0];
  const hit = builds.find((b) => b.name === ref) || builds.find((b) => b.name.toLowerCase() === String(ref).toLowerCase());
  if (!hit) throw new Error(`no gym build named "${ref}". Builds: ${builds.map((b) => b.name).join(', ')}`);
  return hit;
}

function shipFiles(dir) {
  const files = [];
  const walk = (d, rel) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) {
        if (!e.name.endsWith('_DoNotShip')) walk(path.join(d, e.name), r);
      } else if (e.name !== 'Player.log') {
        files.push({ abs: path.join(d, e.name), rel: r });
      }
    }
  };
  walk(dir, '');
  return files;
}

function sha256File(file) {
  return new Promise((resolve, reject) => {
    const h = createHash('sha256');
    fs.createReadStream(file).on('data', (c) => h.update(c)).on('error', reject).on('end', () => resolve(h.digest('hex')));
  });
}

async function packageBuild(build, overwrite) {
  fs.mkdirSync(RELEASE_DIR, { recursive: true });
  const zipPath = path.join(RELEASE_DIR, assetName(build.name));
  const shaPath = `${zipPath}.sha256`;
  if (fs.existsSync(zipPath) && fs.existsSync(shaPath) && !overwrite) {
    const sha = fs.readFileSync(shaPath, 'utf8').trim().split(/\s+/)[0];
    return { zip: zipPath, sha256_file: shaPath, sha256: sha, size_mb: +(fs.statSync(zipPath).size / 1048576).toFixed(1), reused_existing: true };
  }
  const files = shipFiles(build.path);
  const zip = new yazl.ZipFile();
  for (const f of files) zip.addFile(f.abs, `${build.name}/${f.rel}`, { compress: true });
  await new Promise((resolve, reject) => {
    const out = fs.createWriteStream(zipPath);
    zip.outputStream.pipe(out).on('close', resolve).on('error', reject);
    zip.end();
  });
  const sha = await sha256File(zipPath);
  fs.writeFileSync(shaPath, `${sha}  ${path.basename(zipPath)}\n`, 'ascii');
  return { zip: zipPath, sha256_file: shaPath, sha256: sha, files: files.length, size_mb: +(fs.statSync(zipPath).size / 1048576).toFixed(1) };
}

async function verifyDownload(tag, name, expected) {
  const res = await fetch(downloadUrl(tag, name), { redirect: 'follow' });
  if (!res.ok) return { ok: false, error: `download HTTP ${res.status}` };
  const h = createHash('sha256');
  for await (const chunk of res.body) h.update(chunk);
  const got = h.digest('hex');
  return { ok: got === expected, sha256: got };
}

export function register(server) {
  defineTool(server, 'list_releases', {
    title: 'List GitHub releases',
    description: `Releases of ${GITHUB_REPO} with their files, e.g. the gym and demo zips the installer downloads.`,
    readOnly: true,
    input: { limit: z.number().int().min(1).max(30).optional().describe('Default 10.') },
  }, async ({ limit = 10 }) => {
    const rels = JSON.parse(await gh(['release', 'list', '-R', GITHUB_REPO, '--limit', String(limit), '--json',
      'tagName,name,isLatest,isDraft,isPrerelease,publishedAt']));
    const out = [];
    for (const r of rels) {
      const v = await releaseView(r.tagName);
      out.push({
        ...r,
        url: v && v.url,
        assets: (v ? v.assets : []).map((a) => ({ name: a.name, size_mb: +(a.size / 1048576).toFixed(1), downloads: a.downloadCount })),
      });
    }
    return out;
  });

  defineTool(server, 'list_gym_builds', {
    title: 'List gym builds',
    description: `Unity gym builds on this machine (${UNITY_BINARY_DIR} and unity/Builds), newest first, and whether each is `
      + 'already packaged in unity/Builds/release.',
    readOnly: true,
  }, async () => gymBuilds().map((b) => ({
    ...b,
    size_mb: +(dirSize(b.path) / 1048576).toFixed(0),
    packaged_zip: fs.existsSync(path.join(RELEASE_DIR, assetName(b.name))) ? path.join(RELEASE_DIR, assetName(b.name)) : null,
  })));

  defineTool(server, 'package_gym', {
    title: 'Package gym build',
    description: 'Zip a gym build as unity/Builds/release/roboracer-gym-<build>.zip plus .sha256, in the layout the installer '
      + 'expects (one <build>/ folder, player .exe at its top; Player.log and *_DoNotShip folders left out).',
    input: {
      build: z.string().optional().describe('Build folder name (list_gym_builds) or a path. Default: the newest build.'),
      overwrite: z.boolean().optional().describe('Rebuild the zip even if one exists. Default false (reuse it).'),
    },
  }, async ({ build, overwrite }) => {
    const b = resolveBuild(build);
    return { build: b.name, source: b.path, ...(await packageBuild(b, overwrite)) };
  });

  defineTool(server, 'add_gym_to_release', {
    title: 'Add gym to release',
    description: `Package a gym build (if needed) and upload the zip and .sha256 to a GitHub release of ${GITHUB_REPO}: an `
      + 'existing release (default v0.1), or a new one with create_release. Refuses to overwrite a same-named file unless '
      + 'replace. Then downloads it back from the public URL to check the checksum, and can register the build on the Gyms '
      + `tab. Does not edit the installer's gym lists (${INSTALLER_GYM_LISTS}); that is a code change. Large uploads take minutes.`,
    input: {
      build: z.string().optional().describe('Build folder name or path. Default: the newest build.'),
      tag: z.string().optional().describe('Release tag. Default v0.1.'),
      create_release: z.boolean().optional().describe('Create the release (and tag, from the default branch) if it does not exist.'),
      title: z.string().optional().describe('New release title. Default: the tag.'),
      notes: z.string().optional().describe('New release notes.'),
      prerelease: z.boolean().optional(),
      make_latest: z.boolean().optional().describe('New release: mark as Latest. Default true.'),
      replace: z.boolean().optional().describe('Overwrite files of the same name already on the release.'),
      repackage: z.boolean().optional().describe('Rebuild the zip even if one exists.'),
      verify: z.boolean().optional().describe('Download and checksum the uploaded zip. Default true.'),
      register_gym: z.boolean().optional().describe('Also add the build to the Gyms tab (skipped if a gym already points at it).'),
      gym_name: z.string().optional().describe('Gyms tab name. Default: the build name.'),
      gym_description: z.string().optional(),
    },
  }, async (a) => {
    const tag = a.tag || 'v0.1';
    const b = resolveBuild(a.build);
    let release = await releaseView(tag);
    if (!release && !a.create_release) {
      throw new Error(`release ${tag} does not exist in ${GITHUB_REPO}; pass create_release: true to create it`);
    }
    if (release && a.create_release) {
      throw new Error(`release ${tag} already exists; omit create_release to add the gym to it`);
    }
    const name = assetName(b.name);
    if (release && !a.replace) {
      const clash = release.assets.filter((x) => x.name === name || x.name === `${name}.sha256`).map((x) => x.name);
      if (clash.length) throw new Error(`release ${tag} already has ${clash.join(' and ')}; pass replace: true to overwrite`);
    }

    const pkg = await packageBuild(b, a.repackage);
    const uploadTimeout = 60 * 60 * 1000;
    if (release) {
      await gh(['release', 'upload', tag, pkg.zip, pkg.sha256_file, '-R', GITHUB_REPO, ...(a.replace ? ['--clobber'] : [])], { timeoutMs: uploadTimeout });
    } else {
      await gh(['release', 'create', tag, pkg.zip, pkg.sha256_file, '-R', GITHUB_REPO,
        '--title', a.title || tag, '--notes', a.notes || `Unity gym ${b.name}`,
        ...(a.prerelease ? ['--prerelease'] : []),
        ...(a.make_latest === false ? ['--latest=false'] : [])], { timeoutMs: uploadTimeout });
    }
    release = await releaseView(tag);

    const result = {
      release: release && release.url,
      tag,
      created_release: !!a.create_release,
      build: b.name,
      package: pkg,
      urls: [downloadUrl(tag, name), downloadUrl(tag, `${name}.sha256`)],
    };
    if (a.verify !== false) {
      result.verify = release && release.isDraft
        ? { skipped: 'draft releases are not publicly downloadable' }
        : await verifyDownload(tag, name, pkg.sha256);
    }
    if (a.register_gym) {
      const gyms = await list('/get_gyms');
      const existing = gyms.find((g) => path.resolve(String(g.file_path || '')).toLowerCase() === b.exe.toLowerCase());
      if (existing) {
        result.gym = { skipped: `gym ${existing._id} (${existing.name}) already points at this build` };
      } else {
        const res = await post('/add_gym', { name: a.gym_name || b.name, file_path: b.exe, description: a.gym_description || '' });
        result.gym = { created: String(res.insertedId), name: a.gym_name || b.name, file_path: b.exe };
      }
    }
    result.installer_note = `New installs only download gyms named in ${INSTALLER_GYM_LISTS}; add this one there to ship it.`;
    return result;
  });

  defineTool(server, 'delete_release_asset', {
    title: 'Delete release file',
    description: `Remove one file from a GitHub release of ${GITHUB_REPO}. Installers that download it will fail. Requires confirm: true.`,
    destructive: true,
    input: { tag: z.string(), asset: z.string().describe('File name, e.g. roboracer-gym-<build>.zip.'), confirm: confirmArg },
  }, async ({ tag, asset, confirm }) => {
    const release = await releaseView(tag);
    if (!release) throw new Error(`release ${tag} does not exist`);
    if (!release.assets.some((x) => x.name === asset)) {
      throw new Error(`release ${tag} has no file ${asset}. Files: ${release.assets.map((x) => x.name).join(', ')}`);
    }
    requireConfirm(confirm, `Deleting ${asset} from release ${tag}`);
    await gh(['release', 'delete-asset', tag, asset, '-R', GITHUB_REPO, '--yes']);
    return { deleted: asset, tag };
  });
}
