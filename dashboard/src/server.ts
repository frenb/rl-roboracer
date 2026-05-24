import * as express from 'express';
import * as bodyParser from 'body-parser';
import * as path from 'path';
import * as fs from 'fs';
import * as mongoDB from 'mongodb';
const ObjectID = require('mongodb').ObjectID;
const cors = require('cors');
const { spawn } = require('child_process');
import { log, LogLevel } from './log';
import * as morgan from 'morgan';

const WebSocket = require('ws');


export const createServer = (config): express.Application => {
  const app: express.Application = express();
  const databaseName = process.env.DATABASE_NAME || "robotaxi";
  var jobsChanged = true;
  var modelsChanged = true;
  var leaderboardScoresChanged = true;
  // The env_specs collection holds one document per robot_type
  // describing the live env's observation / action spec. Populated by
  // robotaxi.py's publish_env_spec() at every TRAIN / EVAL job start.
  // Used by the Models tab's "Compat" column to flag rows whose
  // training-time spec no longer matches the running env.
  var envSpecsChanged = true;
  // The reward_designs collection holds user-authored reward functions
  // (see rl_agent/reward_designs.py). One document per design with a
  // human-friendly name + Python source code + version. Selected
  // per-job via the New-Job form's "Reward design" dropdown.
  var rewardDesignsChanged = true;
  var MongoClient = mongoDB.MongoClient;
  var url = process.env.MONGODB_URL || "mongodb://root:example@mongo:27017";
  var dbo;
  MongoClient.connect(url, function(err, db) {
    if (err) throw err;
    
    dbo = db.db(databaseName);
    // create a collection object for the collection based on collection_name
    // collectionNames.forEach(function(collectionName) {
    //   const collection = dbo.collection(collectionName);
    //   const changeStream = collection.watch();
  
    //   // listen for changes in the collection
    //   changeStream.on('change', function(change) {
    //     console.log(`Change detected in ${collectionName}:`, change);
    //     // handle the change event here
    //   });
    // });
    const jobs = dbo.collection("jobs");
    const models = dbo.collection("models");
    const leaderboardScores = dbo.collection("leaderboard_scores");
    const envSpecs = dbo.collection("env_specs");
    
    const jobsChangeStream = jobs.watch();
    jobsChangeStream.on('change', (change) => {
      console.log('Change detected:', change);
      jobsChanged=true;
    });

    const modelsChangeStream = models.watch();
    modelsChangeStream.on('change', (change) => {
      console.log('Change detected:', change);
      modelsChanged=true;
    });

    const leaderboardScoresChangeStream = leaderboardScores.watch();
    leaderboardScoresChangeStream.on('change', (change) => {
      console.log('Change detected:', change);
      leaderboardScoresChanged=true;
    });

    const envSpecsChangeStream = envSpecs.watch();
    envSpecsChangeStream.on('change', (change) => {
      console.log('Change detected (env_specs):', change);
      envSpecsChanged=true;
    });

    const rewardDesigns = dbo.collection("reward_designs");
    const rewardDesignsChangeStream = rewardDesigns.watch();
    rewardDesignsChangeStream.on('change', (change) => {
      console.log('Change detected (reward_designs):', change);
      rewardDesignsChanged=true;
    });

  });

  if (config.logging != "none") {
    app.use(morgan(config.logging));
  }

  app.use(cors());
  app.options('*', cors());

  app.use(bodyParser.urlencoded({ extended: true }));
  app.use(bodyParser.json());
  app.use(express.static(path.join(__dirname, '/..')));
  app.get('/leaderboard', (req, res) => {
    const lb: string = path.join(__dirname, '/../leaderboard.html');
    res.sendFile(lb);
  });
  var needsUpdate = function (req, changed) {
    var force = (req.query.force == 'true');
    console.log(`force: ${force} req.query.force: ${req.query.force} jobsChanged: ${changed}`);
    return (changed || force);
  }


  app.get('/leaderboard_scores', (req,res) => {
    if(needsUpdate(req, leaderboardScoresChanged))
    {
      leaderboardScoresChanged=false;
      dbo.collection("leaderboard_scores").find({}).toArray(function(err, result) {
        if (err) throw err;
        //console.log(result);
        console.log(`${result.length} leaderboard scores retrieved`);
        res.json(result);
      });
      return;
    }
    
    console.log(`No leaderboard scores retrieved`);
    res.status(200).send('NO_CHANGES');
  
  });
  app.get('/logs', (req, res) => {
    const lb: string = path.join(__dirname, '/../logs.html');
    res.sendFile(lb);
    // const dockerLogs = spawn('tail', ['-f', '-n', '100', '/python_ws/src/robotaxi.out']);
    // dockerLogs.stdout.on('data', (data) => {
    //   res.write(`${data}\n`);
    // });
    // dockerLogs.stderr.on('data', (data) => {
    //   console.error(`stderr: ${data}`);
    // });
    // dockerLogs.on('close', (code) => {
    //   console.log(`child process exited with code ${code}`);
    //   res.end();
    // });

  })
  app.post('/add_job', (req,res) => {
    console.log("add_job: " + JSON.stringify(req.body));
    dbo.collection("jobs").insertOne(req.body,function(err, result) {
      if (err) throw err;
      //console.log(result);
      res.json(result)
    });
  });
  app.get('/get_jobs', (req,res) => {
    if (needsUpdate(req, jobsChanged)) {
      jobsChanged=false;
      dbo.collection("jobs").find({}).toArray(function(err, result) {
          if (err) throw err;
          //console.log(result);
          console.log(`${result.length} jobs retrieved`)
          res.json(result)
      });
      return;
    }
    
    console.log(`No jobs retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  app.get('/jobs', (req, res) => {
    const lb: string = path.join(__dirname, '/../jobs.html');
    res.sendFile(lb);
  });

  app.post('/update_job_status', (req, res) => {
    console.log("update_job_status: " + JSON.stringify(req.body));
    var job = req.body;
    const myquery = { "_id": ObjectID(job["_id"]) };
    const newvalues = { "$set": { "status": job["status"] } };
    const options = { upsert: false };
    dbo.collection("jobs").updateOne(myquery,newvalues, options, function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });

  app.post('/delete_job', (req,res) => {
    var job = req.body;
    const myquery = { "_id": ObjectID(job["_id"]) };
    console.log(myquery);
    dbo.collection("jobs").deleteOne(myquery, function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });

  app.get('/models', (req, res) => {
    const lb: string = path.join(__dirname, '/../models.html');
    res.sendFile(lb);
  });

  // Side-by-side model comparison view. Reachable both as a standalone
  // page and as a GoldenLayout tab; the heavy lifting (joining models
  // with leaderboard_scores, computing stats) is done client-side from
  // the existing /get_models and /leaderboard_scores endpoints, so no
  // dedicated data endpoint is needed here.
  app.get('/analysis', (req, res) => {
    const lb: string = path.join(__dirname, '/../analysis.html');
    res.sendFile(lb);
  });

  // Reward Design tab. Same pattern as /jobs / /models / /analysis -
  // the static HTML is also served by express.static so this clean
  // route is convenience; the iframe in goldenlayout requests the
  // .html file directly to side-step any rebuild-lag on this route.
  app.get('/reward_designs', (req, res) => {
    const f: string = path.join(__dirname, '/../reward_designs.html');
    res.sendFile(f);
  });
  app.get('/get_models', (req,res) => {
    if(needsUpdate(req, modelsChanged))
    {
      modelsChanged=false;
      // Projection excludes ``reward_design_code`` from the list-view
      // payload. That field stores the verbatim Python source of the
      // model's training-time reward design - small per model (often
      // ~1 KB) but with thousands of historical model docs it
      // dominates the /get_models response size (multi-MB JSON parsed
      // on every dashboard load). The Models tab's "View source"
      // modal fetches the source on click via /get_model_source/:id
      // below, so no UX is lost - the source just isn't pre-loaded.
      //
      // observation_spec / action_spec are kept because the Compat
      // column on the Models tab reads them directly from each row
      // to compare against the live env spec; they're tiny (a few
      // tens of bytes each) so they're not a payload concern.
      dbo.collection("models")
         .find({}, { projection: { reward_design_code: 0 } })
         .toArray(function(err, result) {
        if (err) throw err;
        //console.log(result);
        console.log(`${result.length} models retrieved`)
        res.json(result)
      });
      return;
    }
    console.log(`No models retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  // Per-model source fetch. Returns just the ``reward_design_code``
  // field for one model so the Models tab's source-view modal can
  // populate on click without paying the cost of preloading every
  // model's source through /get_models. Tiny payload (one document,
  // one field) so no projection/index tuning needed.
  //
  // 404 if the id doesn't parse as an ObjectId or no such document
  // exists. 200 with {code: null} when the model exists but has no
  // reward design recorded (legacy / RandomPyPolicy / etc.).
  app.get('/get_model_source/:id', (req, res) => {
    const idStr = String(req.params.id || '');
    let oid;
    try {
      oid = ObjectID(idStr);
    } catch (_e) {
      res.status(400).json({ error: 'invalid id' });
      return;
    }
    dbo.collection("models").findOne(
      { _id: oid },
      { projection: { reward_design_code: 1, reward_design_name: 1, reward_design_version: 1, reward_design_id: 1 } },
      function(err, doc) {
        if (err) {
          console.error('get_model_source failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        if (!doc) {
          res.status(404).json({ error: 'not found' });
          return;
        }
        res.json({
          _id: idStr,
          reward_design_id: doc.reward_design_id || null,
          reward_design_name: doc.reward_design_name || null,
          reward_design_version: doc.reward_design_version || null,
          reward_design_code: doc.reward_design_code || null,
        });
      });
  });

  // env_specs: one document per robot_type, written by
  // publish_env_spec() in rl_agent/robotaxi.py whenever a TRAIN or
  // EVAL job starts. The Models tab fetches this alongside
  // /get_models and /leaderboard_scores so the "Compat" column can
  // compare each model's stored observation/action spec against the
  // live env's. Same NO_CHANGES short-circuit as the other read-only
  // endpoints to keep the dashboard polling cheap.
  app.get('/get_env_specs', (req,res) => {
    if(needsUpdate(req, envSpecsChanged))
    {
      envSpecsChanged=false;
      dbo.collection("env_specs").find({}).toArray(function(err, result) {
        if (err) throw err;
        console.log(`${result.length} env_specs retrieved`)
        res.json(result)
      });
      return;
    }
    console.log(`No env_specs retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  // ---- reward_designs CRUD ----------------------------------------
  //
  // User-authored reward function modules (see rl_agent/reward_designs.py
  // for the contract). The dashboard's future Reward Design tab and the
  // New-Job form's reward-design dropdown both consume these.
  //
  // /get_reward_designs       : list current designs (with NO_CHANGES short-
  //                             circuit, same as other read endpoints)
  // /add_reward_design        : insert a new design
  // /update_reward_design     : update an existing design (bumps version)
  // /archive_reward_design    : soft-delete (archived: true) so historical
  //                             models keep resolving their reward design
  // /lint_reward_design       : compile-check the code via the sim-controller
  //                             container without saving anything

  app.get('/get_reward_designs', (req, res) => {
    if (needsUpdate(req, rewardDesignsChanged)) {
      rewardDesignsChanged = false;
      dbo.collection("reward_designs").find({}).toArray(function(err, result) {
        if (err) throw err;
        console.log(`${result.length} reward_designs retrieved`);
        res.json(result);
      });
      return;
    }
    console.log(`No reward_designs retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  app.post('/add_reward_design', (req, res) => {
    // Body shape: { name, description, code, author?, archived? (default false) }
    // We stamp created_at / updated_at / version=1 here so the client
    // never has to set them; subsequent edits go through
    // /update_reward_design which bumps version.
    const body = req.body || {};
    if (!body.name || !body.code) {
      res.status(400).json({ error: "name and code are required" });
      return;
    }
    const now = new Date();
    const doc = {
      name: String(body.name),
      description: String(body.description || ''),
      code: String(body.code),
      author: String(body.author || ''),
      archived: !!body.archived,
      version: 1,
      created_at: now,
      updated_at: now,
    };
    dbo.collection("reward_designs").insertOne(doc, function(err, result) {
      if (err) {
        console.error('add_reward_design failed:', err);
        res.status(500).json({ error: String(err.message || err) });
        return;
      }
      console.log('reward_design inserted', result.insertedId);
      res.json(result);
    });
  });

  app.post('/update_reward_design', (req, res) => {
    // Body shape: { _id, name?, description?, code?, archived? }
    // Edits bump version and updated_at. The previous version's code
    // is NOT preserved here (that would require a separate
    // reward_design_versions collection); historical models record
    // their training-time code on the model document itself, so a
    // user's accidental edit doesn't break reproducibility of
    // already-trained models.
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try {
      idFilter = { "_id": ObjectID(body._id) };
    } catch (e) {
      // Could be the string id used by the canonical seed (see
      // rl_agent/reward_designs.PASSTHROUGH_DESIGN_ID); accept either
      // shape so the dashboard can edit a seeded design too.
      idFilter = { "_id": String(body._id) };
    }
    const update: any = {
      "$set": { updated_at: new Date() },
      "$inc": { version: 1 },
    };
    if (body.name        !== undefined) update["$set"].name        = String(body.name);
    if (body.description !== undefined) update["$set"].description = String(body.description);
    if (body.code        !== undefined) update["$set"].code        = String(body.code);
    if (body.archived    !== undefined) update["$set"].archived    = !!body.archived;
    dbo.collection("reward_designs").updateOne(
      idFilter, update, { upsert: false },
      function(err, result) {
        if (err) {
          console.error('update_reward_design failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        console.log('reward_design updated', body._id, result);
        res.json(result);
      });
  });

  app.post('/archive_reward_design', (req, res) => {
    // Soft-delete. We do NOT physically remove the document because
    // historical model records reference reward_design_id; deleting
    // the design would orphan that reference. Archived designs are
    // hidden from the New-Job dropdown but still resolve when an
    // in-flight job looks them up.
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try {
      idFilter = { "_id": ObjectID(body._id) };
    } catch (e) {
      idFilter = { "_id": String(body._id) };
    }
    dbo.collection("reward_designs").updateOne(
      idFilter,
      { "$set": { archived: true, updated_at: new Date() } },
      { upsert: false },
      function(err, result) {
        if (err) {
          console.error('archive_reward_design failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        console.log('reward_design archived', body._id);
        res.json(result);
      });
  });

  app.post('/lint_reward_design', (req, res) => {
    // Cheap structural lint of a candidate reward-design code string.
    //
    // The authoritative compile-check is what `load_reward_design()` does
    // inside the sim-controller (rl_agent/reward_designs.py): exec'd in
    // a restricted namespace, with the right argument-set enforced. We
    // can't reach Python from inside this Node container without
    // adding docker-CLI plus a socket mount, so we approximate just
    // enough to give the editor useful feedback before save:
    //
    //   - reject obviously empty input,
    //   - require at least one of the three reward function names,
    //   - flag use of obviously-disallowed builtins so users see them
    //     here instead of waiting until job start.
    //
    // A pass here still doesn't guarantee the design loads cleanly;
    // any syntactic error caught only by Python's compile() will surface
    // at the next job pickup as a RewardDesignError on that job (see
    // do_job's TRAIN branch in rl_agent/robotaxi.py).
    const body = req.body || {};
    if (typeof body.code !== 'string') {
      res.status(400).json({ ok: false, error: "code (string) is required" });
      return;
    }
    const code: string = body.code;
    if (!code.trim()) {
      res.json({ ok: false, error: "code is empty", funcs: [] });
      return;
    }
    const allowedFuncs = ['reward_standard', 'reward_success', 'reward_failure'];
    const found: string[] = [];
    for (const name of allowedFuncs) {
      // top-level `def reward_X(...)` declaration. We match any
      // amount of leading whitespace except a leading non-blank
      // character so we still catch a deliberately-indented def
      // (which would mean the function isn't top-level and the loader
      // would skip it anyway, but the user can see they're warned).
      const re = new RegExp('(^|\\n)\\s*def\\s+' + name + '\\s*\\(', 'm');
      if (re.test(code)) found.push(name);
    }
    if (!found.length) {
      res.json({
        ok: false,
        error: "no reward functions defined. Expected one or more of: " +
               allowedFuncs.join(', '),
        funcs: [],
      });
      return;
    }
    // Look for disallowed module access. This is a *hint*, not a
    // security boundary - the real sandbox is the restricted exec
    // builtins in load_reward_design(). We just want to fail fast
    // here when someone tries `import os` because the trainer would
    // also reject it.
    const disallowedImports = [
      /(^|\n)\s*import\s+(os|sys|subprocess|socket|requests|urllib)\b/,
      /(^|\n)\s*from\s+(os|sys|subprocess|socket|requests|urllib)\b/,
    ];
    for (const re of disallowedImports) {
      const m = code.match(re);
      if (m) {
        res.json({
          ok: false,
          error: "import of restricted module: " + m[2] +
                 " (the trainer's sandbox blocks os / sys / subprocess / sockets / network).",
          funcs: found,
        });
        return;
      }
    }
    res.json({ ok: true, error: null, funcs: found });
  });

  // ---- TensorBoard comparison bucket ------------------------------
  //
  // /tmp/tb_compare/ is the on-demand bucket scanned by the
  // TensorBoard server (see sim-controller's --logdir_spec compare:
  // ... in docker-compose.yml). The dashboard manages its contents
  // through this endpoint:
  //
  //   POST /set_tb_compare_jobs  { jobIds: [...] }
  //     1. ensure /tmp/tb_compare exists (mkdir -p)
  //     2. remove every existing symlink (NOT real directories - the
  //        bucket should only contain symlinks, but be defensive)
  //     3. for each jobId, symlink /tmp/jobsdata/<jobId> ->
  //        /tmp/tb_compare/<jobId> if the source exists
  //     4. return { linked: [...], missing: [...] }
  //
  // The dashboard's Models -> Compare flow calls this BEFORE
  // re-pointing the TB iframe at a filter URL, so TB sees exactly
  // the user's selection on next reload_interval (~5s).
  //
  // /tmp is bind-mounted from the host into BOTH the dashboard and
  // sim-controller containers via docker-compose.yml's `source:
  // ../tmp; target: /tmp` blocks, so a symlink we create here is
  // visible to TB inside sim-controller.
  app.post('/set_tb_compare_jobs', (req, res) => {
    const COMPARE_ROOT = '/tmp/tb_compare';
    const ARCHIVED_ROOT = '/tmp/jobsdata';

    const body = req.body || {};
    const rawIds = Array.isArray(body.jobIds) ? body.jobIds : [];
    // Path-safety: an attacker-controlled jobId of '../..' would let
    // them point a symlink at any host path. We restrict to BSON
    // ObjectId hex (or the seeded passthrough sentinel) and reject
    // everything else.
    const safeIdRe = /^[a-zA-Z0-9_-]{1,64}$/;
    const jobIds: string[] = rawIds
      .map((x: any) => String(x || '').trim())
      .filter((s: string) => s.length > 0 && safeIdRe.test(s));

    try {
      fs.mkdirSync(COMPARE_ROOT, { recursive: true });
    } catch (e: any) {
      // mkdir failure is fatal - we can't manage the bucket.
      res.status(500).json({
        ok: false,
        error: `mkdir ${COMPARE_ROOT} failed: ${String(e && e.message || e)}`,
        linked: [], missing: [],
      });
      return;
    }

    // Sweep the bucket. We ONLY remove symlinks (and broken-symlink
    // dangling entries). If somehow a real directory ended up in
    // here, leave it - we shouldn't be deleting real data the
    // dashboard didn't put here.
    try {
      const entries = fs.readdirSync(COMPARE_ROOT);
      for (const name of entries) {
        const p = path.join(COMPARE_ROOT, name);
        let st: any;
        try { st = fs.lstatSync(p); }
        catch (_e) { continue; }
        if (st.isSymbolicLink()) {
          try { fs.unlinkSync(p); }
          catch (e) {
            console.warn(`set_tb_compare_jobs: failed to unlink ${p}: ${e}`);
          }
        }
      }
    } catch (e) {
      // readdir failure isn't fatal; we'll just stack new symlinks
      // alongside whatever was there.
      console.warn(`set_tb_compare_jobs: readdir ${COMPARE_ROOT} failed: ${e}`);
    }

    const linked: string[] = [];
    const missing: string[] = [];
    for (const id of jobIds) {
      const src = path.join(ARCHIVED_ROOT, id);
      const dst = path.join(COMPARE_ROOT, id);
      if (!fs.existsSync(src)) {
        // Could be a model whose job was never archived (e.g., the
        // currently-training job - its data is in /tmp/active, not
        // /tmp/jobsdata, so the regex filter on the TB iframe URL
        // will hit it under the "current" experiment instead).
        missing.push(id);
        continue;
      }
      try {
        // Type 'dir' makes Windows happy if this ever runs on a
        // Windows container; on Linux it's ignored. fs.symlinkSync
        // is sync because the whole endpoint is sub-100ms and an
        // async chain isn't worth the complexity.
        fs.symlinkSync(src, dst, 'dir');
        linked.push(id);
      } catch (e: any) {
        console.error(`set_tb_compare_jobs: symlink ${src} -> ${dst} failed: ${e}`);
        missing.push(id);
      }
    }

    res.json({ ok: true, linked, missing });
  });

  app.get('/', (req, res) => {
    const indexPagePath: string = path.join(__dirname, '/../index.html');
    fs.access(indexPagePath, (err) => {
      if (err) {
        log(LogLevel.warn, `Can't find file ' ${indexPagePath}`);
        res.status(404).send(`Can't find file ${indexPagePath}`);
      } else {
        res.sendFile(indexPagePath);
      }
    });
  });
  return app;
};
// create websocket server for log tailing (used by logs.html)
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  const dockerLogs = spawn('tail', ['-f', '-n', '100', '/python_ws/src/robotaxi.out']);

  dockerLogs.stdout.on('data', (data) => {
    ws.send(data.toString());
  });

  dockerLogs.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  dockerLogs.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });

  ws.on('close', () => {
    dockerLogs.kill();
  });
});

