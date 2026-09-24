// Audits every experiment design against the live schema and reports
// designs that do not configure what their name or provenance claims.
//
//   docker compose exec -T dashboard node /dashboard/tools/audit_experiment_designs.js
//
// Three classes of finding:
//   NAME     name promises a setting the effective config does not deliver
//   CONFIG   fields are internally inconsistent (e.g. AWAC with no demos)
//   INHERIT  a derived auto:* design dropped fields from its stated base
//
// Effective config = trainer defaults overlaid with the design's own
// fields, matching what apply_to_main_kwargs() hands to main().
const { MongoClient, ObjectId } = require('/dashboard/node_modules/mongodb');

const MONGO = process.env.MONGO_URL || 'mongodb://root:example@mongo:27017/?authSource=admin';
const SCHEMA_URL = process.env.SCHEMA_URL || 'http://localhost/get_experiment_design_schema';

// Metadata, not tunables: excluded when comparing a design's fields.
const META = new Set(['_id', 'name', 'description', 'author', 'archived', 'version',
  'created_at', 'updated_at', 'create_date', 'proposal_id', 'proposal_arm',
  'base_design_id']);

// The fields the dashboard used to drop silently. Tracked separately so a
// re-run can tell a pre-existing gap from one introduced by later work.
const FORMERLY_DROPPED = ['env_discount', 'eval_time_fraction', 'eval_train_interval_sec',
  'awac_lambda', 'awac_beta', 'awac_weight_clip', 'awac_lambda_decay_steps',
  'curriculum_stages', 'curriculum_start_stage', 'corner_radius',
  'curvature_difficulty', 'chicanes_north', 'chicanes_east', 'chicanes_south',
  'chicanes_west'];

const tunables = (d) => (d ? Object.keys(d).filter((k) => !META.has(k)) : []);
const asId = (s) => (/^[0-9a-fA-F]{24}$/.test(String(s)) ? new ObjectId(String(s)) : String(s));

async function main() {
  const { fields, source } = await (await fetch(SCHEMA_URL)).json();
  const defaults = {};
  for (const e of fields) if (e.kind === 'field') defaults[e.name] = e.default;

  const cli = await MongoClient.connect(MONGO);
  const db = cli.db('robotaxi');
  const designs = await db.collection('experiment_designs').find({}).sort({ name: 1 }).toArray();
  const byId = new Map(designs.map((d) => [String(d._id), d]));

  const findings = [];
  let carrying = 0;

  for (const d of designs) {
    const name = String(d.name || '');
    const note = (kind, msg) => findings.push({ name, kind, msg });
    const eff = { ...defaults };
    for (const k of Object.keys(defaults)) {
      if (d[k] !== undefined && d[k] !== null) eff[k] = d[k];
    }
    if (FORMERLY_DROPPED.some((k) => d[k] !== undefined && d[k] !== null)) carrying++;

    const num = (re) => { const m = name.match(re); return m ? Number(m[1]) : null; };
    const stages = Array.isArray(eff.curriculum_stages) ? eff.curriculum_stages : null;

    // --- NAME: does the design deliver what it advertises? ---
    if (/awac/i.test(name) && !(eff.awac_lambda > 0)) {
      note('NAME', `name says AWAC but awac_lambda=${eff.awac_lambda} (0 disables AWAC)`);
    }
    // "(no curriculum)" is a promise of absence, so only flag the positive claim.
    if (/curricul/i.test(name) && !/no[-_ ]curricul/i.test(name) && !(stages && stages.length)) {
      note('NAME', `name says curriculum but curriculum_stages=${JSON.stringify(eff.curriculum_stages)}`);
    }
    if (/no[-_ ]?bc/i.test(name) && eff.bc_pretrain_steps !== 0) {
      note('NAME', `name says No-BC but bc_pretrain_steps=${eff.bc_pretrain_steps}`);
    }
    if (/protected[-_ ]?demos/i.test(name) && !(eff.demo_min_keep > 0)) {
      note('NAME', `name says protected demos but demo_min_keep=${eff.demo_min_keep}`);
    }
    const checks = [
      [num(/reward[-_ ]?scale[-_ ]?(\d+)/i), 'reward_scale_factor', 'reward_scale'],
      [num(/batch[-_ ]?(\d+)/i), 'batch_size', 'batch'],
      [num(/critic[-_ ]?(\d+)/i), 'critic_fc_layers_x', 'critic width'],
    ];
    for (const [want, field, label] of checks) {
      if (want !== null && Number(eff[field]) !== want) {
        note('NAME', `name says ${label} ${want} but ${field}=${eff[field]}`);
      }
    }
    const nStages = num(/\((\d+)[-_ ]?stage/i);
    if (nStages !== null && stages && stages.length !== nStages) {
      note('NAME', `name says ${nStages} stages but curriculum_stages has ${stages.length}`);
    }

    // --- CONFIG: internally inconsistent regardless of the name ---
    if (eff.awac_lambda > 0 && !(eff.demo_min_keep > 0)) {
      note('CONFIG', `awac_lambda=${eff.awac_lambda} needs demo_min_keep>0, got ${eff.demo_min_keep} - AWAC has no demos to sample`);
    }
    if (eff.demo_sample_ratio > 0 && !(eff.demo_min_keep > 0)) {
      note('CONFIG', `demo_sample_ratio=${eff.demo_sample_ratio} only applies in two-table mode, but demo_min_keep=${eff.demo_min_keep}`);
    }
    if (eff.awac_lambda_decay_steps > 0 && !(eff.awac_lambda > 0)) {
      note('CONFIG', `awac_lambda_decay_steps=${eff.awac_lambda_decay_steps} with awac_lambda=0 - nothing to decay`);
    }
    if (stages && stages.length && eff.curriculum_start_stage >= stages.length) {
      note('CONFIG', `curriculum_start_stage=${eff.curriculum_start_stage} out of range for ${stages.length} stages`);
    }
    if (eff.demo_min_keep > eff.demo_prefill_count) {
      note('CONFIG', `demo_min_keep=${eff.demo_min_keep} exceeds demo_prefill_count=${eff.demo_prefill_count}`);
    }
    if (typeof d.curriculum_stages === 'string') {
      note('CONFIG', `curriculum_stages stored as a string, not an array`);
    }

    // --- INHERIT: derived designs do not copy their base's fields ---
    if (d.base_design_id) {
      const base = byId.get(String(d.base_design_id))
        || await db.collection('experiment_designs').findOne({ _id: asId(d.base_design_id) });
      if (!base) {
        note('INHERIT', `base_design_id=${d.base_design_id} does not resolve`);
      } else {
        const dropped = tunables(base).filter((k) => d[k] === undefined);
        // Only material if the base actually differed from the defaults the
        // trainer falls back to; a base equal to stock defaults loses nothing.
        const material = dropped.filter((k) => JSON.stringify(base[k]) !== JSON.stringify(defaults[k]));
        if (material.length) {
          note('INHERIT', `declares base '${base.name}' but drops ${material.length} field(s) that differ from defaults: `
            + material.map((k) => `${k} ${JSON.stringify(base[k])}->${JSON.stringify(defaults[k])}`).join(', '));
        }
      }
    }
  }

  const usage = async (d) => {
    const ids = [String(d._id), d._id];
    return [await db.collection('jobs').countDocuments({ experiment_design_id: { $in: ids } }),
      await db.collection('models').countDocuments({ experiment_design_id: { $in: ids } })];
  };

  const lines = [];
  lines.push(`schema source=${source} (${Object.keys(defaults).length} fields), designs=${designs.length}`);
  lines.push(`designs carrying >=1 of the ${FORMERLY_DROPPED.length} formerly-dropped fields: ${carrying}`);
  lines.push('');

  if (!findings.length) {
    lines.push('No findings.');
  } else {
    const grouped = new Map();
    for (const f of findings) {
      if (!grouped.has(f.name)) grouped.set(f.name, []);
      grouped.get(f.name).push(f);
    }
    lines.push(`${findings.length} finding(s) across ${grouped.size} design(s):`);
    for (const [name, fs] of grouped) {
      const d = designs.find((x) => x.name === name);
      const [jobs, models] = await usage(d);
      lines.push('');
      lines.push(`  ${name}   [jobs=${jobs} models=${models}]`);
      for (const f of fs) lines.push(`     ${f.kind}: ${f.msg}`);
    }
  }
  console.log(lines.join('\n'));
  await cli.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
