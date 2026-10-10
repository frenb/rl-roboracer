// Inserts the starter gyms, reward designs and experiment designs from
// seed.json (copied into the mongo container by Install.ps1, with
// {{UNITY_BINARY_DIR}} already replaced). Documents whose _id already
// exists are left alone, so re-running never overwrites user edits.
const seed = EJSON.parse(require('fs').readFileSync('/tmp/rl-seed.json', 'utf8'));
const counts = {};
for (const coll of Object.keys(seed)) {
  let inserted = 0;
  for (const doc of seed[coll]) {
    const r = db.getCollection(coll).updateOne(
      { _id: doc._id }, { $setOnInsert: doc }, { upsert: true });
    inserted += r.upsertedCount;
  }
  counts[coll] = `${inserted}/${seed[coll].length}`;
}
print('SEEDED ' + JSON.stringify(counts));
