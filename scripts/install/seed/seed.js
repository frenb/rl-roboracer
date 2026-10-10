// Inserts the starter gyms, reward designs and experiment designs from
// seed.json (copied into the mongo container by Install.ps1, with
// {{UNITY_BINARY_DIR}} already replaced). A document whose _id already
// exists only gains the top-level fields it is missing (e.g. a gym's
// default designs added in a later release); fields it already has are
// never overwritten, so re-running keeps user edits.
const seed = EJSON.parse(require('fs').readFileSync('/tmp/rl-seed.json', 'utf8'));
const counts = {};
for (const coll of Object.keys(seed)) {
  let inserted = 0, filled = 0;
  for (const doc of seed[coll]) {
    const existing = db.getCollection(coll).findOne({ _id: doc._id });
    if (!existing) {
      db.getCollection(coll).insertOne(doc);
      inserted++;
      continue;
    }
    const missing = {};
    for (const k of Object.keys(doc)) {
      if (!(k in existing)) missing[k] = doc[k];
    }
    if (Object.keys(missing).length) {
      db.getCollection(coll).updateOne({ _id: doc._id }, { $set: missing });
      filled++;
    }
  }
  counts[coll] = `${inserted}/${seed[coll].length}` + (filled ? ` (+fields on ${filled})` : '');
}
print('SEEDED ' + JSON.stringify(counts));
