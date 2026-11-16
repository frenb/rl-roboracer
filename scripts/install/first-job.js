// Queues the installer's first TRAIN job (mongosh, database robotaxi): a
// short job on the default gym with that gym's seeded course, reward design
// and experiment design, stamped the way the New-job form stamps them; no
// first eval, and no BC pretrain unless the experiment design asks for it.
// An experiment design's fields win over the job's, so it must not set
// num_iterations. A failed earlier first job is set aside and a new one
// queued; one in progress or done is left alone. Prints
// "FIRST_JOB <id> <status>". Used by install.md (step firstjob); the same
// job as Install.ps1's first-job phase.
//
//   FIRST_JOB_GYM_ID      gym _id (default: the v13 starter gym)
//   FIRST_JOB_ITERATIONS  training iterations (default 2000)

const gymId = process.env.FIRST_JOB_GYM_ID || '6aafa685c2736b990584403f';
const iterations = parseInt(process.env.FIRST_JOB_ITERATIONS || '2000', 10);

db.jobs.updateMany({installer_first_job: true, status: 'FAILED'}, {$set: {installer_first_job: false}});
const existing = db.jobs.findOne({installer_first_job: true}, {status: 1});
if (existing) {
  print('FIRST_JOB ' + existing._id.toString() + ' ' + existing.status);
} else {
  const g = db.gyms.findOne({_id: ObjectId(gymId)}) || {};
  const byId = (c, id) => id ? (db.getCollection(c).findOne({_id: id}) ||
    db.getCollection(c).findOne({_id: ObjectId.isValid(id) ? ObjectId(id) : id})) : null;
  const rd = byId('reward_designs', g.default_reward_design_id);
  const ed = byId('experiment_designs', g.default_experiment_design_id);
  const job = {
    job_type: 'TRAIN', model_type: 'SacAgent', robot_type: 'robotaxi',
    num_iterations: iterations, status: 'NOT_STARTED', create_date: new Date(),
    demo_job_id: '', pass_through_actions: false, nn_size_x: '', nn_size_y: '', seed: 0,
    skip_first_eval: true, bc_pretrain_steps: 0,
    gym_id: g._id ? g._id.toString() : '', gym_name: g.name || '', gym_file_path: g.file_path || '',
    percent_complete: 0, installer_first_job: true,
  };
  if (g.default_course_type) job.course_type = g.default_course_type;
  if (rd) { job.reward_design_id = rd._id.toString(); job.reward_design_name = rd.name; }
  if (ed) { job.experiment_design_id = ed._id.toString(); job.experiment_design_name = ed.name; }
  print('designs: ' + (job.course_type || 'trainer default') + ' / ' +
    (job.reward_design_name || 'course default reward') + ' / ' + (job.experiment_design_name || 'trainer defaults'));
  print('FIRST_JOB ' + db.jobs.insertOne(job).insertedId.toString() + ' NOT_STARTED');
}
