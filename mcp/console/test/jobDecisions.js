// Offline check of create_job's open-setting questions. Needs no dashboard.
import assert from 'node:assert/strict';
import { planJob } from '../src/jobDecisions.js';

const gym = { _id: 'g000000000000000000000a1', name: 'fences-v4', default_reward_design_id: 'rd_gym' };
const other = { _id: 'g000000000000000000000b2', name: 'old-gym' };
const duckGym = { _id: 'microduck:Mjlab-Velocity-Flat-MicroDuck', robot: 'microduck', task_id: 'Mjlab-Velocity-Flat-MicroDuck', name: 'flat' };
const data = {
  gyms: [gym, other, duckGym],
  demos: [
    { _id: 'd000000000000000000000c1', course: 'donut', gym_name: 'fences-v4', num_demos: 20, num_steps: 9000 },
    { _id: 'd000000000000000000000c2', course: 'donut', gym_name: 'old-gym', num_demos: 40, num_steps: 99000 },
    { _id: 'd000000000000000000000c3', course: 'donut_camera', gym_name: 'old-gym', num_demos: 10, num_steps: 5000 },
    { _id: 'd000000000000000000000c4', course: 'donut_camera', gym_name: 'old-gym', num_demos: 12, num_steps: 6000 },
  ],
  jobs: [{
    _id: 'j000000000000000000000d1', job_type: 'TRAIN', status: 'DONE', course_type: 'donut', gym_id: gym._id,
    num_iterations: 25000, training_steps: 1000, started_at: '2026-11-12T00:00:00Z', ended_at: '2026-11-12T00:20:00Z',
    reward_design_id: 'rd_v4', experiment_design_id: 'ed_awac',
  }],
  models: [{ _id: 'm1', course_type: 'donut', location: '/m/1', reward_design_id: 'rd_best', experiment_design_id: 'ed_awac' }],
  scores: [{ model_id: 'm1', scores: [10, 12, 14] }],
  rewardDesigns: [{ _id: 'rd_gym', name: 'Gym reward' }, { _id: 'rd_v4', name: 'v4' }, { _id: 'rd_best', name: 'Best reward' }],
  experimentDesigns: [
    { _id: 'experiment-default', name: 'Default' },
    { _id: 'ed_awac', name: 'AWAC + BC', bc_pretrain_steps: 5000, awac_lambda: 1 },
    { _id: 'ed_plain', name: 'Plain SAC' },
  ],
};
const settings = (plan) => plan.questions.map((q) => q.setting);

// Everything open on a flat course: the only demo set on the gym is picked, the rest asked.
let plan = planJob({ job_type: 'TRAIN', gym: 'fences-v4' }, data);
assert.deepEqual(settings(plan), ['reward_design', 'experiment_design', 'num_iterations']);
assert.equal(plan.resolved.demo, 'd000000000000000000000c1');
const reward = plan.questions[0].options;
assert.equal(reward[0].value, 'rd_best');
assert.equal(reward[0].recommended, true);
assert.deepEqual(reward.map((o) => o.value), ['rd_best', 'rd_v4', 'rd_gym', undefined]);
assert.equal(plan.questions[2].options[0].value, 25000);
assert.match(plan.questions[2].options[0].label, /about 8\.3 h/);

// All given: nothing to ask.
plan = planJob({ job_type: 'TRAIN', gym: 'fences-v4', reward_design: 'rd_v4', experiment_design: 'ed_awac',
  num_iterations: 5000, demo: 'd000000000000000000000c1' }, data);
assert.deepEqual(settings(plan), []);

// use_defaults accepts a default instead of asking.
plan = planJob({ job_type: 'TRAIN', gym: 'fences-v4', use_defaults: ['reward_design', 'experiment_design', 'num_iterations'] }, data);
assert.deepEqual(settings(plan), []);

// A camera course with two candidate sets on another gym: asked, with no course default to fall back on.
plan = planJob({ job_type: 'TRAIN', gym: 'fences-v4', course_type: 'donut_camera', reward_design: 'rd_v4',
  experiment_design: 'ed_awac', num_iterations: 5000 }, data);
assert.deepEqual(settings(plan), ['demo']);
assert.deepEqual(plan.questions[0].options.map((o) => o.value), ['d000000000000000000000c4', 'd000000000000000000000c3']);
assert.match(plan.questions[0].note, /No donut_camera demo set was recorded on fences-v4/);

// A design that uses no demos doesn't ask about them.
plan = planJob({ job_type: 'TRAIN', gym: 'fences-v4', course_type: 'donut_camera', reward_design: 'rd_v4',
  experiment_design: 'ed_plain', num_iterations: 5000 }, data);
assert.deepEqual(settings(plan), []);

// Camera demos on a flat course are refused.
assert.throws(() => planJob({ job_type: 'TRAIN', gym: 'fences-v4', demo: 'd000000000000000000000c3',
  reward_design: 'rd_v4', experiment_design: 'ed_awac', num_iterations: 5000 }, data), /flat observations/);

// Not asked: EVAL and DEMO jobs.
assert.deepEqual(settings(planJob({ job_type: 'EVAL', location: '/m/1' }, data)), []);
assert.deepEqual(settings(planJob({ job_type: 'DEMO', gym: 'fences-v4' }, data)), []);

// microduck TRAIN asks its own three.
plan = planJob({ job_type: 'TRAIN', gym: 'Mjlab-Velocity-Flat-MicroDuck' }, data);
assert.deepEqual(settings(plan), ['reward_design', 'experiment_design', 'max_iterations']);

console.log('jobDecisions: ok');
