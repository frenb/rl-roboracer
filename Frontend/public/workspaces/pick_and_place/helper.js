async function executePose(pose) {
    var plan = await api.getPlan(pose);
    await api.doTrajectory(plan.trajectory);
    await api.waitNextResult();
}