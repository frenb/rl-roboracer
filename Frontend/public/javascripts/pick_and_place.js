async function start() {
    api.clearResult();
    log("Starting...");
    
    log("Getting Latest Scene Data...");
    scene_data = await api.getSceneData();
    log("Received Scene Data " + JSON.stringify(scene_data));

    log("Planning pick trajectory...")
    posePoint = scene_data.object_location;
    posePoint.z += 0.10; // stop just above object.
    posePoint.y -= 0.01;
    posePoint.x -= 0.01;
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    plan = await api.getPlan(pose);

    log("Executing pick move with plan...");
    await api.doTrajectory(plan.trajectory);
    await api.waitNextResult();
    log("Done");

    log("Opening gripper...");
    await api.doOpenGripper();
    await api.waitNextResult();
    log("Done");

    log("Planning lowering of arm over object...");
    pose.position.z -= 0.04;
    plan = await api.getPlan(pose);

    log("Lowering arm over object...");
    await api.doTrajectory(plan.trajectory);
    await api.waitNextResult();
    log("Done");

    log("Closing gripper...");
    await api.doCloseGripper();
    await api.waitNextResult();
    log("Done");

    log("Planning place trajectory...")
    posePoint = scene_data.target_location;
    posePoint.z += 0.15; // stop just above target.
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    plan = await api.getPlan(pose);


    log("Executing place move with plan...");
    await api.doTrajectory(plan.trajectory);
    await api.waitNextResult();
    log("Done");

    log("Opening gripper...");
    await api.doOpenGripper();
    await api.waitNextResult();
    log("Done");
}