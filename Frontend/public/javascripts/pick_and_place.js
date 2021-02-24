async function start() {
    api.clearResult();
    console.log("Starting...");
    
    console.log("Getting Latest Scene Data...");
    scene_data = await api.getSceneData();
    console.log("Received Scene Data " + JSON.stringify(scene_data));

    console.log("Planning pick trajectory...")
    posePoint = scene_data.object_location;
    posePoint.z += 0.10; // stop just above object.
    posePoint.y -= 0.01;
    posePoint.x -= 0.01;
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    plan = await api.getPlan(pose);

    console.log("Executing pick move with plan...");
    await api.doTrajectory(plan.trajectory);
    await api.waitNextResult();
    console.log("Done");

    console.log("Opening gripper...");
    await api.doOpenGripper();
    await api.waitNextResult();
    console.log("Done");

    console.log("Planning lowering of arm over object...");
    pose.position.z -= 0.04;
    plan = await api.getPlan(pose);

    console.log("Lowering arm over object...");
    await api.doTrajectory(plan.trajectory);
    await api.waitNextResult();
    console.log("Done");

    console.log("Closing gripper...");
    await api.doCloseGripper();
    await api.waitNextResult();
    console.log("Done");

    console.log("Planning place trajectory...")
    posePoint = scene_data.target_location;
    posePoint.z += 0.15; // stop just above target.
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    plan = await api.getPlan(pose);


    console.log("Executing place move with plan...");
    await api.doTrajectory(plan.trajectory);
    await api.waitNextResult();
    console.log("Done");

    console.log("Opening gripper...");
    await api.doOpenGripper();
    await api.waitNextResult();
    console.log("Done");
}