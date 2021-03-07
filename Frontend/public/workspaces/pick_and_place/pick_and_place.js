async function start() {
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
    var pose = {position: posePoint, orientation: poseOrientation};
    await executePose(pose);
    log("Done");

    log("Opening gripper...");
    await api.doOpenGripper();
    log("Done");

    log("Planning lowering of arm over object...");
    pose.position.z -= 0.04;
    await executePose(pose);
    log("Done");

    log("Closing gripper...");
    await api.doCloseGripper();
    log("Done");

    log("Executing place trajectory...")
    posePoint = scene_data.target_location;
    posePoint.z += 0.15; // stop just above target.
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    await executePose(pose);
    log("Done");

    log("Opening gripper...");
    await api.doOpenGripper();
    log("Done");
}