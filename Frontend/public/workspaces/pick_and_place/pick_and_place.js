async function start() {
    
    while (true) {
        var scene_data = await api.getSceneData();
        
        positions = {
            joint_00: scene_data.joint_00 + (Math.random() * 2 -1) / 100,
            joint_01: scene_data.joint_01 + (Math.random() * 2 -1) / 100,
            joint_02: scene_data.joint_02 + (Math.random() * 2 -1) / 100,
            joint_03: scene_data.joint_03 + (Math.random() * 2 -1) / 100,
            joint_04: scene_data.joint_04 + (Math.random() * 2 -1) / 100,
            joint_05: scene_data.joint_05 + (Math.random() * 2 -1) / 100
        }
        
        log(JSON.stringify(positions));
        
        var position_cmd = {
                cmd_type: 4 /* positions */,
                positions: positions
            };
        
        await api.doMove({ cmd: position_cmd });
        await api.waitNextResult();
    }
        
    /*
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
    var pose = {position: posePoint, orientation: poseOrientation};
    await executePose(pose);
    log("Done");

    log("Opening gripper...");
    await api.doOpenGripper();
    await api.waitNextResult();
    log("Done");

    log("Planning lowering of arm over object...");
    pose.position.z -= 0.04;
    await executePose(pose);
    log("Done");

    log("Closing gripper...");
    await api.doCloseGripper();
    await api.waitNextResult();
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
    await api.waitNextResult();
    log("Done");*/
}