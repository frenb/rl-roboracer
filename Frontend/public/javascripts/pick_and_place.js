

async function getSceneData() {
    return $.ajax({
        dataType: "json",
        url: '/sceneData/latest',
      }).promise();
}

async function getPlan(pose) {
    return $.ajax({
        dataType: "json",
        contentType: 'application/json',
        type: 'POST',
        url: '/plan',
        data: JSON.stringify(pose)

      }).promise();
}

async function doMove(action) {
    return $.ajax({
        contentType: 'application/json',
        type: 'POST',
        url: '/move',
        data: JSON.stringify(action)

      }).promise();
}

async function getPickTrajectory(scene_data) {
    posePoint = scene_data.object_location;
    posePoint.z += 0.10; // stop just above object.
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    return getPlan(pose);
}

async function doTrajectory(trajectory) {
    move_command = {
        cmd_type: 1 /* trajectory */,
        trajectory: trajectory
    };
    return doMove({cmd: move_command});
}

async function start() {
    console.log("Starting...");
    
    console.log("Getting Latest Scene Data...");
    scene_data = await getSceneData();
    console.log("Received Scene Data " + JSON.stringify(scene_data));

    console.log("Planning pick trajectory...")
    plan = await getPickTrajectory(scene_data);
    console.log("Received pick trajectory " + JSON.stringify(plan))

    console.log("Executing pick move with plan...");
    await doTrajectory(plan.trajectory);
}