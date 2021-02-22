
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}  

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

async function doTrajectory(trajectory) {
    move_command = {
        cmd_type: 1 /* trajectory */,
        trajectory: trajectory
    };
    return doMove({cmd: move_command});
}

async function doOpenGripper() {
    move_command = {
        cmd_type: 2 /* open */
    };
    return doMove({cmd: move_command});
}

async function doCloseGripper() {
    move_command = {
        cmd_type: 3 /* close */
    };
    return doMove({cmd: move_command});
}

async function getLatestResult() {
    return $.ajax({
        dataType: "json",
        contentType: 'application/json',
        type: 'GET',
        url: '/result/latest'

      }).promise();
}

async function waitUntilNewResult(last_result) {
    while(true) {
        last_ts = last_result.timestamp || 0;
        current_result = await getLatestResult();
        current_ts = current_result.timestamp || 0;
        if (current_ts > last_ts) {
            return current_result;
        }
        await sleep(1000);
    }
}

async function start() {
    last_result = await getLatestResult();
    console.log("Starting...");
    
    console.log("Getting Latest Scene Data...");
    scene_data = await getSceneData();
    console.log("Received Scene Data " + JSON.stringify(scene_data));

    console.log("Planning pick trajectory...")
    posePoint = scene_data.object_location;
    posePoint.z += 0.10; // stop just above object.
    posePoint.y -= 0.01;
    posePoint.x -= 0.01;
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    plan = await getPlan(pose);

    console.log("Executing pick move with plan...");
    await doTrajectory(plan.trajectory);
    last_result = await waitUntilNewResult(last_result);
    console.log("Done");

    console.log("Opening gripper...");
    await doOpenGripper();
    last_result = await waitUntilNewResult(last_result);
    console.log("Done");

    console.log("Planning lowering of arm over object...");
    pose.position.z -= 0.04;
    plan = await getPlan(pose);

    console.log("Lowering arm over object...");
    await doTrajectory(plan.trajectory);
    last_result = await waitUntilNewResult(last_result);
    console.log("Done");

    console.log("Closing gripper...");
    await doCloseGripper();
    last_result = await waitUntilNewResult(last_result);
    console.log("Done");

    console.log("Planning place trajectory...")
    posePoint = scene_data.target_location;
    posePoint.z += 0.15; // stop just above target.
    poseOrientation = {x : -0.5, y : -0.5, z : 0.5, w : -0.5}; // perpendicular to table.
    pose = {position: posePoint, orientation: poseOrientation};
    plan = await getPlan(pose);


    console.log("Executing place move with plan...");
    await doTrajectory(plan.trajectory);
    last_result = await waitUntilNewResult(last_result);
    console.log("Done");

    console.log("Opening gripper...");
    await doOpenGripper();
    last_result = await waitUntilNewResult(last_result);
    console.log("Done");
}