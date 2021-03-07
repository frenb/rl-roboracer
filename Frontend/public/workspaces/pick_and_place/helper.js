async function executePose(pose) {
    var plan = await api.getPlan(pose);
    await api.doTrajectory(plan.trajectory);
}

async function randomWalk() {
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
                cmd_type: 4 ,
                positions: positions
            };
        
        await api.doMove({ cmd: position_cmd });
    }
}