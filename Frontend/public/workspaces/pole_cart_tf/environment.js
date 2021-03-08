class Action {
    constructor(joint_delta) {
        this.joint_delta = joint_delta;
    }
}

class Environment {
    constructor(scene_data) {
        this._time_start = Date.now();
        updateFromSceneData(scene_data);
    }
    
    updateFromSceneData(scene_data) {
        this.joint_00 = scene_data.joint_00;
        this.pole_x = scene_data.pole_pose.position.x;
        this.pole_y = scene_data.pole_pose.position.y;
        this.pole_z = scene_data.pole_pose.position.z;
        this.pole_o_x = scene_data.pole_pose.orientation.x;
        this.pole_o_y = scene_data.pole_pose.orientation.y;
        this.pole_o_z = scene_data.pole_pose.orientation.z;
        this.pole_o_w = scene_data.pole_pose.orientation.w;
        this.pole_upright = scene_data.pole_upright;
        if (this.pole_upright) {
            this.upright_millis = Date.now() - this._time_start;
        }
    }
    
    async update(action) {
        var scene_data = await api.getSceneData();
        
        positions = {
            joint_00: scene_data.joint_00 + action.joint_delta,
            joint_01: scene_data.joint_01,
            joint_02: scene_data.joint_02,
            joint_03: scene_data.joint_03,
            joint_04: scene_data.joint_04,
            joint_05: scene_data.joint_05
        }
        
        var position_cmd = {
                cmd_type: 4 /* positions */,
                positions: positions
            };
        
        await api.doMove({ cmd: position_cmd });
        
        var new_scene_data = await api.getSceneData();
        updateFromSceneData(new_scene_data);
        
        return !this.pole_upright
    }
    
    getStateTensor() {
        return tf.tensor([
            this.joint_00,
            this.pole_x,
            this.pole_y,
            this.pole_z,
            this.pole_o_x,
            this.pole_o_y,
            this.pole_o_z,
            this.pole_o_w]);
    }
}