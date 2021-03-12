
const MAX_JOINT_ANGLE = 90;



class Environment {
    constructor(scene_data) {
        this.updateFromSceneData(scene_data);
    }
    
    updateFromSceneData(scene_data) {
        scene_data = scene_data.data
        // Save entire scene data for sending move commands with correct angles
        this.scene_data = scene_data;       
        // State variables.
        this.pole_hand_angle =      scene_data.pole_cart.pole_hand_angle;
        this.pole_hand_angle_b =    scene_data.pole_cart.pole_hand_angle_b;
        this.pole_angular_speed =   scene_data.pole_cart.pole_angular_speed;
        this.pole_angular_speed_b = scene_data.pole_cart.pole_angular_speed_b;
    }
    
    
    async update(action) {
        let positions = {
            joint_00: this.scene_data.joint_00 + action[0] * 3.14 / 180.0,
            joint_01: this.scene_data.joint_01 + action[1] * 3.14 / 180.0,
            joint_02: this.scene_data.joint_02 + action[2] * 3.14 / 180.0,
            joint_03: this.scene_data.joint_03,
            joint_04: this.scene_data.joint_04,
            joint_05: this.scene_data.joint_05
        }
        
        var position_cmd = {
                cmd_type: 4 /* positions */,
                positions: positions
            };
        
        await api.doMove({ cmd: position_cmd });
        var new_scene_data = await api.getSceneData();
        this.updateFromSceneData(new_scene_data);
        
        return !this.scene_data.pole_cart.upright || Math.abs(this.scene_data.joint_00) * 180.0 / 3.14 > MAX_JOINT_ANGLE;
    }
    
    getStateTensor() {
        return tf.tensor2d([[
            this.discretizePoleHandAngle(this.pole_hand_angle),
            this.discretizePoleHandAngle(this.pole_hand_angle_b),
            this.discretizePoleAngularSpeed(this.pole_angular_speed),
            this.discretizePoleAngularSpeed(this.pole_angular_speed_b)
            ]]);
    }
    
    discretizePoleHandAngle(angle) {
        return this.discretize(angle, -30, 30, 60);
    }
    
    discretizePoleAngularSpeed(speed) {
        return this.discretize(speed, -1.5, 1.5, 5);    
    }
    
    //discretizeJointAngle(angle) {
    //    let angle_deg = angle * 180.0 / 3.14;
    //    return this.discretize(angle_deg, -MAX_JOINT_ANGLE, MAX_JOINT_ANGLE, 6);
    //}
    
    //discretizeHandTangentSpeed(speed) {
    //    return this.discretize(speed, -0.5, 0.5, 3);
    //}
    
    // TODO: map to actual variable domain instead of a bucket number.
    discretize(x, min, max, buckets) {
        x = Math.max(x, min);
        x = Math.min(x, max);
        let bucket = Math.round(((x - min) / (max - min) * buckets));
        return bucket;
    }
    
}