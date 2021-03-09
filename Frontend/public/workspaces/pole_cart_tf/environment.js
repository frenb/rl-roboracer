
const MAX_JOINT_ANGLE = 30;

class Environment {
    constructor(scene_data) {
        this.updateFromSceneData(scene_data);
    }
    
    updateFromSceneData(scene_data) {
        scene_data = scene_data.data
        // Save entire scene data for sending move commands with correct angles
        this.scene_data = scene_data;       
        // State variables.
        this.joint_00 =             scene_data.joint_00;
        this.hand_tangent_speed =   scene_data.pole_cart.hand_tangent_speed;
        this.pole_hand_angle =      scene_data.pole_cart.pole_hand_angle;
        this.pole_angular_speed =   scene_data.pole_cart.pole_angular_speed;
        this.upright =              scene_data.pole_cart.upright;
    }
    
    // Actions:
    //.   -1 = add -2 degree to joint_00
    //.    0 = nothing
    //.    1 = add 2 degree to joint_00
    async update(action) {
        let positions = {
            joint_00: this.scene_data.joint_00 + action * 2 * 3.14 / 180.0,
            joint_01: this.scene_data.joint_01,
            joint_02: this.scene_data.joint_02,
            joint_03: this.scene_data.joint_03,
            joint_04: this.scene_data.joint_04,
            joint_05: this.scene_data.joint_05
        }
        
        var position_cmd = {
                cmd_type: 4 /* positions */,
                positions: positions
            };
        
        await api.doMove({ cmd: position_cmd });
        api.latest_scene_data = null
        var new_scene_data = await api.getSceneData();
        this.updateFromSceneData(new_scene_data);
        
        return !this.upright || Math.abs(this.joint_00) * 180.0 / 3.14 > MAX_JOINT_ANGLE;
    }
    
    getStateTensor() {
        return tf.tensor2d([[
            this.discretizeJointAngle(this.joint_00),
            this.discretizeHandTangentSpeed(this.hand_tangent_speed),
            this.discretizePoleHandAngle(this.pole_hand_angle),
            this.discretizePoleAngularSpeed(this.pole_angular_speed)]]);
    }
    
    discretizeJointAngle(angle) {
        let angle_deg = angle * 180.0 / 3.14;
        return this.discretize(angle_deg, -MAX_JOINT_ANGLE, MAX_JOINT_ANGLE, 6);
    }
    
    discretizeHandTangentSpeed(speed) {
        return this.discretize(speed, -0.5, 0.5, 3);
    }
    
    discretizePoleHandAngle(angle) {
        return this.discretize(angle, -30, 30, 6);
    }
    
    discretizePoleAngularSpeed(speed) {
        return this.discretize(speed, -1.5, 1.5, 6);    
    }
    
    // TODO: map to actual variable domain instead of a bucket number.
    discretize(x, min, max, buckets) {
        x = Math.max(x, min);
        x = Math.min(x, max);
        let bucket = Math.round(((x - min) / (max - min) * buckets));
        return bucket;
    }
    
}