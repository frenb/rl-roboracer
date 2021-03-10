const WAIT_COMMAND_DEADLINE_MS = 200;

class API {
    constructor() {
        this.waiting = {};
        this.waiting_scene_data = null;
        this.waiting_sim_restarted = null;
        this.latest_scene_data = null;
        this.next_cmd_id = 1;
        // Listen to move action results.
        socket.emit('subscribe', 'move_action/result');
        socket.on('move_action/result', result => this.onMoveActionResult(result));
        // Listen to scene data.
        socket.emit('subscribe', 'scene_data');
        socket.on('scene_data', scene_data => this.onSceneData(scene_data));
        // Listen to simulation status updates (e.g. restarted )
        socket.emit('subscribe', 'sim_status');
        socket.on('sim_status', sim_status => this.onSimStatus(sim_status));

       // this.rtt_history = new Array();

    }

    //updateRttHistory(rtt) {
    //    this.rtt_history.push(rtt);
    //    if (this.rtt_history.length > 20) {
    //        this.rtt_history.shift();
    //    }

    //    let sum = 0;
    //    this.rtt_history.forEach(val => sum += val);

    //    console.log("running rtt: " + sum / this.rtt_history.length);
    //}

    onSimStatus(sim_status) {
        if (sim_status.data.status == 1 /* restarted */) {
            if (this.waiting_sim_restarted) {
                this.waiting_sim_restarted();
            }
        }
    }

    onSceneData(scene_data) {
        if (this.waiting_scene_data) {
            this.waiting_scene_data(scene_data)
        }
        if (this.waiting[scene_data.data.last_executed_cmd_id]) {
            this.waiting[scene_data.data.last_executed_cmd_id][1] = true;
            this.maybeReleaseCmdWaiter(scene_data.data.last_executed_cmd_id);
        }
        this.latest_scene_data = scene_data;
    }

    onMoveActionResult(result) {
        // Wake up waiters for cmd id.
        if (this.waiting[result.data.cmd_id]) {
            this.waiting[result.data.cmd_id][0] = true;
            this.maybeReleaseCmdWaiter(result.data.cmd_id);
        }
    }
    
    maybeReleaseCmdWaiter(cmd_id) {
        if (this.waiting[cmd_id][0] && this.waiting[cmd_id][1]) {
            this.waiting[cmd_id][2]()
            delete this.waiting[cmd_id];
        }
    }

    waitOnCmd(id) {
        // Will wait until a result for the command is returned and we have seen a scene_data with command's results.
        return new Promise(resolve => {
            this.waiting[id] = [true /* returned result */, true /* reflected in scene data */ , resolve];
            setTimeout(() => {delete this.waiting[id]; resolve()}, WAIT_COMMAND_DEADLINE_MS);
        });
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async getSceneData() {
        if (this.latest_scene_data) {
            return this.latest_scene_data;
        }
        // Not populated yet.
        return new Promise(resolve => this.waiting_scene_data = resolve);
    }

    async getPlan(pose) {
        return $.ajax({
            dataType: "json",
            contentType: 'application/json',
            type: 'POST',
            url: '/plan',
            data: JSON.stringify(pose)

        }).promise();
    }

    async doMove(action) {
        //let start = Date.now();
        let id = this.next_cmd_id++;
        let waitPromise = this.waitOnCmd(id);
        action.cmd_id = id;
        await $.ajax({
            contentType: 'application/json',
            type: 'POST',
            url: '/move',
            data: JSON.stringify(action)

        });
        await waitPromise;
        //this.updateRttHistory(Date.now() - start);
    }

    async doSimCommand(command) {
        return $.ajax({
            contentType: 'application/json',
            type: 'POST',
            url: '/simCommand',
            data: JSON.stringify(command)

        }).promise();
    }

    async doReset() {
        let waitPromise = new Promise(resolve => this.waiting_sim_restarted = resolve);
        let cmd = {cmd: 0 /* reset */};
        await this.doSimCommand(cmd);
        return waitPromise;
    }

    async doTrajectory(trajectory) {
        let move_command = {
            cmd_type: 1 /* trajectory */,
            trajectory: trajectory
        };
        return this.doMove({cmd: move_command});
    }

    async doOpenGripper() {
        let move_command = {
            cmd_type: 2 /* open */
        };
        return this.doMove({cmd: move_command});
    }

    async doCloseGripper() {
        let move_command = {
            cmd_type: 3 /* close */
        };
        return this.doMove({cmd: move_command});  
    }
}


var api = new API();