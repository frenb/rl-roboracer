class API {
    constructor() {
        this.waiting = {};
        this.waiting_scene_data = null;
        this.waiting_sim_restarted = null;
        this.latest_scene_data = null;
        this.next_cmd_id = 0;
        // Listen to move action results.
        socket.emit('subscribe', 'move_action/result');
        socket.on('move_action/result', result => this.onMoveActionResult(result));
        // Listen to scene data.
        socket.emit('subscribe', 'scene_data');
        socket.on('scene_data', scene_data => this.onSceneData(scene_data));
        // Listen to simulation status updates (e.g. restarted )
        socket.emit('subscribe', 'sim_status');
        socket.on('scene_data', sim_status => this.onSimStatus(sim_status));

    }

    onSimStatus(sim_status) {
        if (sim_status.data.status == 1 /* restarted */) {
            if (this.waiting_sim_restarted) {
                this.waiting_sim_restarted.resolve();
            }
        }
    }

    onSceneData(scene_data) {
        if (this.waiting_scene_data) {
            this.waiting_scene_data.resolve(scene_data)
        }
        this.latest_scene_data = scene_data;
    }

    onMoveActionResult(result) {
        // Wake up waiters for cmd id.
        if (this.waiting[result.data.cmd_id]) {
            this.waiting[result.data.cmd_id](result)
            delete this.waiting[result.data.cmd_id];
        }
    }

    waitOnCmd(id) {
        return new Promise(resolve => this.waiting[id] = resolve);
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
        let id = this.next_cmd_id++;
        let waitPromise = this.waitOnCmd(id);
        action.cmd_id = id;
        await $.ajax({
            contentType: 'application/json',
            type: 'POST',
            url: '/move',
            data: JSON.stringify(action)

        });
        return waitPromise;
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