class API {
    constructor() {
        this.waiting = {};
        this.next_cmd_id = 0;
        // Listen to move action results.
        socket.emit('subscribe', 'move_action/result');
        socket.on('move_action/result', result => this.onMoveActionResult(result));
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
        return $.ajax({
            dataType: "json",
            url: '/sceneData/latest',
        }).promise();
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
        let cmd = {cmd: 0 /* reset */};
        return this.doSimCommand(cmd);
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