class API {
    constructor() {
        this.waiting_sim_restarted = null;
        // Listen to simulation status updates (e.g. restarted )
        socket.emit('subscribe', 'sim_status');
        socket.on('sim_status', sim_status => this.onSimStatus(sim_status));
    }

    onSimStatus(sim_status) {
        if (sim_status.data.status == 1 /* restarted */) {
            if (this.waiting_sim_restarted) {
                this.waiting_sim_restarted();
            }
        }
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
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
}


var api = new API();