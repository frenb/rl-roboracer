var api = {
    sleep: function (ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    getSceneData: async function () {
        return $.ajax({
            dataType: "json",
            url: '/sceneData/latest',
        }).promise();
    },

    getPlan: async function (pose) {
        return $.ajax({
            dataType: "json",
            contentType: 'application/json',
            type: 'POST',
            url: '/plan',
            data: JSON.stringify(pose)

        }).promise();
    },

    doMove: async function (action) {
        return $.ajax({
            contentType: 'application/json',
            type: 'POST',
            url: '/move',
            data: JSON.stringify(action)

        }).promise();
    },

    doSimCommand: async function(command) {
        return $.ajax({
            contentType: 'application/json',
            type: 'POST',
            url: '/simCommand',
            data: JSON.stringify(command)

        }).promise();
    },

    doReset: async function () {
        let cmd = {cmd: 0 /* reset */};
        return api.doSimCommand(cmd);
    },

    doTrajectory: async function (trajectory) {
        move_command = {
            cmd_type: 1 /* trajectory */,
            trajectory: trajectory
        };
        return api.doMove({cmd: move_command});
    },

    doOpenGripper: async function () {
        move_command = {
            cmd_type: 2 /* open */
        };
        return api.doMove({cmd: move_command});
    },

    doCloseGripper: async function () {
        move_command = {
            cmd_type: 3 /* close */
        };
        return api.doMove({cmd: move_command});
    },

    getLatestResult: async function () {
        return $.ajax({
            dataType: "json",
            contentType: 'application/json',
            type: 'GET',
            url: '/result/latest'

        }).promise();
    },

    lastResult: {},

    clearResult: function () {
        api.lastResult = {};
    },

    waitNextResult: async function () {
        while(true) {
            last_ts = api.lastResult.timestamp || 0;
            current_result = await api.getLatestResult();
            current_ts = current_result.timestamp || 0;
            if (current_ts > last_ts) {
                api.lastResult = current_result;
                return current_result;
            }
            await api.sleep(1000);
        }
    },
}