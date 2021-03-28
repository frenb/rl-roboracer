var net = require('net');
const {_, subscribers} = require('./subscriber');

// TODO: in production, we would want to replace this with reading from wherever docker logging driver
// sends this.
function createRosLogHandler(socket) {
    return function(data_unused) {
        socket.ros_log_client = new net.Socket();
        
        socket.ros_log_client.connect(60061, 'localhost', function() {
            console.log('connected to ros_log on behalf of ' + socket.id);
        });

        socket.on('disconnect', function() {
            socket.ros_log_client.destroy();
        });

        socket.ros_log_client.on('data', function(data) {
            socket.emit('ros_log', {data: data.toString()});
        });
    };
}

function createPythonWorkspaceHandler(socket) {
    return function(data_unused) {
        socket.workspace_socket = new net.Socket();
        socket.workspace_socket.connect(60062, 'localhost', function() {
            console.log('connected to python_workspace on behalf of ' + socket.id);
        });

        socket.on('disconnect', function() {
            socket.workspace_socket.destroy();
        });

        socket.on('python_run', function(src) {
            cmd = {cmd: 'run', src: src};
            console.log("python_run - " + JSON.stringify(cmd));
            socket.workspace_socket.write(JSON.stringify(cmd));
        });

        socket.on('python_stop', function(data_unused) {
            cmd = {cmd: 'stop'};
            console.log("python_stop - " + JSON.stringify(cmd));
            socket.workspace_socket.write(JSON.stringify(cmd));
        });

        socket.workspace_socket.on('data', function(data) {
            let escape = '!escape!';
            if (data.toString().includes(escape)) {
                // Camera Annotation Line
                let seq = data.toString();
                let start = seq.indexOf(escape) + escape.length;
                let end = seq.lastIndexOf(escape); 
                let annot = seq.substring(start, end);       
                socket.emit('python_annotation', {data: annot});
            } else {
                // Normal output line
                socket.emit('python_output', {data: data.toString()});
            }
        });
    }
}

function createTopicSubscriberHandler(socket) {
    return function(topic) {
        socket.data_cb = data => {
            socket.emit(topic, data)
        };
        subscribers[topic].addDataCallback(socket.data_cb);
        socket.on('disconnect', function() {
            subscribers[topic].removeDataCallback(socket.data_cb);
        });
    };
}

function addHandlers(io) {
    io.on('connection', function(socket) {
        socket.on('ros_log', createRosLogHandler(socket));
        socket.on('subscribe', createTopicSubscriberHandler(socket));
        socket.on('python_workspace', createPythonWorkspaceHandler(socket))
    });
}

module.exports = addHandlers