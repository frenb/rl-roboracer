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
    });
}

module.exports = addHandlers