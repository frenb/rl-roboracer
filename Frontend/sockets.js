var net = require('net');

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

function addHandlers(io) {
    io.on('connection', function(socket) {
        socket.on('ros_log', createRosLogHandler(socket));
    });
}

module.exports = addHandlers