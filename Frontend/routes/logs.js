var express = require('express');
var router = express.Router();

var i = 0;
var writeSocket = function(io) {
    io.emit('console_data', i++);
    setTimeout(writeSocket, 1000, io);
}

router.get('/', function(req, res, next) {
    setTimeout(writeSocket, 100, req.app.get('socketio'));
    res.render('logs');
});

module.exports = router;