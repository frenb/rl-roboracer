var grpc = require('@grpc/grpc-js');
var services = require('../proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('../proto/virtual_endpoint/proto/ros_service_pb');

var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('sceneData', { title: 'SceneData' });
});

router.get('/latest', function(req, res, next) {
    var data = req.ros.sceneData.latest;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(data, null, 3));
});

module.exports = router;
