var grpc = require('@grpc/grpc-js');
var services = require('../proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('../proto/virtual_endpoint/proto/ros_service_pb');

var express = require('express');
var router = express.Router();

router.post('/', function(req, res, next) {
    console.log("POST move: " + JSON.stringify(req.body, null, 3));
    req.ros.moveGoalPublisher.publish(req.body);
    res.sendStatus(200);
  });

module.exports = router;