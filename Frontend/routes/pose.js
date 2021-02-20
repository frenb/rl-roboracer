var grpc = require('@grpc/grpc-js');
var services = require('../proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('../proto/virtual_endpoint/proto/ros_service_pb');

var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  // Get or assign default pose values.
  var point_x = parseFloat(req.query.x) || -0.15;
  var point_y = parseFloat(req.query.y) || -0.21;
  var point_z = parseFloat(req.query.z) || 0.75;
  var orientation_x = parseFloat(req.query.o_x) || -0.5;
  var orientation_y = parseFloat(req.query.o_y) || -0.5;
  var orientation_z = parseFloat(req.query.o_z) || 0.5;
  var orientation_w = parseFloat(req.query.o_w) || -0.5;

  var posePoint = {x: point_x, y: point_y, z: point_z}
  var poseOrientation = {x: orientation_x, y: orientation_y, z: orientation_z, w: orientation_w};
  var poseValue = {position: posePoint, orientation: poseOrientation};

  // Send request to ROS
  var serviceRequest = new messages.ServiceRequest();
  serviceRequest.setServiceName('pose_executor');
  serviceRequest.setServiceType('niryo_moveit/PoseExecutorService');
  serviceRequest.setRequest(JSON.stringify({pose: poseValue}));

  req.ros.client.callService(serviceRequest, function(err, result) {
    if (err) {
      console.log('error calling pose service: ' + err);
    } else {
      console.log('succeeded calling pose service');
    }
  })

  res.render('pose', { title: JSON.stringify({pose: poseValue}) });
});

module.exports = router;
