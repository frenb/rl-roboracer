var grpc = require('@grpc/grpc-js');
var services = require('../proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('../proto/virtual_endpoint/proto/ros_service_pb');

var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  // Get or assign default pose values. 
  var point_x = req.params.x || -0.15;
  var point_y = req.params.y || -0.21;
  var point_z = req.params.z || 0.64;
  var orientation_x = req.params.o_x || 0.0;
  var orientation_y = req.params.o_y || 0.0;
  var orientation_z = req.params.o_z || 0.0;
  var orientation_w = req.params.o_w || 0.0;

  var posePoint = {x: point_x, y: point_y, z: point_z}
  var poseOrientation = {x: orientation_x, y: orientation_y, z: orientation_z, w: orientation_w};
  var poseValue = {position: posePoint, orientation: poseOrientation};

  // Send request to ROS
  var serviceRequest = new messages.ServiceRequest();
  serviceRequest.setServiceName('pose_executor');
  serviceRequest.setServiceType('niryo_moveit/PoseExecutorService');
  serviceRequest.setRequest(JSON.stringify({pose: poseValue}));

  req.rpc.client.callService(serviceRequest, function(err, result) {
    if (err) {
      console.log('error calling pose service: ' + err);
    } else {
      console.log('succeeded calling pose service');
    }
  })

  res.render('pose', { title: JSON.stringify({pose: poseValue}) });
});

module.exports = router;
