var grpc = require('@grpc/grpc-js');
var services = require('../proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('../proto/virtual_endpoint/proto/ros_service_pb');

var express = require('express');
var router = express.Router();

router.get('/', function(req, res, next) {
    // Parse desired pose from request,
    let point_x = parseFloat(req.query.x) || -0.15;
    let point_y = parseFloat(req.query.y) || -0.21;
    let point_z = parseFloat(req.query.z) || 0.75;
    let orientation_x = parseFloat(req.query.o_x) || -0.5;
    let orientation_y = parseFloat(req.query.o_y) || -0.5;
    let orientation_z = parseFloat(req.query.o_z) || 0.5;
    let orientation_w = parseFloat(req.query.o_w) || -0.5;
  
    let posePoint = {x: point_x, y: point_y, z: point_z}
    let poseOrientation = {x: orientation_x, y: orientation_y, z: orientation_z, w: orientation_w};
    let poseValue = {position: posePoint, orientation: poseOrientation};
  
    // Build pose request.
    let posePlanRequest = {};
    sceneData = req.ros.sceneDataQueue.peek();
    posePlanRequest.joint_00 = sceneData.joint_00;
    posePlanRequest.joint_01 = sceneData.joint_01;
    posePlanRequest.joint_02 = sceneData.joint_02;
    posePlanRequest.joint_03 = sceneData.joint_03;
    posePlanRequest.joint_04 = sceneData.joint_04;
    posePlanRequest.joint_05 = sceneData.joint_05;
    posePlanRequest.pose = poseValue;

    let serviceRequest = new messages.ServiceRequest();
    serviceRequest.setServiceName('pose_planner');
    serviceRequest.setServiceType('niryo_moveit/PosePlanner');
    serviceRequest.setRequest(JSON.stringify(posePlanRequest));
  
    req.ros.client.callService(serviceRequest, function(err, result) {
      if (err) {
        console.log('error calling pose service: ' + err);
        res.status(500)
        res.render('error', {message: err});
      } else {
        res.setHeader('Content-Type', 'application/json');
        res.end(result.getResponse());
      }
    })
  });

  module.exports = router;