var grpc = require('@grpc/grpc-js');
var services = require('../proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('../proto/virtual_endpoint/proto/ros_service_pb');

var express = require('express');
var router = express.Router();

router.post('/', function(req, res, next) {
    // default value
    let defaultPose = {
      position: {
        x: -0.15,
        y: -0.21,
        z: 0.75,
      },
      orientation: {
        x: -0.5,
        y: -0.5,
        z: 0.5,
        w: -0.5
      }
    };
    let poseValue = req.body || defaultPose;
  
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