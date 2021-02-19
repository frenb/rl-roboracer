var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var grpc = require('@grpc/grpc-js');
var services = require('./proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('./proto/virtual_endpoint/proto/ros_service_pb');

var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');

var client = new services.RosNodeClient('localhost:50051',grpc.credentials.createInsecure());
// Subscribe to scene data topic and print streaming scene data to console.
var subscribeRequest = new messages.SubscribeRequest();
subscribeRequest.setTopic('scene_data');
subscribeRequest.setMsgType('niryo_moveit/SceneData');
var call = client.subscribe(subscribeRequest);
call.on('data', function(topicMessage) {
  // console.log('Received message: ' + topicMessage.toString());
});
call.on('end', function() {
  console.log('end');

});
call.on('error', function(e) {
  console.log('error: ' + e);

});
call.on('status', function(status) {
  console.log('status: ' + status);
});

// Call Pose Executor.
var serviceRequest = new messages.ServiceRequest();

serviceRequest.setServiceName('pose_executor');
serviceRequest.setServiceType('niryo_moveit/PoseExecutorService');

var posePoint = {x: -0.15, y: -0.21, z: 0.64}
var poseOrientation = {x: 0.0, y: 0.0, z: 0.0, w: 0.0};
var poseValue = {position: posePoint, orientation: poseOrientation};
serviceRequest.setRequest(JSON.stringify({pose: poseValue}));

client.callService(serviceRequest, function(err, result) {
  if (err) {
    console.log('error calling pose service: ' + err);
  } else {
    console.log('succeeded calling pose service');
  }
})

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'pug');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
