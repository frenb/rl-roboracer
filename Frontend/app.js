var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var grpc = require('@grpc/grpc-js');
var services = require('./proto/virtual_endpoint/proto/ros_service_grpc_pb');
var messages = require('./proto/virtual_endpoint/proto/ros_service_pb');

const SubscriberQueue = require('./subscriber');
const Publisher = require('./publisher');
var indexRouter = require('./routes/index');
var poseRouter = require('./routes/pose');
var usersRouter = require('./routes/users');
var sceneDataRouter = require('./routes/sceneData');
var planRouter = require('./routes/plan');
var moveRouter = require('./routes/move');
var resultRouter = require('./routes/result');

// Initialize ROS Node GRPC Connection
var client = new services.RosNodeClient('localhost:50051',grpc.credentials.createInsecure());
console.log("Connected to ROS node");

// Scene Data Subsriber - gets data objects in the scene.
var sceneDataRequest = new messages.SubscribeRequest();
sceneDataRequest.setTopic('scene_data');
sceneDataRequest.setMsgType('niryo_moveit/SceneData');
var sceneDataQueue = new SubscriberQueue("scene_data", client.subscribe(sceneDataRequest));

// Move Result Subscriber - gets information on the success / error of move action requests
var moveResultRequest = new messages.SubscribeRequest();
moveResultRequest.setTopic('move_action/result');
moveResultRequest.setMsgType('niryo_moveit/MoveActionResult');
var moveResultQueue = new SubscriberQueue("move_action/result", client.subscribe(moveResultRequest));

// Move Feedback Subscriber - gets information about the current progress of a move action request
var moveFeedbackRequest = new messages.SubscribeRequest();
moveFeedbackRequest.setTopic('move_action/result');
moveFeedbackRequest.setMsgType('niryo_moveit/MoveActionResult');
var moveFeedbackQueue = new SubscriberQueue("move_action/feedback", client.subscribe(moveFeedbackRequest));

// Publisher for sending move commands.
var moveGoalPublisher = new Publisher("move_action/goal", 'niryo_moveit/MoveActionGoal', client);

var ros_obj = {
  client: client,
  sceneDataQueue: sceneDataQueue,
  moveResultQueue: moveResultQueue,
  moveFeedbackQueue: moveFeedbackQueue,
  moveGoalPublisher: moveGoalPublisher
}

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'pug');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/users', usersRouter);
app.use('/pose', function (req, res, next) {
  req.ros = ros_obj
  next();
}, poseRouter);
app.use('/sceneData', function (req, res, next) {
  req.ros = ros_obj
  next();
}, sceneDataRouter);
app.use('/plan', function (req, res, next) {
  req.ros = ros_obj
  next();
}, planRouter);
app.use('/move', function (req, res, next) {
  req.ros = ros_obj
  next();
}, moveRouter);
app.use('/result', function (req, res, next) {
  req.ros = ros_obj
  next();
}, resultRouter);
app.get('/pickAndPlace',  function(req, res, next) {
  res.render('pickAndPlace', { title: 'Pick and Place' });
});
app.get('/editor',  function(req, res, next) {
  res.render('editor');
});
app.use('/', indexRouter);

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
