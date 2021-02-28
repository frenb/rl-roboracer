var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var grpc = require('@grpc/grpc-js');
var services = require('./proto/virtual_endpoint/proto/ros_service_grpc_pb');

const SubscriberQueue = require('./subscriber');
const Publisher = require('./publisher');

var indexRouter = require('./routes/index');
var sceneDataRouter = require('./routes/sceneData');
var planRouter = require('./routes/plan');
var moveRouter = require('./routes/move');
var resultRouter = require('./routes/result');

// Initialize ROS Node GRPC Connection
var client = new services.RosNodeClient('localhost:50051',grpc.credentials.createInsecure());
console.log("Connected to ROS node");

// Initialize ROS publishers and subscribers.
var sceneDataQueue = new SubscriberQueue("scene_data", 'niryo_moveit/SceneData', client);
var moveResultQueue = new SubscriberQueue("move_action/result", 'niryo_moveit/MoveActionResult', client);
var moveFeedbackQueue = new SubscriberQueue("move_action/feedback", 'niryo_moveit/MoveActionFeedback', client);
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

app.set('ros', ros_obj);
app.use('/sceneData', sceneDataRouter);
app.use('/plan', planRouter);
app.use('/move', moveRouter);
app.use('/result', resultRouter);
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
