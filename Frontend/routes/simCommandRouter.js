var express = require('express');
var router = express.Router();

router.post('/', function(req, res, next) {
    var ros = req.app.get('ros');
    ros.simCommandPublisher.publish(req.body);
    res.sendStatus(200);
  });

module.exports = router;