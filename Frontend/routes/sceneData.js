var express = require('express');
var router = express.Router();

router.get('/', function(req, res, next) {
  res.render('sceneData', { title: 'SceneData' });
});

router.get('/latest', function(req, res, next) {
    var ros = req.app.get('ros');
    var data = ros.sceneDataQueue.peek().data;
    res.set('Cache-Control', 'no-store');
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(data, null, 3));
});

module.exports = router;
