var express = require('express');
var router = express.Router();

router.get('/latest', function(req, res, next) {
    var ros = req.app.get('ros');
    let result = ros.moveResultQueue.peek() || {};
    res.set('Cache-Control', 'no-store');
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(result, null, 3));
});

module.exports = router;
