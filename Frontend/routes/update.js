var express = require('express');
var router = express.Router();
const fs = require('fs');
var path = require('path');

// TODO: unsafe. demo only.
router.post('/*', function(req, res, next) {
    let file_path = path.join(__dirname, '../public', req.path);
    console.log("upating file: " + file_path);
    fs.writeFile(file_path, req.body.txt, (err) => {
        if (!err) {
            res.sendStatus(200);
        } else {
            console.log(err);
            res.sendStatus(500);
        }
    });
  });

module.exports = router;