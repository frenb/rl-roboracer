var express = require('express');
var router = express.Router();
const fs = require('fs').promises;
var path = require('path');


router.post('/*', async function(req, res, next) {
    var file_path = path.join(__dirname, '../public', req.path);
    console.log("upating file: " + file_path);

    // Read workspaces file and validate the path is specified there in order
    // to avoid arbitarily clobbering file system.
    var allowed = new Set();
    var workspaces_json = await fs.readFile(path.join(__dirname, '../public/workspaces/workspaces.json'));
    var workspaces = JSON.parse(workspaces_json);
    Object.keys(workspaces).forEach(id => {
        workspaces[id].sources.forEach(source => allowed.add(source.get));
    });

    if (!allowed.has(req.path)) {
        res.sendStatus(403);
        return;
    }

    await fs.writeFile(file_path, req.body.txt);
    res.sendStatus(200);
  });

module.exports = router;