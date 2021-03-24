"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var express = require("express");
var bodyParser = require("body-parser");
var path = require("path");
var fs = require("fs");
var signaling_1 = require("./signaling");
var cors = require('cors');
var log_1 = require("./log");
var morgan = require("morgan");
exports.createServer = function (config) {
    var app = express();
    app.set('isPrivate', config.mode == "private");
    // logging http access
    if (config.logging != "none") {
        app.use(morgan(config.logging));
    }
    // TODO: REMOVE!
    app.use(cors());
    app.options('*', cors());
    var corsOptions = {
        origin: '*',
        exposedHeaders: 'Date'
    };
    // const signal = require('./signaling');
    app.use(bodyParser.urlencoded({ extended: true }));
    app.use(bodyParser.json());
    app.get('/protocol', cors(corsOptions), function (req, res) { return res.json({ useWebSocket: config.websocket }); });
    app.use('/signaling', cors(corsOptions), signaling_1.default);
    app.use(express.static(path.join(__dirname, '/../public/stylesheets')));
    app.use(express.static(path.join(__dirname, '/../public/scripts')));
    app.use(express.static(path.join(__dirname, '/../bower_components')));
    app.use(express.static(path.join(__dirname, '/..')));
    app.use('/images', express.static(path.join(__dirname, '/../public/images')));
    app.get('/videoplayer', function (req, res) {
        var videPagePath = path.join(__dirname, '/../videoplayer.html');
        res.sendFile(videPagePath);
    });
    app.get('/', function (req, res) {
        var indexPagePath = path.join(__dirname, '/../index.html');
        fs.access(indexPagePath, function (err) {
            if (err) {
                log_1.log(log_1.LogLevel.warn, "Can't find file ' " + indexPagePath);
                res.status(404).send("Can't find file " + indexPagePath);
            }
            else {
                res.sendFile(indexPagePath);
            }
        });
    });
    return app;
};
