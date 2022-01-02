"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var express = require("express");
var bodyParser = require("body-parser");
var path = require("path");
var fs = require("fs");
var signaling_1 = require("./signaling");
var mongoDB = require("mongodb");
var ObjectID = require('mongodb').ObjectID;
var cors = require('cors');
var log_1 = require("./log");
var morgan = require("morgan");
exports.createServer = function (config) {
    var app = express();
    var MongoClient = mongoDB.MongoClient;
    var url = "mongodb://root:example@mongo:27017";
    //var connectionstring = "mongodb+srv://root:example@mongo"
    var dbo;
    MongoClient.connect(url, function (err, db) {
        if (err)
            throw err;
        dbo = db.db("local");
    });
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
    app.get('/leaderboard', function (req, res) {
        var lb = path.join(__dirname, '/../leaderboard.html');
        res.sendFile(lb);
    });
    app.get('/leaderboard_scores', function (req, res) {
        dbo.collection("leaderboard_scores").find({}).toArray(function (err, result) {
            if (err)
                throw err;
            console.log(result);
            res.json(result);
        });
    });
    app.post('/add_job', function (req, res) {
        console.log("add_job: " + JSON.stringify(req.body));
        dbo.collection("jobs").insertOne(req.body, function (err, result) {
            if (err)
                throw err;
            console.log(result);
            res.json(result);
        });
    });
    app.get('/get_jobs', function (req, res) {
        dbo.collection("jobs").find({}).toArray(function (err, result) {
            if (err)
                throw err;
            console.log(result);
            res.json(result);
        });
    });
    app.get('/jobs', function (req, res) {
        var lb = path.join(__dirname, '/../jobs.html');
        res.sendFile(lb);
    });
    app.post('/update_job_status', function (req, res) {
        console.log("update_job_status: " + JSON.stringify(req.body));
        var job = req.body;
        var myquery = { "_id": ObjectID(job["_id"]) };
        var newvalues = { "$set": { "status": job["status"] } };
        var options = { upsert: false };
        dbo.collection("jobs").updateOne(myquery, newvalues, options, function (err, result) {
            if (err)
                throw err;
            console.log(result);
            res.json(result);
        });
    });
    app.post('/delete_job', function (req, res) {
        var job = req.body;
        var myquery = { "_id": ObjectID(job["_id"]) };
        console.log(myquery);
        dbo.collection("jobs").deleteOne(myquery, function (err, result) {
            if (err)
                throw err;
            console.log(result);
            res.json(result);
        });
    });
    app.get('/models', function (req, res) {
        var lb = path.join(__dirname, '/../models.html');
        res.sendFile(lb);
    });
    app.get('/get_models', function (req, res) {
        dbo.collection("models").find({}).toArray(function (err, result) {
            if (err)
                throw err;
            console.log(result);
            res.json(result);
        });
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
