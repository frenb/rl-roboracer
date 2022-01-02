import * as express from 'express';
import * as bodyParser from 'body-parser';
import * as path from 'path';
import * as fs from 'fs';
import signaling from './signaling';
import * as mongoDB from 'mongodb';
import * as dotenv from 'dotenv';
const ObjectID = require('mongodb').ObjectID;
const cors = require('cors');

import { log, LogLevel } from './log';
import * as morgan from 'morgan';

export const createServer = (config): express.Application => {
  const app: express.Application = express();
  var MongoClient = mongoDB.MongoClient;
  var url = "mongodb://root:example@mongo:27017";
  //var connectionstring = "mongodb+srv://root:example@mongo"
  var dbo;
  MongoClient.connect(url, function(err, db) {
    if (err) throw err;
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
  }

  // const signal = require('./signaling');
  app.use(bodyParser.urlencoded({ extended: true }));
  app.use(bodyParser.json());
  app.get('/protocol', cors(corsOptions), (req, res) => res.json({ useWebSocket: config.websocket }));
  app.use('/signaling', cors(corsOptions),signaling);
  app.use(express.static(path.join(__dirname, '/../public/stylesheets')));
  app.use(express.static(path.join(__dirname, '/../public/scripts')));
  app.use(express.static(path.join(__dirname, '/../bower_components')));
  app.use(express.static(path.join(__dirname, '/..')));
  app.use('/images', express.static(path.join(__dirname, '/../public/images')));
  app.get('/videoplayer', (req, res) => {
    const videPagePath: string = path.join(__dirname, '/../videoplayer.html');
    res.sendFile(videPagePath);
  });
  app.get('/leaderboard', (req, res) => {
    const lb: string = path.join(__dirname, '/../leaderboard.html');
    res.sendFile(lb);
  });
  app.get('/leaderboard_scores', (req,res) => {
    dbo.collection("leaderboard_scores").find({}).toArray(function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });
  app.post('/add_job', (req,res) => {
    console.log("add_job: " + JSON.stringify(req.body));
    dbo.collection("jobs").insertOne(req.body,function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });
  app.get('/get_jobs', (req,res) => {
    dbo.collection("jobs").find({}).toArray(function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });

  app.get('/jobs', (req, res) => {
    const lb: string = path.join(__dirname, '/../jobs.html');
    res.sendFile(lb);
  });

  app.post('/update_job_status', (req, res) => {
    console.log("update_job_status: " + JSON.stringify(req.body));
    var job = req.body;
    const myquery = { "_id": ObjectID(job["_id"]) };
    const newvalues = { "$set": { "status": job["status"] } };
    const options = { upsert: false };
    dbo.collection("jobs").updateOne(myquery,newvalues, options, function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });

  app.post('/delete_job', (req,res) => {
    var job = req.body;
    const myquery = { "_id": ObjectID(job["_id"]) };
    console.log(myquery);
    dbo.collection("jobs").deleteOne(myquery, function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });

  app.get('/models', (req, res) => {
    const lb: string = path.join(__dirname, '/../models.html');
    res.sendFile(lb);
  });
  app.get('/get_models', (req,res) => {
    dbo.collection("models").find({}).toArray(function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });
 
  app.get('/', (req, res) => {
    const indexPagePath: string = path.join(__dirname, '/../index.html');
    fs.access(indexPagePath, (err) => {
      if (err) {
        log(LogLevel.warn, `Can't find file ' ${indexPagePath}`);
        res.status(404).send(`Can't find file ${indexPagePath}`);
      } else {
        res.sendFile(indexPagePath);
      }
    });
  });
  return app;
};
