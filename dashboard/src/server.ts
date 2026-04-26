import * as express from 'express';
import * as bodyParser from 'body-parser';
import * as path from 'path';
import * as fs from 'fs';
import * as mongoDB from 'mongodb';
const ObjectID = require('mongodb').ObjectID;
const cors = require('cors');
const { spawn } = require('child_process');
import { log, LogLevel } from './log';
import * as morgan from 'morgan';

const WebSocket = require('ws');


export const createServer = (config): express.Application => {
  const app: express.Application = express();
  const databaseName = process.env.DATABASE_NAME || "robotaxi";
  var jobsChanged = true;
  var modelsChanged = true;
  var leaderboardScoresChanged = true;
  var MongoClient = mongoDB.MongoClient;
  var url = process.env.MONGODB_URL || "mongodb://root:example@mongo:27017";
  var dbo;
  MongoClient.connect(url, function(err, db) {
    if (err) throw err;
    
    dbo = db.db(databaseName);
    // create a collection object for the collection based on collection_name
    // collectionNames.forEach(function(collectionName) {
    //   const collection = dbo.collection(collectionName);
    //   const changeStream = collection.watch();
  
    //   // listen for changes in the collection
    //   changeStream.on('change', function(change) {
    //     console.log(`Change detected in ${collectionName}:`, change);
    //     // handle the change event here
    //   });
    // });
    const jobs = dbo.collection("jobs");
    const models = dbo.collection("models");
    const leaderboardScores = dbo.collection("leaderboard_scores");
    
    const jobsChangeStream = jobs.watch();
    jobsChangeStream.on('change', (change) => {
      console.log('Change detected:', change);
      jobsChanged=true;
    });

    const modelsChangeStream = models.watch();
    modelsChangeStream.on('change', (change) => {
      console.log('Change detected:', change);
      modelsChanged=true;
    });

    const leaderboardScoresChangeStream = leaderboardScores.watch();
    leaderboardScoresChangeStream.on('change', (change) => {
      console.log('Change detected:', change);
      leaderboardScoresChanged=true;
    });

  });

  if (config.logging != "none") {
    app.use(morgan(config.logging));
  }

  app.use(cors());
  app.options('*', cors());

  app.use(bodyParser.urlencoded({ extended: true }));
  app.use(bodyParser.json());
  app.use(express.static(path.join(__dirname, '/..')));
  app.get('/leaderboard', (req, res) => {
    const lb: string = path.join(__dirname, '/../leaderboard.html');
    res.sendFile(lb);
  });
  app.get('/job_form', (req, res) => {
    const lb: string = path.join(__dirname, '/../job_form.html');
    res.sendFile(lb);
  });

  var needsUpdate = function (req, changed) {
    var force = (req.query.force == 'true');
    console.log(`force: ${force} req.query.force: ${req.query.force} jobsChanged: ${changed}`);
    return (changed || force);
  }


  app.get('/leaderboard_scores', (req,res) => {
    if(needsUpdate(req, leaderboardScoresChanged))
    {
      leaderboardScoresChanged=false;
      dbo.collection("leaderboard_scores").find({}).toArray(function(err, result) {
        if (err) throw err;
        //console.log(result);
        console.log(`${result.length} leaderboard scores retrieved`);
        res.json(result);
      });
      return;
    }
    
    console.log(`No leaderboard scores retrieved`);
    res.status(200).send('NO_CHANGES');
  
  });
  app.get('/logs', (req, res) => {
    const lb: string = path.join(__dirname, '/../logs.html');
    res.sendFile(lb);
    // const dockerLogs = spawn('tail', ['-f', '-n', '100', '/python_ws/src/robotaxi.out']);
    // dockerLogs.stdout.on('data', (data) => {
    //   res.write(`${data}\n`);
    // });
    // dockerLogs.stderr.on('data', (data) => {
    //   console.error(`stderr: ${data}`);
    // });
    // dockerLogs.on('close', (code) => {
    //   console.log(`child process exited with code ${code}`);
    //   res.end();
    // });

  })
  app.post('/add_job', (req,res) => {
    console.log("add_job: " + JSON.stringify(req.body));
    dbo.collection("jobs").insertOne(req.body,function(err, result) {
      if (err) throw err;
      //console.log(result);
      res.json(result)
    });
  });
  app.get('/get_jobs', (req,res) => {
    if (needsUpdate(req, jobsChanged)) {
      jobsChanged=false;
      dbo.collection("jobs").find({}).toArray(function(err, result) {
          if (err) throw err;
          //console.log(result);
          console.log(`${result.length} jobs retrieved`)
          res.json(result)
      });
      return;
    }
    
    console.log(`No jobs retrieved`);
    res.status(200).send('NO_CHANGES');
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
    if(needsUpdate(req, modelsChanged))
    {
      modelsChanged=false;
      dbo.collection("models").find({}).toArray(function(err, result) {
        if (err) throw err;
        //console.log(result);
        console.log(`${result.length} models retrieved`)
        res.json(result)
      });
      return;
    }
    console.log(`No models retrieved`);
    res.status(200).send('NO_CHANGES');
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
// create websocket server for log tailing (used by logs.html)
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  const dockerLogs = spawn('tail', ['-f', '-n', '100', '/python_ws/src/robotaxi.out']);

  dockerLogs.stdout.on('data', (data) => {
    ws.send(data.toString());
  });

  dockerLogs.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  dockerLogs.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });

  ws.on('close', () => {
    dockerLogs.kill();
  });
});

