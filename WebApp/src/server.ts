import * as express from 'express';
import * as bodyParser from 'body-parser';
import * as path from 'path';
import * as fs from 'fs';
import signaling from './signaling';
const cors = require('cors');

import { log, LogLevel } from './log';
import * as morgan from 'morgan';

export const createServer = (config): express.Application => {
  const app: express.Application = express();
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
