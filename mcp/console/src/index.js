#!/usr/bin/env node
// MCP server for the rl-roboracer console: everything the dashboard tabs do,
// over the dashboard's HTTP API, plus publishing gym builds to GitHub releases.

import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { BASE_URL } from './api.js';
import * as jobs from './tools/jobs.js';
import * as models from './tools/models.js';
import * as designs from './tools/designs.js';
import * as gymsDemos from './tools/gymsDemos.js';
import * as madscientist from './tools/madscientist.js';
import * as releases from './tools/releases.js';
import * as weakness from './tools/weakness.js';
import * as unity from './tools/unity.js';

export function createServer() {
  const server = new McpServer(
    { name: 'roboracer-console', version: '0.1.0' },
    {
      instructions: `Operates the rl-roboracer training console at ${BASE_URL}: job queue, models, leaderboard and analysis, `
        + 'reward and experiment designs, gyms, demos, the MadScientist research agent, gym GitHub releases, and the '
        + 'Unity car clients on this host. Two robots share the queue: roboracer (car, Unity + SAC; the default) and '
        + 'microduck (mjlab + PPO, run by the duck worker). Jobs, models, gyms, reward and experiment designs and logs take '
        + 'a robot argument; the weakness map, demos, releases and Unity tools are roboracer only. '
        + 'Ids may be given in full or as the last 8 characters the dashboard shows; designs and gyms also by name.',
    },
  );
  for (const group of [jobs, weakness, models, designs, gymsDemos, madscientist, releases, unity]) group.register(server);
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await createServer().connect(new StdioServerTransport());
}
