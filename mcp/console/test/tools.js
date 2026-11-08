// Offline check: the server starts and registers its tools. Needs no dashboard.
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const transport = new StdioClientTransport({
  command: process.execPath,
  args: [path.join(here, '..', 'src', 'index.js')],
  env: { ...process.env },
});
const client = new Client({ name: 'tools-check', version: '0.0.0' });
await client.connect(transport);
const { tools } = await client.listTools();
await client.close();

const names = new Set(tools.map((t) => t.name));
const required = ['list_jobs', 'create_job', 'get_weakness_map', 'get_difficult_sections',
  'suggest_start_gates', 'list_models', 'lint_reward_design', 'list_proposals'];
const missing = required.filter((n) => !names.has(n));
const undocumented = tools.filter((t) => !t.description).map((t) => t.name);

console.log(`${tools.length} tools registered`);
if (missing.length || undocumented.length) {
  if (missing.length) console.error(`missing: ${missing.join(', ')}`);
  if (undocumented.length) console.error(`no description: ${undocumented.join(', ')}`);
  process.exit(1);
}
