import { Server } from 'http';
import * as os from 'os';
import { createServer } from './server';
import { AddressInfo } from 'net';

export interface Options {
  port?: number;
  logging?: string;
}

export class Dashboard {
  public static run(argv: string[]): Dashboard {
    const program = require('commander');
    const readOptions = (): Options => {
      if (Array.isArray(argv)) {
        program
          .usage('[options]')
          .option('-p, --port <n>', 'Port to start the server on', process.env.PORT || 80)
          .option('-l, --logging <type>', 'http logging type: combined, dev, short, tiny, none (default dev)', process.env.LOGGING || 'dev')
          .parse(argv);
        return {
          port: program.port,
          logging: program.logging,
        };
      }
    };
    return new Dashboard(readOptions());
  }

  public server?: Server;
  public options: Options;

  constructor(options: Options) {
    this.options = options;
    const app = createServer(this.options);
    this.server = app.listen(this.options.port, () => {
      const { port } = this.server.address() as AddressInfo;
      for (const address of this.getIPAddress()) {
        console.log(`http://${address}:${port}`);
      }
    });
  }

  getIPAddress(): string[] {
    const interfaces = os.networkInterfaces();
    const addresses: string[] = [];
    for (const k in interfaces) {
      for (const k2 in interfaces[k]) {
        const address = interfaces[k][k2];
        if (address.family === 'IPv4') {
          addresses.push(address.address);
        }
      }
    }
    return addresses;
  }
}

Dashboard.run(process.argv);
