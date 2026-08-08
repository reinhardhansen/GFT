// Run the serialized cross-language check under webR.
// Usage: node run-webr-serialized.mjs /absolute/path/to/AlgoGFT
import { WebR } from 'webr';

const root = process.argv[2];
if (!root) {
  console.error('usage: node run-webr-serialized.mjs <AlgoGFT-dir>');
  process.exit(1);
}

const webR = new WebR();
await webR.init();
await webR.FS.mkdir('/host');
await webR.FS.mount('NODEFS', { root }, '/host');

const methods = (process.argv[3] || 'fp,broyden,newton,fpn')
  .split(',').map(m => `'${m.trim()}'`).join(',');

const shelter = await new webR.Shelter();
const res = await shelter.captureR(`
  source('/host/r/R/GFT.R')
  serialized_dir <- '/host/julia/serialized'
  methods_filter <- c(${methods})
  t0 <- proc.time()[3]
  source('/host/r/dev/check_serialized.R')
  cat(sprintf('elapsed: %.1fs\\n', proc.time()[3] - t0))
`);
for (const line of res.output) console.log(`${line.type}: ${line.data}`);
await shelter.purge();
await webR.close();
