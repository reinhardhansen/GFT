// Run the GFT package test suite under webR (R compiled to WebAssembly).
// Development harness for sandboxed environments without native R.
// Usage: node run-webr.mjs /absolute/path/to/r-package-dir
import { WebR } from 'webr';

const root = process.argv[2];
if (!root) {
  console.error('usage: node run-webr.mjs <package-dir>');
  process.exit(1);
}

const webR = new WebR();
await webR.init();
await webR.FS.mkdir('/host');
await webR.FS.mount('NODEFS', { root }, '/host');

const shelter = await new webR.Shelter();
const res = await shelter.captureR(`
  options(warn = 1)
  setwd('/host')
  source('R/GFT.R')
  source('dev/test-shim.R')
  t0 <- proc.time()[3]
  source('tests/testthat/test-gft.R')
  shim_summary()
  cat(sprintf('elapsed: %.1fs\\n', proc.time()[3] - t0))
`);
for (const line of res.output) console.log(`${line.type}: ${line.data}`);
await shelter.purge();
await webR.close();
