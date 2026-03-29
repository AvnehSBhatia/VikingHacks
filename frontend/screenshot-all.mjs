import puppeteer from 'puppeteer';
import { mkdirSync } from 'fs';
import { join } from 'path';

const dir = './temporary screenshots';
mkdirSync(dir, { recursive: true });

const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle0', timeout: 15000 });

// Slow scroll to trigger ALL animations including audit reveal
await page.evaluate(async () => {
  const h = document.body.scrollHeight;
  for (let y = 0; y <= h; y += 300) {
    window.scrollTo(0, y);
    await new Promise(r => setTimeout(r, 150));
  }
  // Wait for audit lines to finish revealing
  await new Promise(r => setTimeout(r, 2000));
});

// Take screenshots at key positions
const sections = [
  [0, 'v2-hero'],
  [900, 'v2-drift-hiw'],
  [1800, 'v2-demo'],
  [2600, 'v2-benchmark-top'],
  [3400, 'v2-benchmark-mid'],
  [4200, 'v2-benchmark-stats'],
  [4800, 'v2-audit'],
  [5400, 'v2-dev-footer'],
];

for (const [y, name] of sections) {
  await page.evaluate((scrollY) => window.scrollTo(0, scrollY), y);
  await new Promise(r => setTimeout(r, 200));
  await page.screenshot({ path: join(dir, `${name}.png`) });
}

await browser.close();
console.log('All screenshots saved.');
