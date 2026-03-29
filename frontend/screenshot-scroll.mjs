import puppeteer from 'puppeteer';
import { mkdirSync } from 'fs';
import { join } from 'path';

const url = process.argv[2] || 'http://localhost:3000';
const yOffset = parseInt(process.argv[3] || '0');
const label = process.argv[4] || 'scroll';
const dir = './temporary screenshots';
mkdirSync(dir, { recursive: true });

const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto(url, { waitUntil: 'networkidle0', timeout: 15000 });

// Trigger all animations
await page.evaluate(async () => {
  const h = document.body.scrollHeight;
  for (let y = 0; y <= h; y += 400) {
    window.scrollTo(0, y);
    await new Promise(r => setTimeout(r, 80));
  }
});
await page.evaluate((y) => window.scrollTo(0, y), yOffset);
await new Promise(r => setTimeout(r, 500));

await page.screenshot({ path: join(dir, `${label}.png`) });
await browser.close();
console.log(`Done: ${label}.png`);
