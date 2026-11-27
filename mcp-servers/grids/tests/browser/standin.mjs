// Headless run of the grid editor in the stand-in host: see run_standin.py.
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [url, out, width, height] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: Number(width), height: Number(height) } });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(url);
const app = page.frameLocator('#app');
const canvas = app.locator('#field');
await app.locator('#edits').waitFor({ state: 'attached' });
await page.waitForFunction(() => document.getElementById('app').contentWindow.gridEditor?.shown);
const box = await canvas.boundingBox();
const at = (fx, fy) => [box.x + box.width * fx, box.y + box.height * fy];
// Drag a box over the middle-right of the basin and set it to land.
await page.mouse.move(...at(0.62, 0.35));
await page.mouse.down();
await page.mouse.move(...at(0.78, 0.55), { steps: 6 });
await page.mouse.up();
await app.locator('#op').selectOption('set');
await app.locator('#value').fill('0');
await app.locator('#note').fill('close the shelf');
await app.locator('#apply').click();
await app.locator('#edits li').first().waitFor();
await page.screenshot({ path: `${out}/editor-box-${width}x${height}.png` });
// Pick one cell, set it, then undo it.
await app.locator('#mode').selectOption('points');
await app.locator('#clear-selection').click();
await page.mouse.click(...at(0.3, 0.5));
await app.locator('#value').fill('5');
await app.locator('#apply').click();
await app.locator('#edits li').nth(1).waitFor();
await app.locator('#undo').click();
await page.waitForFunction(() => document.getElementById('app').contentWindow.gridEditor.edits.length === 1);
// Remove enclosed lakes across the whole grid.
await app.locator('#clear-selection').click();
await app.locator('#op').selectOption('fill_enclosed');
await app.locator('#wet').selectOption('positive');
await app.locator('#land').fill('0');
await app.locator('#apply').click();
await page.waitForFunction(() => document.getElementById('app').contentWindow.gridEditor.edits.length === 2);
await page.screenshot({ path: `${out}/editor-${width}x${height}.png` });
const status = await app.locator('#status').textContent();
const edits = await app.locator('#edits li').allTextContents();
console.log(JSON.stringify({ status, edits, errors }));
await browser.close();
if (errors.length) process.exit(1);
