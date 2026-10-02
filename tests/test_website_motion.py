"""Exercise the scroll path without requiring a browser or animation clock."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_overview_logo_scroll_path_and_reduced_motion():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for website checks")
    source = (
        Path(__file__).resolve().parents[1]
        / "website/src/components/Marketplace/overviewMotion.js"
    )
    subprocess.run(
        [node, "--input-type=module", "-", str(source)],
        input=r"""
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const {logoPosition, attachLogoJourney} = await import(
  'data:text/javascript;base64,' + readFileSync(process.argv[2]).toString('base64'));
const slot = (x, top, size) => ({x, y: top + size / 2, left: x - size / 2,
  top, bottom: top + size, width: size, height: size});
const geometry = {hero: slot(1000, 200, 440), height: 900, width: 1400, header: 64,
  chapters: [800, 1800, 2800].map((top, i) => ({top, bottom: top + 1000,
    padding: 140, slot: slot(i % 2 ? 300 : 1000, top + 180, 300)}))};
const at = focus => logoPosition(geometry, focus - 405, true);
assert.equal(logoPosition(geometry, 0, true).size, 440);
assert.equal(at(1300).x, 1000);
assert.equal(at(2300).x, 300);
assert.equal(at(3300).x, 1000);
assert.equal(at(1800).size, 140);
assert.equal(at(3800), null);
// Adjacent scroll positions must not jump when entering/leaving a bridge.
for (let focus = 950; focus < 3500; focus++) {
  const a = at(focus), b = at(focus + 1);
  assert.ok(Math.abs(a.x - b.x) < 10);
  assert.ok(Math.abs(a.size - b.size) < 3);
}
const mobile = {...geometry, width: 390, height: 844,
  hero: slot(195, 180, 240), chapters: geometry.chapters.map(c =>
    ({...c, slot: slot(195, c.top + 60, 180)}))};
for (let scroll = 0; scroll < 4000; scroll += 10) {
  const p = logoPosition(mobile, scroll, false);
  if (!p) continue;
  assert.ok(p.x - p.size / 2 >= 0 && p.x + p.size / 2 <= mobile.width);
  assert.ok(p.opacity >= 0 && p.opacity <= 1);
  assert.ok([mobile.hero, ...mobile.chapters.map(c => c.slot)].some(s =>
    p.y + scroll - p.size / 2 >= s.top && p.y + scroll + p.size / 2 <= s.bottom));
}
// Reduced motion retains static artwork; cleanup releases observers/listeners.
const media = {matches: true, addEventListener() {}, removeEventListener() {}};
const listeners = new Set();
globalThis.window = {matchMedia: () => media,
  addEventListener: name => listeners.add(name),
  removeEventListener: name => listeners.delete(name)};
let render, disconnected = false, cancelled = false;
globalThis.requestAnimationFrame = fn => {render = fn; return 1;};
globalThis.cancelAnimationFrame = () => {cancelled = true;};
globalThis.ResizeObserver = class {observe() {} disconnect() {disconnected = true;}};
const root = {dataset: {logoJourney: 'active'}}, image = {hidden: false};
const cleanup = attachLogoJourney(root, image);
render();
assert.equal(image.hidden, true);
assert.equal(root.dataset.logoJourney, undefined);
cleanup();
assert.equal(listeners.size, 0);
assert.ok(disconnected && cancelled);
""",
        text=True,
        check=True,
    )
