// Real textlayer engine + browser + ZIP contents; no translation provider or model download.
// Run: node tests/e2e/markdown-zip.e2e.mjs
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const PYTHON = path.join(ROOT, 'backend/.venv/bin/python');
const DATA = mkdtempSync(path.join(tmpdir(), 'markdown-zip-e2e-'));
const PDF = path.join(DATA, 'photos.pdf');
const SHOTS = path.join(ROOT, 'frontend/tests/e2e/shots');
let backend;
let browser;
let logs = '';

function python(source, args = []) {
  const result = spawnSync(PYTHON, ['-c', source, ...args], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr || result.error?.message);
  return result.stdout;
}

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function ready(base) {
  for (let i = 0; i < 150; i++) {
    assert.equal(backend.exitCode, null, logs.slice(-4000));
    try {
      const response = await fetch(`${base}/api/health`);
      if (response.ok) return response.json();
    } catch { /* startup */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Backend did not start: ${logs.slice(-4000)}`);
}

function inspectZip(file) {
  return JSON.parse(python(`
import json, re, sys, zipfile
from PIL import Image
with zipfile.ZipFile(sys.argv[1]) as z:
    names = z.namelist()
    assert z.testzip() is None
    assert 'result.md' in names
    assert [n for n in names if n.startswith('pages/')] == ['pages/page_0001.png', 'pages/page_0002.png']
    assert 'meta.json' not in names and 'source.pdf' not in names
    assert not any(n.startswith('result.') and n != 'result.md' for n in names)
    md = z.read('result.md').decode('utf-8')
    assert 'Original language page 1' in md and 'Original language page 2' in md
    refs = re.findall(r'!\\[[^\\]]*\\]\\((images/[^)]+)\\)', md)
    assert refs, md
    for ref in refs:
        assert ref in names, ref
        with z.open(ref) as image:
            Image.open(image).verify()
    print(json.dumps({'pages': 2, 'figures': len(refs)}))
`, [file]));
}

try {
  python(`
import io, sys
import pymupdf as fitz
from PIL import Image, ImageDraw
doc = fitz.open()
image = Image.new('RGB', (200, 100), 'white')
ImageDraw.Draw(image).rectangle((20, 20, 180, 80), fill='blue')
buf = io.BytesIO()
image.save(buf, format='PNG')
for i in range(2):
    page = doc.new_page(width=400, height=500)
    page.insert_text((30, 50), f'Original language page {i + 1}')
    page.insert_text((30, 80), 'This document is converted without a translation service.')
    if i == 0:
        page.insert_image(fitz.Rect(30, 120, 230, 220), stream=buf.getvalue())
doc.save(sys.argv[1])
doc.close()
`, [PDF]);
  const base = `http://127.0.0.1:${await freePort()}`;
  backend = spawn(PYTHON, ['-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend',
    '--host', '127.0.0.1', '--port', new URL(base).port], {
    cwd: ROOT,
    env: {
      ...process.env, DISABLE_DOTENV: '1', DATA_DIR: path.join(DATA, 'data'),
      OCR_ENGINE: 'textlayer', OCR_DEVICE: 'cpu', PRELOAD_MODEL: '0', NATIVE_TEXT_THRESHOLD: '1',
      OPENAI_API_KEY: '', LLM_OPENAI_API_KEY: '', LLM_LOCAL_OPENAI_API_KEY: '',
      OPENAI_BASE_URL: 'https://api.openai.com/v1', LLM_OPENAI_BASE_URL: 'https://api.openai.com/v1',
      LLM_LOCAL_OPENAI_BASE_URL: '', LLM_PROVIDER: 'openai-responses',
      ALLOWED_HOSTS: 'localhost,127.0.0.1', PDF_WORKER_MODE: 'process',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  backend.stdout.on('data', (chunk) => { logs += chunk; });
  backend.stderr.on('data', (chunk) => { logs += chunk; });
  const health = await ready(base);
  assert.equal(health.translate_available, false);
  assert.equal(health.engine, 'textlayer');
  browser = await chromium.launch({ executablePath: process.env.E2E_CHROMIUM_PATH || undefined });

  for (const polling of [false, true]) {
    const context = await browser.newContext({ viewport: { width: polling ? 390 : 1280, height: 900 } });
    if (polling) await context.addInitScript(() => { window.EventSource = undefined; });
    const page = await context.newPage();
    const requests = [];
    const errors = [];
    page.on('request', (request) => { requests.push(new URL(request.url()).pathname); });
    page.on('pageerror', (error) => { errors.push(error.message); });
    await page.goto(base);
    await page.locator('input[name="workflow"][value="markdown_zip"]').check();
    await page.setInputFiles('#file-input', PDF);
    const uploaded = page.waitForResponse((response) => response.request().method() === 'POST'
      && new URL(response.url()).pathname === '/api/jobs');
    await page.click('#upload-btn');
    const response = await uploaded;
    assert.equal(response.status(), 202);
    const { job_id: id } = await response.json();
    assert.equal((await (await fetch(`${base}/api/jobs/${id}`)).json()).workflow, 'markdown_zip');
    await page.waitForSelector('#result-section:not([hidden])', { timeout: 60000 });
    await page.waitForSelector('#preview-body img', { timeout: 10000 });
    assert.equal(await page.locator('#viewer-open').isVisible(), false);
    assert.equal(await page.locator('[data-tab="qa"]').isVisible(), false);
    assert.match(await page.locator('#dl-zip').getAttribute('href'), /scope=original/);
    assert.doesNotMatch(await page.locator('#dl-md').getAttribute('href'), /lang=/);
    const downloadEvent = page.waitForEvent('download');
    await page.click('#dl-zip');
    const download = await downloadEvent;
    assert.equal(download.suggestedFilename(), 'photos.markdown.zip');
    const zip = path.join(DATA, `result-${polling}.zip`);
    await download.saveAs(zip);
    const contents = inspectZip(zip);
    await page.reload();
    await page.waitForSelector('#markdown-zip-help:not([hidden])');
    assert.equal(await page.locator('input[name="workflow"][value="markdown_zip"]').isChecked(), true);
    assert.equal(new URL(page.url()).hash, `#${id}`);
    assert.equal(requests.some((url) => /\/translate(?:\/|$)|\/qa$|\/providers$|\/viewer-manifest$/.test(url)), false,
      `Unexpected reader/LLM request: ${requests.join(', ')}`);
    assert.deepEqual(errors, []);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    mkdirSync(SHOTS, { recursive: true });
    await page.screenshot({ path: path.join(SHOTS, `markdown-zip-${polling ? 'mobile' : 'desktop'}.png`), fullPage: true });
    console.log(`PASS ${polling ? 'mobile polling' : 'desktop SSE'}: original ZIP, ${contents.pages} pages, ${contents.figures} figure(s), reload, no translation`);
    if (!polling) {
      await page.locator('input[name="workflow"][value="reader"]').check();
      await page.setInputFiles('#file-input', PDF);
      await page.click('#upload-btn');
      await page.waitForSelector('#viewer-open:not([hidden])', { timeout: 60000 });
      assert.equal(await page.locator('[data-tab="reader"]').isVisible(), true);
      assert.doesNotMatch(await page.locator('#dl-zip').getAttribute('href'), /scope=original/);
      await page.goto(`${base}/#${id}`);
      await page.waitForSelector('#markdown-zip-help:not([hidden])');
      assert.equal(await page.locator('#viewer-open').isVisible(), false);
      assert.match(await page.locator('#dl-zip').getAttribute('href'), /scope=original/);
      assert.deepEqual(errors, []);
      console.log('PASS reader service and Markdown history restore after service switch');
    }
    await context.close();
  }
} catch (error) {
  console.error(error);
  console.error(logs.slice(-6000));
  throw error;
} finally {
  await browser?.close();
  if (backend && backend.exitCode === null && backend.signalCode === null) {
    backend.kill('SIGTERM');
    await Promise.race([
      new Promise((resolve) => backend.once('exit', resolve)),
      new Promise((resolve) => setTimeout(resolve, 5000)),
    ]);
    if (backend.exitCode === null && backend.signalCode === null) {
      backend.kill('SIGKILL');
      await new Promise((resolve) => backend.once('exit', resolve));
    }
  }
  rmSync(DATA, { recursive: true, force: true });
}
