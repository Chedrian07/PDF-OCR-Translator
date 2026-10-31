// Explicitly opt-in: real Unlimited-OCR-2 and real configured translation provider.
// npm run test:e2e-live (E2E_OCR_DEVICE=cuda, OCR_SIDECAR_URL must reach the service).
// Uses public synthetic en-mixed.pdf by default. Never substitutes a mock provider.
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const PYTHON = process.env.E2E_PYTHON || path.join(ROOT, 'backend/.venv/bin/python');
const PDF = process.env.E2E_PDF || path.join(ROOT, 'benchmark_docs/unlimited2/en-mixed.pdf');
const TIMEOUT = Number(process.env.E2E_TIMEOUT_S || 900) * 1000;
mkdirSync(path.join(ROOT, 'tmp'), { recursive: true });
const OUT = mkdtempSync(path.join(ROOT, 'tmp/live-e2e-'));
const report = { status: 'running', checks: [], downloads: [], mock_provider: false };
let stage = 'startup';
let backend;
let browser;
let code = 1;

function passed(name) { report.checks.push(name); console.log(`PASS ${name}`); }
function blocked(status) { const error = new Error(status); error.liveStatus = status; throw error; }
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function ready(base) {
  const deadline = Date.now() + TIMEOUT;
  while (Date.now() < deadline) {
    assert(backend.exitCode === null && !backend.spawnFailed, 'Isolated backend exited');
    try {
      const response = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(10000) });
      if (response.ok) {
        const health = await response.json();
        assert(health.engine === 'unlimited2', 'Real unlimited2 engine required');
        if (health.model_loaded) return health;
      }
    } catch { /* cold startup; bounded by deadline */ }
    await pause(500);
  }
  throw new Error('Real backend readiness timed out');
}

function inspectArtifacts(translated) {
  const result = spawnSync(PYTHON, ['-c', `
import json, pathlib, re, sys, zipfile
from html.parser import HTMLParser
import pymupdf
root = pathlib.Path(sys.argv[1]); translated = sys.argv[2] == '1'
class Structure(HTMLParser):
    def __init__(self):
        super().__init__(); self.math = self.images = self.tables = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.images += tag == 'img'
        self.tables += tag == 'table'
        self.math += tag == 'span' and any(value in attrs.get('class', '').split() for value in ('math-inline', 'math-display'))
original = Structure(); original.feed((root / 'original.html').read_text())
with zipfile.ZipFile(root / ('translated.zip' if translated else 'original.zip')) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert 'result.md' in names
    if translated:
        assert 'result.ko.md' in names
        assert re.search('[가-힣]', archive.read('result.ko.md').decode())
    md = archive.read('result.md').decode()
    for ref in re.findall(r'!\\[[^\\]]*\\]\\((images/[^)]+)\\)', md):
        assert ref in names
if translated:
    html = (root / 'translated.html').read_text()
    assert '/api/jobs/' not in html and 'data:image/png;base64,' in html
    rendered = Structure(); rendered.feed(html)
    assert rendered.math == original.math
    assert rendered.images == original.images
    assert rendered.tables == original.tables
    with pymupdf.open(sys.argv[3]) as source:
        source_pages = len(source)
    with pymupdf.open(root / 'translated.pdf') as pdf:
        assert len(pdf) == source_pages and not pdf.is_encrypted
        assert any(re.search('[가-힣]', page.get_text()) for page in pdf)
print(json.dumps({'valid': True}))
`, OUT, translated ? '1' : '0', PDF], { encoding: 'utf8', timeout: 60000 });
  assert(result.status === 0, 'Downloaded ZIP/PDF contents failed validation');
}

try {
  assert(Number.isFinite(TIMEOUT) && TIMEOUT > 0, 'E2E_TIMEOUT_S must be positive');
  assert(readFileSync(PDF).subarray(0, 5).toString() === '%PDF-', 'Public synthetic PDF fixture required');
  const base = `http://127.0.0.1:${await freePort()}`;
  backend = spawn(PYTHON, ['-m', 'uvicorn', 'e2e_live_app:create_live_app', '--factory',
    '--app-dir', path.join(ROOT, 'backend'), '--host', '127.0.0.1', '--port', new URL(base).port,
    '--timeout-graceful-shutdown', '5'], {
    cwd: ROOT, env: { ...process.env, E2E_DATA_DIR: path.join(OUT, 'data') }, stdio: 'ignore',
  });
  backend.on('error', () => { backend.spawnFailed = true; });
  await ready(base);
  passed('real-backend-ready');
  browser = await chromium.launch({ executablePath: process.env.E2E_CHROMIUM_PATH || undefined });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  context.setDefaultTimeout(30000);
  const page = await context.newPage();
  const browserErrors = [];
  page.on('pageerror', () => { browserErrors.push('pageerror'); });
  await page.addInitScript(() => {
    window.__liveOcrEvents = { connected: 0, token: 0, replay: 0, done: 0 };
    const NativeEventSource = window.EventSource;
    window.EventSource = class extends NativeEventSource {
      constructor(url, options) {
        super(url, options);
        if (/\/api\/jobs\/[^/]+\/events$/.test(new URL(url, location.href).pathname)) {
          this.addEventListener('open', () => { window.__liveOcrEvents.connected += 1; });
          for (const name of ['token', 'replay', 'done']) {
            this.addEventListener(name, () => { window.__liveOcrEvents[name] += 1; });
          }
        }
      }
    };
  });
  stage = 'browser-upload-and-ocr';
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('input[name="workflow"][value="reader"]').check();
  await page.setInputFiles('#file-input', PDF);
  const upload = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === '/api/jobs');
  await page.click('#upload-btn');
  const response = await upload;
  assert(response.status() === 202, 'Upload must be accepted');
  const { job_id: id } = await response.json();
  report.job_id = id;
  passed('browser-upload');
  await page.waitForSelector('#result-section:not([hidden])', { timeout: TIMEOUT });
  const job = await (await context.request.get(`${base}/api/jobs/${id}`)).json();
  assert(job.status === 'done', 'OCR must finish successfully');
  const events = await page.evaluate(() => window.__liveOcrEvents);
  assert(events.connected > 0 && events.token + events.replay > 0 && events.done > 0, 'Real OCR stream must deliver text and completion');
  report.ocr_stream = events;
  passed('ocr-stream-and-result');
  await page.click('button[data-tab="preview"]');
  await page.waitForFunction(() => (document.getElementById('preview-body')?.innerText || '').length > 100);
  await page.screenshot({ path: path.join(OUT, 'original-preview.png') });
  passed('browser-result-display');

  async function uiDownload(selector, name) {
    const event = page.waitForEvent('download');
    await page.click(selector);
    const download = await event;
    await download.saveAs(path.join(OUT, name));
    assert(!(await download.failure()), 'Browser download must complete');
    report.downloads.push(name);
  }
  async function apiDownload(route, name) {
    const downloaded = await context.request.get(`${base}/api/jobs/${id}/${route}`, { timeout: TIMEOUT });
    assert(downloaded.ok(), 'Artifact API download must succeed');
    writeFileSync(path.join(OUT, name), await downloaded.body());
    report.downloads.push(name);
  }

  stage = 'original-downloads';
  await uiDownload('#dl-doc', 'original.html');
  await uiDownload('#dl-md', 'original.md');
  const originalHtml = readFileSync(path.join(OUT, 'original.html'), 'utf8');
  assert(originalHtml.includes('data:image/png;base64,') && !originalHtml.includes('/api/jobs/'), 'Original HTML must be standalone');
  assert(readFileSync(path.join(OUT, 'original.md'), 'utf8').length > 100, 'Original Markdown must contain OCR text');
  await apiDownload('archive', 'original.zip');
  inspectArtifacts(false);
  passed('original-html-markdown-zip-downloads');

  stage = 'live-provider-probe';
  const probe = await context.request.post(`${base}/e2e/live-provider-check`, { timeout: 90000 });
  const probeState = await probe.json();
  assert(['ready', 'blocked-auth', 'blocked-config', 'blocked-provider'].includes(probeState.status), 'Harness probe endpoint must return a recognized status');
  report.provider_probe = {
    status: probeState.status,
    ...(new Set(['authentication', 'output-limit', 'timeout', 'connection', 'provider-error', 'no-korean-output']).has(probeState.reason)
      ? { reason: probeState.reason } : {}),
    ...(Number.isInteger(probeState.http_status) && probeState.http_status >= 100 && probeState.http_status <= 599
      ? { http_status: probeState.http_status } : {}),
  };
  if (probeState.status !== 'ready') blocked(probeState.status);
  assert(probe.ok(), 'Live provider probe must succeed');
  passed('real-provider-authenticated');

  stage = 'browser-translation';
  const translating = page.waitForResponse((r) => r.request().method() === 'POST'
    && new URL(r.url()).pathname === `/api/jobs/${id}/translate`);
  await page.click('#translate-btn');
  assert((await translating).status() === 202, 'Browser must start real translation');
  const deadline = Date.now() + TIMEOUT;
  let state;
  while (Date.now() < deadline) {
    state = await (await context.request.get(`${base}/api/jobs/${id}/translate/state?lang=ko`)).json();
    if (state.status === 'done') break;
    if (state.status === 'error') blocked(/인증 실패/.test(String(state.error)) ? 'blocked-auth' : 'blocked-provider');
    await pause(500);
  }
  assert(state?.status === 'done', 'Real translation must finish');
  await page.waitForSelector('#lang-toggle:not([hidden])');
  await page.click('#lang-ko');
  await page.waitForFunction(() => /[가-힣]/.test(document.getElementById('preview-body')?.innerText || ''));
  await page.click('#viewer-open');
  await page.waitForFunction(() => /[가-힣]/.test(document.getElementById('reader-content')?.innerText || ''));
  await page.screenshot({ path: path.join(OUT, 'translated-viewer.png') });
  await page.click('#viewer-close');
  passed('real-translation-and-browser-display');

  stage = 'translated-downloads';
  await uiDownload('#dl-doc-ko', 'translated.html');
  await uiDownload('#dl-md', 'translated.md');
  assert(/[가-힣]/.test(readFileSync(path.join(OUT, 'translated.html'), 'utf8')), 'Translated HTML must contain Korean');
  assert(/[가-힣]/.test(readFileSync(path.join(OUT, 'translated.md'), 'utf8')), 'Translated Markdown must contain Korean');
  await apiDownload('archive', 'translated.zip');
  await apiDownload('pdf?lang=ko', 'translated.pdf');
  inspectArtifacts(true);
  assert(browserErrors.length === 0, 'Browser must not have unhandled errors');
  passed('translated-html-markdown-zip-pdf-downloads');
  report.status = 'passed';
  code = 0;
} catch (error) {
  report.status = error.liveStatus || 'failed';
  report.stage = stage;
  report.error_type = error.constructor.name;
  code = error.liveStatus ? 2 : 1;
  // Never print provider bodies, process environments, or raw backend logs.
  console.error(`${report.status.toUpperCase()} ${stage}`);
} finally {
  await browser?.close();
  if (backend && backend.exitCode === null && !backend.spawnFailed) {
    backend.kill('SIGTERM');
    const stopped = new Promise((resolve) => backend.once('exit', resolve));
    await Promise.race([stopped, pause(8000)]);
    if (backend.exitCode === null && backend.signalCode === null) backend.kill('SIGKILL');
  }
  writeFileSync(path.join(OUT, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`Live E2E evidence: ${path.join(OUT, 'report.json')}`);
}
process.exitCode = code;
