// 외부 호출 없는 전체 브라우저 E2E 하네스.
// 로컬 mock OpenAI 번역(/v1/responses) + 테스트 전용 OpenAI Responses Q&A
// router, FakeEngine 백엔드를 띄우고 번역/PDF/Q&A 확장 분기를 실행한다.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer as createNetServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FINAL = path.resolve(HERE, '../../..');
const BACKEND = path.join(FINAL, 'backend');
const PYTHON = path.join(BACKEND, '.venv', 'bin', 'python');
const DATA = mkdtempSync(path.join(tmpdir(), 'uocr-mock-e2e-'));

function respond(res, status, body) {
  const data = Buffer.from(JSON.stringify(body));
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': String(data.length),
  });
  res.end(data);
}

async function readJson(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 2_000_000) throw new Error('mock request too large');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
}

function sourceFromPrompt(input) {
  const marker = '[번역할 원문]\n';
  const at = String(input || '').lastIndexOf(marker);
  if (at >= 0) return String(input).slice(at + marker.length).trim();
  const repair = '[원문 — 아래 꺾쇠 태그가 정답이다]\n';
  const ra = String(input || '').indexOf(repair);
  if (ra >= 0) return String(input).slice(ra + repair.length).split('\n\n[수정할 번역문]')[0].trim();
  return String(input || '').trim();
}

function mockTranslation(input) {
  const src = sourceFromPrompt(input);
  if (src.startsWith('#')) {
    const m = src.match(/^(#+\s*)([\s\S]*)$/);
    return `${m[1]}모의 번역: ${m[2]}`;
  }
  return `모의 번역: ${src}`;
}

const mock = createServer(async (req, res) => {
  try {
    if (req.method === 'POST' && req.url === '/v1/responses') {
      const payload = await readJson(req);
      const glossary = String(payload.instructions || '').includes('용어집 편집자');
      respond(res, 200, {
        status: 'completed', model: 'mock-translate',
        output_text: glossary ? '[]' : mockTranslation(payload.input),
      });
      return;
    }
    respond(res, 404, { error: { message: 'mock route not found' } });
  } catch (error) {
    respond(res, 400, { error: { message: String(error && error.message || error) } });
  }
});

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

async function freePort() {
  const server = createNetServer();
  const port = await listen(server);
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function waitHealth(url, proc) {
  for (let i = 0; i < 120; i += 1) {
    if (proc.exitCode != null) throw new Error(`backend exited early (${proc.exitCode})`);
    try {
      const response = await fetch(`${url}/api/health`);
      if (response.ok) return;
    } catch (_) { /* cold start */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('mock E2E backend health timeout');
}

const mockPort = await listen(mock);
const backendPort = await freePort();
const mockOrigin = `http://127.0.0.1:${mockPort}`;
const base = `http://127.0.0.1:${backendPort}`;
let logs = '';
const backend = spawn(PYTHON, [
  '-m', 'uvicorn', 'e2e_mock_app:app', '--app-dir', BACKEND,
  '--host', '127.0.0.1', '--port', String(backendPort),
], {
  cwd: FINAL,
  env: {
    ...process.env,
    OCR_ENGINE: 'fake', OCR_DEVICE: 'cpu', PRELOAD_MODEL: '0', FAKE_DELAY: '0',
    DATA_DIR: DATA, ALLOWED_HOSTS: 'localhost,127.0.0.1',
    OPENAI_BASE_URL: mockOrigin, // bare origin 자동 /v1 보완도 함께 검증
    OPENAI_API_KEY: 'mock-only', OPENAI_MODEL: 'mock-translate',
    TRANSLATE_API_MODE: 'responses', TRANSLATE_CONCURRENCY: '1',
    TRANSLATE_MAX_RETRIES: '0', TRANSLATE_CONTEXT: '0',
    LLM_PROVIDER: 'openai-responses',
  },
  stdio: ['ignore', 'pipe', 'pipe'],
});
backend.stdout.on('data', (chunk) => { logs += chunk; });
backend.stderr.on('data', (chunk) => { logs += chunk; });

let code = 1;
try {
  await waitHealth(base, backend);
  const child = spawn(process.execPath, [path.join(HERE, 'ui.e2e.mjs')], {
    cwd: path.resolve(HERE, '../..'),
    env: {
      ...process.env,
      E2E_BASE_URL: base,
      E2E_TIMEOUT_S: '60',
      E2E_VERIFY_MOCK_LLM: '1',
    },
    stdio: 'inherit',
  });
  code = await new Promise((resolve) => child.once('exit', (value) => resolve(value ?? 1)));
} catch (error) {
  console.error(error);
  console.error(logs.slice(-8_000));
} finally {
  backend.kill('SIGTERM');
  await Promise.race([
    new Promise((resolve) => backend.once('exit', resolve)),
    new Promise((resolve) => setTimeout(resolve, 3_000)),
  ]);
  await new Promise((resolve) => mock.close(resolve));
  // DATA는 이 프로세스가 mkdtempSync로 만든 단일 테스트 디렉터리다.
  rmSync(DATA, { recursive: true, force: true });
}

process.exit(code);
