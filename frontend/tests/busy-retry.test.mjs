// 503(바쁨) 제한 재시도와 "한국어 보기에서 일시 장애가 언어를 뒤집지 않는다" 계약.
//
//   node --test frontend/tests/busy-retry.test.mjs
//
// 서버 계약(lane f1-api): 503 + Retry-After = 빌드·예열 중 → Retry-After를 지켜
// 제한 재시도하고 진행을 보여 준다. 404/409만 "번역본 없음" → 원문 폴백.
// 그 밖의 실패(네트워크·5xx)는 언어를 유지한 채 다시 시도를 안내한다(frontend-9).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  BUSY_RETRY_MAX, busyRetryDelay, busyWaitMessage, langFetchVerdict,
} from '../js/core.js';
import { POLL_TIMEOUT_MS, apiGet, busyRetryClock, fetchTextWithBusyRetry } from '../js/api.js';
import { el, state } from '../js/state.js';
import { loadDocLayout, loadMarkdown, loadPreview } from '../js/tabs.js';
import { loadReader } from '../js/reader.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

/* ---------------- 순수 정책 ---------------- */

test('langFetchVerdict: 404/409만 "없음", 503은 "바쁨", 나머지는 일시 장애', () => {
  assert.equal(langFetchVerdict(200), 'ok');
  assert.equal(langFetchVerdict(204), 'ok');
  assert.equal(langFetchVerdict(404), 'missing');
  assert.equal(langFetchVerdict(409), 'missing');
  assert.equal(langFetchVerdict(503), 'busy');
  for (const status of [0, 400, 429, 500, 502, 504, undefined, null, 'x']) {
    assert.equal(langFetchVerdict(status), 'error', `status ${String(status)}`);
  }
});

test('busyRetryDelay: Retry-After(초)를 지키고 1..60초로 묶는다', () => {
  assert.equal(busyRetryDelay(503, '30', 0), 30);
  assert.equal(busyRetryDelay(503, '7', 2), 7);
  assert.equal(busyRetryDelay(503, '86400', 0), 60);
  // 헤더가 없거나 깨지면 단계별 기본 대기(5→10→20→30)
  assert.deepEqual([0, 1, 2, 3].map((n) => busyRetryDelay(503, null, n)), [5, 10, 20, 30]);
  for (const header of ['', 'soon', '0', '-3', undefined]) {
    const wait = busyRetryDelay(503, header, 0);
    assert.ok(wait >= 1 && wait <= 60, `header ${String(header)} -> ${wait}`);
  }
});

test('busyRetryDelay: HTTP-date Retry-After는 남은 초로 환산한다', () => {
  const now = Date.parse('2026-10-01T00:00:00Z');
  assert.equal(busyRetryDelay(503, 'Thu, 01 Oct 2026 00:00:12 GMT', 0, BUSY_RETRY_MAX, now), 12);
  assert.equal(busyRetryDelay(503, 'Wed, 30 Sep 2026 23:00:00 GMT', 0, BUSY_RETRY_MAX, now), 1);
});

test('busyRetryDelay: 재시도는 유한하다 — 상한에 닿으면 0', () => {
  for (let attempt = 0; attempt < BUSY_RETRY_MAX; attempt += 1) {
    assert.ok(busyRetryDelay(503, '5', attempt) > 0, `attempt ${attempt}`);
  }
  assert.equal(busyRetryDelay(503, '5', BUSY_RETRY_MAX), 0);
  assert.equal(busyRetryDelay(503, '5', 0, 0), 0);
});

test('busyRetryDelay: 503이 아니면 재시도하지 않는다', () => {
  for (const status of [0, 200, 400, 404, 409, 429, 500, 502]) {
    assert.equal(busyRetryDelay(status, '5', 0), 0, `status ${status}`);
  }
});

test('busyWaitMessage: 무엇을 얼마나 기다리는지와 시도 횟수를 보여 준다', () => {
  assert.equal(busyWaitMessage('한국어 레이아웃', 30, 1, 4),
    '한국어 레이아웃 준비 중… 30초 뒤 다시 시도합니다 (1/4)');
});

/* ---------------- 런타임: fetchTextWithBusyRetry ---------------- */

function response(status, body = '', headers = {}) {
  const lower = Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get(name) { return lower[String(name).toLowerCase()] ?? null; } },
    async text() { return body; },
    async json() { return JSON.parse(body); },
  };
}

// URL 접두사별 응답 대기열. 함수 항목은 throw(네트워크 오류) 등을 흉내 낸다.
function routeFetch(t, routes) {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    calls.push(url);
    const key = Object.keys(routes).find((prefix) => String(url).startsWith(prefix));
    if (!key) throw new Error(`unrouted fetch ${url}`);
    const queue = routes[key];
    const next = queue.length > 1 ? queue.shift() : queue[0];
    return typeof next === 'function' ? next() : next;
  });
  return calls;
}

// 모듈의 진짜 대기 함수 — 스텁을 여러 번 겹쳐도 항상 이것으로 되돌린다.
const REAL_SLEEP = busyRetryClock.sleep;

function stubSleep(t) {
  const waits = [];
  busyRetryClock.sleep = async (ms) => { waits.push(ms); };
  t.after(() => { busyRetryClock.sleep = REAL_SLEEP; });
  return waits;
}

test('fetchTextWithBusyRetry: 503 뒤 Retry-After만큼 기다렸다 성공을 돌려준다', async (t) => {
  const waits = stubSleep(t);
  routeFetch(t, { '/x': [response(503, '', { 'Retry-After': '3' }), response(200, '<p>ok</p>')] });
  const notes = [];
  const r = await fetchTextWithBusyRetry('/x', { onWait: (...args) => notes.push(args) });
  assert.deepEqual(r, { status: 200, text: '<p>ok</p>' });
  assert.deepEqual(waits, [3000]);
  assert.deepEqual(notes, [[3, 1, BUSY_RETRY_MAX]]);
});

test('fetchTextWithBusyRetry: 바쁨이 계속되면 상한 뒤 503을 돌려준다', async (t) => {
  const waits = stubSleep(t);
  const calls = routeFetch(t, { '/x': [response(503, '', { 'Retry-After': '1' })] });
  const r = await fetchTextWithBusyRetry('/x');
  assert.deepEqual(r, { status: 503, text: null });
  assert.equal(calls.length, BUSY_RETRY_MAX + 1);
  assert.equal(waits.length, BUSY_RETRY_MAX);
});

test('fetchTextWithBusyRetry: 대기 중 잡·언어가 바뀌면 다시 묻지 않는다', async (t) => {
  let current = true;
  busyRetryClock.sleep = async () => { current = false; };
  t.after(() => { busyRetryClock.sleep = REAL_SLEEP; });
  const calls = routeFetch(t, { '/x': [response(503, '', { 'Retry-After': '1' })] });
  const r = await fetchTextWithBusyRetry('/x', { isCurrent: () => current });
  assert.equal(r.text, null);
  assert.equal(calls.length, 1);
});

test('fetchTextWithBusyRetry: 네트워크 오류는 status 0, 404는 재시도 없이 그대로', async (t) => {
  stubSleep(t);
  routeFetch(t, { '/net': [() => { throw new TypeError('Failed to fetch'); }], '/gone': [response(404)] });
  assert.deepEqual(await fetchTextWithBusyRetry('/net'), { status: 0, text: null });
  assert.deepEqual(await fetchTextWithBusyRetry('/gone'), { status: 404, text: null });
});

/* ---------------- 런타임: 결과 탭·리더 로더 ---------------- */

function setupResultView(t) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const key of [
    'doclayoutBody', 'previewBody', 'readerContent', 'viewerRoot', 'readerOutline', 'toast',
  ]) el[key] = mount(doc, 'div');
  el.mdCode = mount(doc, 'code');
  el.readerPageInput = mount(doc, 'input');
  for (const key of ['langOrig', 'langKo', 'viewerLangOrig', 'viewerLangKo']) el[key] = mount(doc, 'button');
  for (const key of ['dlMd', 'dlZip', 'dlDoc', 'viewerDlHtml']) el[key] = mount(doc, 'a');
  el.tabs = [];
  Object.assign(state, {
    currentJobId: 'job-a', currentLang: 'ko', openGen: 7, translateState: 'done',
    docLayoutLoaded: false, previewLoaded: false, markdownLoaded: false,
    resultHasLayout: true, layoutCapability: 'full', currentJobEngine: 'fake',
    healthEngine: 'fake', resultUrls: {}, currentBaseName: 'paper',
    readerPages: { orig: null, ko: null }, readerOutline: { orig: null, ko: null },
    readerAlignmentRetryTimers: new Map(), readerAlignmentRetryCounts: new Map(),
    readerAlignmentBackoff: new Set(), toastTimer: 0,
  });
  t.after(() => {
    clearTimeout(state.toastTimer);
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return doc;
}

const BUSY = () => response(503, '{"detail":"PDF 내보내기 대기열이 가득 찼습니다"}', { 'Retry-After': '2' });

test('레이아웃 탭(한국어): 503 동안 진행을 보이고 끝나면 한국어 그대로 렌더', async (t) => {
  setupResultView(t);
  const waits = stubSleep(t);
  const notes = [];
  const saved = busyRetryClock.sleep;
  busyRetryClock.sleep = async (ms) => { notes.push(el.doclayoutBody.textContent); return saved(ms); };
  routeFetch(t, { '/api/jobs/job-a/layout?lang=ko': [BUSY(), BUSY(), response(200, '<div class="layout-canvas"></div>')] });
  await loadDocLayout();
  assert.equal(state.currentLang, 'ko', '일시적인 503이 전역 언어를 원문으로 되돌리면 안 된다');
  assert.equal(state.docLayoutLoaded, true);
  assert.equal(el.doclayoutBody.innerHTML, '<div class="layout-canvas"></div>');
  assert.deepEqual(waits, [2000, 2000]);
  assert.deepEqual(notes, [
    '한국어 레이아웃 준비 중… 2초 뒤 다시 시도합니다 (1/4)',
    '한국어 레이아웃 준비 중… 2초 뒤 다시 시도합니다 (2/4)',
  ]);
  assert.equal(el.toast.textContent, '');
});

test('레이아웃 탭(한국어): 바쁨이 끝내 안 풀려도 언어를 유지하고 다시 시도를 준다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  const calls = routeFetch(t, { '/api/jobs/job-a/layout?lang=ko': [BUSY()] });
  await loadDocLayout();
  assert.equal(state.currentLang, 'ko');
  assert.equal(state.docLayoutLoaded, false, '실패는 캐시하지 않는다 — 다시 시도할 수 있어야 한다');
  assert.equal(calls.length, BUSY_RETRY_MAX + 1);
  const buttons = el.doclayoutBody.querySelectorAll('button').map((b) => b.textContent);
  assert.deepEqual(buttons, ['다시 시도', '원문 보기']);
  assert.match(el.doclayoutBody.textContent, /한국어 레이아웃을 불러오지 못했습니다/);

  // [다시 시도] → 같은 로더가 다시 묻고, 이번에는 성공한다.
  t.mock.restoreAll();
  stubSleep(t);
  routeFetch(t, { '/api/jobs/job-a/layout?lang=ko': [response(200, '<div>ko</div>')] });
  el.doclayoutBody.querySelector('button').click();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(el.doclayoutBody.innerHTML, '<div>ko</div>');
  assert.equal(state.currentLang, 'ko');
});

test('레이아웃 탭(한국어): 네트워크 오류도 언어를 뒤집지 않는다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  routeFetch(t, { '/api/jobs/job-a/layout?lang=ko': [() => { throw new TypeError('Failed to fetch'); }] });
  await loadDocLayout();
  assert.equal(state.currentLang, 'ko');
  assert.ok(el.doclayoutBody.querySelector('.load-failure'));
});

test('레이아웃 탭(한국어): 404(번역본 없음)일 때만 원문으로 폴백한다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  const calls = routeFetch(t, {
    '/api/jobs/job-a/layout?lang=ko': [response(404, '{"detail":"한국어 번역본이 없습니다"}')],
    '/api/jobs/job-a/layout': [response(200, '<div>orig</div>')],
  });
  await loadDocLayout();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.currentLang, 'orig');
  assert.match(el.toast.textContent, /원문을 표시합니다/);
  assert.deepEqual(calls, ['/api/jobs/job-a/layout?lang=ko', '/api/jobs/job-a/layout']);
  assert.equal(el.doclayoutBody.innerHTML, '<div>orig</div>');
});

test('레이아웃 탭: 같은 잡·언어 로드는 바쁨 대기 중에도 하나만 돈다', async (t) => {
  setupResultView(t);
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  busyRetryClock.sleep = () => gate;
  t.after(() => { busyRetryClock.sleep = REAL_SLEEP; });
  const calls = routeFetch(t, { '/api/jobs/job-a/layout?lang=ko': [BUSY(), response(200, '<div>ko</div>')] });
  const first = loadDocLayout();
  await new Promise((resolve) => setImmediate(resolve));
  await loadDocLayout(); // 탭 재클릭 — 겹쳐 보내지 않는다
  assert.equal(calls.length, 1);
  release();
  await first;
  assert.equal(calls.length, 2);
  assert.equal(el.doclayoutBody.innerHTML, '<div>ko</div>');
});

test('미리보기 탭(한국어): 503 뒤 성공하면 한국어 미리보기를 그린다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  routeFetch(t, { '/api/jobs/job-a/html?lang=ko': [BUSY(), response(200, '<p>한국어</p>')] });
  await loadPreview();
  assert.equal(state.currentLang, 'ko');
  assert.equal(state.previewLoaded, true);
  assert.equal(el.previewBody.innerHTML, '<p>한국어</p>');
});

test('Markdown 탭(한국어): 5xx는 언어를 유지하고 재시도 안내만 남긴다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  routeFetch(t, { '/api/jobs/job-a/markdown?lang=ko': [response(500, 'boom')] });
  await loadMarkdown();
  assert.equal(state.currentLang, 'ko');
  assert.equal(state.markdownLoaded, false);
  assert.match(el.mdCode.textContent, /다시 시도합니다/);
});

test('리더(한국어): 네트워크 오류는 언어를 유지하고 다시 시도·원문 보기를 준다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  routeFetch(t, {
    '/api/jobs/job-a/outline?lang=ko': [response(200, '{"items":[]}')],
    '/api/jobs/job-a/html?lang=ko': [() => { throw new TypeError('Failed to fetch'); }],
  });
  await loadReader();
  assert.equal(state.currentLang, 'ko');
  const box = el.readerContent.querySelector('.reader-load-error');
  assert.ok(box, 'reader-load-error가 보여야 한다');
  assert.deepEqual(box.querySelectorAll('button').map((b) => b.textContent), ['다시 시도', '원문 보기']);
  assert.equal(el.viewerRoot.hasAttribute('aria-busy'), false);
  assert.equal(state.readerRailKey, '', '비운 레일의 서명은 무효화된다');
});

test('리더(한국어): 404일 때만 원문으로 폴백한다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  routeFetch(t, {
    '/api/jobs/job-a/outline': [response(404, '{}')],
    '/api/jobs/job-a/html?lang=ko': [response(404, '{"detail":"한국어 번역본이 없습니다"}')],
    // 원문 재로드는 이 테스트의 관심사가 아니다 — 네트워크 오류로 끊어 둔다.
    '/api/jobs/job-a/html': [() => { throw new TypeError('offline'); }],
  });
  await loadReader();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.currentLang, 'orig');
  assert.match(el.toast.textContent, /번역본이 없어 원문을 표시합니다/);
});

test('리더 개요: 일시 실패는 캐시하지 않고, 404만 빈 개요로 확정한다', async (t) => {
  setupResultView(t);
  stubSleep(t);
  state.currentLang = 'orig';
  routeFetch(t, {
    '/api/jobs/job-a/outline': [response(503, '{}'), response(404, '{}')],
    '/api/jobs/job-a/html': [() => { throw new TypeError('offline'); }],
  });
  await loadReader();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.readerOutline.orig, null, '503 개요를 빈 목록으로 굳히지 않는다');
  await loadReader();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(state.readerOutline.orig, []);
});

/* ---------------- apiGet 시간 제한 (폴링 직렬화의 짝) ---------------- */

test('apiGet: timeoutMs가 있으면 시간 제한 신호를 붙이고, 응답이 없으면 실패한다', async (t) => {
  // Node의 AbortSignal.timeout 타이머는 unref라 이벤트 루프를 붙잡지 않는다 — 시간 제한이
  // 터질 때까지 테스트 프로세스가 살아 있도록 잠깐 붙잡아 둔다.
  const keepAlive = setInterval(() => {}, 10);
  t.after(() => clearInterval(keepAlive));
  const seen = [];
  t.mock.method(globalThis, 'fetch', (url, init) => {
    seen.push(init && init.signal);
    if (!init || !init.signal) return Promise.resolve(response(200, '{"ok":true}'));
    return new Promise((_, reject) => {
      init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
    });
  });
  assert.deepEqual(await apiGet('/api/jobs'), { ok: true });
  assert.equal(seen[0], undefined, '시간 제한이 없으면 신호도 없다(기존 호출부 그대로)');
  await assert.rejects(apiGet('/api/jobs', { timeoutMs: 30 }), (err) => err && err.name === 'TimeoutError');
  assert.ok(seen[1], '폴링 호출은 시간 제한 신호를 붙인다');
  assert.ok(POLL_TIMEOUT_MS >= 10_000 && POLL_TIMEOUT_MS <= 60_000);
});
