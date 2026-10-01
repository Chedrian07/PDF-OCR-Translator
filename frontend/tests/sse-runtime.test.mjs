// SSE 연결 → 폴링 강등 → 재승격 상태 기계 — 실제 js/sse.js·js/translate.js를 가짜 DOM,
// 가짜 EventSource, node:test mock timers로 돌린다 (tests-baseline-8).
//
//   node --test frontend/tests/sse-runtime.test.mjs
//
// 지키는 계약:
//  · 첫 연결이 비-200이면(EventSource CLOSED, error 1회) 즉시 폴링으로 강등하고 재승격을 예약한다
//    (frontend-1). 연결 도중 끊김(CONNECTING)은 2회 규칙을 유지한다.
//  · 재승격 open → 폴링 해제·백오프 리셋.
//  · 폴링은 한 번에 한 요청, 재승격 사이에 도착한 낡은 스냅샷은 적용하지 않는다(frontend-12).
//  · 404 → 잡 제거와 빈 화면, 터미널 → 폴링·재승격 정리 후 결과 렌더.
//  · teardownConnections가 es와 모든 타이머를 정리한다. 번역 SSE도 CLOSED면 바로 폴링한다.
//  · DELETE의 종료 SSE {deleted:true}는 취소 화면이 아니라 잡을 닫는다(목록·이어 읽기·메모 정리).
//  · POST /cancel 202 {status:'canceled'}(대기 잡 즉시 마감)는 종료 이벤트 없이도 바로 마감한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { readerNotesKey, readerPosKey } from '../js/constants.js';
import { EL_IDS, el, state } from '../js/state.js';
import {
  handleSseConnError, startFallbackPolling, startStream, teardownConnections,
} from '../js/sse.js';
import { deleteJob, requestCancel } from '../js/jobs.js';
import { connectTranslateEvents, teardownTranslate } from '../js/translate.js';
import { installFakeDom, installFakeStorage, mount } from './helpers/fake-dom.mjs';

class MockEventSource {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0; // CONNECTING
    this.listeners = new Map();
    this.closed = false;
    MockEventSource.instances.push(this);
  }
  addEventListener(name, fn) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(fn);
  }
  close() { this.closed = true; this.readyState = 2; }
  emit(name, data) {
    for (const fn of this.listeners.get(name) || []) fn(data === undefined ? {} : { data: JSON.stringify(data) });
  }
  // 브라우저 동작: 비-200 첫 응답 → CLOSED + error 1회 / 도중 끊김 → CONNECTING + error
  failClosed() { this.readyState = 2; this.emit('error'); }
  failReconnecting() { this.readyState = 0; this.emit('error'); }
  open() { this.readyState = 1; this.emit('open'); }
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

function response(status, data) {
  return {
    ok: status >= 200 && status < 300, status, headers: { get: () => null },
    async text() { return JSON.stringify(data); },
  };
}

function setup(t, { fetchImpl } = {}) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const key of Object.keys(EL_IDS)) el[key] = mount(doc, 'div', EL_IDS[key]);
  el.tabs = [];
  el.panels = [];
  el.modeRadios = [];
  el.qaSuggestions = [];
  Object.assign(state, {
    currentJobId: 'job-a', displayedStatus: 'running', displayedPhase: 'ocr', openGen: 1,
    jobs: [{ job_id: 'job-a', filename: 'a.pdf', status: 'running' }],
    es: null, sseErrorCount: 0, fallbackActive: false, fallbackTimer: 0,
    ssePromoteTimer: 0, ssePromoteAttempts: 0, streamConnected: false,
    translateEs: null, translatePollTimer: 0, translateSseErrors: 0, translateGen: 0,
    readerImgTimers: new Map(), readerAlignmentRetryTimers: new Map(),
    readerAlignmentRetryCounts: new Map(), readerAlignmentBackoff: new Set(),
    toastTimer: 0, rafId: 0, previewTimer: 0,
  });
  MockEventSource.instances = [];
  const savedES = Object.getOwnPropertyDescriptor(globalThis, 'EventSource');
  globalThis.EventSource = MockEventSource;
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push(String(url));
    return fetchImpl ? fetchImpl(String(url), init) : response(200, { jobs: [] });
  });
  t.mock.timers.enable({ apis: ['setInterval', 'setTimeout'] });
  t.after(() => {
    teardownConnections();
    t.mock.timers.reset();
    if (savedES) Object.defineProperty(globalThis, 'EventSource', savedES);
    else delete globalThis.EventSource;
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return { doc, calls };
}

test('첫 연결이 비-200(CLOSED)이면 error 1회만으로 즉시 폴링 강등 + 재승격 예약', (t) => {
  setup(t);
  startStream('job-a');
  const es = MockEventSource.instances[0];
  es.failClosed();
  assert.equal(state.fallbackActive, true, '폴링이 켜진다');
  assert.ok(state.fallbackTimer, '폴링 타이머');
  assert.ok(state.ssePromoteTimer, '재승격 타이머');
  assert.equal(state.es, null);
  assert.equal(es.closed, true);
  assert.match(el.streamPane.textContent, /상태 폴링으로 전환/);
});

test('연결 도중 끊김(CONNECTING)은 2회째에 강등한다 — 브라우저 자동 재접속을 존중', (t) => {
  setup(t);
  startStream('job-a');
  const es = MockEventSource.instances[0];
  es.failReconnecting();
  assert.equal(state.fallbackActive, false);
  assert.equal(state.sseErrorCount, 1);
  es.failReconnecting();
  assert.equal(state.fallbackActive, true);
});

test('재승격: 백오프 뒤 다시 연결하고 open이면 폴링을 풀고 백오프를 리셋한다', (t) => {
  setup(t, { fetchImpl: () => new Promise(() => {}) }); // 폴링 응답은 오지 않는다
  startStream('job-a');
  MockEventSource.instances[0].failClosed();
  assert.equal(state.ssePromoteAttempts, 1);
  t.mock.timers.tick(10_000); // 첫 재승격 시도(10초)
  assert.equal(MockEventSource.instances.length, 2, '재승격 시도가 새 연결을 연다');
  const promoted = MockEventSource.instances[1];
  assert.equal(state.fallbackActive, true, 'open 전까지 폴링 유지');
  promoted.open();
  assert.equal(state.fallbackActive, false);
  assert.equal(state.fallbackTimer, 0);
  assert.equal(state.ssePromoteTimer, 0);
  assert.equal(state.ssePromoteAttempts, 0);
  assert.ok(state.es === promoted);
});

test('재승격 시도가 실패하면 폴링은 유지하고 다음 백오프로 다시 예약한다', (t) => {
  setup(t, { fetchImpl: () => new Promise(() => {}) });
  startStream('job-a');
  MockEventSource.instances[0].failClosed();
  t.mock.timers.tick(10_000);
  MockEventSource.instances[1].failClosed();
  assert.equal(state.fallbackActive, true);
  assert.equal(state.es, null);
  assert.equal(state.ssePromoteAttempts, 2);
  t.mock.timers.tick(19_999);
  assert.equal(MockEventSource.instances.length, 2, '두 번째 백오프는 20초');
  t.mock.timers.tick(1);
  assert.equal(MockEventSource.instances.length, 3);
});

test('폴링은 한 번에 한 요청 — 느린 응답 동안 1초 틱이 겹쳐 보내지 않는다', async (t) => {
  let release;
  const { calls } = setup(t, {
    fetchImpl: (url) => (url === '/api/jobs/job-a'
      ? new Promise((resolve) => { release = () => resolve(response(200, {
        job_id: 'job-a', status: 'running', progress: { phase: 'ocr', current_page: 2, total_pages: 5 },
      })); })
      : response(200, { jobs: [] })),
  });
  startFallbackPolling('job-a');
  t.mock.timers.tick(1000);
  t.mock.timers.tick(1000);
  t.mock.timers.tick(1000);
  assert.equal(calls.filter((u) => u === '/api/jobs/job-a').length, 1);
  release();
  await flush();
  assert.equal(el.progressCount.textContent, '2 / 5 페이지', '진행 스냅샷 적용');
  t.mock.timers.tick(1000);
  assert.equal(calls.filter((u) => u === '/api/jobs/job-a').length, 2, '응답 뒤에는 다시 묻는다');
});

test('폴링 응답 도중 SSE가 재승격되면 낡은 스냅샷(진행·터미널 모두)을 적용하지 않는다', async (t) => {
  let release;
  setup(t, {
    fetchImpl: (url) => (url === '/api/jobs/job-a'
      ? new Promise((resolve) => { release = (body) => resolve(response(200, body)); })
      : response(200, { jobs: [] })),
  });
  startStream('job-a');
  MockEventSource.instances[0].failClosed();
  t.mock.timers.tick(1000); // 폴링 요청 출발
  t.mock.timers.tick(9000); // 재승격 시도
  MockEventSource.instances[1].open(); // 응답 전에 SSE가 살아난다
  el.progressCount.textContent = 'sentinel';
  release({ job_id: 'job-a', status: 'error', error: 'late', progress: { phase: 'ocr', current_page: 9 } });
  await flush();
  assert.equal(el.progressCount.textContent, 'sentinel', '낡은 진행 스냅샷 미적용');
  assert.equal(state.displayedStatus, 'running', '터미널 렌더는 살아난 스트림이 직접 전달한다');
});

test('폴링 404 → 잡을 목록에서 지우고 빈 화면으로', async (t) => {
  setup(t, {
    fetchImpl: (url) => (url === '/api/jobs/job-a'
      ? response(404, { detail: '잡을 찾을 수 없습니다' })
      : response(200, { jobs: [] })),
  });
  startFallbackPolling('job-a');
  t.mock.timers.tick(1000);
  await flush();
  assert.equal(state.currentJobId, null);
  assert.deepEqual(state.jobs, []);
  assert.equal(el.emptyState.hidden, false);
  assert.equal(el.jobView.hidden, true);
  assert.equal(state.fallbackTimer, 0);
  assert.equal(state.fallbackActive, false);
});

test('폴링이 터미널(error)을 보면 폴링·재승격을 정리하고 결과를 렌더한다', async (t) => {
  const { calls } = setup(t, {
    fetchImpl: (url) => (url === '/api/jobs/job-a'
      ? response(200, { job_id: 'job-a', status: 'error', error: '변환 실패', progress: {} })
      : response(200, { jobs: [{ job_id: 'job-a', status: 'error' }] })),
  });
  startStream('job-a');
  MockEventSource.instances[0].failClosed();
  t.mock.timers.tick(1000);
  await flush();
  await flush();
  assert.equal(state.displayedStatus, 'error');
  assert.equal(state.fallbackActive, false);
  assert.equal(state.fallbackTimer, 0);
  assert.equal(state.ssePromoteTimer, 0);
  assert.equal(el.errorMessage.textContent, '변환 실패');
  assert.ok(calls.includes('/api/jobs'), '목록도 갱신한다');
});

test('teardownConnections: es와 폴링·재승격 타이머를 모두 정리한다', (t) => {
  setup(t, { fetchImpl: () => new Promise(() => {}) });
  startStream('job-a');
  const es = MockEventSource.instances[0];
  es.failClosed();
  teardownConnections();
  assert.equal(state.es, null);
  assert.equal(state.fallbackTimer, 0);
  assert.equal(state.ssePromoteTimer, 0);
  assert.equal(state.fallbackActive, false);
  t.mock.timers.tick(60_000);
  assert.equal(MockEventSource.instances.length, 1, '정리 뒤에는 재승격하지 않는다');
});

test('교체된 옛 연결의 늦은 오류는 새 연결의 카운트를 올리지 않는다', (t) => {
  setup(t);
  startStream('job-a');
  const old = MockEventSource.instances[0];
  startStream('job-a'); // 같은 잡 재구독 — 옛 연결은 닫힌다
  handleSseConnError('job-a', old);
  assert.equal(state.sseErrorCount, 0);
  assert.equal(state.fallbackActive, false);
});

test('번역 SSE도 첫 연결이 CLOSED면 즉시 상태 폴링으로 전환한다', (t) => {
  setup(t, { fetchImpl: () => new Promise(() => {}) });
  for (const key of ['translateBtn', 'readerTranslateBtn']) el[key].hidden = true;
  connectTranslateEvents('job-a');
  const es = MockEventSource.instances[0];
  assert.match(es.url, /\/translate\/events/);
  es.failClosed();
  assert.equal(state.translateEs, null);
  assert.ok(state.translatePollTimer, '번역 상태 폴링이 시작된다');
  teardownTranslate();
});

/* ---------------- 삭제·취소의 종료 처리 ---------------- */

const DELETED = { message: '삭제된 작업입니다', canceled: true, deleted: true };

test('DELETE의 종료 SSE {deleted:true}는 취소 화면 대신 잡을 닫고 로컬 흔적을 지운다', async (t) => {
  const { calls } = setup(t);
  const storage = installFakeStorage(t);
  storage.setItem(readerPosKey('job-a'), '7');
  storage.setItem(readerNotesKey('job-a'), '{"v":1,"items":[]}');
  storage.setItem(readerPosKey('job-b'), '2');
  Object.assign(state, {
    displayedStatus: 'queued',
    jobs: [{ job_id: 'job-a', status: 'queued' }, { job_id: 'job-b', status: 'done' }],
  });
  startStream('job-a');
  const es = MockEventSource.instances[0];
  es.open();
  es.emit('error', DELETED); // 다른 탭(또는 API)이 이 대기 잡을 지웠다
  await flush();
  assert.equal(state.currentJobId, null);
  assert.equal(el.emptyState.hidden, false, '빈 화면으로');
  assert.equal(el.jobView.hidden, true);
  assert.notEqual(el.errorTitle.textContent, '취소됨', '취소 화면을 그리지 않는다');
  assert.deepEqual(state.jobs.map((j) => j.job_id), ['job-b']);
  assert.equal(storage.getItem(readerPosKey('job-a')), null, '이어 읽기 위치 정리');
  assert.equal(storage.getItem(readerNotesKey('job-a')), null, '인용·하이라이트 정리');
  assert.equal(storage.getItem(readerPosKey('job-b')), '2', '다른 잡은 그대로');
  assert.equal(es.closed, true);
  assert.equal(state.es, null);
  assert.match(el.toast.textContent, /삭제되었습니다/, '왜 닫혔는지 알린다');
  assert.ok(!calls.some((u) => u.startsWith('/api/jobs/job-a')), `사라진 산출물을 묻지 않는다: ${calls}`);
});

test('이 탭이 지운 잡의 {deleted:true}가 DELETE 응답보다 먼저 와도 "삭제됨" 안내는 띄우지 않는다', async (t) => {
  let releaseDelete;
  setup(t, {
    fetchImpl: (url, init) => {
      if (init && init.method === 'DELETE') {
        return new Promise((resolve) => { releaseDelete = () => resolve(response(204, null)); });
      }
      return response(200, { jobs: [] });
    },
  });
  installFakeStorage(t);
  state.displayedStatus = 'queued';
  startStream('job-a');
  MockEventSource.instances[0].open();
  const deleting = deleteJob('job-a');
  MockEventSource.instances[0].emit('error', DELETED); // 서버는 응답 전에 이벤트부터 보낸다
  await flush();
  assert.equal(state.currentJobId, null);
  assert.equal(el.emptyState.hidden, false);
  releaseDelete();
  await deleting;
  assert.doesNotMatch(el.toast.textContent || '', /삭제되었습니다/);
});

function cancelSetup(t, { cancelStatus, job }) {
  return setup(t, {
    fetchImpl: (url, init) => {
      if (url === '/api/jobs/job-a/cancel' && init && init.method === 'POST') {
        return response(202, { job_id: 'job-a', status: cancelStatus });
      }
      if (url === '/api/jobs/job-a') return response(200, job);
      return response(200, { jobs: [job] });
    },
  });
}

test('POST /cancel 202 {status:canceled}(대기 잡 즉시 마감)는 종료 이벤트 없이 바로 취소 화면', async (t) => {
  const job = {
    job_id: 'job-a', filename: 'a.pdf', status: 'canceled',
    error: '사용자에 의해 취소되었습니다', progress: {},
  };
  const { calls } = cancelSetup(t, { cancelStatus: 'canceled', job });
  state.displayedStatus = 'queued';
  startFallbackPolling('job-a'); // 폴링 강등 중 — 다음 틱(1초)도, 종료 SSE도 기다리지 않는다
  await requestCancel();
  assert.equal(state.displayedStatus, 'canceled');
  assert.equal(el.errorSection.hidden, false);
  assert.equal(el.errorTitle.textContent, '취소됨');
  assert.equal(el.errorMessage.textContent, '사용자에 의해 취소되었습니다');
  assert.equal(state.cancelRequestedFor, null);
  assert.equal(el.jobStop.hidden, true, '정지 버튼이 남지 않는다');
  assert.equal(state.fallbackTimer, 0, '폴링도 정리한다');
  assert.ok(calls.includes('/api/jobs/job-a'), '최종 상태를 한 번 다시 받는다');
});

test('POST /cancel 202 {status:canceling}(실행 중)은 러너의 종료 이벤트를 기다린다', async (t) => {
  const { calls } = cancelSetup(t, {
    cancelStatus: 'canceling', job: { job_id: 'job-a', status: 'running', progress: {} },
  });
  state.displayedStatus = 'running';
  await requestCancel();
  assert.equal(state.displayedStatus, 'running');
  assert.equal(state.cancelRequestedFor, 'job-a');
  assert.equal(el.jobStop.disabled, true);
  assert.equal(el.jobStopLabel.textContent, '취소 중…');
  assert.ok(!calls.includes('/api/jobs/job-a'));
});

test('종료 SSE가 cancel 응답보다 먼저 마감했으면 다시 그리지 않는다', async (t) => {
  let releaseCancel;
  const { calls } = setup(t, {
    fetchImpl: (url, init) => {
      if (url === '/api/jobs/job-a/cancel') {
        return new Promise((resolve) => {
          releaseCancel = () => resolve(response(202, { job_id: 'job-a', status: 'canceled' }));
        });
      }
      return response(200, { jobs: [] });
    },
  });
  state.displayedStatus = 'queued';
  startStream('job-a');
  const es = MockEventSource.instances[0];
  es.open();
  const canceling = requestCancel();
  es.emit('error', { message: '사용자에 의해 취소되었습니다', canceled: true });
  assert.equal(state.displayedStatus, 'canceled');
  releaseCancel();
  await canceling;
  assert.ok(!calls.includes('/api/jobs/job-a'), `중복 마감 조회 없음: ${calls}`);
  assert.equal(el.errorTitle.textContent, '취소됨');
});
