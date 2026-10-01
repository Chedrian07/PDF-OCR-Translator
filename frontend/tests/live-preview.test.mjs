// 라이브 미리보기 백로그 렌더 — 순서 보장 동시 요청 + 실패 시 앞서 받은 페이지 보존.
//
//   node --test frontend/tests/live-preview.test.mjs
//
// 예전에는 중간에 연 잡의 확정 페이지 백로그를 RTT마다 한 장씩 순차 POST했고, 한 장만
// 실패해도 그 사이클에서 받은 수십 장을 모두 버리고 처음부터 다시 보냈다(frontend-11).
// /render-preview의 429(남용 방어 — 크기 가중 레이트리밋·동시 렌더 상한)는 장애가 아니라
// "천천히"다: Retry-After만큼 쉬고, 5회 연속 실패 중단에 세지 않는다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { PREVIEW_RENDER_CONCURRENCY } from '../js/constants.js';
import { renderPagesInOrder } from '../js/core.js';
import { EL_IDS, el, state } from '../js/state.js';
import { runPreviewRender } from '../js/live.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

test('renderPagesInOrder: 동시 요청은 상한까지, 반영은 페이지 순서대로', async () => {
  const gates = [deferred(), deferred(), deferred(), deferred()];
  const started = [];
  const committed = [];
  const pages = [{ md: 'a' }, { md: 'b' }, { md: 'c' }, { md: 'd' }];
  const run = renderPagesInOrder(pages, (p) => {
    started.push(p.md);
    return gates[pages.indexOf(p)].promise;
  }, (p, html) => committed.push(`${p.md}:${html}`), { concurrency: 2 });
  await flush();
  assert.deepEqual(started, ['a', 'b'], '동시에 두 장까지만');
  gates[1].resolve({ html: 'B' }); // 뒤 페이지가 먼저 와도
  await flush();
  assert.deepEqual(committed, [], '앞 페이지를 기다린다');
  gates[0].resolve({ html: 'A' });
  await flush();
  assert.deepEqual(committed, ['a:A', 'b:B']);
  assert.deepEqual(started, ['a', 'b', 'c', 'd']);
  gates[2].resolve({ html: 'C' });
  gates[3].resolve({ html: 'D' });
  assert.deepEqual(await run, { failStatus: -1, committed: 4, stale: false });
  assert.deepEqual(committed, ['a:A', 'b:B', 'c:C', 'd:D']);
});

test('renderPagesInOrder: 실패하면 그 앞까지만 반영하고 실패 상태를 돌려준다', async () => {
  const committed = [];
  const result = await renderPagesInOrder(
    [{ md: '1' }, { md: '2' }, { md: '3' }, { md: '4' }],
    async (p) => (p.md === '3' ? { status: 502 } : { html: `<p>${p.md}</p>` }),
    (p) => committed.push(p.md),
    { concurrency: 3 },
  );
  assert.deepEqual(result, { failStatus: 502, committed: 2, stale: false });
  assert.deepEqual(committed, ['1', '2']);
});

test('renderPagesInOrder: 내용 없는 페이지는 요청 없이 빈 HTML로, 예외는 네트워크 오류(0)로', async () => {
  const requested = [];
  const committed = [];
  const result = await renderPagesInOrder(
    [{ md: '' }, { md: 'x' }, { md: 'boom' }],
    async (p) => { requested.push(p.md); if (p.md === 'boom') throw new Error('offline'); return { html: 'X' }; },
    (p, html) => committed.push([p.md, html]),
  );
  assert.deepEqual(requested, ['x', 'boom']);
  assert.deepEqual(committed, [['', ''], ['x', 'X']]);
  assert.equal(result.failStatus, 0);
});

test('renderPagesInOrder: 중간에 잡이 바뀌면(stale) 더 반영하지 않는다', async () => {
  let current = true;
  const committed = [];
  const result = await renderPagesInOrder(
    [{ md: '1' }, { md: '2' }],
    async (p) => { if (p.md === '1') current = false; return { html: p.md }; },
    (p) => committed.push(p.md),
    { isCurrent: () => current },
  );
  assert.equal(result.stale, true);
  assert.deepEqual(committed, []);
});

/* ---------------- 런타임: runPreviewRender ---------------- */

function setup(t, renderFor) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const key of Object.keys(EL_IDS)) el[key] = mount(doc, 'div', EL_IDS[key]);
  Object.assign(state, {
    currentJobId: 'job-a', liveGen: 1, rawText: '', previewDirty: true, previewTimer: 0,
    previewInFlight: false, previewFails: 0, previewStopped: false, previewPageCache: [],
    previewPageNodes: [], previewTailNodes: [], previewTailMd: '', previewTailSep: false,
    previewAutoScroll: false,
  });
  const bodies = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const body = String(init && init.body);
    bodies.push(body);
    // renderFor는 상태 숫자 또는 {status, retryAfter}(429 응답의 Retry-After 헤더)
    const out = renderFor(body);
    const { status, retryAfter = null } = typeof out === 'number' ? { status: out } : out;
    return {
      ok: status === 200, status,
      headers: { get: (name) => (/^retry-after$/i.test(name) ? retryAfter : null) },
      text: async () => `<p>${body}</p>`,
    };
  });
  t.after(() => {
    clearTimeout(state.previewTimer);
    state.previewTimer = 0;
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return { bodies };
}

test('백로그 중 한 장이 실패해도 앞서 받은 페이지는 남고, 다음 사이클은 그 페이지부터', async (t) => {
  let fail = true;
  const { bodies } = setup(t, (body) => (fail && body === 'page 3' ? 502 : 200));
  // 확정 페이지 5개(첫 마커 앞 빈 서두 + 1..4쪽) + 미확정 꼬리
  state.rawText = '<PAGE>page 1<PAGE>page 2<PAGE>page 3<PAGE>page 4<PAGE>tail';
  await runPreviewRender();
  assert.deepEqual(state.previewPageCache, ['', '<p>page 1</p>', '<p>page 2</p>'],
    '실패 앞까지 받은 페이지는 버리지 않는다');
  assert.equal(state.previewFails, 1);
  assert.equal(state.previewDirty, true);
  assert.ok(bodies.filter((b) => b === 'page 1').length === 1);
  assert.ok(bodies.length <= 1 + PREVIEW_RENDER_CONCURRENCY + 1, `동시 요청 상한: ${bodies.join(', ')}`);

  fail = false;
  bodies.length = 0;
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  await runPreviewRender();
  assert.deepEqual(bodies, ['page 3', 'page 4', 'tail'], '받은 페이지는 다시 보내지 않는다');
  assert.equal(state.previewPageCache.length, 5);
  assert.equal(state.previewTailMd, 'tail');
  assert.equal(state.previewFails, 0);
});

/* ---------------- 런타임: 429 백오프 ---------------- */

// 비동기 사슬(fetch → text → 반영)이 다 돌 때까지 마이크로태스크·즉시 큐를 비운다.
async function settle() {
  for (let i = 0; i < 20; i += 1) await flush();
}

test('render-preview 429: 실패로 세지 않고 Retry-After만큼 쉰 뒤 막힌 페이지부터 잇는다', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000_000 });
  t.after(() => t.mock.timers.reset());
  let throttled = true;
  const { bodies } = setup(t, (body) => (
    throttled && body === 'page 2' ? { status: 429, retryAfter: '7' } : 200));
  state.rawText = '<PAGE>page 1<PAGE>page 2<PAGE>page 3<PAGE>tail';
  state.previewFails = 4; // 직전 일시 장애가 4번 쌓여 있어도 429 한 번으로 멈추면 안 된다
  await runPreviewRender();
  assert.deepEqual(state.previewPageCache, ['', '<p>page 1</p>'], '막히기 전 페이지는 반영');
  assert.equal(state.previewFails, 4, '429는 연속 실패 횟수에 더하지 않는다');
  assert.equal(state.previewStopped, false);
  assert.equal(state.previewDirty, true);
  assert.equal(state.previewRetryAt, 1_000_000 + 7_000, 'Retry-After 7초');
  assert.ok(state.previewTimer, '대기 뒤 재시도가 예약된다');

  bodies.length = 0;
  t.mock.timers.tick(6_999);
  await settle();
  assert.deepEqual(bodies, [], 'Retry-After 전에는 보내지 않는다(평소 3초 재시도보다 길다)');
  throttled = false;
  t.mock.timers.tick(1);
  await settle();
  assert.deepEqual(bodies, ['page 2', 'page 3', 'tail'], '받은 페이지는 다시 보내지 않는다');
  assert.equal(state.previewFails, 0);
  assert.equal(state.previewTailMd, 'tail');
});

test('render-preview 429가 계속돼도 미리보기를 멈추지 않는다 — 502는 다섯 번이면 멈춘다', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 5_000_000 });
  t.after(() => t.mock.timers.reset());
  let status = { status: 429, retryAfter: '1' };
  const { bodies } = setup(t, () => status);
  state.rawText = 'tail only';
  await runPreviewRender();
  for (let round = 1; round < 6; round += 1) {
    t.mock.timers.tick(1_000);                 // Retry-After(1초)가 지나 예약된 재시도가 돈다
    await settle();
  }
  assert.equal(bodies.length, 6, '429마다 Retry-After 간격으로 다시 시도한다');
  assert.equal(state.previewStopped, false, '429만으로는 중단하지 않는다');
  assert.equal(state.previewFails, 0);

  status = 502;
  for (let round = 0; round < 5; round += 1) {
    t.mock.timers.tick(3_000);                 // 일시 장애 재시도 간격(600ms → 4회째부터 3초)
    await settle();
  }
  assert.equal(bodies.length, 11);
  assert.equal(state.previewStopped, true, '진짜 실패는 여전히 다섯 번이면 중단');
  assert.match(el.livePreview.textContent, /계속 실패해 중단/);
});

test('render-preview 429 대기 중 앞당겨 불린 실행은 요청 없이 남은 시간만큼 미룬다', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 9_000_000 });
  t.after(() => t.mock.timers.reset());
  const { bodies } = setup(t, () => 200);
  state.rawText = '<PAGE>page 1<PAGE>tail';
  state.previewRetryAt = 9_000_000 + 5_000;   // 직전 429가 5초 쉬라고 했다
  await runPreviewRender();                    // replay·reset 경로가 바로 다시 부른 경우
  assert.deepEqual(bodies, []);
  assert.equal(state.previewDirty, true, '보낼 내용은 그대로 남는다');
  assert.ok(state.previewTimer);
  t.mock.timers.tick(4_999);
  await settle();
  assert.deepEqual(bodies, []);
  t.mock.timers.tick(1);
  await settle();
  assert.deepEqual(bodies, ['page 1', 'tail']);
});
