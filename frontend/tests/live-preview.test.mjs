// 라이브 미리보기 백로그 렌더 — 순서 보장 동시 요청 + 실패 시 앞서 받은 페이지 보존.
//
//   node --test frontend/tests/live-preview.test.mjs
//
// 예전에는 중간에 연 잡의 확정 페이지 백로그를 RTT마다 한 장씩 순차 POST했고, 한 장만
// 실패해도 그 사이클에서 받은 수십 장을 모두 버리고 처음부터 다시 보냈다(frontend-11).

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
    const status = renderFor(body);
    return {
      ok: status === 200, status, headers: { get: () => null },
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
