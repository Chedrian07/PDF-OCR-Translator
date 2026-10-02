// health 재조회 + 로드 실패·워커 중지 표시 — 순수 정책 + 실제 js/health.js(가짜 DOM).
//
//   node --test frontend/tests/health-runtime.test.mjs
//
// 예전에는 한 번 정상이면 health를 다시 묻지 않아 나중에 죽은 sidecar·워커가 배지로 안
// 보였고, model_load_error·worker_alive를 무시해 프리로드 실패가 '모델 로딩 중…'과
// '로딩이 끝나는 대로 자동 변환' 안내로 끝없이 보였다(frontend-8).
// 상시 폴링이 생긴 뒤로는 같은 응답에도 aria-live 배지 영역과 role=alert 안내를 매번 다시 써서
// 스크린리더가 같은 내용을 세션 내내 되풀이해 읽었다 — 바뀐 것만 고친다(frontend-2).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  HEALTH_POLL_FAST_MS, HEALTH_POLL_SLOW_MS, healthPollDelay, healthStatus,
} from '../js/core.js';
import { EL_IDS, el, state } from '../js/state.js';
import { loadHealth, renderHealth, renderHealthError, setupHealthPolling } from '../js/health.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

test('healthStatus: 로딩 중·로드 실패·워커 중지를 구분한다 (필드 부재는 정상)', () => {
  assert.deepEqual(healthStatus({ model_loaded: false }), { loading: true, loadError: '', workerDead: false });
  assert.deepEqual(healthStatus({ model_loaded: false, model_load_error: '  CUDA OOM  ' }),
    { loading: false, loadError: 'CUDA OOM', workerDead: false });
  // 로드가 끝난 뒤 남은 과거 오류 문구는 무시한다
  assert.deepEqual(healthStatus({ model_loaded: true, model_load_error: 'old' }),
    { loading: false, loadError: '', workerDead: false });
  assert.equal(healthStatus({ worker_alive: false }).workerDead, true);
  assert.deepEqual(healthStatus(null), { loading: false, loadError: '', workerDead: false });
  assert.equal(healthStatus({ model_loaded: false, model_load_error: 'x'.repeat(500) }).loadError.length, 300);
});

test('healthPollDelay: 정상은 느리게, 로딩·장애·연결 실패는 빠르게 다시 묻는다', () => {
  assert.equal(healthPollDelay({ model_loaded: true, worker_alive: true }), HEALTH_POLL_SLOW_MS);
  assert.equal(healthPollDelay({}), HEALTH_POLL_SLOW_MS);
  assert.equal(healthPollDelay({ model_loaded: false }), HEALTH_POLL_FAST_MS);
  assert.equal(healthPollDelay({ model_loaded: false, model_load_error: 'OOM' }), HEALTH_POLL_FAST_MS);
  assert.equal(healthPollDelay({ worker_alive: false }), HEALTH_POLL_FAST_MS);
  assert.equal(healthPollDelay({ provider: 'local-sidecar', provider_health: { status: 'error' } }),
    HEALTH_POLL_FAST_MS);
  assert.equal(healthPollDelay({ model_loaded: true }, true), HEALTH_POLL_FAST_MS);
  assert.ok(HEALTH_POLL_FAST_MS < HEALTH_POLL_SLOW_MS);
});

function setup(t, bodies) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const key of Object.keys(EL_IDS)) el[key] = mount(doc, 'div', EL_IDS[key]);
  Object.assign(state, { healthTimer: 0, healthInFlight: false, modelLoadError: '', translateState: 'none' });
  const queue = [...bodies];
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    calls.push(String(url));
    const body = queue.length > 1 ? queue.shift() : queue[0];
    return { ok: true, status: 200, headers: { get: () => null }, text: async () => JSON.stringify(body) };
  });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  t.after(() => {
    clearTimeout(state.healthTimer);
    t.mock.timers.reset();
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return { doc, calls };
}

const badges = () => el.healthBadges.querySelectorAll('.badge').map((b) => b.textContent);

test('프리로드 실패: "로딩 중" 대신 실패 배지와 사유가 담긴 업로드 안내', (t) => {
  setup(t, [{}]);
  renderHealth({ model_loaded: false, model_load_error: 'CUDA out of memory', device: 'cuda' });
  assert.ok(badges().includes('모델 로드 실패'));
  assert.ok(!badges().some((b) => b.includes('모델 로딩 중')));
  const badge = el.healthBadges.querySelectorAll('.badge-error').find((b) => b.textContent === '모델 로드 실패');
  assert.match(badge.getAttribute('title'), /CUDA out of memory/);
  assert.equal(el.uploadModelNotice.hidden, false);
  assert.ok(el.uploadModelNotice.classList.contains('is-error'));
  assert.equal(el.uploadModelNotice.getAttribute('role'), 'alert');
  assert.match(el.uploadModelNoticeText.textContent, /불러오지 못했습니다 \(CUDA out of memory\)/);
  assert.equal(el.uploadModelNoticeSpinner.hidden, true);
});

test('로딩 중에는 기존 안내, 로드되면 안내를 숨긴다', (t) => {
  setup(t, [{}]);
  renderHealth({ model_loaded: false });
  assert.ok(badges().includes('모델 로딩 중…'));
  assert.equal(el.uploadModelNotice.hidden, false);
  assert.ok(!el.uploadModelNotice.classList.contains('is-error'));
  assert.match(el.uploadModelNoticeText.textContent, /로딩이 끝나는 대로/);
  renderHealth({ model_loaded: true });
  assert.equal(el.uploadModelNotice.hidden, true);
});

test('워커 스레드가 죽으면 "작업 처리기 중지됨" 배지', (t) => {
  setup(t, [{}]);
  renderHealth({ model_loaded: true, worker_alive: false });
  assert.ok(badges().includes('작업 처리기 중지됨'));
  renderHealth({ model_loaded: true, worker_alive: true });
  assert.ok(!badges().includes('작업 처리기 중지됨'));
});

test('정상이어도 30초마다 다시 묻고, 그 사이 죽은 sidecar를 배지로 띄운다', async (t) => {
  const { calls } = setup(t, [
    { model_loaded: true, worker_alive: true, provider: 'local-sidecar', provider_health: { status: 'ok' } },
    { model_loaded: true, worker_alive: true, provider: 'local-sidecar', provider_health: { status: 'error', error: 'connection refused' } },
  ]);
  await loadHealth();
  assert.equal(calls.length, 1);
  assert.ok(!badges().includes('엔진 서버 연결 안 됨'));
  t.mock.timers.tick(HEALTH_POLL_SLOW_MS - 1);
  assert.equal(calls.length, 1, '정상 상태는 30초 간격');
  t.mock.timers.tick(1);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls.length, 2);
  assert.ok(badges().includes('엔진 서버 연결 안 됨'), '사후 sidecar 장애가 드러난다');
  t.mock.timers.tick(HEALTH_POLL_FAST_MS);
  assert.equal(calls.length, 3, '장애 중에는 10초 간격');
});

test('숨은 탭에서는 예약하지 않고, 다시 보이면 바로 묻는다', async (t) => {
  const { doc, calls } = setup(t, [{ model_loaded: true }]);
  setupHealthPolling();
  doc.hidden = true;
  await loadHealth();
  assert.equal(state.healthTimer, 0, '숨은 탭은 폴링 타이머를 걸지 않는다');
  t.mock.timers.tick(HEALTH_POLL_SLOW_MS * 3);
  assert.equal(calls.length, 1);
  doc.hidden = false;
  doc.dispatch('visibilitychange');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls.length, 2);
  assert.ok(state.healthTimer, '보이면 다시 예약한다');
});

test('조회가 진행 중이면 겹쳐 보내지 않는다', async (t) => {
  const { calls } = setup(t, [{ model_loaded: true }]);
  const first = loadHealth();
  const second = loadHealth();
  await Promise.all([first, second]);
  assert.equal(calls.length, 1);
});

test('MLX 디바이스는 Apple GPU 배지(칩 이름·Metal 스타일)로 보인다', (t) => {
  setup(t, [{}]);
  const device = () => el.healthBadges.querySelectorAll('.badge-device')[0];
  renderHealth({ model_loaded: true, device: 'mlx', gpu_name: 'Apple M4 Max', dtype: 'bfloat16+q8' });
  assert.equal(device().textContent, 'MLX · M4 Max');
  assert.ok(device().classList.contains('is-metal'));
  assert.ok(!device().classList.contains('is-cpu'));
  assert.match(device().getAttribute('title'), /dtype: bfloat16\+q8/);
  // 칩 이름이 없어도 회색 CPU 배지가 아니다
  renderHealth({ model_loaded: true, device: 'mlx' });
  assert.equal(device().textContent, 'MLX');
  assert.ok(device().classList.contains('is-metal'));
  // 기존 디바이스 배지는 그대로
  renderHealth({ model_loaded: true, device: 'metal', gpu_name: 'Apple M4 Max' });
  assert.equal(device().textContent, 'Metal · M4 Max');
  renderHealth({ model_loaded: true, device: 'cpu', gpu_name: 'ignored' });
  assert.equal(device().textContent, 'CPU');
  assert.ok(device().classList.contains('is-cpu'));
});

/* ---------------- 같은 응답은 DOM을 건드리지 않는다 (frontend-2) ---------------- */

// 노드와 그 하위 요소에 쓰기 기록기를 단다 — 속성·클래스·hidden·글자·자식 목록을 바꾸는 모든
// 호출을 log에 남긴다. 브라우저는 같은 값으로 setAttribute해도 변이 기록을 만들므로 값 비교
// 없이 호출 자체를 센다. classList 토글은 실제로 바뀔 때만 센다(브라우저와 같다).
const elementsUnder = (root) => root.children.flatMap((child) => [child, ...elementsUnder(child)]);

function watchWrites(root, log) {
  const nodes = [root, ...elementsUnder(root)];
  for (const node of nodes) {
    const label = node.dataset.badge || node.id || node.className || node.tagName;
    for (const method of ['setAttribute', 'removeAttribute', 'appendChild', 'insertBefore', 'removeChild', 'append', 'prepend']) {
      const orig = node[method];
      node[method] = function watched(...args) { log.push(`${label}.${method}(${args[0]})`); return orig.apply(this, args); };
    }
    for (const prop of ['hidden', 'textContent', 'className', 'innerHTML', 'title']) {
      let owner = node;
      let desc;
      while (owner && !(desc = Object.getOwnPropertyDescriptor(owner, prop))) owner = Object.getPrototypeOf(owner);
      let value = desc && 'value' in desc ? desc.value : undefined;
      Object.defineProperty(node, prop, {
        configurable: true,
        get() { return desc && desc.get ? desc.get.call(node) : value; },
        set(v) {
          log.push(`${label}.${prop}=`);
          if (desc && desc.set) desc.set.call(node, v); else value = v;
        },
      });
    }
    const list = node.classList;
    for (const method of ['add', 'remove', 'toggle']) {
      const orig = list[method].bind(list);
      list[method] = (...args) => {
        const before = list.toString();
        const out = orig(...args);
        if (list.toString() !== before) log.push(`${label}.classList.${method}(${args[0]})`);
        return out;
      };
    }
  }
}

const FAILING = {
  model_loaded: false, model_load_error: 'CUDA out of memory', worker_alive: false, engine: 'fake',
  device: 'cuda', gpu_name: 'NVIDIA GeForce RTX 4090', provider: 'local-sidecar',
  provider_health: { status: 'error', error: 'connection refused' },
};

test('같은 health 응답을 다시 받으면 배지·업로드 안내를 전혀 다시 쓰지 않는다', (t) => {
  setup(t, [{}]);
  for (const data of [FAILING, { model_loaded: true, worker_alive: true, device: 'mlx', gpu_name: 'Apple M4 Max' }]) {
    renderHealth(data);
    const nodes = elementsUnder(el.healthBadges);
    const noticeText = el.uploadModelNoticeText.firstChild;
    const log = [];
    watchWrites(el.healthBadges, log);
    watchWrites(el.uploadModelNotice, log);
    watchWrites(el.uploadModelNoticeText, log);
    watchWrites(el.uploadModelNoticeSpinner, log);
    renderHealth(data);
    renderHealth({ ...data });
    assert.deepEqual(log, [], `다시 쓴 곳: ${log.join(', ')}`);
    const after = elementsUnder(el.healthBadges);
    assert.ok(after.length === nodes.length && after.every((n, i) => n === nodes[i]), '배지 노드가 그대로다');
    assert.ok(el.uploadModelNoticeText.firstChild === noticeText, '안내 글자 노드가 그대로다');
  }
});

test('상태가 바뀌면 바뀐 배지만 고친다 — 나머지 배지 노드는 그대로', (t) => {
  setup(t, [{}]);
  const healthy = { model_loaded: true, worker_alive: true, device: 'cuda', gpu_name: 'RTX 4090',
    provider: 'local-sidecar', provider_health: { status: 'ok' } };
  renderHealth(healthy);
  const [model, device] = el.healthBadges.children;
  assert.deepEqual(badges(), ['baidu/Unlimited-OCR', 'CUDA · RTX 4090']);
  renderHealth({ ...healthy, provider_health: { status: 'error', error: 'refused' } });
  assert.deepEqual(badges(), ['baidu/Unlimited-OCR', 'CUDA · RTX 4090', '엔진 서버 연결 안 됨']);
  assert.ok(el.healthBadges.children[0] === model && el.healthBadges.children[1] === device,
    '그대로인 배지는 다시 만들지 않는다');
  // 사유(툴팁)만 바뀌면 같은 노드의 title만 고친다 — 글자를 다시 쓰지 않는다
  const provider = el.healthBadges.children[2];
  const text = provider.firstChild;
  renderHealth({ ...healthy, provider_health: { status: 'error', error: 'timeout' } });
  assert.ok(el.healthBadges.children[2] === provider && provider.firstChild === text);
  assert.match(provider.getAttribute('title'), /timeout/);
  // 로딩 → 로드 실패: 같은 자리 배지가 새 글자로 바뀐다(전이는 한 번 알린다)
  renderHealth({ ...healthy, model_loaded: false });
  assert.ok(badges().includes('모델 로딩 중…'));
  renderHealth({ ...healthy, model_loaded: false, model_load_error: 'OOM' });
  assert.ok(badges().includes('모델 로드 실패') && !badges().includes('모델 로딩 중…'));
  assert.ok(el.healthBadges.children[0] === model, '모델 배지는 끝까지 같은 노드');
});

test('연결 실패 배지도 반복 조회마다 다시 쓰지 않고, 회복하면 정상 배지로 돌아간다', (t) => {
  setup(t, [{}]);
  renderHealthError();
  const log = [];
  watchWrites(el.healthBadges, log);
  renderHealthError();
  assert.deepEqual(log, []);
  assert.deepEqual(badges(), ['서버 연결 실패']);
  renderHealth({ model_loaded: true, device: 'cpu' });
  assert.deepEqual(badges(), ['baidu/Unlimited-OCR', 'CPU']);
});
