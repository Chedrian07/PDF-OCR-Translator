// health 재조회 + 로드 실패·워커 중지 표시 — 순수 정책 + 실제 js/health.js(가짜 DOM).
//
//   node --test frontend/tests/health-runtime.test.mjs
//
// 예전에는 한 번 정상이면 health를 다시 묻지 않아 나중에 죽은 sidecar·워커가 배지로 안
// 보였고, model_load_error·worker_alive를 무시해 프리로드 실패가 '모델 로딩 중…'과
// '로딩이 끝나는 대로 자동 변환' 안내로 끝없이 보였다(frontend-8).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  HEALTH_POLL_FAST_MS, HEALTH_POLL_SLOW_MS, healthPollDelay, healthStatus,
} from '../js/core.js';
import { EL_IDS, el, state } from '../js/state.js';
import { loadHealth, renderHealth, setupHealthPolling } from '../js/health.js';
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
