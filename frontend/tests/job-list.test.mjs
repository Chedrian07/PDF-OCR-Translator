// 작업 목록 증분 렌더 + 목록 폴링 직렬화 — 실제 js/jobs.js를 가짜 DOM에서 돌린다.
//
//   node --test frontend/tests/job-list.test.mjs
//
// 지키는 계약 (frontend-4, frontend-12):
//  · 같은 데이터로 다시 렌더하면 DOM을 전혀 건드리지 않는다(포커스·스크린리더 위치 보존).
//  · 바뀐 줄은 같은 <li>·같은 버튼을 제자리에서 고친다 — 키보드 포커스와 2단계 삭제
//    무장(armed)이 5초 폴링을 넘어 유지된다.
//  · 줄이 옮겨지거나 지워져 포커스가 빠지면 같은 잡(없으면 같은 자리)으로 돌려준다.
//  · 목록 갱신은 한 번에 하나, 주기 폴링은 진행 중이면 건너뛴다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { jobRowSignature } from '../js/core.js';
import { armTimers, el, state } from '../js/state.js';
import { pollJobs, refreshJobs, renderJobList } from '../js/jobs.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

function job(id, status = 'done', extra = {}) {
  return { job_id: id, filename: `${id}.pdf`, status, created_at: '2026-10-01T09:00:00Z', ...extra };
}

function setup(t, jobs) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  el.jobList = mount(doc, 'ul');
  el.jobListEmpty = mount(doc, 'p');
  el.toast = mount(doc, 'div');
  Object.assign(state, { jobs, currentJobId: null, toastTimer: 0 });
  t.after(() => {
    for (const entry of armTimers.values()) clearTimeout(entry.t);
    armTimers.clear();
    clearTimeout(state.toastTimer);
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return doc;
}

const rows = () => [...el.jobList.children];
const ids = () => rows().map((li) => li.dataset.jobId);
const control = (id, cls) => rows().find((li) => li.dataset.jobId === id).querySelector(`.${cls}`);

function countMutations(t, node) {
  const counter = { n: 0 };
  for (const name of ['appendChild', 'insertBefore', 'removeChild']) {
    const original = node[name].bind(node);
    node[name] = (...args) => { counter.n += 1; return original(...args); };
  }
  t.after(() => { for (const name of ['appendChild', 'insertBefore', 'removeChild']) delete node[name]; });
  return counter;
}

test('jobRowSignature: 표시 필드가 바뀔 때만 달라진다', () => {
  const a = job('a', 'running');
  assert.equal(jobRowSignature(a, false), jobRowSignature({ ...a, progress: { current_page: 3 } }, false));
  assert.notEqual(jobRowSignature(a, false), jobRowSignature(a, true));
  assert.notEqual(jobRowSignature(a, false), jobRowSignature({ ...a, status: 'done' }, false));
  assert.notEqual(jobRowSignature(job('q', 'queued', { queue_position: 1 }), false),
    jobRowSignature(job('q', 'queued', { queue_position: 2 }), false));
});

test('같은 데이터로 다시 렌더하면 DOM을 건드리지 않고 포커스가 남는다', (t) => {
  const doc = setup(t, [job('a'), job('b'), job('c')]);
  renderJobList();
  assert.deepEqual(ids(), ['a', 'b', 'c']);
  const del = control('b', 'ji-del');
  del.focus();
  const mutations = countMutations(t, el.jobList);
  state.jobs = [job('a'), job('b'), job('c')]; // 폴링 — 새 객체, 같은 내용
  renderJobList();
  assert.equal(mutations.n, 0);
  assert.equal(doc.activeElement, del);
});

test('상태가 바뀐 줄은 제자리에서 갱신되고 포커스·삭제 무장이 유지된다', (t) => {
  const doc = setup(t, [job('a'), job('b', 'running')]);
  renderJobList();
  const item = rows()[1];
  const del = control('b', 'ji-del');
  del.focus();
  del.click(); // 1단계 — 무장
  assert.ok(del.classList.contains('armed'));
  state.jobs = [job('a'), job('b', 'done')];
  renderJobList();
  assert.equal(rows()[1], item, '같은 <li>를 재사용한다');
  assert.equal(control('b', 'ji-del'), del, '같은 삭제 버튼을 재사용한다');
  assert.equal(doc.activeElement, del, '키보드 포커스가 body로 날아가지 않는다');
  assert.ok(del.classList.contains('armed'), '무장이 재렌더를 넘어 유지된다');
  assert.equal(item.querySelector('.chip').textContent, '완료');
  assert.ok(item.querySelector('.ji-read'), '완료되면 바로 읽기 버튼이 생긴다');
  assert.equal(armTimers.get('b').btn, del);
});

test('줄 순서가 바뀌어 포커스가 빠지면 같은 잡의 같은 컨트롤로 돌려준다', (t) => {
  const doc = setup(t, [job('a'), job('b'), job('c')]);
  renderJobList();
  control('c', 'ji-del').focus();
  state.jobs = [job('c'), job('a'), job('b')];
  renderJobList();
  assert.deepEqual(ids(), ['c', 'a', 'b']);
  assert.equal(doc.activeElement, control('c', 'ji-del'));
});

test('새 잡이 맨 위에 생겨도 기존 줄은 옮기지 않는다', (t) => {
  setup(t, [job('a'), job('b')]);
  renderJobList();
  const before = rows();
  state.jobs = [job('n', 'queued'), job('a'), job('b')];
  renderJobList();
  assert.deepEqual(ids(), ['n', 'a', 'b']);
  assert.equal(rows()[1], before[0]);
  assert.equal(rows()[2], before[1]);
});

test('포커스된 잡이 목록에서 사라지면 같은 자리 줄의 열기 버튼으로 넘긴다', (t) => {
  const doc = setup(t, [job('a'), job('b'), job('c')]);
  renderJobList();
  control('b', 'ji-del').focus();
  state.jobs = [job('a'), job('c')];
  renderJobList();
  assert.deepEqual(ids(), ['a', 'c']);
  assert.equal(doc.activeElement, control('c', 'ji-open'));
});

test('빈 목록이면 안내 문구를 보이고 줄을 모두 지운다', (t) => {
  setup(t, [job('a')]);
  renderJobList();
  assert.equal(el.jobListEmpty.hidden, true);
  state.jobs = [];
  renderJobList();
  assert.equal(rows().length, 0);
  assert.equal(el.jobListEmpty.hidden, false);
});

test('키보드 2단계 삭제: 폴링 재렌더 뒤 두 번째 Enter가 실제로 삭제한다', async (t) => {
  setup(t, [job('a'), job('b')]);
  renderJobList();
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push(`${(init && init.method) || 'GET'} ${url}`);
    if (init && init.method === 'DELETE') return { ok: true, status: 204, headers: { get: () => null }, text: async () => '' };
    return { ok: true, status: 200, headers: { get: () => null }, text: async () => '{"jobs":[]}' };
  });
  const del = control('b', 'ji-del');
  del.focus();
  del.click();
  state.jobs = [job('a'), job('b')]; // 무장 창 안에 5초 틱이 온다
  renderJobList();
  document.activeElement.click(); // 같은 포커스에서 두 번째 Enter
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(calls.includes('DELETE /api/jobs/b'), calls.join(', '));
});

test('목록 갱신은 한 번에 하나 — 진행 중 명시적 갱신은 한 번 더, 주기 폴링은 건너뛴다', async (t) => {
  setup(t, []);
  const pending = [];
  t.mock.method(globalThis, 'fetch', () => new Promise((resolve) => { pending.push(resolve); }));
  const reply = (jobs) => ({
    ok: true, status: 200, headers: { get: () => null },
    text: async () => JSON.stringify({ jobs }),
  });
  const first = refreshJobs();
  pollJobs();
  pollJobs();
  assert.equal(pending.length, 1, '주기 폴링은 겹쳐 보내지 않는다');
  const again = refreshJobs(); // 삭제·업로드 직후의 명시적 갱신
  assert.equal(again, first, '진행 중인 갱신에 합류한다');
  assert.equal(pending.length, 1);
  pending[0](reply([job('old')]));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(pending.length, 2, '명시적 갱신은 끝난 뒤 한 번 더 돈다');
  pending[1](reply([job('new')]));
  await first;
  assert.deepEqual(ids(), ['new']);
  // 끝난 뒤에는 다시 새 갱신을 시작할 수 있다.
  const next = pollJobs();
  assert.notEqual(next, first);
  assert.equal(pending.length, 3);
  pending[2](reply([job('new')]));
  await next;
});
