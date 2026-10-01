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
//  · '더 보기': 최신 50건 뒤를 before 커서로 한 쪽씩 잇고, 폴링은 넓어진 창을 유지한다.
//    커서 잡이 지워지면(422) 처음부터 다시 받는다. 서버 limit 상한(500)을 넘는 창은 커서로 잇는다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  JOB_LIST_PAGE, appendJobPage, jobListMoreLabel, jobListUrl, jobRowSignature, normalizeJobPage,
} from '../js/core.js';
import { armTimers, el, state } from '../js/state.js';
import { loadMoreJobs, pollJobs, refreshJobs, renderJobList } from '../js/jobs.js';
import { assertSameNode, installFakeDom, mount } from './helpers/fake-dom.mjs';

function job(id, status = 'done', extra = {}) {
  return { job_id: id, filename: `${id}.pdf`, status, created_at: '2026-10-01T09:00:00Z', ...extra };
}

function setup(t, jobs) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  el.jobList = mount(doc, 'ul');
  el.jobListMore = mount(doc, 'button');
  el.jobListEmpty = mount(doc, 'p');
  el.toast = mount(doc, 'div');
  Object.assign(state, {
    jobs, currentJobId: null, toastTimer: 0,
    jobListLimit: JOB_LIST_PAGE, jobsHasMore: false, jobsTotal: null, jobsLoadingMore: false,
  });
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
  assertSameNode(assert, doc.activeElement, del, '포커스가 그대로');
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
  assertSameNode(assert, rows()[1], item, '같은 <li>를 재사용한다');
  assertSameNode(assert, control('b', 'ji-del'), del, '같은 삭제 버튼을 재사용한다');
  assertSameNode(assert, doc.activeElement, del, '키보드 포커스가 body로 날아가지 않는다');
  assert.ok(del.classList.contains('armed'), '무장이 재렌더를 넘어 유지된다');
  assert.equal(item.querySelector('.chip').textContent, '완료');
  assert.ok(item.querySelector('.ji-read'), '완료되면 바로 읽기 버튼이 생긴다');
  assertSameNode(assert, armTimers.get('b').btn, del, '무장 타이머가 현재 버튼을 가리킨다');
});

test('줄 순서가 바뀌어 포커스가 빠지면 같은 잡의 같은 컨트롤로 돌려준다', (t) => {
  const doc = setup(t, [job('a'), job('b'), job('c')]);
  renderJobList();
  control('c', 'ji-del').focus();
  state.jobs = [job('c'), job('a'), job('b')];
  renderJobList();
  assert.deepEqual(ids(), ['c', 'a', 'b']);
  assertSameNode(assert, doc.activeElement, control('c', 'ji-del'), '같은 잡의 삭제 버튼으로 복원');
});

test('새 잡이 맨 위에 생겨도 기존 줄은 옮기지 않는다', (t) => {
  setup(t, [job('a'), job('b')]);
  renderJobList();
  const before = rows();
  state.jobs = [job('n', 'queued'), job('a'), job('b')];
  renderJobList();
  assert.deepEqual(ids(), ['n', 'a', 'b']);
  assertSameNode(assert, rows()[1], before[0], '기존 첫 줄 재사용');
  assertSameNode(assert, rows()[2], before[1], '기존 둘째 줄 재사용');
});

test('포커스된 잡이 목록에서 사라지면 같은 자리 줄의 열기 버튼으로 넘긴다', (t) => {
  const doc = setup(t, [job('a'), job('b'), job('c')]);
  renderJobList();
  control('b', 'ji-del').focus();
  state.jobs = [job('a'), job('c')];
  renderJobList();
  assert.deepEqual(ids(), ['a', 'c']);
  assertSameNode(assert, doc.activeElement, control('c', 'ji-open'), '같은 자리 줄의 열기 버튼');
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

/* ---------------- '더 보기' 페이지 ---------------- */

test('jobListUrl: 기본 창은 예전 URL, 그 밖은 limit(1–500)과 before 커서', () => {
  assert.equal(jobListUrl(), '/api/jobs');
  assert.equal(jobListUrl(50), '/api/jobs');
  assert.equal(jobListUrl(100), '/api/jobs?limit=100');
  assert.equal(jobListUrl(50, 'j_abc'), '/api/jobs?limit=50&before=j_abc');
  assert.equal(jobListUrl(9999), '/api/jobs?limit=500', '서버 상한으로 묶는다');
  assert.equal(jobListUrl(0), '/api/jobs', '비정상 값은 기본 창');
  assert.equal(jobListUrl(10, 'a&b=c'), '/api/jobs?limit=10&before=a%26b%3Dc', '커서는 인코딩');
});

test('normalizeJobPage·appendJobPage·jobListMoreLabel: 구버전 응답과 겹친 경계를 견딘다', () => {
  assert.deepEqual(normalizeJobPage({ jobs: [job('a')] }), { jobs: [job('a')], hasMore: false, total: null });
  assert.deepEqual(normalizeJobPage({ jobs: [job('a'), null, { status: 'done' }], has_more: true, total: '7' }),
    { jobs: [job('a')], hasMore: true, total: 7 });
  assert.deepEqual(normalizeJobPage(null), { jobs: [], hasMore: false, total: null });
  const merged = appendJobPage([job('a'), job('b')], [job('b'), job('c')]);
  assert.deepEqual(merged.map((j) => j.job_id), ['a', 'b', 'c'], '경계에 걸친 잡은 한 번만');
  assert.equal(jobListMoreLabel(50, 132), '더 보기 (50/132)');
  assert.equal(jobListMoreLabel(50, null), '더 보기');
  assert.equal(jobListMoreLabel(50, 50), '더 보기');
});

const many = (from, count) => Array.from({ length: count }, (_, i) => job(`j${String(from + i).padStart(4, '0')}`));

// 서버 흉내 — 최신순 목록에서 limit·before로 자른다(api.list_jobs와 같은 규칙).
function serverJobs(t, all, { missingCursor = () => false } = {}) {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    calls.push(String(url));
    const u = new URL(String(url), 'http://x');
    const limit = Number(u.searchParams.get('limit') || 50);
    const before = u.searchParams.get('before');
    let rest = all();
    if (before != null) {
      const at = rest.findIndex((j) => j.job_id === before);
      if (at < 0 || missingCursor(before)) {
        return { ok: false, status: 422, headers: { get: () => null }, text: async () => '{"detail":"before"}' };
      }
      rest = rest.slice(at + 1);
    }
    const body = { jobs: rest.slice(0, limit), has_more: rest.length > limit, total: all().length };
    return { ok: true, status: 200, headers: { get: () => null }, text: async () => JSON.stringify(body) };
  });
  return calls;
}

test("'더 보기': 마지막 잡 다음 한 쪽을 before로 잇고 폴링은 넓어진 창을 유지한다", async (t) => {
  setup(t, []);
  const all = many(0, 120);
  const calls = serverJobs(t, () => all);
  await refreshJobs();
  assert.equal(rows().length, 50);
  assert.equal(el.jobListMore.hidden, false, '뒤에 더 있으면 버튼이 보인다');
  assert.equal(el.jobListMore.textContent, '더 보기 (50/120)');
  const firstRows = rows();
  await loadMoreJobs();
  assert.equal(calls.at(-1), '/api/jobs?limit=50&before=j0049', '마지막 잡 다음부터');
  assert.equal(rows().length, 100);
  assertSameNode(assert, rows()[0], firstRows[0], '앞 줄은 다시 만들지 않는다');
  assert.equal(el.jobListMore.textContent, '더 보기 (100/120)');
  await refreshJobs(); // 5초 폴링
  assert.equal(calls.at(-1), '/api/jobs?limit=100', '넓어진 창을 그대로 다시 받는다');
  assert.equal(rows().length, 100);
  await loadMoreJobs();
  assert.equal(rows().length, 120);
  assert.equal(el.jobListMore.hidden, true, '끝까지 받으면 버튼이 사라진다');
});

test("'더 보기' 커서 잡이 그사이 지워지면(422) 넓힌 창을 처음부터 다시 받는다", async (t) => {
  setup(t, []);
  let all = many(0, 80);
  const calls = serverJobs(t, () => all);
  await refreshJobs();
  all = all.filter((j) => j.job_id !== 'j0049'); // 다른 탭이 목록 끝 잡을 지웠다
  await loadMoreJobs();
  assert.deepEqual(calls.slice(-2), ['/api/jobs?limit=50&before=j0049', '/api/jobs?limit=100']);
  assert.equal(rows().length, 79);
  assert.ok(!ids().includes('j0049'));
  assert.equal(el.jobListMore.hidden, true);
  assert.doesNotMatch(el.toast.textContent || '', /불러오지 못했습니다/, '422는 오류로 알리지 않는다');
});

test("'더 보기' 응답을 기다리는 사이 목록 끝이 밀리면 틈 없이 처음부터 다시 받는다", async (t) => {
  setup(t, []);
  let all = many(1, 120);
  const calls = serverJobs(t, () => all);
  await refreshJobs();
  const more = loadMoreJobs();
  all = [job('j0000'), ...all]; // 새 업로드 — 50건 창의 끝(j0050)이 51번째로 밀린다
  await refreshJobs();          // 그사이 폴링이 창을 다시 받았다
  await more;
  assert.equal(calls.at(-1), '/api/jobs?limit=100');
  assert.deepEqual(ids().slice(48, 52), ['j0048', 'j0049', 'j0050', 'j0051'], '경계에 틈이 없다');
});

test('서버 limit 상한(500)을 넘는 창은 before 커서로 이어 받는다', async (t) => {
  setup(t, []);
  const all = many(0, 620);
  const calls = serverJobs(t, () => all);
  state.jobListLimit = 550;
  await refreshJobs();
  assert.deepEqual(calls, ['/api/jobs?limit=500', '/api/jobs?limit=50&before=j0499']);
  assert.equal(rows().length, 550);
  assert.equal(state.jobsHasMore, true);
});

test("구버전 서버(has_more 없음)에는 '더 보기'를 보이지 않는다", async (t) => {
  setup(t, []);
  t.mock.method(globalThis, 'fetch', async () => ({
    ok: true, status: 200, headers: { get: () => null },
    text: async () => JSON.stringify({ jobs: many(0, 50) }),
  }));
  await refreshJobs();
  assert.equal(rows().length, 50);
  assert.equal(el.jobListMore.hidden, true);
});

test("'더 보기'로 끝까지 받으면 포커스를 새로 붙은 첫 줄로 넘긴다", async (t) => {
  const doc = setup(t, []);
  serverJobs(t, () => many(0, 60));
  await refreshJobs();
  el.jobListMore.focus();
  const loading = loadMoreJobs();
  assert.equal(el.jobListMore.getAttribute('aria-busy'), 'true', '받는 중 표시');
  assert.equal(el.jobListMore.disabled, false, '비활성으로 바꿔 포커스를 잃게 하지 않는다');
  await loadMoreJobs(); // 연타 — 두 번째 요청은 보내지 않는다
  await loading;
  assert.equal(rows().length, 60);
  assert.equal(el.jobListMore.hidden, true);
  assertSameNode(assert, doc.activeElement, control('j0050', 'ji-open'), '새 첫 줄의 열기 버튼');
});
