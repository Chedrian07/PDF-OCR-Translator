// 잡 품질 경고 표시 — 순수 정규화/페이지 링크 분해 + 실제 jobs.js 렌더(가짜 DOM).
//
//   node --test frontend/tests/job-warnings.test.mjs
//
// 서버는 실패 플레이스홀더·텍스트 레이어 복구·충실도 예산 소진·페이지 경계 불일치를
// job.warnings에 남긴다(frontend-2, pipeline-ocr-10). 예전에는 UI 어디에도 보이지 않았다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { JOB_NOTE_MAX_ITEMS, jobNotices, warningSegments } from '../js/core.js';
import { el, state } from '../js/state.js';
import {
  jobListItem, renderJobWarnings, toggleJobWarnings, updateJobListItem,
} from '../js/jobs.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

/* ---------------- 순수 ---------------- */

test('jobNotices: 문자열 경고를 정리하고 중복·빈 값·비문자열을 버린다', () => {
  assert.deepEqual(jobNotices({ warnings: [' a ', 'a', '', null, 3, 'b'] }), {
    warnings: ['a', 'b'], notices: [],
  });
  assert.deepEqual(jobNotices(null), { warnings: [], notices: [] });
  assert.deepEqual(jobNotices({ warnings: 'not-a-list' }), { warnings: [], notices: [] });
});

test('jobNotices: notices와 level=info 항목은 참고로 분리한다 (Phase 2 대비)', () => {
  const r = jobNotices({
    warnings: [{ level: 'warn', message: '3페이지: 복구' }, { level: 'info', message: '페이지별 처리' }],
    notices: ['충실도 개선 채택', { text: '참고 2' }],
  });
  assert.deepEqual(r.warnings, ['3페이지: 복구']);
  assert.deepEqual(r.notices, ['페이지별 처리', '충실도 개선 채택', '참고 2']);
});

test('jobNotices: 항목 수에 상한이 있다', () => {
  const many = Array.from({ length: JOB_NOTE_MAX_ITEMS + 50 }, (_, i) => `w${i}`);
  assert.equal(jobNotices({ warnings: many }).warnings.length, JOB_NOTE_MAX_ITEMS);
});

const pages = (text) => warningSegments(text).filter((s) => s.type === 'page').map((s) => s.page);

test('warningSegments: 실제 경고 문구의 페이지 언급을 링크 조각으로 나눈다', () => {
  assert.deepEqual(pages('3페이지: single OCR 실패 후 PDF 내장 텍스트 레이어로 복구'), [3]);
  assert.deepEqual(pages('3–5페이지: 변환 실패로 플레이스홀더 삽입 (RuntimeError: boom)'), [3]);
  assert.deepEqual(pages('충실도 미달이지만 재처리 예산이 모자라 건너뛴 페이지: 3, 5, 17 (상한 조정)'), [3, 5, 17]);
  assert.deepEqual(pages('1페이지 청크: 페이지 마커 3개 (기대 4) — 빈 페이지로 보정'), [1]);
});

test('warningSegments: 분수(2/10페이지)·개수(페이지 4개)는 링크하지 않는다', () => {
  assert.deepEqual(pages('2/10페이지 렌더에 실패해 흰 페이지로 대체했습니다 (3, 5 외 2쪽)'), []);
  assert.deepEqual(pages('result.md 페이지 경계 불일치: 분할 3개 ≠ 페이지 4개'), []);
  assert.deepEqual(pages('엔진은 페이지 단위 모델이라 문서를 페이지별로 처리했습니다'), []);
});

test('warningSegments: 조각을 이어 붙이면 원문 그대로다', () => {
  for (const text of ['3–5페이지: x', '건너뛴 페이지: 3, 5', '', '페이지 없음', '12페이지']) {
    assert.equal(warningSegments(text).map((s) => s.value).join(''), text);
  }
});

/* ---------------- 런타임: 헤더 칩·목록 ---------------- */

function setup(t) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  el.jobWarningsChip = mount(doc, 'button');
  el.jobWarnings = mount(doc, 'section');
  el.jobWarningsList = mount(doc, 'ul');
  el.jobNotices = mount(doc, 'div');
  el.jobNoticesList = mount(doc, 'ul');
  el.readerPagePane = mount(doc, 'div');
  el.resultSection = mount(doc, 'section');
  el.resultSection.hidden = true; // 리더가 아직 안 보이는 상태에서 링크를 누른다
  for (const key of ['readerContent', 'viewerRoot', 'readerOutline']) el[key] = mount(doc, 'div');
  el.readerPageInput = mount(doc, 'input');
  const readerTab = mount(doc, 'button');
  readerTab.dataset.tab = 'reader';
  el.tabs = [readerTab];
  el.panels = [];
  Object.assign(state, {
    currentJobId: 'job-a', currentLang: 'orig', openGen: 1, displayedStatus: 'done',
    jobWarningsOpen: false, readerPage: 1, readerTotalHint: 20,
    readerPages: { orig: null, ko: null }, readerOutline: { orig: null, ko: null },
    readerAlignmentRetryTimers: new Map(), readerAlignmentRetryCounts: new Map(),
    readerAlignmentBackoff: new Set(), readerAnchorRaf: 0,
  });
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('offline'); });
  t.after(() => { Object.assign(state, savedState); Object.assign(el, savedEls); });
  return { doc, readerTab };
}

const WARNINGS = [
  '3페이지: single OCR 실패 후 PDF 내장 텍스트 레이어로 복구 (이미지·정밀 레이아웃 제외)',
  '충실도 미달이지만 재처리 예산이 모자라 건너뛴 페이지: 7, 9 (OCR_FIDELITY_MAX_RETRY_RATIO로 상한 조정)',
];

test('경고가 있으면 "주의 N건" 칩을 띄우고, 누르면 목록이 펼쳐진다', (t) => {
  setup(t);
  renderJobWarnings({ job_id: 'job-a', status: 'done', warnings: WARNINGS });
  assert.equal(el.jobWarningsChip.hidden, false);
  assert.equal(el.jobWarningsChip.textContent, '주의 2건');
  assert.ok(el.jobWarningsChip.classList.contains('chip-warn'));
  assert.equal(el.jobWarnings.hidden, true, '처음에는 접혀 있다');
  assert.equal(el.jobWarningsChip.getAttribute('aria-expanded'), 'false');
  toggleJobWarnings();
  assert.equal(el.jobWarnings.hidden, false);
  assert.equal(el.jobWarningsChip.getAttribute('aria-expanded'), 'true');
  const items = el.jobWarningsList.querySelectorAll('li');
  assert.equal(items.length, 2);
  assert.equal(items.map((li) => li.textContent).join('\n'), WARNINGS.join('\n'));
  assert.deepEqual(el.jobWarningsList.querySelectorAll('.warning-page-link').map((b) => b.textContent),
    ['3페이지', '7', '9']);
  assert.equal(el.jobNotices.hidden, true);
});

test('"N페이지" 링크는 읽기 탭을 열고 그 페이지로 간다', (t) => {
  const { readerTab } = setup(t);
  renderJobWarnings({ job_id: 'job-a', status: 'done', warnings: WARNINGS });
  toggleJobWarnings();
  el.jobWarningsList.querySelectorAll('.warning-page-link')[1].click(); // '7'
  assert.equal(state.readerPage, 7);
  assert.ok(readerTab.classList.contains('active'), '읽기 탭이 활성화된다');
  assert.equal(document.activeElement, el.readerPagePane, '포커스가 리더 원문 면으로 간다');
});

test('완료 전 잡의 경고는 평문으로 보인다 (리더가 없으므로 링크 없음)', (t) => {
  setup(t);
  renderJobWarnings({ job_id: 'job-a', status: 'canceled', warnings: WARNINGS });
  assert.equal(el.jobWarningsList.querySelectorAll('.warning-page-link').length, 0);
  assert.equal(el.jobWarningsList.querySelectorAll('li').length, 2);
});

test('참고(notices)만 있으면 흐린 "참고 N건" 칩과 별도 목록', (t) => {
  setup(t);
  renderJobWarnings({ job_id: 'job-a', status: 'done', warnings: [], notices: ['페이지별로 처리했습니다'] });
  assert.equal(el.jobWarningsChip.textContent, '참고 1건');
  assert.ok(el.jobWarningsChip.classList.contains('chip-note'));
  assert.ok(!el.jobWarningsChip.classList.contains('chip-warn'));
  assert.equal(el.jobNotices.hidden, false);
  assert.equal(el.jobNoticesList.querySelectorAll('li').length, 1);
  assert.equal(el.jobWarningsList.querySelectorAll('li').length, 0);
});

test('경고가 없는 잡으로 바뀌면 칩과 목록을 숨긴다', (t) => {
  setup(t);
  renderJobWarnings({ job_id: 'job-a', status: 'done', warnings: WARNINGS });
  toggleJobWarnings();
  renderJobWarnings({ job_id: 'job-b', status: 'done', warnings: [] });
  assert.equal(el.jobWarningsChip.hidden, true);
  assert.equal(el.jobWarnings.hidden, true);
  assert.equal(el.jobWarningsList.querySelectorAll('li').length, 0);
});

test('작업 목록 줄에도 경고 개수를 표시하고, 경고가 사라지면 지운다', (t) => {
  setup(t);
  const item = jobListItem({ job_id: 'job-a', filename: 'a.pdf', status: 'done', warnings: WARNINGS });
  assert.equal(item.querySelector('.ji-warn').textContent, '주의 2');
  updateJobListItem(item, { job_id: 'job-a', filename: 'a.pdf', status: 'done', warnings: [WARNINGS[0]] });
  assert.equal(item.querySelectorAll('.ji-warn').length, 1);
  assert.equal(item.querySelector('.ji-warn').textContent, '주의 1');
  updateJobListItem(item, { job_id: 'job-a', filename: 'a.pdf', status: 'done', warnings: [] });
  assert.equal(item.querySelector('.ji-warn'), null);
});
