// 번역 참고 사항(warnings) — GET /translate/state가 report.json의 warnings(문자열 목록)를
// 함께 주면 결과 화면 번역 요약 아래에 흐린 접이식 목록으로 보인다.
//
//   node --test frontend/tests/translate-warnings.test.mjs
//
// 지키는 계약:
//  · warnings가 없거나(구버전 서버) 비정상이면 목록을 숨긴다 — 빈 상자를 남기지 않는다.
//  · 문자열(또는 {message|text})만 받고, 같은 문장은 한 번만, 개수는 상한까지.
//  · 요약 칩의 상세(title·클릭 토스트)에는 개수만 — 긴 문장은 목록에서 읽는다.
//  · SSE done(개수만)·잡 전환(reset)은 목록을 비우고, state 조회가 다시 채운다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  TRANSLATE_WARNING_MAX_ITEMS, translateKeptSummary, translateWarnings,
} from '../js/core.js';
import { EL_IDS, el, state } from '../js/state.js';
import {
  initTranslateForJob, renderTranslateSummary, resetTranslateUI, teardownTranslate,
} from '../js/translate.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

const CACHE_NOTE = '기존 캐시 12건이 하나도 적중하지 않아 유닛 40개를 전량 재번역했습니다';
const REF_NOTE = '참고문헌 규칙 불일치 — 같은 원문이 한쪽 경로에서만 원문 유지됩니다';
const STATE = {
  lang: 'ko', status: 'done', current: 9, total: 9,
  skip_reasons: { references: 2 }, kept_reasons: {}, warnings: [CACHE_NOTE, REF_NOTE],
};

test('translateWarnings: 필드 부재·비정상 값은 빈 목록, 문자열만 중복 없이', () => {
  assert.deepEqual(translateWarnings({ status: 'done' }), []);
  assert.deepEqual(translateWarnings(null), []);
  assert.deepEqual(translateWarnings({ warnings: 'not a list' }), []);
  assert.deepEqual(
    translateWarnings({ warnings: [` ${CACHE_NOTE} `, CACHE_NOTE, '', 7, null, { message: REF_NOTE }] }),
    [CACHE_NOTE, REF_NOTE],
  );
  const flood = Array.from({ length: TRANSLATE_WARNING_MAX_ITEMS + 9 }, (_, i) => `참고 ${i}`);
  assert.equal(translateWarnings({ warnings: flood }).length, TRANSLATE_WARNING_MAX_ITEMS);
});

test('translateKeptSummary: 참고 사항은 상세에 개수만 더한다', () => {
  const s = translateKeptSummary(STATE);
  assert.ok(s.detail.includes('번역 참고 사항 2건'), s.detail);
  assert.ok(!s.detail.includes(CACHE_NOTE), '긴 문장은 칩 상세에 넣지 않는다');
  assert.equal(s.tone, '', '참고 사항만으로 경고 톤이 되지 않는다');
  const plain = translateKeptSummary({ ...STATE, warnings: undefined });
  assert.ok(!plain.detail.includes('참고 사항'), plain.detail);
});

function setup(t) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const key of Object.keys(EL_IDS)) {
    const tag = key === 'translateWarnings' ? 'details'
      : key === 'translateWarningsSummary' ? 'summary'
        : key === 'translateWarningsList' ? 'ul' : 'div';
    el[key] = mount(doc, tag, EL_IDS[key]);
  }
  el.tabs = [];
  el.panels = [];
  Object.assign(state, {
    currentJobId: 'job-a', displayedStatus: 'done', translateGen: 0, translateState: 'none',
    translateEs: null, translatePollTimer: 0, translateSseErrors: 0, toastTimer: 0,
    viewerOpen: false,
  });
  t.after(() => {
    teardownTranslate();
    clearTimeout(state.toastTimer);
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return doc;
}

const items = () => el.translateWarningsList.children.map((li) => li.textContent);

test('번역 요약 아래 참고 사항 목록 — 있으면 보이고, done 이벤트·잡 전환에서는 비운다', (t) => {
  setup(t);
  renderTranslateSummary(STATE);
  assert.equal(el.translateWarnings.hidden, false);
  assert.equal(el.translateWarningsSummary.textContent, '번역 참고 사항 2건');
  assert.deepEqual(items(), [CACHE_NOTE, REF_NOTE]);
  assert.equal(el.translateSummary.hidden, false, '요약 칩도 함께');

  // SSE done 페이로드에는 개수만 있다 — 옛 목록을 남기지 않고, 이어지는 state 조회가 채운다.
  renderTranslateSummary({ counts: { total: 9, skipped: 2, kept_original: 0 } });
  assert.equal(el.translateWarnings.hidden, true);
  assert.deepEqual(items(), []);

  renderTranslateSummary(STATE);
  resetTranslateUI(); // 다른 잡으로 전환
  assert.equal(el.translateWarnings.hidden, true);
  assert.deepEqual(items(), []);
});

test('구버전 서버(warnings 없음)에는 빈 목록 상자를 보이지 않는다', (t) => {
  setup(t);
  renderTranslateSummary({ ...STATE, warnings: undefined });
  assert.equal(el.translateWarnings.hidden, true);
  assert.equal(el.translateSummary.hidden, false);
});

test('번역이 끝난 잡을 열면 translate/state의 warnings가 목록으로 보인다', async (t) => {
  setup(t);
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    calls.push(String(url));
    return {
      ok: true, status: 200, headers: { get: () => null },
      text: async () => JSON.stringify(STATE),
    };
  });
  await initTranslateForJob();
  assert.ok(calls.includes('/api/jobs/job-a/translate/state?lang=ko'), calls.join(', '));
  assert.equal(state.translateState, 'done');
  assert.equal(el.translateWarnings.hidden, false);
  assert.deepEqual(items(), [CACHE_NOTE, REF_NOTE]);
});
