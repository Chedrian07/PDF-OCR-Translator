// 번역 결과 요약 — "왜 이 문단이 영어로 남았나"를 사용자에게 보여 주는 순수 코어.
//
//   node --test frontend/tests/translate-report.test.mjs
//   cd frontend && npm test
//
// 서버 계약(backend/app/api.py · backend/app/translate/engine.py):
//  · GET /api/jobs/{id}/translate/state?lang=ko 는 state.json에 report.json의
//    skip_reasons·kept_reasons·reference_rule을 병합해 준다(_TRANSLATE_REASON_KEYS).
//  · GET /api/jobs/{id}/translate/report 는 report.json 전체(kept_original 목록 포함).
//  · SSE done 이벤트에는 counts{total,translated,cached,skipped,kept_original}가 실린다.
// 어느 페이로드가 오든 같은 요약으로 접혀야 하고, 근거가 없으면 아무 말도 하면 안 된다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { translateKeptSummary } from '../app.js';

// 실제 report.json 형태 (backend/tests/test_api_translate.py::_REPORT와 같은 모양)
const STATE_WITH_REASONS = {
  lang: 'ko',
  status: 'done',
  current: 9,
  total: 9,
  error: null,
  skip_reasons: { references: 2, 'already-korean': 1 },
  kept_reasons: { 'gate-rejected': 1 },
  reference_rule: { md_only: 0, layout_only: 4 },
};

test('translateKeptSummary: 원문 유지 개수와 사유를 함께 보여 준다', () => {
  const s = translateKeptSummary(STATE_WITH_REASONS);
  assert.equal(s.kept, 1);
  assert.equal(s.skipped, 3);
  assert.equal(s.text, '원문 유지 1 · 건너뜀 3');
  assert.ok(s.detail.includes('번역 품질 기준 미달 1'), s.detail);
  assert.ok(s.detail.includes('참고문헌 2'), s.detail);
  assert.ok(s.detail.includes('이미 한국어 1'), s.detail);
  assert.ok(s.detail.includes('총 9개 문단'), s.detail);
  assert.equal(s.tone, 'warn', '번역 실패로 남은 문단이 있으면 경고 톤');
});

test('translateKeptSummary: SSE done의 counts만으로도 개수를 낸다', () => {
  // 사유별 집계가 도착하기 전(done 이벤트 직후)에도 숫자는 즉시 보여야 한다.
  const s = translateKeptSummary({
    phase: 'translate',
    lang: 'ko',
    counts: { total: 40, translated: 34, cached: 0, skipped: 5, kept_original: 1 },
  });
  assert.equal(s.kept, 1);
  assert.equal(s.skipped, 5);
  assert.equal(s.total, 40);
  assert.equal(s.text, '원문 유지 1 · 건너뜀 5');
  assert.ok(s.detail.includes('원문 그대로 남은 문단 1개'), s.detail);
});

test('translateKeptSummary: counts와 사유별 집계 중 큰 쪽을 쓴다 (개수를 잃지 않는다)', () => {
  const s = translateKeptSummary({
    counts: { total: 10, skipped: 0, kept_original: 0 },
    kept_reasons: { 'empty-output': 2, 'api-rejected': 1 },
    skip_reasons: { references: 4 },
  });
  assert.equal(s.kept, 3);
  assert.equal(s.skipped, 4);
  assert.ok(s.detail.indexOf('빈 응답 2') < s.detail.indexOf('API 거절 1'), '많은 사유 먼저');
});

test('translateKeptSummary: 건너뜀만 있으면 경고 톤이 아니다 (참고문헌 등은 정상)', () => {
  const s = translateKeptSummary({ skip_reasons: { references: 7 }, kept_reasons: {} });
  assert.equal(s.kept, 0);
  assert.equal(s.text, '건너뜀 7');
  assert.equal(s.tone, '');
  assert.ok(s.detail.includes('의도적으로 번역하지 않은 문단 7개'), s.detail);
});

test('translateKeptSummary: 전부 번역됐으면 그렇게 말한다', () => {
  const s = translateKeptSummary({ skip_reasons: {}, kept_reasons: {} });
  assert.equal(s.text, '전부 번역됨');
  assert.equal(s.tone, '');
  assert.equal(s.kept, 0);
  assert.equal(s.skipped, 0);
});

test('translateKeptSummary: 미지의 사유 키도 개수를 잃지 않고 그대로 노출', () => {
  const s = translateKeptSummary({ kept_reasons: { unknown: 1, 'future-reason': 2 } });
  assert.equal(s.kept, 3);
  assert.ok(s.detail.includes('future-reason 2'), s.detail);
  assert.ok(s.detail.includes('원인 미상 1'), s.detail);
});

test('translateKeptSummary: 근거가 없으면 null — 거짓 안심을 주지 않는다', () => {
  // report.json이 없는 구버전 잡의 state에는 사유 키가 아예 없다.
  assert.equal(translateKeptSummary({ lang: 'ko', status: 'done', total: 9 }), null);
  assert.equal(translateKeptSummary(null), null);
  assert.equal(translateKeptSummary(undefined), null);
  assert.equal(translateKeptSummary('done'), null);
});

test('translateKeptSummary: 비정상 값(음수·문자열·NaN)은 0으로 방어', () => {
  const s = translateKeptSummary({
    counts: { total: 'x', skipped: -3, kept_original: NaN },
    kept_reasons: { 'gate-rejected': -1, 'empty-output': 'two' },
    skip_reasons: null,
  });
  assert.equal(s.kept, 0);
  assert.equal(s.skipped, 0);
  assert.equal(s.total, 0);
  assert.equal(s.text, '전부 번역됨');
  assert.ok(!s.detail.includes('총 0개'), s.detail);
});
