// docs/ARCHITECTURE.md의 SSE 폴백 설명 ↔ 실제 js/sse.js·js/translate.js 타이머 (감사 frontend-10).
//
//   node --test frontend/tests/docs-sse-fallback.test.mjs
//
// 문서는 '번역 SSE도 같은 규칙'(재승격)이라고 적었지만, 번역은 상태 폴링을 번역이 끝날 때까지
// 유지하고 재승격하지 않는다. 숫자는 실제 코드를 돌려 얻는다 — 타이머를 바꾸면 이 테스트가
// 문서 갱신을 요구한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { state } from '../js/state.js';
import {
  clearSsePromote, scheduleSsePromote, startFallbackPolling, stopFallbackPolling,
} from '../js/sse.js';
import { handleTranslateConnError, teardownTranslate } from '../js/translate.js';

const ARCH = fs.readFileSync(new URL('../../docs/ARCHITECTURE.md', import.meta.url), 'utf8');

// '- SSE 폴백:' 항목(다음 형제 항목 전까지)을 공백 하나로 편 한 줄.
function fallbackNote() {
  const start = ARCH.indexOf('- SSE 폴백:');
  assert.ok(start >= 0, 'docs/ARCHITECTURE.md에 "SSE 폴백" 항목이 없다');
  const end = ARCH.indexOf('\n  - ', start);
  return ARCH.slice(start, end === -1 ? undefined : end).replace(/\s+/g, ' ');
}

const sec = (ms) => `${ms / 1000}초`;

// 타이머를 걸지 않고 지연(ms)만 기록한다 — 콜백은 돌지 않는다.
function captureTimers(t) {
  const intervals = [];
  const timeouts = [];
  t.mock.method(globalThis, 'setInterval', (_fn, ms) => intervals.push(ms));
  t.mock.method(globalThis, 'clearInterval', () => {});
  t.mock.method(globalThis, 'setTimeout', (_fn, ms) => timeouts.push(ms));
  t.mock.method(globalThis, 'clearTimeout', () => {});
  return { intervals, timeouts };
}

function patchState(t, patch) {
  const saved = { ...state };
  Object.assign(state, patch);
  t.after(() => { Object.assign(state, saved); });
}

test('잡 SSE 폴백: 문서의 상태 폴링 간격·재승격 백오프가 sse.js와 같다', (t) => {
  const { intervals, timeouts } = captureTimers(t);
  patchState(t, {
    currentJobId: 'job-a', fallbackActive: false, fallbackTimer: 0,
    ssePromoteTimer: 0, ssePromoteAttempts: 0,
  });
  startFallbackPolling('job-a');
  for (let i = 0; i < 4; i += 1) scheduleSsePromote('job-a');
  stopFallbackPolling();
  clearSsePromote();

  assert.equal(intervals.length, 1);
  const [a, b, c, d] = timeouts;
  assert.ok(a < b && b < c && c === d, `백오프는 늘다가 상한에서 멈춘다: ${timeouts}`);
  const note = fallbackNote();
  assert.ok(note.includes(`${sec(intervals[0])} 상태 폴링`),
    `잡 폴링 간격 ${sec(intervals[0])}이 문서에 없다: ${note}`);
  assert.ok(note.includes(`${sec(a)}→${sec(b)}→${sec(c)}`),
    `재승격 백오프 ${sec(a)}→${sec(b)}→${sec(c)}가 문서에 없다: ${note}`);
});

test('번역 SSE 폴백: 재승격 없이 translate.js의 간격으로 끝까지 폴링한다고 적는다', (t) => {
  const { intervals, timeouts } = captureTimers(t);
  const es = { readyState: 2, close() {} }; // 첫 응답이 비-200 → CLOSED, error 1회
  patchState(t, {
    currentJobId: 'job-a', translateEs: es, translatePollTimer: 0,
    translateSseErrors: 0, translateGen: 0,
  });
  handleTranslateConnError('job-a', es);
  teardownTranslate();

  assert.equal(intervals.length, 1, '강등하면 상태 폴링을 하나 건다');
  assert.equal(timeouts.length, 0, '번역에는 SSE 재승격 타이머가 없다');
  const note = fallbackNote();
  assert.doesNotMatch(note, /번역 SSE도 같은 규칙/);
  const at = note.indexOf('번역 SSE');
  assert.ok(at >= 0, `SSE 폴백 항목에 번역 SSE 설명이 없다: ${note}`);
  const translate = note.slice(at);
  assert.ok(translate.includes(sec(intervals[0])),
    `번역 폴링 간격 ${sec(intervals[0])}이 문서에 없다: ${translate}`);
  assert.match(translate, /재승격하지 않는다/);
});
