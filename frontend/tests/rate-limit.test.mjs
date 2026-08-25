// Unlimited-OCR frontend — 429(남용 방어 상한) 안내 문구 테스트.
//
// 실행: npm test --prefix frontend  /  node --test frontend/tests/rate-limit.test.mjs
//
// 서버(backend/app/api.py)는 QA·번역 상한 초과 시 429 + Retry-After 헤더 +
// {"detail": "…"} 로 거절한다. Retry-After는 delta-seconds 또는 HTTP-date이며,
// 비정상 값(음수·거대값·쓰레기 문자열)도 UI를 영구히 잠그면 안 된다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  applyRetryLock,
  parseRetryAfter,
  rateLimitNotice,
  RETRY_AFTER_FALLBACK_S,
  RETRY_AFTER_MAX_S,
} from '../app.js';

const NOW = Date.parse('2026-01-01T00:00:00Z');

test('parseRetryAfter: delta-seconds 정수를 그대로 쓴다', () => {
  assert.equal(parseRetryAfter('30', NOW), 30);
  assert.equal(parseRetryAfter('5', NOW), 5);
  assert.equal(parseRetryAfter(' 12 ', NOW), 12);
});

test('parseRetryAfter: HTTP-date는 현재 시각 기준 남은 초로 환산한다', () => {
  assert.equal(parseRetryAfter('Thu, 01 Jan 2026 00:00:45 GMT', NOW), 45);
  // 소수 초는 올림 — 재시도가 아직 이른 순간에 다시 429를 맞지 않게
  assert.equal(parseRetryAfter('Thu, 01 Jan 2026 00:00:10 GMT', NOW + 500), 10);
});

test('parseRetryAfter: 헤더 없음·빈값·파싱 실패는 기본값으로 강등', () => {
  assert.equal(parseRetryAfter(null, NOW), RETRY_AFTER_FALLBACK_S);
  assert.equal(parseRetryAfter(undefined, NOW), RETRY_AFTER_FALLBACK_S);
  assert.equal(parseRetryAfter('', NOW), RETRY_AFTER_FALLBACK_S);
  assert.equal(parseRetryAfter('   ', NOW), RETRY_AFTER_FALLBACK_S);
  assert.equal(parseRetryAfter('soon', NOW), RETRY_AFTER_FALLBACK_S);
  assert.equal(parseRetryAfter('NaN', NOW), RETRY_AFTER_FALLBACK_S);
});

test('parseRetryAfter: 음수·0·과거 시각은 최소 1초', () => {
  assert.equal(parseRetryAfter('0', NOW), 1);
  assert.equal(parseRetryAfter('-5', NOW), 1);
  assert.equal(parseRetryAfter('Thu, 01 Jan 2020 00:00:00 GMT', NOW), 1);
});

test('parseRetryAfter: 거대값은 상한으로 잘린다 (UI 영구 잠금 방지)', () => {
  assert.equal(parseRetryAfter('999999999', NOW), RETRY_AFTER_MAX_S);
  assert.equal(parseRetryAfter('Sat, 01 Jan 2050 00:00:00 GMT', NOW), RETRY_AFTER_MAX_S);
  assert.ok(RETRY_AFTER_MAX_S <= 600);
});

test('rateLimitNotice: 서버 detail 앞부분 + 구체적인 대기 초', () => {
  const res = rateLimitNotice('30', '요청이 너무 잦습니다 — 잠시 후 다시 시도하세요', NOW);
  assert.equal(res.seconds, 30);
  assert.equal(res.message, '요청이 너무 잦습니다 — 30초 후 다시 시도해 주세요.');
  assert.ok(!res.message.includes('잠시 후')); // 모호한 절은 대체된다
});

test('rateLimitNotice: 번역/질문 동시 실행 상한 detail도 그대로 살린다', () => {
  const t = rateLimitNotice('30', '동시에 실행 중인 번역이 너무 많습니다 — 잠시 후 다시 시도하세요', NOW);
  assert.equal(t.message, '동시에 실행 중인 번역이 너무 많습니다 — 30초 후 다시 시도해 주세요.');
  const q = rateLimitNotice('5', '동시에 처리 중인 질문이 너무 많습니다 — 잠시 후 다시 시도하세요', NOW);
  assert.equal(q.seconds, 5);
  assert.equal(q.message, '동시에 처리 중인 질문이 너무 많습니다 — 5초 후 다시 시도해 주세요.');
});

test('rateLimitNotice: detail이 없거나 비문자열이면 기본 문구', () => {
  const res = rateLimitNotice(null, null, NOW);
  assert.equal(res.seconds, RETRY_AFTER_FALLBACK_S);
  assert.equal(res.message, `요청이 많습니다 — ${RETRY_AFTER_FALLBACK_S}초 후 다시 시도해 주세요.`);
  assert.equal(rateLimitNotice('7', { detail: 'x' }, NOW).message,
    '요청이 많습니다 — 7초 후 다시 시도해 주세요.');
  assert.equal(rateLimitNotice('7', '   ', NOW).message,
    '요청이 많습니다 — 7초 후 다시 시도해 주세요.');
});

test('rateLimitNotice: 문구는 항상 초를 포함하고 잠금은 유한하다', () => {
  for (const header of [null, '', 'garbage', '-1', '0', '999999999', '45']) {
    const { seconds, message } = rateLimitNotice(header, '요청이 너무 잦습니다', NOW);
    assert.ok(seconds >= 1 && seconds <= RETRY_AFTER_MAX_S, `범위 밖: ${header} → ${seconds}`);
    assert.ok(message.includes(`${seconds}초 후`));
  }
});

/* ---------------- applyRetryLock ---------------- */
// 잠금은 잡이 아니라 클라이언트(IP) 단위다 — 잡을 바꿔도 서버는 계속 429를 준다.
// 그런데 잡 전환·재렌더(showTranslateButton / setQaBusyUI(false))는 버튼을 되살린다.
// 그 상태를 그대로 두면 "눌리지만 요청은 나가지 않는 버튼"이 된다.

test('applyRetryLock: 남은 시간이 있으면 버튼을 다시 비활성으로 되돌린다', () => {
  const btns = [{ disabled: false }, { disabled: false }];
  assert.equal(applyRetryLock(12, btns), true);
  assert.deepEqual(btns.map((b) => b.disabled), [true, true]);
});

test('applyRetryLock: 잠금이 끝났으면 아무것도 건드리지 않는다', () => {
  const btns = [{ disabled: false }];
  assert.equal(applyRetryLock(0, btns), false);
  assert.equal(btns[0].disabled, false, '만료 뒤에는 다시 활성 상태 그대로');
  assert.equal(applyRetryLock(-3, btns), false);
  assert.equal(applyRetryLock(null, btns), false);
  assert.equal(applyRetryLock(NaN, btns), false);
  assert.equal(btns[0].disabled, false);
});

test('applyRetryLock: 다른 사유로 이미 비활성인 버튼을 활성화하지 않는다', () => {
  // 프로바이더 미설정으로 비활성인 버튼을 잠금 해제가 되살리면 안 된다 —
  // 이 함수는 오직 "더 잠그는" 방향으로만 움직인다.
  const unavailable = { disabled: true };
  applyRetryLock(0, [unavailable]);
  assert.equal(unavailable.disabled, true);
  applyRetryLock(5, [unavailable]);
  assert.equal(unavailable.disabled, true);
});

test('applyRetryLock: 빈 목록·null 항목에도 던지지 않는다', () => {
  assert.equal(applyRetryLock(5, []), true);
  assert.equal(applyRetryLock(5, [null, undefined]), true);
  assert.equal(applyRetryLock(5, null), true);
});
