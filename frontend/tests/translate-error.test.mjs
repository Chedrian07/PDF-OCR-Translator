// 번역 실패 사유 한 줄 — state.json error·SSE error 메시지를 버튼 옆 칩·토스트용으로 접는다.
//
//   node --test frontend/tests/translate-error.test.mjs
//
// 지키는 계약:
//  · 상류 오류 본문(200자에서 잘린 파이썬 dict 미리보기)은 상태코드까지만 보이고 전문은 detail.
//  · 이미 읽을 수 있는 한국어 문장은 그대로, 너무 길면 단어 경계에서 줄인다.
//  · 사유가 없어도 실패했다는 사실은 말한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { translateErrorText } from '../js/core.js';

test('an upstream 5xx body is folded to its status code', () => {
  const raw = "번역 API 오류 (HTTP 500): {'error': {'message': \"litellm.BadRequestError: litellm.Co";
  assert.deepEqual(translateErrorText(raw), {
    text: '번역 API 서버 오류 (HTTP 500) — 잠시 후 다시 시도해 주세요', detail: raw,
  });
});

test('an upstream 4xx points at the model settings', () => {
  const raw = "번역 API 오류 (HTTP 400): {'error': {'message': 'unknown model'}}";
  assert.equal(translateErrorText(raw).text, '번역 API가 요청을 거부했습니다 (HTTP 400) — 모델명·설정을 확인해 주세요');
});

test('a readable message is kept and a long one is cut at a word boundary', () => {
  const auth = '번역 API 인증 실패 — OPENAI_API_KEY를 확인하세요';
  assert.equal(translateErrorText(auth).text, auth);

  const long = `번역 API가 응답하지 않습니다 — ${'성공 없이 유닛이 연속으로 시간 초과됐습니다 '.repeat(8)}`;
  const { text, detail } = translateErrorText(long);
  assert.ok(text.length <= 161 && text.endsWith('…'), text);
  assert.ok(!/\s…$/.test(text), text);
  assert.equal(detail, long.trim());
});

test('a failure without a reason still says so', () => {
  assert.deepEqual(translateErrorText(''), { text: '서버가 사유를 보내지 않았습니다', detail: '' });
  assert.deepEqual(translateErrorText(null), { text: '서버가 사유를 보내지 않았습니다', detail: '' });
});
