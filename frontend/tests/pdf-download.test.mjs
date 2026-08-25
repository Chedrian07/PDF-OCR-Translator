// PDF 다운로드 버튼의 판단 로직 — DOM 없이 Node에서 직접 검증한다.
//
//   node --test frontend/tests/pdf-download.test.mjs
//
// 여기서 지키는 계약:
//  · 503(내보내기 대기열 포화)은 Retry-After만큼 기다렸다가 딱 한 번 재시도한다.
//  · 그 외 상태 코드는 절대 재시도하지 않는다 — 404/409는 기다려도 안 바뀐다.
//  · 진행 문구는 Content-Length가 없어도 "멈춰 있지 않다"를 보여 준다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { PDF_RETRY_MAX, pdfProgressLabel, pdfRetryDelay } from '../app.js';

/* ---------------- pdfRetryDelay ---------------- */

test('pdfRetryDelay: 503은 Retry-After만큼 기다렸다 재시도한다', () => {
  assert.equal(pdfRetryDelay(503, '7', 0), 7);
  assert.equal(pdfRetryDelay(503, '30', 0), 30);
});

test('pdfRetryDelay: 재시도는 한 번뿐 — 무한 재시도로 서버를 더 밀지 않는다', () => {
  assert.ok(pdfRetryDelay(503, '5', 0) > 0);
  assert.equal(pdfRetryDelay(503, '5', PDF_RETRY_MAX), 0);
});

test('pdfRetryDelay: 503이 아니면 재시도하지 않는다', () => {
  // 404(번역본 없음)·409(미완료)·500은 기다린다고 달라지지 않는다.
  for (const status of [200, 400, 404, 409, 500, 502]) {
    assert.equal(pdfRetryDelay(status, '5', 0), 0, `status ${status}`);
  }
});

test('pdfRetryDelay: Retry-After가 없거나 쓰레기여도 합리적인 값으로 떨어진다', () => {
  for (const header of [null, '', 'soon', '0', '-3', undefined]) {
    const wait = pdfRetryDelay(503, header, 0);
    assert.ok(wait >= 1 && wait <= 60, `header ${String(header)} -> ${wait}`);
  }
});

test('pdfRetryDelay: 터무니없이 긴 Retry-After는 상한에 걸린다', () => {
  // 서버가 잘못된 값을 줘도 버튼이 몇 시간 잠기면 안 된다.
  assert.equal(pdfRetryDelay(503, '86400', 0), 60);
});

/* ---------------- pdfProgressLabel ---------------- */

test('pdfProgressLabel: Content-Length가 있으면 퍼센트를 보여 준다', () => {
  assert.equal(pdfProgressLabel(0, 1000), '내려받는 중… 0%');
  assert.equal(pdfProgressLabel(500, 1000), '내려받는 중… 50%');
  assert.equal(pdfProgressLabel(1000, 1000), '내려받는 중… 100%');
});

test('pdfProgressLabel: 길이를 모르면 받은 MB를 보여 준다', () => {
  assert.equal(pdfProgressLabel(1048576, 0), '내려받는 중… 1.0MB');
  assert.equal(pdfProgressLabel(0, 0), '내려받는 중… 0.0MB');
});

test('pdfProgressLabel: 퍼센트는 0..100을 벗어나지 않는다', () => {
  // 서버가 실제 본문보다 작은 Content-Length를 보내도 "180%"가 뜨면 안 된다.
  assert.equal(pdfProgressLabel(2000, 1000), '내려받는 중… 100%');
  assert.equal(pdfProgressLabel(-5, 1000), '내려받는 중… 0%');
});
