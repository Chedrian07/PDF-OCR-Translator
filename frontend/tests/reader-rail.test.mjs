// 리더 레일 재진입 렌더 범위 — 실제 js/reader.js를 가짜 DOM에서 돌린다.
//
//   node --test frontend/tests/reader-rail.test.mjs
//
// 지키는 계약 (frontend-5):
//  · 탭 재활성화·뷰어 열기(레일이 그대로)는 이미 그린 섹션을 다시 그리지 않는다 —
//    카드·KaTeX 노드·하이라이트 <mark>가 그대로 남는다.
//  · 아직 빈 섹션(새로 도착한 캐시)만 채운다.
//  · 레일을 새로 만들 때(잡·언어·총 페이지 변경)는 캐시된 페이지를 전부 그린다.
//  · 활성 블록 표시는 새 카드에 직접 붙는다 — 페이지 하나를 그릴 때 레일 전체를 훑지 않는다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { railPagesToRender } from '../js/core.js';
import { el, state } from '../js/state.js';
import { renderRailPage, renderReaderDocument } from '../js/reader.js';
import { assertNotSameNode, assertSameNode } from './helpers/fake-dom.mjs';
import { alignment, setupReader } from './helpers/reader-setup.mjs';

test('railPagesToRender: 새 레일이면 전부, 그대로면 아직 안 그린 페이지만', () => {
  const rendered = new Set([1, 3]);
  const isRendered = (page) => rendered.has(page);
  assert.deepEqual(railPagesToRender([1, 2, 3], true, isRendered), [1, 2, 3]);
  assert.deepEqual(railPagesToRender([1, 2, 3], false, isRendered), [2]);
  assert.deepEqual(railPagesToRender(new Map([[4, null]]).keys(), false, isRendered), [4]);
  assert.deepEqual(railPagesToRender(null, false, isRendered), []);
});

test('재진입: 이미 그린 카드·흐름 본문·하이라이트를 다시 만들지 않는다', (t) => {
  const doc = setupReader(t);
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1', 'p1-b2']));
  state.readerAlignments.orig.set(2, null); // 좌표 없는 페이지 — 흐름 본문
  renderReaderDocument();
  const card = el.readerContent.querySelector('[data-block-id="p1-b1"]');
  assert.ok(card, '정렬 카드가 그려진다');
  const flowBody = state.readerRailEls.get(2).lastElementChild;
  const mark = doc.createElement('mark');
  mark.className = 'reader-highlight';
  flowBody.appendChild(mark); // 사용자가 칠해 둔 하이라이트

  renderReaderDocument(); // 질문 탭 → 읽기 탭 재활성화, 뷰어 열기
  assertSameNode(assert, el.readerContent.querySelector('[data-block-id="p1-b1"]'), card, '카드 노드가 그대로');
  assert.ok(mark.isConnected, '하이라이트가 사라지지 않는다');
  assertSameNode(assert, state.readerRailEls.get(2).lastElementChild, flowBody, '흐름 본문이 그대로');
});

test('재진입: 새로 캐시된(아직 빈) 섹션만 채운다', (t) => {
  setupReader(t, { total: 3 });
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1']));
  renderReaderDocument();
  const card = el.readerContent.querySelector('[data-block-id="p1-b1"]');
  assert.ok(state.readerRailEls.get(3).querySelector('.reader-rail-pending'), '3쪽은 아직 자리표시자');
  state.readerAlignments.orig.set(3, alignment(3, ['p3-b1']));
  renderReaderDocument();
  assert.ok(el.readerContent.querySelector('[data-block-id="p3-b1"]'), '새 캐시 페이지는 그린다');
  assertSameNode(assert, el.readerContent.querySelector('[data-block-id="p1-b1"]'), card, '기존 페이지는 그대로');
});

test('레일을 새로 만들면(언어·총 페이지 변경) 캐시된 페이지를 전부 다시 그린다', (t) => {
  setupReader(t);
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1']));
  renderReaderDocument();
  const card = el.readerContent.querySelector('[data-block-id="p1-b1"]');
  state.readerRailKey = ''; // 레일 서명 무효화 — 새 레일
  renderReaderDocument();
  const rebuilt = el.readerContent.querySelector('[data-block-id="p1-b1"]');
  assert.ok(rebuilt);
  assertNotSameNode(assert, rebuilt, card, '새 레일은 카드를 새로 만든다');
});

test('활성 블록: 새 카드에 바로 붙고, 다른 페이지를 그릴 때 레일 전체를 훑지 않는다', (t) => {
  setupReader(t);
  state.readerActiveBlock = 'p1-b2';
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1', 'p1-b2']));
  state.readerAlignments.orig.set(2, alignment(2, ['p2-b1']));
  renderReaderDocument();
  const active = el.readerContent.querySelector('[data-block-id="p1-b2"]');
  assert.ok(active.classList.contains('is-active'));
  assert.equal(active.getAttribute('aria-current'), 'true');
  assert.ok(!el.readerContent.querySelector('[data-block-id="p1-b1"]').classList.contains('is-active'));
  // 2쪽만 다시 그려도 1쪽 카드는 건드리지 않는다(예전에는 레일 전 카드 토글).
  active.classList.remove('is-active');
  renderRailPage(2);
  assert.ok(!active.classList.contains('is-active'), '다른 페이지 렌더가 1쪽 카드를 훑지 않는다');
});
