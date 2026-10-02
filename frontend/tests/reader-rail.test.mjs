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

import { railPagesToRender, splitHtmlTables } from '../js/core.js';
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

/* ---------------- 표 블록 카드 (fresh-user-2) ----------------
 * Unlimited-OCR는 표를 HTML로 낸다. 정렬 API는 그 원문을 그대로 싣는데, 카드가 텍스트 노드로만
 * 그려 기본 화면(읽기 탭·논문 뷰어 레일)에 '<table><tr><td>Mode</td>…'가 글자 그대로 보였다.
 * 구조 태그만 읽어 안전한 표로 다시 만든다 — 업로드 PDF에서 온 다른 태그는 요소가 되지 않는다. */

const SAMPLE_TABLE = '<table><tr><td>Mode</td><td>base_size</td></tr><tr><td>gundam</td><td>1024</td></tr></table>';
const cell = (text, extra = {}) => ({ header: false, text, colspan: 1, rowspan: 1, ...extra });

test('splitHtmlTables: 구조 태그만 행·셀로 읽고 셀 글자의 엔티티를 푼다', () => {
  assert.deepEqual(splitHtmlTables(SAMPLE_TABLE), [{
    type: 'table', rows: [[cell('Mode'), cell('base_size')], [cell('gundam'), cell('1024')]],
  }]);
  const rich = '<table border="1"><thead><tr><th colspan="2">Head</th></tr></thead>'
    + '<tbody><tr><td rowspan="2">a &amp; b&lt;c&#62;</td><td>\\(x<y\\)<br/>z</td></tr></tbody></table>';
  assert.deepEqual(splitHtmlTables(rich), [{
    type: 'table',
    rows: [
      [cell('Head', { header: true, colspan: 2 })],
      [cell('a & b<c>', { rowspan: 2 }), cell('\\(x<y\\)\nz')],
    ],
  }]);
});

test('splitHtmlTables: 표 앞뒤 글자·표 아닌 블록·잘린 표·과한 병합을 견딘다', () => {
  assert.deepEqual(splitHtmlTables('Table 1: caption <table><tr><td>a</td></tr></table> tail'), [
    { type: 'text', value: 'Table 1: caption ' },
    { type: 'table', rows: [[cell('a')]] },
    { type: 'text', value: ' tail' },
  ]);
  assert.deepEqual(splitHtmlTables('plain <b>text</b> with <tr> tag'),
    [{ type: 'text', value: 'plain <b>text</b> with <tr> tag' }]);
  assert.deepEqual(splitHtmlTables('<table><tr><td>a<td>b'), [{ type: 'table', rows: [[cell('a'), cell('b')]] }],
    '잘린 표(닫는 태그 없음)도 셀로 읽는다');
  const [wide] = splitHtmlTables('<table><tr><td colspan="999" rowspan="0">x</td></tr></table>');
  assert.deepEqual(wide.rows[0][0], cell('x', { colspan: 64 }), '병합 수는 1~64로 묶는다');
  const [nested] = splitHtmlTables('<table><tr><td>a<table><tr><td>in</td></tr></table>b</td><td>c</td></tr></table>');
  assert.deepEqual(nested.rows, [[cell('a'), cell('in'), cell('b'), cell('c')]], '중첩 표는 펼친다');
});

test('splitHtmlTables: 셀 안의 다른 태그는 요소가 아니라 글자다', () => {
  const [table] = splitHtmlTables('<table><tr><td><img src=x onerror=alert(1)>cell<script>x</script></td></tr></table>');
  assert.deepEqual(table.rows, [[cell('<img src=x onerror=alert(1)>cell<script>x</script>')]]);
});

function tableAlignment(page, { source, target = source, translated = false }) {
  return {
    page,
    lang: 'orig',
    blocks: [{
      id: `p${page}-b4`, index: 4, type: 'table', source, target, translated,
      rect: { left: 10, top: 10, width: 50, height: 10 },
    }],
  };
}

test('표 블록 카드: 원시 HTML 대신 표를 그린다 — 셀 글자는 텍스트 노드', (t) => {
  setupReader(t, { total: 1 });
  state.readerAlignments.orig.set(1, tableAlignment(1, {
    source: '<table><tr><td>Mode</td><td><img src=x onerror=alert(1)></td></tr></table>',
  }));
  renderReaderDocument();
  const card = el.readerContent.querySelector('[data-block-id="p1-b4"]');
  const target = card.querySelector('.reader-map-target');
  assert.ok(!target.textContent.includes('<table'), target.textContent);
  const cells = target.querySelectorAll('td');
  assert.deepEqual(cells.map((c) => c.textContent), ['Mode', '<img src=x onerror=alert(1)>']);
  assert.equal(target.querySelectorAll('img').length, 0, '업로드 PDF의 태그가 요소가 되지 않는다');
  assert.equal(card.querySelectorAll('table').length, 1);
});

test('표 블록 카드: 한국어 레일은 번역된 표와 ORIGINAL 표를 둘 다 표로 그린다', (t) => {
  setupReader(t, { total: 1 });
  state.currentLang = 'ko';
  state.readerPages.ko = state.readerPages.orig;
  state.readerAlignments.ko.set(1, tableAlignment(1, {
    source: SAMPLE_TABLE,
    target: '<table><tr><th>모드</th><th>기본 크기</th></tr><tr><td>gundam</td><td>1024</td></tr></table>',
    translated: true,
  }));
  renderReaderDocument();
  const card = el.readerContent.querySelector('[data-block-id="p1-b4"]');
  const [translatedTable, originalTable] = card.querySelectorAll('table');
  assert.ok(translatedTable && originalTable, '번역 표와 원문 표');
  assert.deepEqual(translatedTable.querySelectorAll('th').map((c) => c.textContent), ['모드', '기본 크기']);
  assert.deepEqual(originalTable.querySelectorAll('td').map((c) => c.textContent),
    ['Mode', 'base_size', 'gundam', '1024']);
  assert.ok(!card.textContent.includes('<t'), card.textContent);
});
