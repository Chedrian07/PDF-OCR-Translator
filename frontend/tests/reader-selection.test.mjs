// 리더 선택 문장 정리 + 카드 경계를 넘는 하이라이트 — 실제 js/reader.js(가짜 DOM·가짜 Range).
//
//   node --test frontend/tests/reader-selection.test.mjs
//
// selection.toString()은 카드 머리말('01 본문 원문 보기')과 KaTeX 글자 두 벌을 섞는다.
// 설명·인용 문장은 본문만, 수식은 원래 TeX로 만들고, 하이라이트는 칠한 조각 글자를
// 메모로 남겨 다시 그린 레일에서 같은 조각을 찾게 한다(frontend-6, frontend-10).
// 선택이 걸린 레일을 다시 그리면(언어 전환 등) 그 선택은 버리고, 메모 언어는 문장을 고른
// 레일의 언어로 남긴다 — 예전에는 다른 언어 레일의 선택이 지금 언어로 저장됐다(frontend-7).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { el, state } from '../js/state.js';
import { readerNotesKey } from '../js/constants.js';
import {
  captureReaderSelection, highlightReaderSelection, rangeReadableText, renderRailPage,
  renderReaderDocument, saveReaderCitation,
} from '../js/reader.js';
import { installFakeStorage, mount } from './helpers/fake-dom.mjs';
import { alignment, setupReader } from './helpers/reader-setup.mjs';

// 문서 순서로 노드를 비교하는 최소 Range (start/end는 텍스트 노드 기준).
function fakeRange(doc, startNode, startOffset, endNode, endOffset) {
  const order = [];
  const walk = (node) => { order.push(node); for (const c of node.childNodes) walk(c); };
  walk(doc.documentElement);
  const index = (node) => order.indexOf(node);
  const last = (node) => { let n = node; while (n.lastChild) n = n.lastChild; return n; };
  let common = startNode;
  while (common && !common.contains(endNode)) common = common.parentNode;
  return {
    startContainer: startNode, startOffset, endContainer: endNode, endOffset,
    commonAncestorContainer: common,
    intersectsNode(node) {
      return index(last(node)) >= index(startNode) && index(node) <= index(endNode);
    },
  };
}

function card(doc, number, parts) {
  const article = doc.createElement('article');
  article.className = 'reader-map-card';
  const head = doc.createElement('div');
  head.className = 'reader-map-card-head';
  head.append(number, ' 본문 ');
  const locate = doc.createElement('button');
  locate.append('원문 보기');
  head.appendChild(locate);
  const target = doc.createElement('p');
  target.className = 'reader-map-target';
  for (const part of parts) target.appendChild(part);
  article.append(head, target);
  return { article, target };
}

function katexSpan(doc, tex, rendered) {
  const math = doc.createElement('span');
  math.className = 'katex';
  const mathml = doc.createElement('span');
  mathml.className = 'katex-mathml';
  const annotation = doc.createElement('annotation');
  annotation.append(tex);
  mathml.appendChild(annotation);
  const html = doc.createElement('span');
  html.className = 'katex-html';
  html.append(rendered);
  math.append(mathml, html);
  return math;
}

function build(t) {
  const doc = setupReader(t);
  installFakeStorage(t);
  for (const key of ['readerNotesList', 'readerNotesEmpty', 'readerNotesBadge', 'readerNotesCopy', 'readerNotesExport']) {
    el[key] = mount(doc, 'div');
  }
  state.readerNotes = [];
  const section = doc.createElement('section');
  section.className = 'reader-rail-page';
  section.dataset.page = '3';
  const body = doc.createElement('div');
  body.className = 'reader-rail-body';
  section.appendChild(body);
  el.readerContent.appendChild(section);
  const first = card(doc, '01', [doc.createTextNode('Alpha beta')]);
  const math = katexSpan(doc, 'x^2', 'x2');
  const second = card(doc, '02', [doc.createTextNode('Gamma '), math, doc.createTextNode(' delta')]);
  body.append(first.article, second.article);
  return { doc, first, second };
}

test('rangeReadableText: 카드 머리말·버튼은 빼고 수식은 TeX 한 번으로', (t) => {
  const { doc, first, second } = build(t);
  const range = fakeRange(doc, first.target.firstChild, 6, second.target.lastChild, 3);
  assert.equal(rangeReadableText(range), 'beta Gamma \\(x^2\\) de');
});

test('선택 캡처: 정리된 문장과 실제 페이지를 기억한다', (t) => {
  const { doc, first, second } = build(t);
  const range = fakeRange(doc, first.target.firstChild, 0, second.target.firstChild, 5);
  globalThis.getSelection = () => ({
    isCollapsed: false, rangeCount: 1, getRangeAt: () => range,
    toString: () => 'Alpha beta 02 본문 원문 보기 Gamma', removeAllRanges() {},
  });
  t.after(() => { delete globalThis.getSelection; });
  captureReaderSelection();
  assert.equal(state.readerSelection, 'Alpha beta Gamma');
  assert.equal(state.readerSelectionPage, 3);
});

test('하이라이트: 카드 경계를 넘어도 조각마다 감싸고 메모 문장은 칠한 본문 그대로', (t) => {
  const { doc, first, second } = build(t);
  const range = fakeRange(doc, first.target.firstChild, 6, second.target.lastChild, 3);
  let cleared = false;
  globalThis.getSelection = () => ({
    isCollapsed: false, rangeCount: 1, getRangeAt: () => range,
    toString: () => 'beta 02 본문 원문 보기 Gamma x^2x2 de', removeAllRanges() { cleared = true; },
  });
  t.after(() => { delete globalThis.getSelection; });
  state.readerSelection = 'beta Gamma \\(x^2\\) de';
  state.readerSelectionPage = 3;
  highlightReaderSelection();
  const marks = el.readerContent.querySelectorAll('mark.reader-highlight');
  assert.deepEqual(marks.map((m) => m.textContent), ['beta', 'Gamma ', ' de']);
  assert.ok(marks.every((m) => !m.closest('.reader-map-card-head') && !m.closest('.katex')),
    '머리말·수식 안은 칠하지 않는다');
  assert.equal(el.readerContent.querySelectorAll('.reader-map-card').length, 2, '카드를 복제하지 않는다');
  assert.equal(el.readerContent.querySelectorAll('.reader-rail-page').length, 1, '섹션을 복제하지 않는다');
  assert.equal(state.readerNotes.length, 1);
  assert.equal(state.readerNotes[0].text, 'beta Gamma de');
  assert.ok(marks.every((m) => m.dataset.noteId === state.readerNotes[0].id));
  assert.equal(second.target.querySelector('.katex annotation').textContent, 'x^2', '수식 구조는 그대로');
  assert.ok(cleared);
});

test('도구를 펼치느라 문서 선택이 지워져도 잡아 둔 범위로 하이라이트한다', (t) => {
  const { doc, first } = build(t);
  const range = fakeRange(doc, first.target.firstChild, 0, first.target.firstChild, 5);
  let live = true;
  globalThis.getSelection = () => (live
    ? { isCollapsed: false, rangeCount: 1, getRangeAt: () => range, toString: () => 'Alpha', removeAllRanges() {} }
    : { isCollapsed: true, rangeCount: 0, getRangeAt: () => null, toString: () => '', removeAllRanges() {} });
  t.after(() => { delete globalThis.getSelection; });
  captureReaderSelection();          // mouseup — 'Alpha'를 잡는다
  live = false;                      // [선택 문장 도구] summary 클릭이 선택을 지운다
  highlightReaderSelection();
  const marks = el.readerContent.querySelectorAll('mark.reader-highlight');
  assert.deepEqual(marks.map((m) => m.textContent), ['Alpha']);
  assert.equal(state.readerNotes.length, 1);
  assert.equal(state.readerSelection, '');
  highlightReaderSelection();        // 한 번 칠한 범위는 다시 쓰지 않는다
  assert.equal(el.readerContent.querySelectorAll('mark.reader-highlight').length, 1);
});

/* ---------------- 언어 전환·재렌더 뒤의 선택 (frontend-7) ---------------- */

// 실제 레일(renderReaderDocument)을 그리고 1페이지 첫 카드 본문의 앞 5글자를 선택한다.
function railWithSelection(t, lang) {
  const doc = setupReader(t);
  const storage = installFakeStorage(t);
  for (const key of ['readerNotesList', 'readerNotesEmpty', 'readerNotesBadge', 'readerNotesCopy', 'readerNotesExport']) {
    el[key] = mount(doc, 'div');
  }
  state.readerNotes = [];
  state.toastTimer = 0;
  t.after(() => { clearTimeout(state.toastTimer); delete globalThis.getSelection; });
  state.readerPages.ko = state.readerPages.orig;
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1']));
  state.readerAlignments.ko.set(1, alignment(1, ['p1-b1']));
  state.currentLang = lang;
  renderReaderDocument();
  const text = el.readerContent.querySelector('[data-block-id="p1-b1"] .reader-map-target').firstChild;
  const range = fakeRange(doc, text, 0, text, 5);
  let live = true;
  globalThis.getSelection = () => (live
    ? { isCollapsed: false, rangeCount: 1, getRangeAt: () => range, toString: () => 'Block', removeAllRanges() {} }
    : { isCollapsed: true, rangeCount: 0, getRangeAt: () => null, toString: () => '', removeAllRanges() {} });
  captureReaderSelection();
  live = false; // 도구를 누르면 문서 선택은 지워진다
  return { doc, storage, range };
}

const stored = (storage) => {
  const raw = storage.getItem(readerNotesKey('job-a'));
  return raw ? JSON.parse(raw).items : [];
};

test('언어를 바꿔 레일을 다시 그리면 이전 레일의 선택을 버린다 — 다른 언어로 인용되지 않는다', (t) => {
  const { storage } = railWithSelection(t, 'ko');
  assert.equal(state.readerSelection, 'Block');
  assert.equal(el.readerCite.disabled, false);
  state.currentLang = 'orig'; // 원문 보기로 전환 — 레일을 새로 만든다
  renderReaderDocument();
  assert.equal(state.readerSelection, '', '사라진 레일의 선택이 도구에 남았다');
  assert.equal(el.readerCite.disabled, true);
  assert.equal(el.readerHighlight.disabled, true);
  assert.match(el.readerSelection.textContent, /문장을 선택하세요/);
  saveReaderCitation();
  assert.deepEqual(stored(storage), [], '예전에는 한국어 문장이 원문 인용으로 저장됐다');
});

test('인용 언어는 저장 시점의 보기 언어가 아니라 문장을 고른 레일의 언어다', (t) => {
  const { storage } = railWithSelection(t, 'ko');
  state.currentLang = 'orig'; // 레일을 다시 그리기 전에 저장해도
  saveReaderCitation();
  assert.deepEqual(stored(storage).map((n) => [n.kind, n.lang, n.text]), [['citation', 'ko', 'Block']]);
});

test('선택이 걸린 페이지를 다시 그리면 그 선택도 버린다', (t) => {
  railWithSelection(t, 'orig');
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1', 'p1-b2'])); // 정렬이 새로 도착
  renderRailPage(1);
  assert.equal(state.readerSelection, '');
  assert.equal(el.readerHighlight.disabled, true);
});

test('저장해 둔 범위가 접혔으면(그 DOM이 바뀌었다) 칠한 것 없는 하이라이트를 저장하지 않는다', (t) => {
  const { storage, range } = railWithSelection(t, 'orig');
  // 브라우저의 live range는 감싼 노드가 바뀌면 컨테이너 위치로 접힌다.
  Object.assign(range, {
    collapsed: true, startContainer: el.readerContent, startOffset: 0,
    endContainer: el.readerContent, endOffset: 0, commonAncestorContainer: el.readerContent,
  });
  highlightReaderSelection();
  assert.deepEqual(stored(storage), [], '예전에는 칠한 조각 0개로 하이라이트를 저장하고 토스트를 띄웠다');
  assert.equal(el.readerContent.querySelectorAll('mark.reader-highlight').length, 0);
  assert.equal(state.readerSelection, '', '칠할 수 없는 선택은 도구에서 비운다');
});
