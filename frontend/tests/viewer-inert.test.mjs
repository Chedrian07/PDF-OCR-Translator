// 전체 화면 뷰어(aria-modal) 배경 격리 — 실제 js/viewer.js(가짜 DOM).
//
//   node --test frontend/tests/viewer-inert.test.mjs
//
// 예전에는 inert로 둘 배경을 선택자로 하나씩 나열해, 나중에 결과 아래에 생긴 details(번역
// 참고 사항·PDF 생성 리포트)가 빠졌다 — 뷰어가 열린 동안에도 그 summary가 탭 순서에 남아
// 키보드 포커스가 가려진 배경으로 샜다(frontend-3). 여기서 지키는 계약:
//  · 뷰어 밖의 모든 영역(앞으로 생길 영역 포함)은 inert·aria-hidden이다.
//  · 뷰어 자신과 그 조상, 알림(#toast)은 그대로다.
//  · 닫으면 이 함수가 단 표시만 걷는다(원래 aria-hidden이던 영역은 그대로).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { el, state } from '../js/state.js';
import { setViewerBackgroundInert } from '../js/viewer.js';
import { installFakeDom } from './helpers/fake-dom.mjs';

// index.html의 뼈대 — 태그·id·class만 같게 만든다.
function buildPage(doc) {
  const make = (tag, attrs = {}, ...kids) => {
    const node = doc.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    for (const kid of kids) node.appendChild(kid);
    return node;
  };
  const button = (id) => make('button', id ? { id, type: 'button' } : { type: 'button' });
  const details = (id, cls, summaryId) => make('details', { id, class: cls },
    make('summary', summaryId ? { id: summaryId } : {}), make('ul'));
  const viewer = make('div', { id: 'production-viewer', class: 'tab-panel production-viewer' },
    button('viewer-close'),
    make('div', { id: 'reader-content', tabindex: '0' }),
    make('details', { class: 'reader-tools-wrap' }, make('summary'), button('reader-cite')));
  const result = make('section', { id: 'result-section', class: 'result-section' },
    make('div', { class: 'result-actions' }, button('viewer-open')),
    details('translate-warnings', 'result-notes', 'translate-warnings-summary'),
    details('pdf-report', 'result-notes', 'pdf-report-summary'),
    make('div', { class: 'tabs' }, button()),
    make('div', { class: 'tab-panels' }, viewer,
      make('div', { class: 'tab-panel', 'data-panel': 'preview' }, button('copy-md'))));
  const jobView = make('div', { id: 'job-view', class: 'job-view' },
    make('div', { class: 'job-head' }, button('job-delete')),
    make('section', { id: 'job-warnings', class: 'job-warnings' }, button()),
    make('section', { id: 'progress-section', class: 'progress-section' }),
    details('live-details', 'live-details'),
    make('section', { id: 'error-section', class: 'error-section' }),
    result);
  const main = make('main', { class: 'main' },
    make('div', { id: 'empty-state', class: 'empty-state', 'aria-hidden': 'true' }), jobView);
  doc.body.append(
    make('header', { class: 'app-header' }, button('theme-toggle')),
    make('div', { class: 'layout' }, make('aside', { class: 'sidebar' }, button('upload-btn')), main),
    make('div', { id: 'toast', role: 'status' }),
    make('script', { type: 'module' }));
  return { viewer, toast: doc.getElementById('toast') };
}

const allElements = (root) => root.children.flatMap((child) => [child, ...allElements(child)]);
const insideInert = (node) => {
  for (let cur = node; cur && cur.nodeType === 1; cur = cur.parentNode) if (cur.inert === true) return true;
  return false;
};
const insideHidden = (node) => !!node.closest('[aria-hidden="true"]');
const label = (node) => `${node.tagName}#${node.id || ''}.${node.className}`;

function setup(t) {
  const doc = installFakeDom(t);
  const savedEls = { ...el };
  const savedOpen = state.viewerOpen;
  const page = buildPage(doc);
  el.viewerRoot = page.viewer;
  t.after(() => { Object.assign(el, savedEls); state.viewerOpen = savedOpen; });
  return { doc, ...page };
}

test('뷰어가 열리면 뷰어 밖의 모든 영역이 inert·aria-hidden — 결과 아래 details 포함', (t) => {
  const { doc, viewer, toast } = setup(t);
  setViewerBackgroundInert(true);
  const outside = allElements(doc.body)
    .filter((node) => !viewer.contains(node) && !node.contains(viewer) && node !== toast
      && node.tagName !== 'SCRIPT');
  const leaks = outside.filter((node) => !insideInert(node) || !insideHidden(node)).map(label);
  assert.deepEqual(leaks, [], '뷰어 밖에 inert가 아닌 영역이 남았다');
  for (const id of ['translate-warnings-summary', 'pdf-report-summary']) {
    assert.ok(insideInert(doc.getElementById(id)), `${id}가 탭 순서에 남는다 (예전 회귀)`);
  }
  // 뷰어 안·뷰어의 조상·알림은 그대로다
  const inViewer = allElements(viewer).filter(insideInert).map(label);
  assert.deepEqual(inViewer, []);
  for (let cur = viewer; cur && cur !== doc.body; cur = cur.parentNode) {
    assert.ok(cur.inert !== true && cur.getAttribute('aria-hidden') !== 'true', `${label(cur)}가 가려졌다`);
  }
  assert.ok(!insideInert(toast) && !insideHidden(toast), '알림은 뷰어 위에서도 읽혀야 한다');
});

test('닫으면 이 함수가 단 표시만 걷고, 원래 aria-hidden이던 영역은 그대로 둔다', (t) => {
  const { doc } = setup(t);
  setViewerBackgroundInert(true);
  setViewerBackgroundInert(true); // 두 번 열어도(복원 경로) 표시를 겹쳐 달지 않는다
  setViewerBackgroundInert(false);
  const left = allElements(doc.body).filter((node) => node.inert === true || node.dataset.viewerInert);
  assert.deepEqual(left.map(label), []);
  assert.equal(doc.getElementById('empty-state').getAttribute('aria-hidden'), 'true',
    '원래 aria-hidden이던 영역은 닫아도 가려진 채다');
  assert.equal(doc.getElementById('translate-warnings').getAttribute('aria-hidden'), null);
  assert.equal(doc.querySelector('.sidebar').getAttribute('aria-hidden'), null);
  assert.equal(doc.querySelector('.app-header').getAttribute('aria-hidden'), null);
});
