// 레이아웃 탭 shrink-to-fit 재실행 시점 — 실제 js/tabs.js + layout-fit.js 멱등성.
//
//   node --test frontend/tests/layout-fit.test.mjs
//
// layout-fit은 보이는 블록만 잴 수 있다. 주입 직후 rAF에서 한 번만 돌면 숨은 탭에서 도착한
// 레이아웃이나 늦게 로드된 웹폰트 뒤에 넘친 블록이 잘린 채 남았다(frontend-13).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EL_IDS, el, state } from '../js/state.js';
import { activateTab, loadDocLayout, refitDocLayout } from '../js/tabs.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function setup(t) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const key of Object.keys(EL_IDS)) el[key] = mount(doc, 'div', EL_IDS[key]);
  const panel = mount(doc, 'div');
  panel.className = 'tab-panel';
  panel.dataset.panel = 'doclayout';
  panel.appendChild(el.doclayoutBody);
  const tab = mount(doc, 'button');
  tab.dataset.tab = 'doclayout';
  el.tabs = [tab];
  el.panels = [panel];
  Object.assign(state, {
    currentJobId: 'job-a', currentLang: 'orig', openGen: 3, docLayoutLoaded: false,
    layoutCapability: 'full', readerAnchorRaf: 0, readerBands: [],
  });
  const fits = [];
  globalThis.uocrFitLayout = (root) => fits.push(root);
  t.after(() => {
    delete globalThis.uocrFitLayout;
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return { doc, panel, fits };
}

test('이미 불러온 레이아웃 탭을 다시 보이면 맞춤을 다시 돌린다', (t) => {
  const { panel, fits } = setup(t);
  state.docLayoutLoaded = true;
  panel.hidden = true;
  refitDocLayout();
  assert.equal(fits.length, 0, '숨은 패널은 잴 수 없다 — 건너뛴다');
  activateTab('doclayout'); // 탭 전환이 패널을 보이게 한 뒤 다시 맞춘다
  assert.equal(panel.hidden, false);
  assert.equal(fits.length, 1);
  assert.ok(fits[0] === el.doclayoutBody);
});

test('불러오기 전에는 맞춤을 돌리지 않는다', (t) => {
  const { fits } = setup(t);
  state.docLayoutLoaded = false;
  refitDocLayout();
  assert.equal(fits.length, 0);
});

test('주입 직후에 한 번, 웹폰트 로드가 끝나면 한 번 더 맞춘다', async (t) => {
  const { doc, fits } = setup(t);
  let fontsReady;
  doc.fonts = { ready: new Promise((resolve) => { fontsReady = resolve; }) };
  t.mock.method(globalThis, 'fetch', async () => ({
    ok: true, status: 200, headers: { get: () => null },
    text: async () => '<div class="layout-canvas"><div class="layout-block">x</div></div>',
  }));
  await loadDocLayout();
  assert.equal(fits.length, 1, '주입 직후');
  fontsReady();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(fits.length, 2, '폰트 로드 뒤 다시');
});

test('layout-fit.js: 같은 블록에 여러 번 돌려도 원래 크기에서 다시 축소한다(멱등)', () => {
  // 클래식 스크립트를 최소 window/getComputedStyle 스텁으로 평가한다.
  const src = fs.readFileSync(path.join(FRONTEND, 'layout-fit.js'), 'utf8');
  const win = {};
  new Function('window', 'getComputedStyle', 'requestAnimationFrame', src)(
    win, () => ({ fontSize: '16px' }), undefined);
  const block = {
    classList: { contains: () => false },
    className: 'layout-block',
    dataset: {},
    style: { fontSize: '2cqw' },
    children: [],
    querySelectorAll: () => [],
    // 폰트가 base의 80% 이하일 때만 넘치지 않는 블록
    get scrollHeight() { return parseFloat(this.style.fontSize) > 1.6 ? 120 : 100; },
    clientHeight: 100,
    clientWidth: 300,
  };
  const root = { querySelectorAll: () => [block] };
  win.uocrFitLayout(root);
  const first = block.style.fontSize;
  win.uocrFitLayout(root);
  win.uocrFitLayout(root);
  assert.equal(block.style.fontSize, first, '재실행이 축소를 누적하지 않는다');
  assert.equal(block.dataset.uocrBaseFs, '2cqw');
  assert.ok(parseFloat(first) <= 1.6 && parseFloat(first) >= 1.1, first);
});
