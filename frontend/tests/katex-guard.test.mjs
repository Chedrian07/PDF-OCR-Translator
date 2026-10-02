// 내려받는 standalone HTML의 KaTeX 가드(katex-guard.js) ↔ 앱 구현(core.js·constants.js) 대조.
//
//   node --test frontend/tests/katex-guard.test.mjs
//
// standalone 파일은 ES 모듈을 쓸 수 없어 앱의 크기 묶기(core.clampTexSizes)·조판 결과 상한
// (core.katexStyleOversized)·옵션(constants.katexOptions)을 클래식 스크립트로 옮겼다. 예전
// 인라인 조판은 KaTeX를 직접 불러 \raisebox{-4000em}{x} 하나로 파일이 수만 px로 늘어났다
// (감사 frontend-4). 한쪽만 바꾸면 이 대조가 깨진다.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { KATEX_MAX_BOX_EM, KATEX_MAX_SIZE_EM, katexOptions } from '../js/constants.js';
import { clampTexSizes, katexStyleOversized } from '../js/core.js';

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const GUARD_SRC = fs.readFileSync(path.join(FRONTEND, 'katex-guard.js'), 'utf8');

function loadGuard(win = {}) {
  // eslint-disable-next-line no-new-func
  new Function('window', GUARD_SRC)(win);
  return win.uocrKatexGuard;
}

function loadKatex() {
  const src = fs.readFileSync(path.join(FRONTEND, 'vendor', 'katex', 'katex.min.js'), 'utf8');
  const mod = { exports: {} };
  // eslint-disable-next-line no-new-func
  new Function('module', 'exports', src)(mod, mod.exports);
  return mod.exports;
}

const styleList = (html) => [...html.matchAll(/style="([^"]*)"/g)].map((m) => m[1]);
const maxEm = (html) => Math.max(0, ...styleList(html)
  .flatMap((st) => [...st.matchAll(/(-?(?:\d+(?:\.\d*)?|\.\d+))em/g)].map((m) => Math.abs(Number(m[1])))));

const ABUSE = [
  '\\raisebox{-4000em}{x}', '\\rule[-3000em]{1em}{1em}', '\\kern{-5000em}x', '\\kern-5000em x',
  'a\\\\[-900em] b', '\\hspace{-300em}x', '\\hspace*{-400em}x', '\\mkern-9000mu x',
  '\\raisebox{- 4000 em}{x}', '\\raisebox{-4000cm}{x}', '\\kern{ - 5000 em }x',
];

test('클래식 스크립트다 — 모듈 문법 없이 window.uocrKatexGuard 하나만 내보낸다', () => {
  assert.doesNotMatch(GUARD_SRC, /^\s*(import|export)\b/m);
  const win = {};
  const guard = loadGuard(win);
  assert.deepEqual(Object.keys(win), ['uocrKatexGuard']);
  for (const name of ['clampTexSizes', 'styleOversized', 'options', 'renderMath', 'typesetMath']) {
    assert.equal(typeof guard[name], 'function', name);
  }
});

test('옵션이 앱 katexOptions와 같고 호출마다 새 객체다', () => {
  const guard = loadGuard();
  for (const display of [false, true, 0, 1]) {
    assert.deepEqual(guard.options(display), katexOptions(display));
  }
  assert.notEqual(guard.options(false), guard.options(false));
});

test('크기 묶기가 앱 clampTexSizes와 같다 — 고정 표와 무작위 입력', () => {
  const guard = loadGuard();
  const fixed = [
    ...ABUSE, '\\text{150 cm} + 300 mm', 'h = 150\\,\\mathrm{cm}', '\\kern-.5em x', '\\\\[2pt]',
    '\\hspace{1em}', '\\def\\x{-4000em}\\kern\\x y', 'x^2', '', '\\kern-4000qq x',
    '\\raisebox{-4000pt}{x}', '\\rule{2000em}{2000em}', '\\hskip -3e4em x', null, undefined,
  ];
  for (const tex of fixed) assert.equal(guard.clampTexSizes(tex), clampTexSizes(tex), String(tex));

  let seed = 20261002;
  const rand = (n) => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; };
  const pick = (list) => list[rand(list.length)];
  const units = ['em', 'ex', 'mu', 'pt', 'mm', 'cm', 'in', 'bp', 'pc', 'dd', 'cc', 'nd', 'nc', 'sp', 'px', 'zz'];
  const size = () => `${pick(['', '-', '+', '- ', ' -'])}${pick(['0', '.5', '12', '400', '4000.25', '99999'])}${pick(['', ' '])}${pick(units)}`;
  const forms = [
    (s) => `\\kern${s} x`, (s) => `\\kern{${s}}x`, (s) => `\\mkern${s}`, (s) => `\\hskip ${s}`,
    (s) => `\\mskip{${s}}`, (s) => `\\hspace{${s}}`, (s) => `\\hspace*{${s}}`, (s) => `\\raisebox{${s}}{y}`,
    (s) => `\\rule[${s}]{${size()}}{${size()}}`, (s) => `a\\\\[${s}] b`, (s) => `\\text{${s}}`,
    (s) => `\\kernel ${s}`, (s) => `x_{${s}}`,
  ];
  for (let i = 0; i < 2000; i += 1) {
    const tex = Array.from({ length: 1 + rand(3) }, () => pick(forms)(size())).join(' ');
    assert.equal(guard.clampTexSizes(tex), clampTexSizes(tex), tex);
  }
});

test('조판 결과 상한 판정이 앱 katexStyleOversized와 같다', () => {
  const guard = loadGuard();
  const styles = [
    `height:${KATEX_MAX_BOX_EM + 1}em`, `vertical-align:-${KATEX_MAX_BOX_EM}em`, 'margin-right:-1e+21em',
    'height:0.5em;width:2em', 'top:-4000.5em', 'width:100.0001EM', '', null, 'margin:3em 1e2em',
    'padding:.5em', 'x:-0em', 'height:101px',
  ];
  for (const style of styles) assert.equal(guard.styleOversized(style), katexStyleOversized(style), String(style));
  // 같은 문자열을 거듭 판정해도 결과가 같다(전역 정규식 상태가 남지 않는다)
  for (let i = 0; i < 3; i += 1) assert.equal(guard.styleOversized('top:-4000em'), true);
});

test('실제 KaTeX: 음수·거대 크기도 묶은 TeX로 조판하면 ±상한 안이고 오류가 없다', () => {
  const guard = loadGuard();
  const katex = loadKatex();
  assert.ok(maxEm(katex.renderToString('\\raisebox{-4000em}{x}', guard.options(false))) >= 4000,
    '대조군 — 옵션만으로는 그대로 수천 em');
  for (const tex of ABUSE) {
    for (const display of [false, true]) {
      const html = katex.renderToString(guard.clampTexSizes(tex), guard.options(display));
      assert.ok(maxEm(html) <= 2 * KATEX_MAX_SIZE_EM, `${tex} → ${maxEm(html)}em`);
      assert.ok(!html.includes('katex-error'), `${tex}: 묶은 TeX도 조판된다`);
    }
  }
});

// 브라우저 katex.render 대신: 실제 번들의 renderToString 결과를 대상에 남기고, [style] 조회는
// 그 HTML의 style 속성으로 답한다.
function stubTarget(tex, display = false) {
  return {
    textContent: tex,
    html: '',
    attrs: {},
    classList: { contains: (name) => name === (display ? 'math-display' : 'math-inline') },
    setAttribute(name, value) { this.attrs[name] = value; },
    querySelectorAll(selector) {
      assert.equal(selector, '[style]');
      return styleList(this.html).map((style) => ({ getAttribute: () => style }));
    },
  };
}

function realKatexStub() {
  const real = loadKatex();
  const seen = [];
  return {
    seen,
    render(tex, target, options) {
      seen.push({ tex, options });
      target.html = real.renderToString(tex, options);
      target.textContent = '';
    },
  };
}

test('renderMath: 크기를 묶어 조판하고, 매크로로 만든 거대 박스는 원문 TeX로 되돌린다', () => {
  const guard = loadGuard();
  const katex = realKatexStub();

  const normal = stubTarget('\\frac{a}{b}');
  assert.equal(guard.renderMath(katex, normal, '\\frac{a}{b}', true), true);
  assert.ok(normal.html.includes('katex'), '정상 수식은 조판 결과가 남는다');
  assert.deepEqual(normal.attrs, {});
  assert.deepEqual(katex.seen.at(-1).options, katexOptions(true));

  const negative = stubTarget('\\raisebox{-4000em}{x}');
  assert.equal(guard.renderMath(katex, negative, '\\raisebox{-4000em}{x}', false), true);
  assert.equal(katex.seen.at(-1).tex, '\\raisebox{-10em}{x}', '조판 전에 크기 인자를 묶는다');
  assert.ok(maxEm(negative.html) <= 2 * KATEX_MAX_SIZE_EM, negative.html.slice(0, 200));
  assert.deepEqual(negative.attrs, {});

  const macro = '\\def\\x{-4000em}\\raisebox{\\x}{y}';
  const target = stubTarget(macro);
  assert.equal(guard.renderMath(katex, target, macro, true), true);
  assert.equal(target.textContent, macro, '상한을 넘는 조판 결과는 원문 TeX로 되돌린다');
  assert.equal(target.attrs['data-math-fallback'], 'oversized');

  const broken = stubTarget('x');
  const throwing = { render() { throw new Error('boom'); } };
  assert.equal(guard.renderMath(throwing, broken, '\\frac{', false), false);
  assert.equal(broken.textContent, '\\frac{', '렌더 불가 TeX는 원문 유지');
});

test('typesetMath: 문서의 인라인·디스플레이 수식을 모두 window.katex로 조판한다', () => {
  const katex = realKatexStub();
  const guard = loadGuard({ katex });
  const nodes = [stubTarget('x^2'), stubTarget('\\kern-5000em y', true)];
  guard.typesetMath({
    querySelectorAll(selector) {
      assert.equal(selector, '.math-inline,.math-display');
      return nodes;
    },
  });
  assert.deepEqual(katex.seen.map((s) => [s.tex, s.options.displayMode]), [
    ['x^2', false], ['\\kern-10em y', true],
  ]);
  // KaTeX 자산이 없으면 아무것도 하지 않는다(원문 LaTeX가 그대로 보인다)
  const bare = loadGuard({});
  assert.doesNotThrow(() => bare.typesetMath({ querySelectorAll: () => { throw new Error('호출되면 안 된다'); } }));
});
