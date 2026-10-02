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

import {
  KATEX_MAX_BOX_EM, KATEX_MAX_SIZE_EM, KATEX_MAX_TEX_CHARS, katexOptions,
} from '../js/constants.js';
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

// 크기 묶기는 입력 길이에 선형이어야 한다 — 상한 없는 \s*·[^\]]* 스캔은 닫는 괄호 없는 '\\['
// 반복·'\kern'+긴 공백에서 길이의 제곱 시간이 걸려, 적대적 PDF 한 쪽(수식 스팬 45만 자)이
// 미리보기·리더·내려받은 HTML을 23초 멈췄다(감사 delta-api-frontend-infra-1). 백엔드
// test_render_math_safety와 같은 판정: 절대 예산 안이면 통과, 넘으면 1/4 크기 대비 시간 비로
// 2차 비용만 잡는다(선형 ~4배, 2차 ~16배).
const LINEAR_SIZE = 200 * 1024;
const LINEAR_BUDGET_MS = 1000;
const LINEAR_MAX_SCALING = 8;
const PATHOLOGICAL_TEX = {
  '닫는 괄호 없는 \\\\[ 반복': (n) => `\\[${'\\\\['.repeat(Math.floor(n / 3))}`,
  '\\kern + 긴 공백(단위 없음)': (n) => `\\kern${' '.repeat(n)}x`,
  '\\kern + 긴 공백 + 부호': (n) => `\\kern${' '.repeat(n)}-5000em`,
  '닫는 괄호 없는 \\rule[ 반복': (n) => '\\rule['.repeat(Math.floor(n / 6)),
  '\\\\ + 긴 공백': (n) => `a\\\\${' '.repeat(n)}[-900em] b`,
  '닫는 중괄호 없는 \\raisebox{ 반복': (n) => '\\raisebox{-4000em '.repeat(Math.floor(n / 18)),
};

function elapsedMs(fn, input) {
  const start = performance.now();
  fn(input);
  return performance.now() - start;
}

function assertLinear(fn, build, label) {
  const elapsed = elapsedMs(fn, build(LINEAR_SIZE));
  if (elapsed < LINEAR_BUDGET_MS) return;
  const quarter = elapsedMs(fn, build(LINEAR_SIZE / 4));
  const ratio = elapsed / Math.max(quarter, 0.001);
  assert.ok(ratio < LINEAR_MAX_SCALING,
    `${label}: ${LINEAR_SIZE}자 ${elapsed.toFixed(0)}ms, 1/4 크기 ${quarter.toFixed(0)}ms(×${ratio.toFixed(1)}) — 2차 비용`);
}

for (const [label, build] of Object.entries(PATHOLOGICAL_TEX)) {
  test(`크기 묶기는 입력 길이에 선형이다 — ${label} (앱·내려받기 둘 다)`, () => {
    const guard = loadGuard();
    assertLinear(clampTexSizes, build, `core ${label}`);
    assertLinear(guard.clampTexSizes, build, `guard ${label}`);
    const small = build(4096);
    assert.equal(guard.clampTexSizes(small), clampTexSizes(small), '두 구현의 결과가 같다');
  });
}

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

test('renderMath: 상한보다 긴 TeX는 조판하지 않고 원문으로 둔다 — 앱 상한과 같은 값', () => {
  // KaTeX 자체도 긴 입력에 초선형이다('x+' 20만 자 = 11초) — 상한이 앱(ui.renderMath)과 같아야
  // 내려받은 HTML과 앱 화면이 같은 수식을 같은 모양으로 보인다.
  const guard = loadGuard();
  const katex = realKatexStub();
  const atCap = 'x+'.repeat(KATEX_MAX_TEX_CHARS / 2);
  assert.equal(atCap.length, KATEX_MAX_TEX_CHARS);
  const fits = stubTarget(atCap);
  assert.equal(guard.renderMath(katex, fits, atCap, false), true);
  assert.equal(katex.seen.length, 1, '상한 길이까지는 조판한다');
  assert.deepEqual(fits.attrs, {});

  const tooLong = `${atCap}y`;
  const target = stubTarget(tooLong);
  assert.equal(guard.renderMath(katex, target, tooLong, true), true, '끝 — 다시 시도하지 않는다');
  assert.equal(katex.seen.length, 1, 'KaTeX를 부르지 않는다');
  assert.equal(target.textContent, tooLong);
  assert.equal(target.attrs['data-math-fallback'], 'too-long');
  assert.match(target.attrs.title, /원문 TeX/);
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
