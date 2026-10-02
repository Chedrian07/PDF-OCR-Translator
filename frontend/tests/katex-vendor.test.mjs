// 벤더 KaTeX 사본과 렌더 옵션 계약 — DOM 없이 Node에서 직접 검증한다.
//
//   node --test frontend/tests/katex-vendor.test.mjs
//
// 여기서 지키는 계약:
//  · vendor/katex/VERSION은 katex.min.js 번들의 version 문자열과 같다(드리프트 방지).
//    벤더 사본은 어떤 의존성 스캐너에도 걸리지 않으므로 이 대조가 유일한 경보다.
//  · 번들은 GHSA-238p-pmpm-9mq7(설정 프로토타입 오염) 수정본 0.18.2 이상이다.
//  · katex.min.css가 참조하는 woff2 폰트가 전부 fonts/에 있다(부분 갱신 방지).
//  · 앱의 모든 katex.render 호출은 공용 katexOptions(상한·trust 명시)를 쓴다.
//  · KaTeX maxSize는 양수 크기만 묶는다 — 음수 크기(\raisebox{-4000em} 등)는 clampTexSizes로
//    묶고, 매크로로 만든 거대 박스는 katexStyleOversized로 잡아 원문 TeX로 되돌린다(frontend-4).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  KATEX_MAX_BOX_EM, KATEX_MAX_EXPAND, KATEX_MAX_SIZE_EM, KATEX_MAX_TEX_CHARS, katexOptions,
} from '../js/constants.js';
import { clampTexSizes, katexStyleOversized } from '../js/core.js';
import { renderMath, typesetMath } from '../js/ui.js';
import { mathTextNodes } from '../js/reader.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const VENDOR = path.join(FRONTEND, 'vendor', 'katex');
const GHSA_FIXED = [0, 18, 2]; // GHSA-238p-pmpm-9mq7 patched in 0.18.2

function bundleVersion() {
  const js = fs.readFileSync(path.join(VENDOR, 'katex.min.js'), 'utf8');
  const found = [...js.matchAll(/version:"(\d+\.\d+\.\d+)"/g)].map((m) => m[1]);
  return [...new Set(found)];
}

// UMD 번들을 같은 realm에서 CommonJS로 평가한다 (frontend/package.json이 type:module이라
// require로는 UMD 분기가 깨진다). 같은 realm이어야 Object.prototype 오염 실험이 유효하다.
function loadKatex() {
  const src = fs.readFileSync(path.join(VENDOR, 'katex.min.js'), 'utf8');
  const mod = { exports: {} };
  // eslint-disable-next-line no-new-func
  new Function('module', 'exports', src)(mod, mod.exports);
  return mod.exports;
}

function versionAtLeast(version, floor) {
  const parts = version.split('.').map(Number);
  for (let i = 0; i < floor.length; i += 1) {
    if ((parts[i] || 0) !== floor[i]) return (parts[i] || 0) > floor[i];
  }
  return true;
}

function frontendSources() {
  const files = [path.join(FRONTEND, 'app.js')];
  for (const name of fs.readdirSync(path.join(FRONTEND, 'js'))) {
    if (name.endsWith('.js')) files.push(path.join(FRONTEND, 'js', name));
  }
  return files;
}

test('KaTeX VERSION 파일이 번들의 version 문자열과 일치한다', () => {
  const declared = fs.readFileSync(path.join(VENDOR, 'VERSION'), 'utf8').trim();
  const versions = bundleVersion();
  assert.deepEqual(versions, [declared], `VERSION=${declared}, bundle=${versions.join(',')}`);
});

test('KaTeX 번들은 GHSA-238p-pmpm-9mq7 수정본(0.18.2) 이상이다', () => {
  const [version] = bundleVersion();
  assert.ok(versionAtLeast(version, GHSA_FIXED), `vendored KaTeX ${version} < 0.18.2`);
  assert.equal(loadKatex().version, version);
});

test('KaTeX CSS가 참조하는 woff2 폰트가 모두 벤더 사본에 있다', () => {
  const css = fs.readFileSync(path.join(VENDOR, 'katex.min.css'), 'utf8');
  const fonts = [...new Set([...css.matchAll(/url\((fonts\/[^)]+\.woff2)\)/g)].map((m) => m[1]))];
  assert.ok(fonts.length >= 10, `woff2 참조가 너무 적다: ${fonts.length}`);
  const missing = fonts.filter((rel) => !fs.existsSync(path.join(VENDOR, rel)));
  assert.deepEqual(missing, []);
  assert.ok(fs.existsSync(path.join(VENDOR, 'LICENSE')), 'MIT LICENSE 사본이 있어야 한다');
});

test('katexOptions: 상한과 trust·strict를 명시한 새 객체를 돌려준다', () => {
  const inline = katexOptions(false);
  const display = katexOptions(1);
  assert.deepEqual(inline, {
    displayMode: false,
    throwOnError: false,
    maxSize: KATEX_MAX_SIZE_EM,
    maxExpand: KATEX_MAX_EXPAND,
    strict: 'ignore',
    trust: false,
  });
  assert.equal(display.displayMode, true);
  assert.notEqual(inline, katexOptions(false), '호출마다 새 객체 — KaTeX가 옵션을 바꿔도 공유 상태가 오염되지 않는다');
  assert.ok(Number.isFinite(inline.maxSize) && inline.maxSize > 0);
  assert.ok(Number.isFinite(inline.maxExpand) && inline.maxExpand > 0);
});

test('앱의 katex.render는 ui.renderMath 한 곳뿐이고, 크기를 묶은 TeX와 katexOptions를 쓴다', () => {
  const sites = [];
  for (const file of frontendSources()) {
    const src = fs.readFileSync(file, 'utf8');
    for (const m of src.matchAll(/katex\.render\(([^;]*?)\);/gs)) {
      sites.push({ file: path.relative(FRONTEND, file), args: m[1] });
    }
  }
  // 다른 모듈이 katex.render를 직접 부르면 음수 크기 상한·원문 폴백(frontend-4)을 건너뛴다.
  assert.deepEqual(sites.map((site) => site.file), [path.join('js', 'ui.js')], JSON.stringify(sites));
  assert.match(sites[0].args, /^clampTexSizes\(/);
  assert.match(sites[0].args, /katexOptions\(/);
});

test('maxSize: 거대한 \\rule 박스가 상한으로 잘린다', () => {
  const katex = loadKatex();
  const styles = (html) => [...html.matchAll(/style="([^"]*)"/g)].map((m) => m[1]).join(';');
  const capped = styles(katex.renderToString('\\rule{2000em}{2000em}', katexOptions(false)));
  assert.ok(!capped.includes('2000em'), capped);
  assert.ok(capped.includes(`${KATEX_MAX_SIZE_EM}em`), capped);
  // 대조군: 옵션 없이 렌더하면 그대로 2000em — 위 단정이 공허하지 않음을 보인다.
  const raw = styles(katex.renderToString('\\rule{2000em}{2000em}', { throwOnError: false }));
  assert.ok(raw.includes('2000em'), raw);
});

test('maxExpand: 무한 매크로 루프는 예외 없이 오류 표시로 끝난다', () => {
  const html = loadKatex().renderToString('\\def\\a{\\a}\\a', katexOptions(false));
  assert.match(html, /katex-error/);
  assert.match(html, /Too many expansions/);
});

test('trust: Object.prototype이 오염돼도 \\href가 링크로 렌더되지 않는다', () => {
  const katex = loadKatex();
  // eslint-disable-next-line no-extend-native
  Object.prototype.trust = true;
  try {
    const html = katex.renderToString('\\href{https://evil.example/x}{x}', katexOptions(false));
    assert.ok(!/<a\s/i.test(html), html);
    assert.ok(!html.includes('href="https://evil.example'), html);
  } finally {
    delete Object.prototype.trust;
  }
});

/* ---------------- 음수·거대 크기 (frontend-4) ---------------- */

const styleList = (html) => [...html.matchAll(/style="([^"]*)"/g)].map((m) => m[1]);
const maxEm = (html) => Math.max(0, ...styleList(html)
  .flatMap((st) => [...st.matchAll(/(-?(?:\d+(?:\.\d*)?|\.\d+))em/g)].map((m) => Math.abs(Number(m[1])))));
const ABUSE = [
  '\\raisebox{-4000em}{x}', '\\rule[-3000em]{1em}{1em}', '\\kern{-5000em}x', '\\kern-5000em x',
  'a\\\\[-900em] b', '\\hspace{-300em}x', '\\hspace*{-400em}x', '\\mkern-9000mu x',
  '\\raisebox{- 4000 em}{x}', '\\raisebox{-4000cm}{x}', '\\kern{ - 5000 em }x',
];

test('KaTeX maxSize는 음수 크기를 묶지 않는다 — clampTexSizes를 거치면 ±상한 안', () => {
  const katex = loadKatex();
  // 대조군: 옵션만으로는 그대로 수천 em — 아래 단정이 공허하지 않음을 보인다.
  assert.ok(maxEm(katex.renderToString('\\raisebox{-4000em}{x}', katexOptions(false))) >= 4000);
  for (const tex of ABUSE) {
    for (const display of [false, true]) {
      const html = katex.renderToString(clampTexSizes(tex), katexOptions(display));
      assert.ok(maxEm(html) <= 2 * KATEX_MAX_SIZE_EM, `${tex} → ${maxEm(html)}em`);
      assert.ok(!html.includes('katex-error'), `${tex}: 묶은 TeX도 조판된다`);
    }
  }
});

test('clampTexSizes: 크기 인자 자리만 같은 단위로 묶고, 크기가 아닌 글자·작은 크기는 그대로', () => {
  assert.equal(clampTexSizes('\\raisebox{-4000em}{x}'), '\\raisebox{-10em}{x}');
  assert.equal(clampTexSizes('\\rule[-3000em]{1em}{1em}'), '\\rule[-10em]{1em}{1em}');
  assert.equal(clampTexSizes('\\kern-5000em x'), '\\kern-10em x');
  assert.equal(clampTexSizes('a\\\\[-900em] b'), 'a\\\\[-10em] b');
  assert.equal(clampTexSizes('\\mkern-9000mu x'), '\\mkern-180mu x', 'mu는 mu로 묶는다');
  assert.equal(clampTexSizes('\\raisebox{-4000pt}{x}'), '\\raisebox{-100pt}{x}');
  for (const same of [
    '\\text{150 cm} + 300 mm', 'h = 150\\,\\mathrm{cm}', '\\kern-.5em x', '\\\\[2pt]',
    '\\hspace{1em}', '\\def\\x{-4000em}\\kern\\x y', 'x^2', '', '\\kern-4000qq x',
  ]) {
    assert.equal(clampTexSizes(same), same, same);
  }
  assert.equal(clampTexSizes(null), '');
});

test('katexStyleOversized: 매크로·arraystretch로 만든 거대 박스는 잡고, 정상 큰 수식은 통과', () => {
  const katex = loadKatex();
  const oversized = (tex) => styleList(katex.renderToString(clampTexSizes(tex), katexOptions(true)))
    .some((st) => katexStyleOversized(st));
  for (const tex of [
    '\\def\\x{-4000em}\\kern\\x y', '\\def\\x{-4000em}\\raisebox{\\x}{y}',
    '\\def\\arraystretch{1000}\\begin{array}{c}a\\\\b\\end{array}', '\\def\\x{-40}\\kern\\x00em x',
  ]) {
    assert.equal(oversized(tex), true, tex);
  }
  const aligned = `\\begin{aligned}${Array.from({ length: 30 }, (_, i) => `x_{${i}} &= \\frac{a}{b}`).join('\\\\')}\\end{aligned}`;
  const matrix = `\\begin{pmatrix}${Array.from({ length: 30 }, (_, i) => `a_{${i}}`).join('\\\\')}\\end{pmatrix}`;
  for (const tex of [aligned, matrix, '\\sum_{i=1}^n \\frac{x_i^2}{\\sqrt{y}} \\int_0^1 f(x) dx', ...ABUSE]) {
    assert.equal(oversized(tex), false, tex.slice(0, 60));
  }
  assert.equal(katexStyleOversized(`height:${KATEX_MAX_BOX_EM + 1}em`), true);
  assert.equal(katexStyleOversized(`vertical-align:-${KATEX_MAX_BOX_EM}em`), false);
  assert.equal(katexStyleOversized('margin-right:-1e+21em'), true);
  assert.equal(katexStyleOversized(null), false);
});

// 브라우저 katex.render 대신: 실제 번들의 renderToString 결과 style을 가진 span을 붙인다.
function installKatexStub(t) {
  const real = loadKatex();
  const seen = [];
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'katex');
  globalThis.katex = {
    render(tex, target, options) {
      seen.push(tex);
      const html = real.renderToString(tex, options);
      target.textContent = '';
      for (const style of styleList(html)) {
        const span = document.createElement('span');
        span.setAttribute('style', style);
        target.appendChild(span);
      }
    },
  };
  t.after(() => {
    if (saved) Object.defineProperty(globalThis, 'katex', saved);
    else delete globalThis.katex;
  });
  return seen;
}

test('renderMath·typesetMath: 크기를 묶어 조판하고, 그래도 거대하면 원문 TeX로 되돌린다', (t) => {
  const doc = installFakeDom(t);
  const seen = installKatexStub(t);
  const target = mount(doc, 'span');
  assert.equal(renderMath(target, '\\raisebox{-4000em}{x}', false), true);
  assert.equal(seen.at(-1), '\\raisebox{-10em}{x}', 'KaTeX에는 묶은 TeX가 간다');
  assert.equal(target.dataset.mathFallback, undefined);
  const macro = '\\def\\x{-4000em}\\kern\\x y';
  assert.equal(renderMath(target, macro, true), true);
  assert.equal(target.textContent, macro, '거대 박스 대신 원문 TeX 글자');
  assert.equal(target.dataset.mathFallback, 'oversized');
  assert.match(target.getAttribute('title'), /원문 TeX/);

  const root = mount(doc, 'div');
  const inline = doc.createElement('span');
  inline.className = 'math-inline';
  inline.textContent = '\\def\\arraystretch{1000}\\begin{array}{c}a\\\\b\\end{array}';
  root.appendChild(inline);
  typesetMath(root);
  assert.equal(inline.dataset.mathDone, '1', '되돌린 수식도 다시 조판하지 않는다');
  assert.equal(inline.dataset.mathFallback, 'oversized');
});

test('renderMath: 상한보다 긴 TeX는 KaTeX에 넘기지 않고 원문 글자로 둔다', (t) => {
  // KaTeX 조판은 긴 입력에 초선형이다('x+' 5만 자 0.5초, 20만 자 11초) — 적대적 PDF의 거대
  // 수식 스팬 하나가 미리보기·리더를 멈추지 않게 한다(delta-api-frontend-infra-1).
  const doc = installFakeDom(t);
  const seen = installKatexStub(t);
  const target = mount(doc, 'span');
  const tooLong = 'x+'.repeat(KATEX_MAX_TEX_CHARS / 2 + 1);
  assert.equal(renderMath(target, tooLong, true), true, '끝 — typesetMath가 다시 시도하지 않는다');
  assert.deepEqual(seen, [], 'KaTeX를 부르지 않는다');
  assert.equal(target.textContent, tooLong);
  assert.equal(target.dataset.mathFallback, 'too-long');
  assert.match(target.getAttribute('title'), /원문 TeX/);

  const fits = mount(doc, 'span');
  assert.equal(renderMath(fits, 'x^2', false), true);
  assert.deepEqual(seen, ['x^2']);
  assert.equal(fits.dataset.mathFallback, undefined);
});

test('리더 카드 수식(mathTextNodes)도 renderMath를 거친다 — 크기를 묶고 거대 결과는 원문 TeX', (t) => {
  installFakeDom(t);
  const seen = installKatexStub(t);
  const nodes = mathTextNodes('앞 \\(\\raisebox{-4000em}{x}\\) 뒤 \\[\\def\\x{-4000em}\\kern\\x y\\]');
  assert.deepEqual(seen, ['\\raisebox{-10em}{x}', '\\def\\x{-4000em}\\kern\\x y']);
  const [inline, display] = nodes.filter((node) => node.nodeType === 1);
  assert.equal(inline.className, 'math-inline');
  assert.equal(inline.dataset.mathFallback, undefined);
  assert.equal(display.className, 'math-display');
  assert.equal(display.dataset.mathFallback, 'oversized');
  assert.equal(display.textContent, '\\def\\x{-4000em}\\kern\\x y');
});
