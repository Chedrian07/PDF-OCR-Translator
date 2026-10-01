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

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { KATEX_MAX_EXPAND, KATEX_MAX_SIZE_EM, katexOptions } from '../js/constants.js';

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

test('앱의 모든 katex.render 호출은 katexOptions를 쓴다', () => {
  let calls = 0;
  for (const file of frontendSources()) {
    const src = fs.readFileSync(file, 'utf8');
    for (const m of src.matchAll(/katex\.render\(([^;]*?)\);/gs)) {
      calls += 1;
      assert.match(m[1], /katexOptions\(/, `${path.relative(FRONTEND, file)}: ${m[0].slice(0, 120)}`);
    }
  }
  assert.ok(calls >= 2, `katex.render 호출부를 찾지 못했다 (${calls})`);
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
