// 지원 브라우저 하한 — 앱 모듈 그래프가 파싱 단계에서 깨지는 문법을 막는다(정적 검사).
//
//   node --test frontend/tests/browser-compat.test.mjs
//
// 정규식 리터럴의 lookbehind(후방 탐색)는 Safari 16.4(2023-03) 미만에서 모듈 파싱 단계의
// SyntaxError다. core.js는 app.js를 포함한 거의 모든 모듈이 정적으로 가져오므로, 한 줄의
// lookbehind가 앱 전체를 빈 화면으로 만든다(frontend-5). 런타임 API와 달리 기능 검사로 감쌀
// 수도 없다 — 소스에 아예 두지 않는다.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { warningSegments } from '../js/core.js';

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// 브라우저가 실행하는 모든 스크립트 — 앱 모듈, 클래식 스크립트, 벤더 번들.
function browserScripts() {
  const files = ['app.js', 'layout-fit.js', 'theme-init.js', path.join('vendor', 'katex', 'katex.min.js')];
  for (const name of fs.readdirSync(path.join(FRONTEND, 'js'))) {
    if (name.endsWith('.js')) files.push(path.join('js', name));
  }
  return files;
}

test('브라우저가 실행하는 스크립트에 정규식 lookbehind가 없다 (Safari 16.4 미만 파싱 실패)', () => {
  const found = [];
  for (const rel of browserScripts()) {
    const src = fs.readFileSync(path.join(FRONTEND, rel), 'utf8');
    src.split('\n').forEach((line, i) => {
      if (/\(\?<[=!]/.test(line)) found.push(`${rel}:${i + 1}: ${line.trim().slice(0, 100)}`);
    });
  }
  assert.deepEqual(found, []);
});

const pages = (text) => warningSegments(text).filter((s) => s.type === 'page').map((s) => s.page);

test('warningSegments: lookbehind 없이도 앞 글자 규칙이 같다 — 거절한 자리 바로 뒤부터 다시 찾는다', () => {
  assert.deepEqual(pages('3페이지4페이지'), [3, 4], '붙어 있는 두 언급');
  assert.deepEqual(pages('2/3–5페이지'), [5], '분수 뒤 범위의 끝 쪽은 다시 찾는다');
  assert.deepEqual(pages('1.5페이지'), [], '소수');
  assert.deepEqual(pages('123456페이지'), [], '다섯 자리보다 긴 수의 꼬리');
  assert.deepEqual(pages('(7페이지)'), [7]);
  for (const text of ['3페이지4페이지', '2/3–5페이지', '1.5페이지', '(7페이지) 끝']) {
    assert.equal(warningSegments(text).map((s) => s.value).join(''), text);
  }
});
