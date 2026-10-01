// 서버 렌더 문서 조각의 외부 이미지 차단 + index.html CSP 계약.
//
//   node --test frontend/tests/html-safety.test.mjs
//
// OCR·텍스트 레이어 마크다운의 ![](https://tracker.example/p.png)는 그대로 <img>가 되어
// 문서를 여는 순간 사용자 IP·열람 시각이 제3자·LAN 기기로 샜다(frontend-3). 프런트는
// <template>로 파싱해 붙이기 전에 막고, CSP(img-src 'self' data: blob:)가 최후 방어다.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { imageSrcAllowed } from '../js/core.js';
import { sanitizeImageSources } from '../js/ui.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ORIGIN = 'http://127.0.0.1:8000';

test('imageSrcAllowed: 같은 출처·data:image·blob:만 허용한다', () => {
  for (const src of [
    '/api/jobs/j_1/files/images/0_0.jpg', 'images/a.png', `${ORIGIN}/api/jobs/j/page/1`,
    'data:image/png;base64,AAAA', 'blob:http://127.0.0.1:8000/1234', '', null,
  ]) assert.equal(imageSrcAllowed(src, ORIGIN), true, String(src));
  for (const src of [
    'https://tracker.example/p.png?doc=42', '//evil.example/a.gif', 'http://192.168.0.1/cgi',
    'http://127.0.0.1:9000/x.png', 'HTTPS://TRACKER.EXAMPLE/x', 'data:text/html,<b>x</b>',
    'javascript:alert(1)', 'file:///etc/passwd', 'ftp://example.com/a.png',
  ]) assert.equal(imageSrcAllowed(src, ORIGIN), false, String(src));
});

function setup(t) {
  const doc = installFakeDom(t);
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'location', { value: { origin: ORIGIN }, configurable: true });
  t.after(() => {
    if (saved) Object.defineProperty(globalThis, 'location', saved);
    else delete globalThis.location;
  });
  return doc;
}

function img(doc, parent, attrs) {
  const node = doc.createElement('img');
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent.appendChild(node);
  return node;
}

test('sanitizeImageSources: 외부 이미지는 요청 전에 자리표시로 바꾸고 같은 출처는 둔다', (t) => {
  const doc = setup(t);
  const root = mount(doc, 'div');
  const p = doc.createElement('p');
  root.appendChild(p);
  const local = img(doc, p, { src: '/api/jobs/j/files/images/fig.png', alt: '그림 1' });
  img(doc, p, { src: 'https://tracker.example/p.png?doc=42', alt: 'beacon' });
  img(doc, p, { src: '//evil.example/a.gif' });
  sanitizeImageSources(root);
  const imgs = root.querySelectorAll('img');
  assert.equal(imgs.length, 1);
  assert.ok(imgs[0] === local, '같은 출처 그림은 그대로');
  const blocked = root.querySelectorAll('.blocked-image');
  assert.deepEqual(blocked.map((b) => b.textContent), ['[외부 이미지 차단됨: beacon]', '[외부 이미지 차단됨]']);
  assert.match(blocked[0].getAttribute('title'), /tracker\.example/);
  assert.equal(local.getAttribute('loading'), null, '요청하지 않으면 lazy를 강제하지 않는다');
});

test('sanitizeImageSources: lazyImages면 남은 그림에 loading=lazy·decoding=async', (t) => {
  const doc = setup(t);
  const root = mount(doc, 'div');
  const page = img(doc, root, { src: '/api/jobs/j/files/facsimile/page_0001.png' });
  sanitizeImageSources(root, { lazyImages: true });
  assert.equal(page.getAttribute('loading'), 'lazy');
  assert.equal(page.getAttribute('decoding'), 'async');
});

test('sanitizeImageSources: srcset·다른 미디어의 외부 주소 속성도 지운다', (t) => {
  const doc = setup(t);
  const root = mount(doc, 'div');
  const mixed = img(doc, root, { src: '/a.png', srcset: '/a.png 1x, https://cdn.example/a@2x.png 2x' });
  const okSet = img(doc, root, { src: '/a.png', srcset: '/a.png 1x, /a@2x.png 2x' });
  const video = doc.createElement('video');
  video.setAttribute('poster', 'https://tracker.example/poster.jpg');
  video.setAttribute('src', '/api/jobs/j/files/v.mp4');
  root.appendChild(video);
  const source = doc.createElement('source');
  source.setAttribute('src', 'http://10.0.0.1/x.webm');
  video.appendChild(source);
  sanitizeImageSources(root);
  assert.equal(mixed.hasAttribute('srcset'), false);
  assert.equal(okSet.getAttribute('srcset'), '/a.png 1x, /a@2x.png 2x');
  assert.equal(video.hasAttribute('poster'), false);
  assert.equal(video.getAttribute('src'), '/api/jobs/j/files/v.mp4');
  assert.equal(source.hasAttribute('src'), false);
});

/* ---------------- index.html 계약 ---------------- */

const html = fs.readFileSync(path.join(FRONTEND, 'index.html'), 'utf8');

function cspDirectives() {
  const m = html.match(/<meta\s+http-equiv="Content-Security-Policy"\s+content="([^"]+)"/i);
  assert.ok(m, 'index.html에 CSP meta가 있어야 한다');
  return Object.fromEntries(m[1].split(';').map((part) => part.trim()).filter(Boolean)
    .map((part) => { const [name, ...values] = part.split(/\s+/); return [name, values]; }));
}

test('CSP: 이미지는 같은 출처와 data:·blob:만, 연결도 같은 출처만', () => {
  const csp = cspDirectives();
  assert.deepEqual(csp['img-src'], ["'self'", 'data:', 'blob:']);
  assert.deepEqual(csp['connect-src'], ["'self'"]);
  assert.deepEqual(csp['default-src'], ["'self'"]);
  assert.deepEqual(csp['object-src'], ["'none'"]);
});

test('CSP: 스크립트는 같은 출처 파일만 — 인라인 스크립트가 없어야 그 정책이 성립한다', () => {
  const csp = cspDirectives();
  assert.deepEqual(csp['script-src'], ["'self'"]);
  const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
  assert.ok(scripts.length >= 3);
  for (const [, attrs, body] of scripts) {
    assert.match(attrs, /\bsrc="\.\//, `인라인 스크립트 금지: ${attrs}`);
    assert.equal(body.trim(), '', '스크립트 본문은 비어 있어야 한다');
  }
});

test('CSP meta는 리소스보다 먼저, 테마 부트스트랩은 동기 스크립트로 head에 있다', () => {
  const head = html.slice(0, html.indexOf('</head>'));
  const csp = head.search(/http-equiv="Content-Security-Policy"/i);
  const firstResource = head.search(/<(link|script)\b/i);
  assert.ok(csp >= 0 && csp < firstResource, 'CSP는 첫 link/script보다 앞서야 적용된다');
  const theme = head.match(/<script\b([^>]*)src="\.\/theme-init\.js"([^>]*)>/i);
  assert.ok(theme, 'theme-init.js가 head에 있어야 한다');
  assert.doesNotMatch(theme[0], /\b(defer|async|type="module")\b/, '첫 페인트 전에 실행돼야 한다');
  assert.ok(fs.existsSync(path.join(FRONTEND, 'theme-init.js')));
  assert.match(head, /<meta name="referrer" content="no-referrer">/);
});
