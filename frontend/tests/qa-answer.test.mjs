// 질문(Q&A) 답변 — 서버가 렌더한 조각(answer_html)으로 그린다.
//
//   node --test frontend/tests/qa-answer.test.mjs
//
// 로컬 모델 답변은 마크다운·TeX다('제안하는 방법의 이름은 **TURBOQUANT**입니다.'·'$$D…$$'). 질문 탭은
// node.textContent = answer로 넣어 별표와 TeX 원문이 그대로 보였다(실앱 local-openai). 서버는 /html과
// 같은 안전 렌더러(텍스트 이스케이프·잡 파일 이미지만)로 만든 answer_html을 함께 준다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { renderQaAnswer } from '../js/qa.js';
import { el, state } from '../js/state.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

function mountLog(t) {
  const doc = installFakeDom(t);
  const savedEls = { ...el };
  const savedState = { ...state };
  el.qaLog = mount(doc, 'div');
  Object.assign(state, { qaCatalog: null, qaProvider: 'local-openai' });
  t.after(() => {
    Object.assign(el, savedEls);
    Object.assign(state, savedState);
  });
  return doc;
}

test('renderQaAnswer: answer_html이 있으면 그 조각을 마크다운 본문으로 붙인다', (t) => {
  const doc = mountLog(t);
  const node = mount(doc, 'div');
  node.classList.add('qa-msg', 'assistant', 'loading');
  const html = '<p>이름은 <strong>TURBOQUANT</strong>이다.</p><div class="math-display">x</div>';

  renderQaAnswer(node, {
    answer: '이름은 **TURBOQUANT**이다.\n\n$$x$$', answer_html: html,
    provider: 'local-openai', model: 'default_model',
  });

  assert.equal(node.classList.contains('loading'), false);
  const body = node.querySelector('.qa-answer');
  assert.ok(body, '렌더된 답변 본문이 없다 — 별표·TeX가 글자 그대로 보인다');
  assert.ok(body.classList.contains('markdown-body'));
  assert.equal(body.innerHTML, html);
  assert.ok(!node.textContent.includes('**TURBOQUANT**'), node.textContent);
  assert.ok(node.querySelector('.qa-meta'), '공급자·모델 메타 줄은 그대로');
});

test('renderQaAnswer: answer_html이 없으면(구 서버) 원문 글자 그대로', (t) => {
  const doc = mountLog(t);
  const node = mount(doc, 'div');

  renderQaAnswer(node, { answer: '<b>굵게</b> **별표**', provider: 'ollama', model: 'm' });

  assert.equal(node.querySelector('.qa-answer'), null);
  assert.ok(node.textContent.startsWith('<b>굵게</b> **별표**'), node.textContent);
  assert.equal(node.innerHTML, '', '원문을 HTML로 해석하지 않는다');
});
