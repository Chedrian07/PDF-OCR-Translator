// 질문(Q&A) 첫 공급자 — 로컬 의도는 원격으로 넘어가지 않고, 자동으로 고른 값은 저장하지 않는다.
//
//   node --test frontend/tests/qa-provider.test.mjs
//
// 공급자 가용성은 실시간 프로브다(local-openai는 GET /models 2.5초, Ollama는 /api/tags). 앱을 연
// 순간 로컬 서버가 아직 안 떴거나 바쁘면 '못 씀'이 된다. 예전에는 그때 첫 가용 공급자(원격
// OpenAI)로 바꾸고 그 값을 localStorage에 덮어써, 사용자가 사적 문서 때문에 고른 로컬 공급자가
// 조용히 원격 유료 API로 바뀐 채 굳었다 — 다음 질문부터 페이지 원문이 원격으로 나갔다(감사
// delta-api-frontend-infra-2).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { QA_LS_PROVIDER } from '../js/constants.js';
import { pickQaProvider } from '../js/core.js';
import { loadQaCatalog } from '../js/qa.js';
import { el, state } from '../js/state.js';
import { installFakeDom, installFakeStorage, mount } from './helpers/fake-dom.mjs';

function catalog({ defaultProvider = 'openai-responses', local = false, ollama = false, remote = true } = {}) {
  return {
    default_provider: defaultProvider,
    default_reasoning_effort: 'default',
    providers: [
      { id: 'openai-responses', label: 'OpenAI Responses', available: remote, remote: true,
        supports_reasoning_summary: true, models: ['gpt-5-mini'], default_model: 'gpt-5-mini' },
      { id: 'openai-chat', label: 'OpenAI Chat', available: remote, remote: true,
        supports_reasoning_summary: false, models: ['gpt-5-mini'], default_model: 'gpt-5-mini' },
      { id: 'ollama', label: 'Ollama Local', available: ollama, remote: false,
        supports_reasoning_summary: false, models: ollama ? ['qwen3:4b'] : [], default_model: 'qwen3:4b' },
      { id: 'local-openai', label: 'Local (MLX)', available: local, remote: false,
        supports_reasoning_summary: false, models: ['default_model'], default_model: 'default_model' },
    ],
  };
}

test('pickQaProvider: 저장된 로컬 공급자가 지금 못 써도 원격으로 넘어가지 않는다', () => {
  const busyLocal = catalog({ local: false, remote: true });
  assert.equal(pickQaProvider(busyLocal, 'local-openai'), 'local-openai',
    '예전에는 openai-responses — 문서 원문이 고르지 않은 원격 API로 나갔다');
  assert.equal(pickQaProvider(busyLocal, 'ollama'), 'ollama');
  // 운영자가 서버 기본을 로컬로 둔 배포(LLM_PROVIDER=local-openai)도 같다
  assert.equal(pickQaProvider(catalog({ defaultProvider: 'local-openai', remote: true }), null), 'local-openai');
});

test('pickQaProvider: 로컬 의도는 다른 로컬 공급자로만 넘어간다', () => {
  assert.equal(pickQaProvider(catalog({ ollama: true, remote: true }), 'local-openai'), 'ollama');
  assert.equal(pickQaProvider(catalog({ local: true, remote: true }), 'ollama'), 'local-openai');
});

test('pickQaProvider: 원격 의도를 못 쓰면(키 없음) 쓸 수 있는 로컬로 넘어간다 — 기기 밖으로 나가지 않는 방향', () => {
  const keyless = catalog({ local: true, remote: false });
  assert.equal(pickQaProvider(keyless, null), 'local-openai', '서버 기본(원격, 설정 필요) 대신');
  assert.equal(pickQaProvider(keyless, 'openai-chat'), 'local-openai');
  assert.equal(pickQaProvider(catalog({ local: true, remote: true }), 'openai-chat'), 'openai-chat',
    '쓸 수 있는 저장값은 그대로');
});

function mountQaControls(t) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const [key, tag] of [
    ['qaProvider', 'select'], ['qaModel', 'select'], ['qaEffort', 'select'], ['qaSummary', 'select'],
    ['qaSummaryField', 'div'], ['qaThinking', 'button'], ['toast', 'div'],
  ]) el[key] = mount(doc, tag);
  Object.assign(state, { qaCatalog: null, qaCatalogLoading: false, qaProvider: '', toastTimer: 0 });
  t.after(() => {
    clearTimeout(state.toastTimer);
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
}

function serveCatalog(t, body) {
  t.mock.method(globalThis, 'fetch', async () => ({
    ok: true, status: 200, headers: { get: () => null }, text: async () => JSON.stringify(body),
  }));
}

test('loadQaCatalog: 로컬 서버가 바쁠 때 고른 공급자를 저장하지 않아, 다음 세션도 사용자의 로컬 선택이다', async (t) => {
  mountQaControls(t);
  const storage = installFakeStorage(t);
  storage.setItem(QA_LS_PROVIDER, 'local-openai');
  serveCatalog(t, catalog({ local: false, remote: true }));
  await loadQaCatalog();
  assert.equal(state.qaProvider, 'local-openai', '원격(openai-responses)으로 바뀌지 않는다');
  assert.equal(storage.getItem(QA_LS_PROVIDER), 'local-openai');
  assert.equal(el.qaProvider.value, 'local-openai');
});

test('loadQaCatalog: 자동으로 고른 공급자는 저장하지 않는다 — 저장은 사용자가 바꿀 때만', async (t) => {
  mountQaControls(t);
  const storage = installFakeStorage(t);
  serveCatalog(t, catalog({ local: true, remote: false }));
  await loadQaCatalog();
  assert.equal(state.qaProvider, 'local-openai', '쓸 수 있는 공급자로 시작한다(P4)');
  assert.equal(storage.getItem(QA_LS_PROVIDER), null, '일시 상태로 고른 값이 사용자 선택으로 굳지 않는다');
});
