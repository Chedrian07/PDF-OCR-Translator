import { test } from 'node:test';
import assert from 'node:assert/strict';

import { el, state } from '../js/state.js';
import {
  applyTranslateAvailability, connectTranslateEvents, initTranslateForJob,
  startTranslate, startTranslatePolling, teardownTranslate,
} from '../js/translate.js';

function element() {
  const attributes = new Map();
  return {
    disabled: false, hidden: false, dataset: {}, style: {}, textContent: '',
    classList: { toggle() {} },
    setAttribute(name, value) { attributes.set(name, value); },
    removeAttribute(name) { attributes.delete(name); },
  };
}

function setup(t) {
  const savedState = { ...state };
  const savedEls = { ...el };
  Object.assign(state, {
    currentJobId: 'job-a', displayedStatus: 'done', translateGen: 0,
    translateState: 'none', translateRequestPending: false, translateAvailable: true,
    translateEs: null, translatePollTimer: 0, translateSseErrors: 0,
    translateRetryAt: 0, toastTimer: 0,
  });
  for (const key of [
    'translateBtn', 'readerTranslateBtn', 'translateProgress', 'langToggle',
    'viewerLangToggle', 'translateCancel', 'translateProgressLabel',
    'translateProgressTrack', 'translateProgressFill', 'toast',
  ]) el[key] = element();
  t.after(() => {
    teardownTranslate();
    clearTimeout(state.toastTimer);
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function response(status, data) {
  return { ok: status >= 200 && status < 300, status,
    headers: { get() { return null; } },
    async text() { return JSON.stringify(data); } };
}

test('translation start stays locked during health refresh and rejects duplicate starts', async (t) => {
  setup(t);
  const request = deferred();
  let calls = 0;
  t.mock.method(globalThis, 'fetch', () => { calls += 1; return request.promise; });

  const pending = startTranslate();
  state.translateAvailable = false;
  applyTranslateAvailability();
  state.translateAvailable = true;
  applyTranslateAvailability();
  assert.equal(el.translateBtn.disabled, true);
  assert.equal(el.readerTranslateBtn.disabled, true);
  await startTranslate();
  assert.equal(calls, 1);

  request.resolve(response(503, { detail: 'Provider is unavailable' }));
  await pending;
  assert.equal(state.translateRequestPending, false);
  assert.equal(el.translateBtn.disabled, false);
  assert.equal(el.readerTranslateBtn.disabled, false);
});

test('old translation POST cannot unlock a new request after A to B to A', async (t) => {
  setup(t);
  const oldRequest = deferred();
  const newRequest = deferred();
  let calls = 0;
  t.mock.method(globalThis, 'fetch', () => (++calls === 1 ? oldRequest : newRequest).promise);
  const oldPending = startTranslate();
  teardownTranslate();
  state.currentJobId = 'job-b';
  teardownTranslate();
  state.currentJobId = 'job-a';
  const newPending = startTranslate();

  oldRequest.resolve(response(503, { detail: 'Old failure' }));
  await oldPending;
  assert.equal(state.translateRequestPending, true);
  assert.equal(el.translateBtn.disabled, true);
  assert.equal(el.toast.textContent, '');

  newRequest.resolve(response(503, { detail: 'Current failure' }));
  await newPending;
  assert.equal(state.translateRequestPending, false);
  assert.equal(el.toast.textContent, 'Current failure');
});

test('obsolete translation state response cannot overwrite a reopened job', async (t) => {
  setup(t);
  const request = deferred();
  t.mock.method(globalThis, 'fetch', () => request.promise);
  const pending = initTranslateForJob();
  teardownTranslate();
  state.translateState = 'running';
  el.translateBtn.hidden = true;
  request.resolve(response(200, { status: 'none' }));
  await pending;
  assert.equal(state.translateState, 'running');
  assert.equal(el.translateBtn.hidden, true);
});

test('replaced translation EventSource cannot send progress into its replacement', (t) => {
  setup(t);
  const sources = [];
  class MockEventSource {
    constructor() { this.listeners = new Map(); this.closed = false; sources.push(this); }
    addEventListener(name, callback) { this.listeners.set(name, callback); }
    close() { this.closed = true; }
    emit(name, data) { this.listeners.get(name)({ data: JSON.stringify(data) }); }
  }
  const savedSource = Object.getOwnPropertyDescriptor(globalThis, 'EventSource');
  globalThis.EventSource = MockEventSource;
  t.after(() => {
    if (savedSource) Object.defineProperty(globalThis, 'EventSource', savedSource);
    else delete globalThis.EventSource;
  });
  connectTranslateEvents('job-a');
  connectTranslateEvents('job-a');
  assert.equal(sources[0].closed, true);
  sources[1].emit('progress', { current: 2, total: 3 });
  sources[0].emit('progress', { current: 1, total: 3 });
  assert.equal(el.translateProgressLabel.textContent, '번역 중 2/3');
});

test('translation polls do not overlap and old polls cannot stop a replacement timer', async (t) => {
  setup(t);
  const ticks = [];
  const cleared = [];
  t.mock.method(globalThis, 'setInterval', (callback) => { ticks.push(callback); return ticks.length; });
  t.mock.method(globalThis, 'clearInterval', (timer) => { cleared.push(timer); });
  const oldRequest = deferred();
  let calls = 0;
  t.mock.method(globalThis, 'fetch', () => { calls += 1; return oldRequest.promise; });
  startTranslatePolling('job-a');
  const oldPending = ticks[0]();
  await ticks[0]();
  assert.equal(calls, 1);

  startTranslatePolling('job-a');
  oldRequest.resolve(response(200, { status: 'error', error: 'Old failure' }));
  await oldPending;
  assert.equal(state.translatePollTimer, 2);
  assert.deepEqual(cleared, [1]);
  assert.equal(el.toast.textContent, '');
});

test('polling a missing translation restores an available start button', async (t) => {
  setup(t);
  let tick;
  t.mock.method(globalThis, 'setInterval', (callback) => { tick = callback; return 1; });
  t.mock.method(globalThis, 'clearInterval', () => {});
  t.mock.method(globalThis, 'fetch', async () => response(200, { status: 'none' }));
  state.translateState = 'running';
  startTranslatePolling('job-a');
  await tick();
  assert.equal(state.translateState, 'none');
  assert.equal(el.translateBtn.disabled, false);
  assert.equal(el.readerTranslateBtn.disabled, false);
});
