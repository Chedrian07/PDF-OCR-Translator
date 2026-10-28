import { test } from 'node:test';
import assert from 'node:assert/strict';
import { el, state } from '../js/state.js';
import { applyWorkflowResult } from '../js/results.js';
import { readWorkflow, renderWorkflowChoice } from '../js/workflow.js';
import { startTranslate } from '../js/translate.js';
import { openViewer } from '../js/viewer.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

test('compact result resets service visibility when reopening a legacy reader job', (t) => {
  const doc = installFakeDom(t);
  const savedEls = { ...el }, savedState = { ...state };
  t.after(() => { Object.assign(el, savedEls); Object.assign(state, savedState); });
  for (const key of ['markdownZipHelp', 'dlDoc', 'dlMd', 'dlZip']) el[key] = mount(doc, 'div');
  el.tabs = ['reader', 'preview', 'markdown', 'qa'].map((name) => {
    const tab = mount(doc, 'button'); tab.dataset.tab = name; return tab;
  });
  assert.equal(applyWorkflowResult({ workflow: 'markdown_zip' }), true);
  assert.deepEqual(el.tabs.filter((tab) => !tab.hidden).map((tab) => tab.dataset.tab), ['preview', 'markdown']);
  assert.equal(el.dlDoc.hidden, true);
  assert.equal(el.dlZip.classList.contains('btn-primary'), true);
  assert.equal(el.markdownZipHelp.hidden, false);
  assert.equal(applyWorkflowResult({}), false);
  assert.equal(state.currentWorkflow, 'reader');
  assert.ok(el.tabs.every((tab) => !tab.hidden));
  assert.equal(el.dlDoc.hidden, false);
  assert.equal(el.dlZip.classList.contains('btn-primary'), false);
  assert.equal(el.markdownZipHelp.hidden, true);
});

test('Markdown service rejects translation and fullscreen viewer actions without requests', async (t) => {
  const saved = { ...state };
  t.after(() => Object.assign(state, saved));
  Object.assign(state, { currentWorkflow: 'markdown_zip', currentJobId: 'md-job', displayedStatus: 'done' });
  t.mock.method(globalThis, 'fetch', () => { throw new Error('Unexpected network request'); });
  await startTranslate();
  openViewer();
  assert.equal(state.viewerOpen, saved.viewerOpen);
});

test('upload workflow comes from selection and defaults to existing reader service', (t) => {
  const saved = el.workflowRadios;
  t.after(() => { el.workflowRadios = saved; });
  el.workflowRadios = [];
  assert.equal(readWorkflow(), 'reader');
  el.workflowRadios = [{ checked: true, value: 'markdown_zip' }];
  assert.equal(readWorkflow(), 'markdown_zip');
});


test('service selection updates upload and empty state while active job stays independent', (t) => {
  const doc = installFakeDom(t);
  const savedEls = { ...el }, savedState = { ...state };
  t.after(() => { Object.assign(el, savedEls); Object.assign(state, savedState); });
  for (const key of ['uploadBtn', 'workflowHelp', 'readerFeatures', 'serviceHeading', 'serviceDescription']) el[key] = mount(doc, 'div');
  el.workflowRadios = [{ value: 'reader', checked: true }, { value: 'markdown_zip', checked: false }];
  state.uploading = false;
  state.currentWorkflow = 'reader';
  renderWorkflowChoice('markdown_zip');
  assert.equal(readWorkflow(), 'markdown_zip');
  assert.equal(state.currentWorkflow, 'reader');
  assert.equal(el.uploadBtn.textContent, 'Markdown ZIP 만들기');
  assert.equal(el.workflowHelp.hidden, false);
  assert.equal(el.readerFeatures.hidden, true);
  assert.match(el.serviceDescription.textContent, /모든 원본 페이지/);
  renderWorkflowChoice('reader');
  assert.equal(el.uploadBtn.textContent, '변환 시작');
  assert.equal(el.workflowHelp.hidden, true);
  assert.equal(el.readerFeatures.hidden, false);
});
