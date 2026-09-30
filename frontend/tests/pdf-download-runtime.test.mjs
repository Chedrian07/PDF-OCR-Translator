import { test } from 'node:test';
import assert from 'node:assert/strict';

import { el, state } from '../js/state.js';
import { downloadPdfWithReport } from '../js/reader.js';

function anchor(href, filename) {
  const attributes = new Map([['href', href], ['download', filename]]);
  return {
    dataset: {}, textContent: 'PDF 다운로드',
    classList: { contains() { return false; } },
    getAttribute(name) { return attributes.get(name) || null; },
    setAttribute(name, value) { attributes.set(name, value); },
    removeAttribute(name) { attributes.delete(name); },
  };
}

test('PDF download keeps its original filename when the user opens another job', async (t) => {
  const savedState = { ...state };
  const savedEls = { ...el };
  const savedDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  t.after(() => {
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
    if (savedDocument) Object.defineProperty(globalThis, 'document', savedDocument);
    else delete globalThis.document;
  });
  state.pdfDownloadBusy = false;
  el.dlPdf = anchor('/api/jobs/job-a/pdf?lang=ko&view=dual', 'paper-a.ko.pdf');
  el.viewerDlPdf = anchor('/api/jobs/job-a/pdf?lang=ko&view=dual', 'paper-a.ko.pdf');
  el.toast = { textContent: '', className: '', hidden: true };
  const downloads = [];
  globalThis.document = {
    body: { appendChild() {} },
    createElement() {
      return { click() { downloads.push({ href: this.href, filename: this.download }); }, remove() {} };
    },
  };
  t.mock.method(URL, 'createObjectURL', () => 'blob:job-a');
  t.mock.method(globalThis, 'setTimeout', () => 0);
  let resolve;
  const response = new Promise((done) => { resolve = done; });
  const fetches = [];
  t.mock.method(globalThis, 'fetch', (url) => { fetches.push(url); return response; });

  const pending = downloadPdfWithReport({ preventDefault() {}, currentTarget: el.dlPdf });
  for (const button of [el.dlPdf, el.viewerDlPdf]) {
    button.setAttribute('href', '/api/jobs/job-b/pdf?lang=ko&view=dual');
    button.setAttribute('download', 'paper-b.ko.pdf');
  }
  resolve(new Response('%PDF-1.4\n', { headers: { 'Content-Type': 'application/pdf' } }));
  await pending;
  assert.deepEqual(fetches, ['/api/jobs/job-a/pdf?lang=ko&view=dual']);
  assert.deepEqual(downloads, [{ href: 'blob:job-a', filename: 'paper-a.ko.pdf' }]);
  assert.equal(state.pdfDownloadBusy, false);
});
