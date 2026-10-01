// 리더 런타임 테스트 공용 준비 — 가짜 DOM에 리더 요소를 붙이고 상태를 초기화한다.
// 실제 js/reader.js를 그대로 돌리기 위한 것이다(테스트 파일끼리 서로 import하지 않게
// 여기로 모은다).

import { el, state } from '../../js/state.js';
import { resetReaderDocumentState } from '../../js/reader.js';
import { installFakeDom, mount } from './fake-dom.mjs';

export function alignment(page, ids) {
  return {
    page,
    lang: 'orig',
    blocks: ids.map((id, index) => ({
      id, index, type: 'text', source: `Block ${id}`, target: `Block ${id}`, translated: false,
      rect: { left: 10, top: 10 + index * 20, width: 50, height: 10 },
    })),
  };
}

export function setupReader(t, { total = 2 } = {}) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  for (const key of [
    'readerContent', 'readerPageStage', 'readerPagePane', 'viewerRoot', 'readerOutline',
    'viewerThumbnails', 'readerProgressFill', 'readerMapStatus', 'readerActivity',
    'readerSelection', 'readerVisualLabel', 'readerRailTitle', 'resultSection', 'toast',
  ]) el[key] = mount(doc, 'div');
  el.readerPageInput = mount(doc, 'input');
  el.readerTotal = mount(doc, 'span');
  for (const key of ['readerPrev', 'readerNext', 'readerExplain', 'readerHighlight', 'readerCite']) {
    el[key] = mount(doc, 'button');
  }
  el.tabs = [];
  el.panels = [];
  Object.assign(state, {
    currentJobId: 'job-a', currentLang: 'orig', openGen: 1, readerPage: 1, readerTotalHint: total,
    readerPages: {
      orig: Array.from({ length: total }, (_, i) => ({
        page: i + 1, html: `<section class="doc-page" data-page="${i + 1}"><p>flow ${i + 1}</p></section>`,
      })),
      ko: null,
    },
    readerAlignments: { orig: new Map(), ko: new Map() },
    readerAlignmentPending: new Set(),
    readerAlignmentRetryTimers: new Map(),
    readerAlignmentRetryCounts: new Map(),
    readerAlignmentBackoff: new Set(),
    readerImgTimers: new Map(),
    readerOutline: { orig: [], ko: null },
    readerActiveBlock: '', readerSelection: '', readerSync: true, viewerOpen: false,
    readerResizeObserver: null, readerMeasureRaf: 0, readerAnchorRaf: 0,
  });
  resetReaderDocumentState();
  // 아직 캐시에 없는 페이지의 정렬 GET은 응답하지 않는다(타이머·재시도 없이 대기).
  t.mock.method(globalThis, 'fetch', () => new Promise(() => {}));
  t.after(() => { Object.assign(state, savedState); Object.assign(el, savedEls); });
  return doc;
}
