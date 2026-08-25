import {
  buildViewerSearch, clampReaderPage, readerImageUrl, viewerThumbnailWindow, withLangUrl,
} from './core.js';
import { el, state } from './state.js';
import { h } from './ui.js';
import { setLang } from './translate.js';
import {
  readerLangKey, readerTotal, readerViewportFocus, remeasureReaderWithAnchor,
  renderReaderDocument, setReaderPage,
} from './reader.js';
import { activateTab } from './tabs.js';

export function syncViewerSearch() {
  try {
    const search = buildViewerSearch(location.search, {
      open: state.viewerOpen,
      page: state.readerPage,
      lang: state.viewerOpen && state.viewerIntent && state.viewerIntent.lang === 'ko'
        ? 'ko'
        : state.currentLang,
    });
    history.replaceState(null, '', location.pathname + search + location.hash);
  } catch (_) { /* local file / restricted history contexts */ }
}

export function applyViewerPanelState() {
  el.viewerRoot.classList.toggle('nav-collapsed', state.viewerNavCollapsed);
  el.viewerRoot.classList.toggle('rail-collapsed', state.viewerRailCollapsed);
  el.viewerToggleNav.setAttribute('aria-expanded', state.viewerNavCollapsed ? 'false' : 'true');
  el.viewerToggleNav.setAttribute('aria-pressed', state.viewerNavCollapsed ? 'false' : 'true');
  el.viewerToggleRail.setAttribute('aria-expanded', state.viewerRailCollapsed ? 'false' : 'true');
  el.viewerToggleRail.setAttribute('aria-pressed', state.viewerRailCollapsed ? 'false' : 'true');
}

// 썸네일은 창(window)이 바뀔 때만 다시 만든다 — 연속 스크롤에서는 현재 페이지가
// 자주 바뀌므로, 매번 <img>를 새로 붙이면 깜빡임과 불필요한 요청이 생긴다.
export function renderViewerThumbnails() {
  const total = readerTotal();
  const pages = viewerThumbnailWindow(total, state.readerPage, 2);
  const signature = `${state.currentJobId}|${pages.join(',')}`;
  if (signature !== state.readerThumbSignature) {
    state.readerThumbSignature = signature;
    el.viewerThumbnails.textContent = '';
    let previous = 0;
    for (const page of pages) {
      if (previous && page - previous > 1) {
        el.viewerThumbnails.appendChild(h('span', {
          class: 'viewer-thumb-gap',
          text: '•••',
          'aria-hidden': 'true',
        }));
      }
      const image = h('img', {
        src: readerImageUrl(state.currentJobId, page),
        alt: '',
        loading: page === state.readerPage ? 'eager' : 'lazy',
        decoding: 'async',
      });
      image.addEventListener('error', () => image.classList.add('is-failed'), { once: true });
      const button = h('button', {
        class: 'viewer-thumbnail',
        type: 'button',
        'data-viewer-page': String(page),
        'aria-label': `${page}페이지로 이동`,
        title: `${page}페이지`,
      }, image, h('span', { text: String(page) }));
      button.addEventListener('click', () => {
        setReaderPage(page);
        el.readerPagePane.focus({ preventScroll: true });
      });
      el.viewerThumbnails.appendChild(button);
      previous = page;
    }
  }
  // 현재 표시는 항상 갱신 (창이 그대로여도 페이지는 바뀐다)
  for (const button of el.viewerThumbnails.querySelectorAll('[data-viewer-page]')) {
    const current = Number(button.dataset.viewerPage) === state.readerPage;
    button.classList.toggle('is-current', current);
    if (current) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  }
}

export async function loadViewerManifest() {
  const id = state.currentJobId;
  if (!id) return;
  const lang = state.currentLang;
  const base = (state.resultUrls || {}).viewerManifest ||
    `/api/jobs/${id}/viewer-manifest`;
  try {
    const response = await fetch(withLangUrl(base, lang), {
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) return; // 구버전 서버는 기존 result/pages 힌트로 폴백
    const manifest = await response.json();
    if (state.currentJobId !== id || state.currentLang !== lang) return;
    state.viewerManifest = manifest;
    const count = Number(manifest && manifest.document && manifest.document.page_count) || 0;
    if (count > state.readerTotalHint) {
      state.readerTotalHint = count;
      if (state.readerPages[readerLangKey()]) renderReaderDocument();
    }
    el.viewerRoot.dataset.quality = String(
      (manifest && manifest.quality && manifest.quality.state) || 'unknown',
    );
    renderViewerThumbnails();
  } catch (_) { /* manifest는 additive 최적화 — legacy reader 경로 유지 */ }
}

export function applyViewerIntent() {
  const intent = state.viewerIntent || { open: false, page: 1, lang: 'orig' };
  if (!intent.open || state.displayedStatus !== 'done') return;
  state.readerPage = clampReaderPage(intent.page, state.readerTotalHint || 0);
  openViewer({ restore: true });
}

export function setViewerBackgroundInert(on) {
  for (const node of document.querySelectorAll([
    '.app-header',
    '.sidebar',
    '.job-head',
    '.progress-section',
    '.live-details',
    '.error-section',
    '.result-actions',
    '.tabs',
    '.tab-panel:not(#production-viewer)',
  ].join(','))) {
    node.inert = on;
    if (on) {
      if (!node.hasAttribute('aria-hidden')) node.dataset.viewerAriaHidden = '1';
      node.setAttribute('aria-hidden', 'true');
    } else if (node.dataset.viewerAriaHidden) {
      node.removeAttribute('aria-hidden');
      delete node.dataset.viewerAriaHidden;
    }
  }
}

export function openViewer(options = {}) {
  if (!state.currentJobId || state.displayedStatus !== 'done') return;
  // reader 탭 밖에서 열면 pane은 display:none이라 실측 focus가 없다. 마지막 의미
  // 위치가 현재 페이지와 같을 때 재사용해 페이지 중간에서 맨 위로 튀지 않게 한다.
  const measured = readerViewportFocus();
  const anchor = measured || (
    state.readerLastFocus && state.readerLastFocus.page === state.readerPage
      ? state.readerLastFocus : { page: state.readerPage, fraction: 0 }
  );
  state.viewerReturnFocus = options.restore ? null : document.activeElement;
  const intent = state.viewerIntent || { open: false, page: state.readerPage, lang: 'orig' };
  state.viewerOpen = true;
  state.viewerIntent = { open: true, page: state.readerPage, lang: intent.lang };
  activateTab('reader');
  document.body.classList.add('viewer-mode');
  setViewerBackgroundInert(true);
  el.viewerRoot.classList.add('is-open');
  el.viewerRoot.setAttribute('role', 'dialog');
  el.viewerRoot.setAttribute('aria-modal', 'true');
  el.viewerRoot.setAttribute('aria-label', `${el.viewerFilename.textContent || '논문'} 읽기`);
  el.viewerRoot.tabIndex = -1;
  el.viewerOpen.setAttribute('aria-expanded', 'true');
  applyViewerPanelState();
  renderViewerThumbnails();
  loadViewerManifest();
  if (intent.lang === 'ko' && state.translateState === 'done') setLang('ko');
  syncViewerSearch();
  // 레이아웃이 3열 전체 화면으로 바뀌면 페이지 높이가 달라진다 — 다시 재고
  // 현재 페이지로 정렬한 뒤 스크롤 면에 포커스를 준다(Space/PageDown 즉시 동작).
  remeasureReaderWithAnchor(anchor, () => {
    el.readerPagePane.focus({ preventScroll: true });
  });
}

export function closeViewer(options = {}) {
  const wasOpen = state.viewerOpen;
  const anchor = wasOpen ? readerViewportFocus() : null;
  if (anchor) state.readerLastFocus = anchor;
  state.viewerOpen = false;
  state.viewerIntent = { open: false, page: 1, lang: 'orig' };
  if (typeof document !== 'undefined') document.body.classList.remove('viewer-mode');
  if (typeof document !== 'undefined') setViewerBackgroundInert(false);
  if (el.viewerRoot) {
    el.viewerRoot.classList.remove('is-open');
    el.viewerRoot.setAttribute('role', 'tabpanel');
    el.viewerRoot.removeAttribute('aria-modal');
    el.viewerRoot.removeAttribute('aria-label');
    el.viewerRoot.removeAttribute('tabindex');
  }
  if (el.viewerOpen) el.viewerOpen.setAttribute('aria-expanded', 'false');
  if (options.sync !== false && typeof location !== 'undefined') syncViewerSearch();
  if (wasOpen && options.restoreFocus !== false) {
    const target = state.viewerReturnFocus && state.viewerReturnFocus.isConnected
      ? state.viewerReturnFocus
      : el.viewerOpen;
    requestAnimationFrame(() => target && target.focus());
  }
  if (wasOpen) {
    // 임베디드 레이아웃으로 돌아오면 페이지 높이가 다시 바뀐다.
    remeasureReaderWithAnchor(anchor);
  }
  state.viewerReturnFocus = null;
}
