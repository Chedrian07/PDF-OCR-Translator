import {
  READER_ALIGNMENT_COOLDOWN_MS, READER_DEFAULT_RATIO, READER_FOCUS_RATIO, READER_HYDRATE_RADIUS,
  READER_KEEP_RADIUS, READER_SYNC_KEY, READER_SYNC_QUIET_MS, READER_ZOOM_KEY, READER_ZOOM_MAX,
  READER_ZOOM_MIN, readerPosKey,
} from './constants.js';
import {
  alignmentBatchPlan, alignmentFailureIsPermanent, blockAtFraction, clampReaderPage,
  extractDocPages, livePageImageUrl, normalizeAlignmentPayload, overlayInKeepWindow,
  pdfExportState, pdfReportMessage, railAnchorFrom, railAnchorTarget, readerFocusAt,
  readerHydrationWindow, readerImageUrl, readerRailBandAt, splitInlineMath,
  translatedHtmlExportState, withLangUrl,
} from './core.js';
import { el, state } from './state.js';
import { h, localGet, localSet, nowMs, setDownload, showToast, typesetMath } from './ui.js';
import { apiGet } from './api.js';
import { revertToOriginal } from './translate.js';
import { prefillQaPageFromReader } from './qa.js';
import {
  applyViewerPanelState, closeViewer, loadViewerManifest, renderViewerThumbnails,
  syncViewerSearch,
} from './viewer.js';
import { activateTab } from './tabs.js';

/* ============================ 읽기 (리더) 탭 ============================
 * 완료된 잡의 기본 뷰 — 원문 PDF 페이지를 세로로 이어 붙인 하나의 연속 스크롤
 * 면과, 같은 좌표에 묶인 번역 레일을 나란히 놓는다. 두 면은 스크롤을 공유해
 * (bbox 기준으로) 늘 같은 문단을 마주 본다. 논문을 페이지 버튼 없이 죽 읽을 수
 * 있고, 페이지 컨트롤(번호 입력·◀▶·썸네일·목차)은 "그 페이지로 스크롤"로 남는다.
 * 현재 페이지는 스크롤 위치에서 역산한다(readerPageBands + readerFocusAt).
 *
 * 본문 텍스트에는 두 경로가 있다:
 *   1) 좌표 정렬 — /viewer/pages(배치) 또는 /alignment(단건). 블록 카드 + bbox 오버레이.
 *   2) 폴백 — /html을 extractDocPages로 나눈 페이지 섹션(좌표 없는 잡·구버전).
 * 둘 다 (잡, 언어)별로 캐시하고, 이미지·정렬은 현재 페이지 주변 창(window)만
 * 실제로 붙인다. 50페이지 넘는 논문에서도 네트워크/디코딩 메모리가 일정하다.
 * ==================================================================== */


export function readerLangKey() { return state.currentLang === 'ko' ? 'ko' : 'orig'; }

// 리더 탭이 실제로 보이는 상태인지 (키보드 단축키 가드).
export function readerIsActive() {
  return !el.resultSection.hidden &&
    el.tabs.some((t) => t.dataset.tab === 'reader' && t.classList.contains('active'));
}

// 총 페이지: 잡 메타 힌트(result.pages → progress.total_pages) 우선,
// 없으면 추출된 섹션 수, 그마저 없으면 1.
export function readerTotal() {
  const pages = state.readerPages[readerLangKey()];
  return state.readerTotalHint || (pages ? pages.length : 0) || 1;
}

// 페이지 종횡비(가로/세로). 서버가 알려 준 크기 우선 → 이 문서에서 처음 알게 된
// 크기(대부분의 논문은 전 페이지 동일) → 최후에 A4 근사. 자리표시자 높이가
// 실제 이미지와 맞아야 스크롤 위치가 튀지 않는다.
export function readerPageRatio(page) {
  const known = state.readerPageSizes.get(page) || state.readerPageSizeSeed;
  if (known && known.w > 0 && known.h > 0) return known.w / known.h;
  return READER_DEFAULT_RATIO;
}

export function rememberReaderPageSize(page, width, height) {
  const w = Number(width);
  const h = Number(height);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return false;
  const prev = state.readerPageSizes.get(page);
  if (prev && prev.w === w && prev.h === h) return false;
  state.readerPageSizes.set(page, { w, h });
  if (!state.readerPageSizeSeed) state.readerPageSizeSeed = { w, h };
  const section = state.readerPageEls.get(page);
  if (section) section.style.aspectRatio = `${w} / ${h}`;
  return true;
}


export function teardownReader() {
  for (const timer of state.readerImgTimers.values()) clearTimeout(timer);
  state.readerImgTimers.clear();
  for (const timer of state.readerAlignmentRetryTimers.values()) clearTimeout(timer);
  state.readerAlignmentRetryTimers.clear();
  state.readerAlignmentRetryCounts.clear();
  state.readerAlignmentBackoff.clear();
  clearTimeout(state.readerPosTimer);
  state.readerPosTimer = 0;
  clearTimeout(state.readerUrlTimer);
  state.readerUrlTimer = 0;
  clearTimeout(state.readerJumpTimer);
  state.readerJumpTimer = 0;
  state.readerScrollTarget = null;
  for (const key of ['readerScrollRaf', 'readerRailRaf', 'readerMeasureRaf', 'readerAnchorRaf']) {
    if (state[key]) cancelAnimationFrame(state[key]);
    state[key] = 0;
  }
  state.readerMeasureAnchor = null;
  state.readerPaneQuietUntil = 0;
  state.readerRailQuietUntil = 0;
  if (state.readerResizeObserver) {
    state.readerResizeObserver.disconnect();
    state.readerResizeObserver = null;
  }
}

// 잡 전환 시 리더 상태 초기화 — 페이지 1, 언어별 캐시 폐기, 스택/측정값 정리.
export function resetReaderForJob() {
  teardownReader();
  state.readerPage = 1;
  state.readerTotalHint = 0;
  state.readerPages = { orig: null, ko: null };
  state.readerOutline = { orig: null, ko: null };
  state.readerAlignments = { orig: new Map(), ko: new Map() };
  state.readerAlignmentPending = new Set();
  state.readerActiveBlock = '';
  state.readerSelection = '';
  state.readerSelectionPage = 1;
  state.readerHighlights = [];
  state.readerCitations = [];
  state.viewerManifest = null;
  state.viewerNavCollapsed = localGet('uocr-viewer-nav-collapsed') === '1';
  state.viewerRailCollapsed = localGet('uocr-viewer-rail-collapsed') === '1';
  state.qaPageTouched = false;
  resetReaderDocumentState();
  el.readerContent.innerHTML = '';
  el.readerPageStage.textContent = '';
  el.readerMapStatus.textContent = '좌표 연결 준비 중';
  el.readerActivity.textContent = '0개 블록';
  el.readerPagePane.classList.remove('failed');
  el.viewerRoot.removeAttribute('aria-busy'); // 이전 잡의 로더가 남긴 상태 정리
  el.readerPageInput.disabled = false;
  el.readerPageInput.value = '1';
  el.readerTotal.textContent = '–';
  el.readerPrev.disabled = true;
  el.readerNext.disabled = true;
  el.readerOutline.textContent = '';
  el.viewerThumbnails.textContent = '';
  updateReaderProgress();
  applyViewerPanelState();
  updateReaderResearchTools();
}

// 문서 스택(원문 페이지 + 번역 레일)에 딸린 파생 상태만 비운다.
export function resetReaderDocumentState() {
  state.readerBands = [];
  state.readerLastFocus = null;
  state.readerRailPage = 1;
  state.readerRailAnchor = null;
  state.readerPageEls = new Map();
  state.readerRailEls = new Map();
  state.readerCardEls = new Map();
  state.readerRailBands = [];
  state.readerRailIndexByPage = new Map();
  state.readerRailTabPage = 0;
  state.readerPageSizes = new Map();
  state.readerSourceImagePages = new Set();
  state.readerPageSizeSeed = null;
  state.readerStackKey = '';
  state.readerRailKey = '';
  state.readerThumbSignature = '';
  state.readerOutlineSignature = '';
  state.readerRailIndexDirty = true;
  state.readerMeasureAnchor = null;
}

/* ── 로딩 (본문·목차·좌표) ────────────────────────────────────────── */

// 리더 탭 활성화/언어 전환 진입점 — (잡, 언어)별 최초 1회만 /html을 가져와
// 페이지 섹션으로 분해해 캐시한다. 캐시가 있으면 즉시 문서를 그린다.
export async function loadReader() {
  const id = state.currentJobId;
  if (!id) return;
  const lang = state.currentLang;
  loadReaderOutline(id, lang);
  if (state.readerPages[readerLangKey()]) {
    // 이전 언어 로더가 남긴 로딩 플래그를 걷는다 — 그 로더는 이미 stale이라
    // 아래 잡/언어 가드에서 조기 return하며 스스로 복구하지 못한다.
    el.viewerRoot.removeAttribute('aria-busy');
    el.readerPageInput.disabled = false;
    renderReaderDocument();
    return;
  }
  el.viewerRoot.setAttribute('aria-busy', 'true');
  el.readerPageInput.disabled = true;
  el.readerContent.innerHTML = '';
  el.readerContent.appendChild(h('div', { class: 'reader-loading', role: 'status' },
    h('span', { class: 'spinner', 'aria-hidden': 'true' }),
    h('p', { text: '논문 본문과 페이지 구조를 불러오는 중…' }),
  ));
  let html = null;
  try {
    const res = await fetch(withLangUrl(`/api/jobs/${id}/html`, lang), { headers: { Accept: 'text/html' } });
    if (res.ok) html = await res.text();
  } catch (_) { /* 아래 공통 실패 처리 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return; // 잡/언어 전환 → 최신 로더에 위임
  el.viewerRoot.removeAttribute('aria-busy');
  el.readerPageInput.disabled = false;
  if (html == null) {
    // 한국어 뷰에서 번역본을 못 받으면 조용히 원문으로 폴백 (다른 탭과 동일 규칙).
    if (lang === 'ko' && revertToOriginal('한국어 본문을 불러오지 못해 원문을 표시합니다.')) { loadReader(); return; }
    el.readerContent.innerHTML = '';
    const retry = h('button', { class: 'btn btn-small', type: 'button', text: '다시 시도' });
    retry.addEventListener('click', loadReader);
    el.readerContent.appendChild(h('div', { class: 'reader-load-error', role: 'alert' },
      h('strong', { text: '본문을 불러오지 못했습니다.' }),
      h('p', { text: '네트워크 연결과 작업 상태를 확인한 뒤 다시 시도해 주세요.' }),
      retry,
    ));
    return;
  }
  state.readerPages[lang === 'ko' ? 'ko' : 'orig'] = extractDocPages(html);
  renderReaderDocument();
  loadViewerManifest(); // 총 페이지 확정 — 스택 길이를 뒤늦게라도 맞춘다
}

// 현재 창(window)에 필요한 정렬을 배치로 가져온다. 서버 배치가 없거나 실패하면
// 페이지 단건 /alignment로 폴백한다(좌표 없는 잡은 폴백 본문으로 정상 동작).
export function clearReaderAlignmentRetry(id, key, page) {
  const retryKey = `${id}:${key}:${page}`;
  const timer = state.readerAlignmentRetryTimers.get(retryKey);
  if (timer) clearTimeout(timer);
  state.readerAlignmentRetryTimers.delete(retryKey);
  state.readerAlignmentRetryCounts.delete(retryKey);
  state.readerAlignmentBackoff.delete(retryKey);
}

export function clearReaderAlignmentRetryLanguage(id, lang) {
  if (!id) return;
  const key = lang === 'ko' ? 'ko' : 'orig';
  const prefix = `${id}:${key}:`;
  for (const [retryKey, timer] of state.readerAlignmentRetryTimers) {
    if (!retryKey.startsWith(prefix)) continue;
    clearTimeout(timer);
    state.readerAlignmentRetryTimers.delete(retryKey);
  }
  for (const retryKey of state.readerAlignmentRetryCounts.keys()) {
    if (retryKey.startsWith(prefix)) state.readerAlignmentRetryCounts.delete(retryKey);
  }
  for (const retryKey of state.readerAlignmentBackoff) {
    if (retryKey.startsWith(prefix)) state.readerAlignmentBackoff.delete(retryKey);
  }
}

export function scheduleReaderAlignmentRetry(id, lang, pages, focusPage) {
  const key = lang === 'ko' ? 'ko' : 'orig';
  const retryPages = [...new Set((Array.isArray(pages) ? pages : [pages])
    .map((n) => Math.floor(Number(n))).filter((n) => n >= 1))];
  if (!retryPages.length) return;
  const page = retryPages.includes(focusPage) ? focusPage : retryPages[0];
  const retryKey = `${id}:${key}:${page}`;
  if (state.readerAlignmentRetryTimers.has(retryKey)) return;
  const attempt = state.readerAlignmentRetryCounts.get(retryKey) || 0;
  const backoffKeys = retryPages.map((n) => `${id}:${key}:${n}`);
  for (const backoffKey of backoffKeys) state.readerAlignmentBackoff.add(backoffKey);
  if (attempt >= 2) {
    // 자동 재시도는 정확히 두 번으로 끝낸다. 이후에는 30초 동안 어떤 resize/
    // hydrate 이벤트도 우회 요청하지 못하게 하고, 쿨다운 뒤 다음 사용자 동작에서
    // 새 예산으로 다시 시도한다(타이머 자체는 HTTP를 시작하지 않는다).
    const cooldown = setTimeout(() => {
      state.readerAlignmentRetryTimers.delete(retryKey);
      state.readerAlignmentRetryCounts.delete(retryKey);
      for (const backoffKey of backoffKeys) state.readerAlignmentBackoff.delete(backoffKey);
    }, READER_ALIGNMENT_COOLDOWN_MS);
    state.readerAlignmentRetryTimers.set(retryKey, cooldown);
    return;
  }
  state.readerAlignmentRetryCounts.set(retryKey, attempt + 1);
  const timer = setTimeout(() => {
    state.readerAlignmentRetryTimers.delete(retryKey);
    for (const backoffKey of backoffKeys) state.readerAlignmentBackoff.delete(backoffKey);
    if (state.currentJobId !== id || state.currentLang !== lang) {
      state.readerAlignmentRetryCounts.delete(retryKey);
      return;
    }
    if (state.readerAlignments[key].has(page)) {
      clearReaderAlignmentRetry(id, key, page);
      return;
    }
    loadReaderAlignmentWindow(id, lang, page);
  }, 800 * (2 ** attempt));
  state.readerAlignmentRetryTimers.set(retryKey, timer);
}

export async function loadReaderAlignmentWindow(id, lang, page) {
  const key = lang === 'ko' ? 'ko' : 'orig';
  const cache = state.readerAlignments[key];
  const pending = state.readerAlignmentPending;
  const covered = new Set(cache.keys());
  const total = readerTotal();
  for (let n = 1; n <= total; n += 1) {
    const requestKey = `${id}:${key}:${n}`;
    if (pending.has(requestKey) || state.readerAlignmentBackoff.has(requestKey)) covered.add(n);
  }
  const plan = alignmentBatchPlan(total, page, READER_HYDRATE_RADIUS, covered);
  for (const batch of plan) {
    const requested = Array.from(
      { length: batch.limit }, (_, index) => batch.start + index,
    );
    const requestKeys = requested.map((n) => `${id}:${key}:${n}`);
    if (requestKeys.some((requestKey) => pending.has(requestKey))) continue;
    for (const requestKey of requestKeys) pending.add(requestKey);
    // 실제 cache/DOM이 바뀐 페이지만 다시 그린다. transient 5xx에서 requested
    // 전체를 재렌더하면 ResizeObserver→hydrate가 즉시 같은 요청을 반복한다.
    const changed = new Set();
    const transient = [];
    try {
      let handled = false;
      try {
        const url = withLangUrl(
          `/api/jobs/${id}/viewer/pages?start=${batch.start}&limit=${batch.limit}&include=alignment`,
          lang,
        );
        const res = await fetch(url, { headers: { Accept: 'application/json' } });
        if (res.ok) {
          const body = await res.json();
          if (state.currentJobId !== id || state.currentLang !== lang) return;
          const received = new Set();
          for (const item of Array.isArray(body && body.items) ? body.items : []) {
            const itemPage = Math.floor(Number(item && item.page));
            if (itemPage < 1) continue;
            received.add(itemPage);
            changed.add(itemPage);
            rememberReaderPageSize(itemPage, item.width, item.height);
            cache.set(itemPage, item.alignment ? normalizeAlignmentPayload(item.alignment) : null);
            clearReaderAlignmentRetry(id, key, itemPage);
          }
          // 성공 배치에서 빠진 번호는 좌표 없는 페이지로 확정한다. 그렇지 않으면
          // 같은 구멍을 스크롤할 때마다 배치 GET을 영원히 반복한다.
          for (const n of requested) {
            if (!received.has(n)) {
              cache.set(n, null);
              changed.add(n);
            }
            clearReaderAlignmentRetry(id, key, n);
          }
          handled = true;
        }
      } catch (_) { /* 아래 단건 폴백 */ }
      if (!handled) {
        for (const n of requested) {
          if (cache.has(n)) continue;
          let resolved = false;
          try {
            const res = await fetch(withLangUrl(`/api/jobs/${id}/alignment?page=${n}`, lang), {
              headers: { Accept: 'application/json' },
            });
            if (res.ok) {
              cache.set(n, normalizeAlignmentPayload(await res.json()));
              resolved = true;
            } else if (alignmentFailureIsPermanent(res.status)) {
              // 404 부재 / 409 레이아웃 불일치 등 확정 실패만 세션 캐시 — 흐름형 본문으로 폴백
              cache.set(n, null);
              resolved = true;
            }
          } catch (_) { /* 일시 네트워크 오류 — pending 해제 뒤 다음 창에서 재시도 */ }
          if (state.currentJobId !== id || state.currentLang !== lang) return;
          if (resolved) {
            changed.add(n);
            clearReaderAlignmentRetry(id, key, n);
          } else {
            transient.push(n);
          }
        }
      }
    } finally {
      // 잡/언어가 바뀌어 위에서 조기 return해도 같은 페이지가 영구 pending으로
      // 남지 않는다. reset으로 Set이 교체돼도 이 요청이 등록한 원래 Set만 정리.
      for (const requestKey of requestKeys) pending.delete(requestKey);
    }
    if (state.currentJobId !== id || state.currentLang !== lang) return;
    // 한 배치 실패에는 timer도 하나만 둔다. 그 페이지의 hydration window가 같은
    // 이웃 범위를 다시 덮으므로 페이지별 timer는 빠른 5xx에서 burst만 만든다.
    if (transient.length) {
      // 좌표 정렬 API만 일시 실패한 경우에도 /html 본문은 즉시 읽을 수 있게 한다.
      // cache에는 넣지 않아 아래 bounded retry가 성공하면 정렬 카드로 교체된다.
      const pages = state.readerPages[key] || [];
      const anchor = captureRailAnchor(); // 렌더 전 위치 (개별 모드 스크롤 고정)
      let fallbackChanged = false;
      for (const n of transient) {
        const section = state.readerRailEls.get(n);
        const body = section && section.lastElementChild;
        if (section && body) {
          fallbackChanged = renderRailFlowContent(n, section, body, pages, true) || fallbackChanged;
        }
      }
      if (fallbackChanged) {
        syncReaderInteractiveTabStops();
        updateReaderRailStatus();
        resyncRailToSource(anchor);
      }
      scheduleReaderAlignmentRetry(id, lang, transient, page);
    }
    if (changed.size) {
      const anchor = captureRailAnchor(); // 렌더 전 위치 (개별 모드 스크롤 고정)
      for (const n of changed) renderRailPage(n);
      scheduleReaderMeasure();
      // 방금 채운 내용만큼 레일 높이가 커졌다 — 보고 있던 문단에 다시 맞춘다.
      resyncRailToSource(anchor);
    }
  }
}

// 레일 레이아웃이 바뀌기 직전의 스크롤 앵커. 개별(sync off) 모드에서만 의미가
// 있다 — 연동 모드는 원문 면 눈높이가 기준이라 앵커가 필요 없다.
export function captureRailAnchor() {
  const rail = el.readerContent;
  if (state.readerSync || !rail) return null;
  const layout = readerRailLayoutIndex();
  const anchor = railAnchorFrom(layout.bands, rail.scrollTop, rail.clientHeight);
  // 마지막 앵커를 남긴다 — 렌더 경로가 아닌 늦은 이미지 로드(watchRailLateLayout)도
  // 이 위치로 되돌려야 개별 모드에서 읽던 문단이 아래로 밀리지 않는다.
  state.readerRailAnchor = anchor;
  return anchor;
}

// 원문 면의 현재 눈높이에 레일을 다시 맞춘다 (레일 레이아웃이 바뀐 뒤 호출).
// 개별 모드에서는 대신 captureRailAnchor()로 잡아 둔 위치를 되돌린다.
export function resyncRailToSource(anchor) {
  if (readerJumpActive()) return;
  if (!state.readerSync) { restoreRailAnchor(anchor); return; }
  if (!state.readerBands.length) return;
  const pane = el.readerPagePane;
  syncRailFromSource(readerFocusAt(
    state.readerBands, pane.scrollTop + pane.clientHeight * READER_FOCUS_RATIO,
  ));
}

// 렌더로 늘어난 레일 높이만큼 밀린 스크롤을 앵커 위치로 되돌린다.
export function restoreRailAnchor(anchor) {
  const rail = el.readerContent;
  if (!anchor || !rail) return;
  state.readerRailIndexDirty = true; // 방금 렌더로 밴드가 바뀌었다
  const next = railAnchorTarget(readerRailLayoutIndex().bands, anchor);
  if (next == null || Math.abs(next - rail.scrollTop) < 4) return;
  quietRail();
  rail.scrollTop = next;
}

export async function loadReaderOutline(id, lang) {
  const key = lang === 'ko' ? 'ko' : 'orig';
  if (state.readerOutline[key]) { renderReaderOutline(); return; }
  let items = [];
  try {
    const data = await apiGet(withLangUrl(`/api/jobs/${id}/outline`, lang));
    if (data && Array.isArray(data.items)) items = data.items;
  } catch (_) { /* 레이아웃 없는 잡은 빈 개요로 정상 폴백 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return;
  state.readerOutline[key] = items;
  renderReaderOutline();
}

export function renderReaderOutline() {
  const items = state.readerOutline[readerLangKey()];
  const signature = `${state.currentJobId}|${readerLangKey()}|${(items || []).length}`;
  if (signature !== state.readerOutlineSignature) {
    state.readerOutlineSignature = signature;
    el.readerOutline.textContent = '';
    if (!Array.isArray(items) || !items.length) {
      el.readerOutline.appendChild(h('span', {
        class: 'reader-outline-empty',
        text: '이 문서에는 좌표 기반 제목 정보가 없습니다.',
      }));
      return;
    }
    for (const item of items) {
      const button = h('button', {
        type: 'button',
        text: String(item.text || ''),
        'data-level': String(item.level || 2),
        'data-outline-page': String(Number(item.page) || 1),
        title: `${item.page || 1}페이지로 이동`,
      });
      button.addEventListener('click', () => setReaderPage(item.page));
      el.readerOutline.appendChild(button);
    }
  }
  updateReaderOutlineActive();
}

export function updateReaderOutlineActive() {
  for (const button of el.readerOutline.querySelectorAll('[data-outline-page]')) {
    const active = Number(button.dataset.outlinePage) === state.readerPage;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'location');
    else button.removeAttribute('aria-current');
  }
}

export function alignmentTypeLabel(type) {
  const labels = {
    title: '제목',
    text: '본문',
    list: '목록',
    table: '표',
    equation: '수식',
    caption: '캡션',
    footnote: '각주',
    reference: '참고문헌',
  };
  return labels[type] || '텍스트';
}

/* ── 원문 페이지 연속 스택 ────────────────────────────────────────── */

// 전체 페이지의 자리(종횡비만 잡은 빈 섹션)를 한 번에 만든다. 이미지는
// hydrateReaderPages가 현재 창에만 붙인다 — 스크롤바 길이는 처음부터 정확하다.
export function buildReaderPageStack() {
  const total = readerTotal();
  const key = `${state.currentJobId}|${total}`;
  if (key === state.readerStackKey) return false;
  state.readerStackKey = key;
  state.readerPageEls = new Map();
  el.readerPageStage.textContent = '';
  const frag = document.createDocumentFragment();
  for (let page = 1; page <= total; page += 1) {
    const image = h('img', {
      class: 'reader-page-image',
      alt: `${page}/${total}페이지 원문 PDF`,
      loading: 'lazy',
      decoding: 'async',
      draggable: 'false',
    });
    const section = h('section', {
      class: 'reader-page',
      'data-page': String(page),
      'aria-label': `${page}페이지`,
    }, image, h('div', {
      class: 'reader-map-overlay',
      'data-page': String(page),
      role: 'group',
      'aria-label': `${page}페이지 원문 블록 위치`,
    }), h('span', { class: 'reader-page-tag', text: String(page), 'aria-hidden': 'true' }));
    const ratio = readerPageRatio(page);
    section.style.aspectRatio = `${ratio}`;
    state.readerPageEls.set(page, section);
    frag.appendChild(section);
  }
  el.readerPageStage.appendChild(frag);
  return true;
}

// 현재 창의 페이지에만 실제 이미지를 붙이고, 창을 크게 벗어난 페이지는
// src를 떼어 디코딩된 비트맵 메모리를 브라우저에 돌려준다.
export function hydrateReaderPages() {
  const id = state.currentJobId;
  if (!id) return;
  const total = readerTotal();
  const win = readerHydrationWindow(total, state.readerPage, READER_HYDRATE_RADIUS);
  const keep = readerHydrationWindow(total, state.readerPage, READER_KEEP_RADIUS);
  for (const [page, section] of state.readerPageEls) {
    const image = section.firstElementChild;
    if (!image || image.tagName !== 'IMG') continue;
    if (page >= win.start && page <= win.end) {
      // 이 세션에서 source PNG 폴백이 확인된 페이지는 keep 창을 벗어났다 돌아와도
      // 실패한 primary를 1.5초씩 다시 거치지 않는다.
      const url = state.readerSourceImagePages.has(page)
        ? livePageImageUrl(id, page) : readerImageUrl(id, page);
      if (image.dataset.url !== url && !image.dataset.retried) {
        image.dataset.url = url;
        section.classList.remove('is-failed');
        image.setAttribute('src', url);
      }
      // 창 밖으로 나가며 걷어냈던 오버레이를 되살린다 (정렬은 캐시에 남아 있다).
      const overlay = section.querySelector('.reader-map-overlay');
      if (overlay && !overlay.firstChild) renderPageOverlay(page);
    } else if (page < keep.start || page > keep.end) {
      if (image.dataset.url) {
        delete image.dataset.url;
        delete image.dataset.retried;
        image.removeAttribute('src');
        section.classList.remove('is-failed');
      }
      // 오버레이 버튼도 함께 걷는다 — 긴 논문을 끝까지 읽으면 수천 개가 쌓여
      // 포커스 순환(Tab)과 히트 테스트가 무거워진다. 다시 들어오면 재렌더된다.
      const overlay = section.querySelector('.reader-map-overlay');
      if (overlay && overlay.firstChild) overlay.textContent = '';
    }
  }
  loadReaderAlignmentWindow(id, state.currentLang, state.readerPage);
}

// 스크롤 좌표계(pane.scrollTop 기준)에서의 페이지 밴드를 실측한다.
export function measureReaderBands() {
  const pane = el.readerPagePane;
  if (!pane || !state.readerPageEls.size) { state.readerBands = []; return false; }
  // 숨겨진 탭/display:none 상태의 0px rect로 정상 밴드를 덮어쓰면 다음 진입 때
  // 현재 페이지와 이어읽기 위치가 1쪽으로 오염된다. 보이는 시점의 observer가 재측정.
  if (!pane.isConnected || pane.getClientRects().length === 0 ||
      pane.clientWidth <= 0 || pane.clientHeight <= 0) return false;
  const paneTop = pane.getBoundingClientRect().top;
  const scrollTop = pane.scrollTop;
  const bands = [];
  for (const [page, section] of state.readerPageEls) {
    const rect = section.getBoundingClientRect();
    bands.push({ page, top: rect.top - paneTop + scrollTop, height: rect.height });
  }
  if (!bands.length || bands.every((band) => band.height <= 0)) return false;
  bands.sort((a, b) => a.page - b.page);
  state.readerBands = bands;
  return true;
}

export function readerViewportFocus() {
  const pane = el.readerPagePane;
  if (!pane || !state.readerBands.length || pane.clientHeight <= 0) return null;
  return readerFocusAt(
    state.readerBands, pane.scrollTop + pane.clientHeight * READER_FOCUS_RATIO,
  );
}

export function restoreReaderMeasureAnchor(anchor) {
  if (!anchor) return;
  const pane = el.readerPagePane;
  const band = state.readerBands.find((candidate) => candidate.page === anchor.page);
  if (!pane || !band) return;
  const next = Math.max(0, Math.round(
    band.top + band.height * anchor.fraction - pane.clientHeight * READER_FOCUS_RATIO,
  ));
  quietPane();
  pane.scrollTop = next;
  state.readerPage = clampReaderPage(anchor.page, readerTotal());
  state.readerLastFocus = { page: state.readerPage, fraction: anchor.fraction };
  renderReaderPage();
  hydrateReaderPages();
  if (state.readerSync) syncRailFromSource(anchor);
}

export function remeasureReaderWithAnchor(anchor, after) {
  // 빠른 줌 연타는 마지막 측정 하나로 합치고, 잡 전환 뒤 이전 잡의 앵커가 새
  // 문서에 주입되지 않도록 job/stack 서명을 함께 고정한다.
  if (state.readerAnchorRaf) cancelAnimationFrame(state.readerAnchorRaf);
  const jobId = state.currentJobId;
  const stackKey = state.readerStackKey;
  state.readerAnchorRaf = requestAnimationFrame(() => {
    state.readerAnchorRaf = 0;
    if (state.currentJobId !== jobId || state.readerStackKey !== stackKey) return;
    if (measureReaderBands()) {
      if (anchor) restoreReaderMeasureAnchor(anchor);
      else scrollReaderToPage(state.readerPage, 'auto');
    } else if (anchor) {
      // 탭이 숨겨진 동안이면 다음 visible ResizeObserver/진입 측정에 넘긴다.
      state.readerMeasureAnchor = anchor;
    }
    if (typeof after === 'function') after();
  });
}

export function scheduleReaderMeasure() {
  // ResizeObserver는 크기가 이미 바뀐 뒤 호출되지만 state.readerBands는 직전
  // 레이아웃 좌표다. 그 좌표에서 의미 위치(page+fraction)를 잡아 새 높이에 remap한다.
  if (!state.readerMeasureAnchor && !state.readerScrollTarget) {
    state.readerMeasureAnchor = readerViewportFocus();
  }
  if (state.readerMeasureRaf) return;
  const needsInitialLanding = state.readerBands.length === 0;
  state.readerMeasureRaf = requestAnimationFrame(() => {
    state.readerMeasureRaf = 0;
    const anchor = state.readerMeasureAnchor;
    state.readerMeasureAnchor = null;
    if (!measureReaderBands()) return;
    if (anchor) restoreReaderMeasureAnchor(anchor);
    else if (needsInitialLanding) scrollReaderToPage(state.readerPage, 'auto');
  });
}

/* ── 번역 레일 연속 스택 ──────────────────────────────────────────── */

// 레일도 전 페이지 섹션을 미리 만들어 둔다. 내용은 정렬/폴백이 도착할 때
// 페이지 단위로 채운다 — 스크롤 동기화가 늘 같은 페이지 축을 공유한다.
export function buildReaderRailStack() {
  const total = readerTotal();
  const key = `${state.currentJobId}|${readerLangKey()}|${total}`;
  if (key === state.readerRailKey) return false;
  state.readerRailKey = key;
  state.readerRailEls = new Map();
  state.readerCardEls = new Map();
  state.readerRailTabPage = 0; // 새 섹션으로 교체 — 이전 Tab 대상은 사라졌다
  el.readerContent.textContent = '';
  const frag = document.createDocumentFragment();
  for (let page = 1; page <= total; page += 1) {
    const body = h('div', { class: 'reader-rail-body' },
      h('p', { class: 'reader-rail-pending muted', text: '불러오는 중…' }));
    const section = h('section', {
      class: 'reader-rail-page',
      'data-page': String(page),
      'aria-label': `${page}페이지 번역문`,
    }, h('div', { class: 'reader-rail-head' },
      h('span', { class: 'reader-rail-page-no', text: `${page}` }),
      h('span', { class: 'reader-rail-page-of', text: `/ ${total}` }),
    ), body);
    state.readerRailEls.set(page, section);
    frag.appendChild(section);
  }
  el.readerContent.appendChild(frag);
  state.readerRailIndexDirty = true;
  return true;
}

// 텍스트에 섞인 `\( … \)` / `\[ … \]`를 KaTeX로 조판한 노드 배열로 바꾼다.
// KaTeX가 없거나 조판 실패면 원문 표기 그대로 남는다(그레이스풀 폴백).
export function mathTextNodes(text) {
  const nodes = [];
  for (const part of splitInlineMath(text)) {
    if (part.type === 'text') {
      if (part.value) nodes.push(document.createTextNode(part.value));
      continue;
    }
    const span = document.createElement('span');
    span.className = part.display ? 'math-display' : 'math-inline';
    span.textContent = part.value;
    if (window.katex) {
      try {
        window.katex.render(part.value, span, {
          displayMode: !!part.display,
          throwOnError: false,
        });
      } catch (_) { span.textContent = part.value; }
    }
    nodes.push(span);
  }
  return nodes;
}

export function readerBlockCard(block, displayIndex, page) {
  const number = String(displayIndex + 1).padStart(2, '0');
  const ko = readerLangKey() === 'ko';
  const mainText = ko ? block.target : block.source;
  const status = ko ? (block.translated ? '번역됨' : '원문 유지') : '원문';
  const locate = h('button', {
    class: 'reader-map-locate',
    type: 'button',
    tabindex: page === readerRailFocusPage() ? '0' : '-1',
    'aria-label': `${page}페이지 ${number}번 ${alignmentTypeLabel(block.type)} 블록의 원문 위치 표시`,
    title: '원문 위치 표시',
    text: '원문 보기',
  });
  const target = h('p', { class: 'reader-map-target' });
  target.append(...mathTextNodes(mainText));
  const card = h('article', {
    class: `reader-map-card type-${block.type.replace(/[^a-z0-9_-]/gi, '')}`,
    'data-block-id': block.id,
    'data-page': String(page),
  },
  h('div', { class: 'reader-map-card-head' },
    h('span', { class: 'reader-map-number', text: number }),
    h('span', { class: 'reader-map-type', text: alignmentTypeLabel(block.type) }),
    h('span', {
      class: `reader-map-state${block.translated ? ' is-translated' : ''}`,
      text: status,
    }),
    locate,
  ),
  target);
  if (ko && block.source && block.source !== mainText) {
    const source = h('p', {});
    source.append(...mathTextNodes(block.source));
    card.appendChild(h('div', { class: 'reader-map-source' },
      h('span', { text: 'ORIGINAL' }),
      source,
    ));
  }
  return card;
}

// 레일 본문의 그림은 뒤늦게 로드되며 섹션 높이를 바꾼다. ResizeObserver는
// 스크롤 컨테이너(.reader-content) 자신의 박스만 보므로 이 변화를 못 잡는다 —
// 색인을 무효화하지 않으면 레일→원문 역동기화가 옛 밴드로 엉뚱한 페이지를 가리킨다.
export function watchRailLateLayout(body) {
  for (const img of body.querySelectorAll('img')) {
    if (img.complete) continue;
    img.addEventListener('load', onRailLateLayout, { once: true });
    img.addEventListener('error', onRailLateLayout, { once: true });
  }
}

export function markReaderRailIndexDirty() {
  state.readerRailIndexDirty = true;
}

// 늦게 로드된 그림이 레일 높이를 바꾼다 — 색인 무효화만으로는 부족하다.
// 개별(sync off) 모드에는 위치를 되잡아 줄 기준면이 없어 읽던 문단이 그림 높이만큼
// 아래로 밀린다. 마지막 앵커로 복원한다(연동 모드는 원문 면이 되잡는다).
export function onRailLateLayout() {
  markReaderRailIndexDirty();
  if (state.readerSync || readerJumpActive()) return;
  restoreRailAnchor(state.readerRailAnchor);
}

export function renderRailFlowContent(page, section, body, pages, transient = false) {
  if (transient && section.dataset.transient === '1') return false;
  section.dataset.mode = 'flow';
  if (transient) section.dataset.transient = '1';
  else delete section.dataset.transient;
  body.textContent = '';
  const entry = pages.find((p) => p.page === page) ||
    (pages.length === 1 && page === 1 ? pages[0] : null);
  if (entry) {
    // Trusted server-rendered fragment (/html — 미리보기 탭과 동일한 신뢰 경계).
    body.innerHTML = entry.html;
    if (transient) {
      body.prepend(h('p', {
        class: 'reader-rail-retry-note muted',
        role: 'status',
        'aria-live': 'polite',
        'aria-atomic': 'true',
        text: '본문은 표시했습니다. 원문 좌표 연결을 잠시 후 다시 시도합니다.',
      }));
    }
    typesetMath(body);
    watchRailLateLayout(body);
  } else if (pages.length === 1 && readerTotal() > 1) {
    // 서버가 페이지 구분 없이 한 장으로 렌더한 문서 — 본문은 1페이지에 전부 있다.
    body.appendChild(h('p', {
      class: 'muted',
      text: '이 문서에는 페이지 구분 정보가 없습니다 — 본문 전체는 1페이지에 표시됩니다.',
    }));
  } else {
    body.appendChild(h('p', { class: 'muted', text: '이 페이지에는 표시할 본문이 없습니다.' }));
  }
  // 언어 전환/일시 실패 뒤 현재 언어 좌표가 없으면 이전 bbox를 남기지 않는다.
  renderPageOverlay(page);
  state.readerRailIndexDirty = true;
  return true;
}

// 한 페이지의 레일 내용을 채운다 (정렬 → 카드, 없으면 /html 섹션 폴백).
export function renderRailPage(page) {
  const section = state.readerRailEls.get(page);
  if (!section) return;
  const body = section.lastElementChild;
  const cache = state.readerAlignments[readerLangKey()];
  const pages = state.readerPages[readerLangKey()] || [];
  if (!cache.has(page)) return; // 아직 미도착 — 자리표시자 유지
  const alignment = cache.get(page);
  body.textContent = '';
  delete section.dataset.transient;

  if (alignment && alignment.blocks.length) {
    section.dataset.mode = 'aligned';
    for (const [index, block] of alignment.blocks.entries()) {
      const card = readerBlockCard(block, index, page);
      state.readerCardEls.set(block.id, card);
      body.appendChild(card);
    }
    renderPageOverlay(page);
  } else {
    renderRailFlowContent(page, section, body, pages);
  }
  state.readerRailIndexDirty = true;
  if (state.readerActiveBlock) setReaderActiveBlock(state.readerActiveBlock);
  syncReaderInteractiveTabStops();
  updateReaderRailStatus();
}


// 한 페이지의 bbox 오버레이 버튼을 (다시) 그린다. 창 밖에서 걷어냈다가 돌아올 때도
// 같은 경로를 탄다 — 좌표는 %라 이미지 로드 여부와 무관하다.
export function renderPageOverlay(page) {
  const section = state.readerPageEls.get(page);
  if (!section) return;
  const overlay = section.querySelector('.reader-map-overlay');
  if (!overlay) return;
  overlay.textContent = '';
  // 개별(sync off) 모드의 레일 스크롤은 좌측 페이지를 세운 채 먼 페이지 정렬을
  // 계속 불러온다 — keep 창 밖까지 그리면 hydrate의 걷어내기가 돌지 않아
  // 좌측 스테이지에 오버레이 버튼이 무제한 쌓인다(Tab 순환·히트 테스트 저하).
  if (!overlayInKeepWindow(page, state.readerPage, readerTotal())) return;
  const alignment = state.readerAlignments[readerLangKey()].get(page);
  if (!alignment || !alignment.blocks.length) return;
  for (const [index, block] of alignment.blocks.entries()) {
    const box = h('button', {
      class: 'reader-map-box',
      type: 'button',
      tabindex: page === state.readerPage ? '0' : '-1',
      'data-block-id': block.id,
      'aria-label': `${page}페이지 ${index + 1}번 ${alignmentTypeLabel(block.type)}: ${block.source.slice(0, 90)}`,
      title: `${index + 1} · ${block.source.slice(0, 120)}`,
    });
    box.style.left = `${block.rect.left}%`;
    box.style.top = `${block.rect.top}%`;
    box.style.width = `${block.rect.width}%`;
    box.style.height = `${block.rect.height}%`;
    if (block.id === state.readerActiveBlock) {
      box.classList.add('is-active');
      box.setAttribute('aria-pressed', 'true');
    }
    overlay.appendChild(box);
  }
}

export function updateReaderRailStatus() {
  const cache = state.readerAlignments[readerLangKey()];
  const alignment = cache.get(state.readerPage);
  if (alignment && alignment.blocks.length) {
    el.readerMapStatus.textContent = readerLangKey() === 'ko'
      ? '한국어 ↔ 원문 좌표'
      : '원문 텍스트 ↔ 원문 좌표';
    el.readerActivity.textContent = `${state.readerPage}쪽 · ${alignment.blocks.length}개 블록`;
    return;
  }
  if (!cache.has(state.readerPage)) {
    el.readerMapStatus.textContent = '원문 좌표 연결 중…';
    el.readerActivity.textContent = '불러오는 중';
    return;
  }
  el.readerMapStatus.textContent = '흐름형 본문';
  el.readerActivity.textContent = '좌표 정보 없음';
}

// 레일 스크롤 → 페이지/블록 역산을 위한 위치 색인. 레이아웃이 바뀔 때만
// getBoundingClientRect를 호출하고, 매 scroll frame에는 페이지 이진 탐색만 한다.
// 카드 좌표는 여기서 미리 재지 않는다 — 장문서에서 수천 개의 카드를 매 재색인마다
// 재면 페이지 하나가 도착할 때마다 O(전체 카드) rect 계산이 든다(readerRailCardEntries).
export function readerRailLayoutIndex() {
  if (!state.readerRailIndexDirty) {
    return { bands: state.readerRailBands, byPage: state.readerRailIndexByPage };
  }
  const rail = el.readerContent;
  const origin = rail.getBoundingClientRect().top - rail.scrollTop;
  const bands = [];
  for (const [page, section] of state.readerRailEls) {
    const rect = section.getBoundingClientRect();
    bands.push({ page, top: rect.top - origin, height: rect.height });
  }
  bands.sort((a, b) => a.top - b.top);
  state.readerRailBands = bands;
  state.readerRailIndexByPage = new Map(); // 페이지별 카드 좌표는 요청 시 채운다
  state.readerRailIndexDirty = false;
  return { bands, byPage: state.readerRailIndexByPage };
}

// 한 페이지의 카드 위치 색인(레일 스크롤 좌표계). 색인이 무효화되면 함께 버려진다.
export function readerRailCardEntries(page) {
  const byPage = state.readerRailIndexByPage;
  const cached = byPage.get(page);
  if (cached) return cached;
  const section = state.readerRailEls.get(page);
  if (!section) return [];
  const rail = el.readerContent;
  const origin = rail.getBoundingClientRect().top - rail.scrollTop;
  const entries = [];
  for (const card of section.querySelectorAll('.reader-map-card[data-block-id]')) {
    entries.push({ id: card.dataset.blockId, top: card.getBoundingClientRect().top - origin });
  }
  entries.sort((a, b) => a.top - b.top);
  byPage.set(page, entries);
  return entries;
}

/* ── 좌표 블록 강조 ───────────────────────────────────────────────── */

export function readerMappingNodes(id) {
  const safeId = typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(id) : id.replace(/["\\]/g, '\\$&');
  return {
    card: el.readerContent.querySelector(`[data-block-id="${safeId}"]`),
    box: el.readerPageStage.querySelector(`.reader-map-box[data-block-id="${safeId}"]`),
  };
}

export function scrollReaderMappingTarget(target, container) {
  if (!target || !container) return;
  // scrollIntoView는 조상 스크롤 컨테이너(.main·문서)까지 함께 움직이므로 해당
  // 패널 좌표만 계산한다.
  const offset = target.getBoundingClientRect().top - container.getBoundingClientRect().top;
  const top = container.scrollTop + offset - container.clientHeight * 0.35;
  if (container === el.readerContent) quietRail(); else quietPane();
  container.scrollTo({ top: Math.max(0, top), behavior: readerScrollBehavior() });
}

export function setReaderActiveBlock(id, scrollTarget) {
  state.readerActiveBlock = id || '';
  for (const node of el.readerContent.querySelectorAll('.reader-map-card[data-block-id]')) {
    const active = node.dataset.blockId === state.readerActiveBlock;
    node.classList.toggle('is-active', active);
    if (active) node.setAttribute('aria-current', 'true');
    else node.removeAttribute('aria-current');
  }
  for (const node of el.readerPageStage.querySelectorAll('.reader-map-box[data-block-id]')) {
    const active = node.dataset.blockId === state.readerActiveBlock;
    node.classList.toggle('is-active', active);
    node.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
  if (!id || !scrollTarget) return;
  const nodes = readerMappingNodes(id);
  const target = scrollTarget === 'card' ? nodes.card : nodes.box;
  if (!target && scrollTarget === 'box' && nodes.card) {
    // 긴 문서에서는 먼 페이지 overlay를 lazy-unmount한다. 레일 카드의 원문 보기
    // 클릭 시 해당 페이지를 먼저 hydrate하고 다음 frame에 새 bbox로 이동한다.
    const page = Number(nodes.card.dataset.page) || state.readerPage;
    setReaderPage(page);
    requestAnimationFrame(() => {
      scrollReaderMappingTarget(readerMappingNodes(id).box, el.readerPagePane);
    });
    return;
  }
  if (!target) return;
  const container = scrollTarget === 'card' ? el.readerContent : el.readerPagePane;
  scrollReaderMappingTarget(target, container);
}

export function previewReaderBlock(id, on) {
  if (!id) return;
  const nodes = readerMappingNodes(id);
  if (nodes.card) nodes.card.classList.toggle('is-hovered', on);
  if (nodes.box) nodes.box.classList.toggle('is-hovered', on);
}

/* ── 문서 렌더 + 페이지 이동 ──────────────────────────────────────── */

// 리더 진입/언어 전환/총 페이지 확정 시의 단일 진입점.
export function renderReaderDocument() {
  const id = state.currentJobId;
  if (!id) return;
  const pages = state.readerPages[readerLangKey()];
  if (!pages) return;
  const total = readerTotal();
  state.readerPage = clampReaderPage(state.readerPage, total);

  const builtStack = buildReaderPageStack();
  const builtRail = buildReaderRailStack();
  el.readerVisualLabel.textContent = state.currentLang === 'ko'
    ? '원문 PDF · 한국어 대조'
    : '원문 PDF · 좌표 기준면';
  el.readerRailTitle.textContent = state.currentLang === 'ko'
    ? '원문 위치와 연결된 한국어'
    : '원문 위치와 연결된 텍스트';

  // 이미 캐시된 페이지는 즉시 그린다 (언어 전환 후 재진입 등).
  const cache = state.readerAlignments[readerLangKey()];
  for (const page of cache.keys()) renderRailPage(page);

  hydrateReaderPages();
  observeReaderResize();
  renderReaderPage();
  if (builtStack || builtRail) {
    requestAnimationFrame(() => {
      measureReaderBands();
      scrollReaderToPage(state.readerPage, 'auto');
    });
  } else {
    scheduleReaderMeasure();
  }
}

// 컨트롤 바/썸네일/목차/진행률을 현재 페이지에 맞춘다 (스크롤은 건드리지 않는다).
export function renderReaderPage() {
  const total = readerTotal();
  const n = clampReaderPage(state.readerPage, total);
  state.readerPage = n;
  el.readerPageInput.value = String(n);
  el.readerPageInput.max = String(total);
  el.readerTotal.textContent = String(total);
  el.readerPrev.disabled = n <= 1;
  el.readerNext.disabled = n >= total;
  syncReaderAnchorIds();
  syncReaderInteractiveTabStops();
  renderReaderOutline();
  renderViewerThumbnails();
  updateReaderProgress();
  updateReaderRailStatus();
  updateReaderResearchTools();
}

// #reader-image / #reader-map-overlay 는 "현재 페이지"를 가리키는 안정 앵커다.
// 연속 스택에서도 같은 뜻을 유지하도록 현재 페이지 노드로 id를 옮긴다
// (외부 도구·e2e·스크린리더가 이 두 앵커로 현재 페이지를 찾는다).
export function syncReaderAnchorIds() {
  const previousImage = document.getElementById('reader-image');
  const previousOverlay = document.getElementById('reader-map-overlay');
  const section = state.readerPageEls.get(state.readerPage);
  const image = section ? section.querySelector('.reader-page-image') : null;
  const overlay = section ? section.querySelector('.reader-map-overlay') : null;
  if (previousImage && previousImage !== image) previousImage.removeAttribute('id');
  if (previousOverlay && previousOverlay !== overlay) previousOverlay.removeAttribute('id');
  if (image) image.id = 'reader-image';
  if (overlay) overlay.id = 'reader-map-overlay';
  el.readerImage = image;
  el.readerMapOverlay = overlay;
}

// 레일에서 Tab 순환에 열어 둘 페이지 — 연동이면 원문 면 기준, 개별이면 레일 기준.
export function readerRailFocusPage() {
  return state.readerSync ? state.readerPage : state.readerRailPage;
}

export function setRailLocateTabIndex(page, value) {
  const section = state.readerRailEls.get(page);
  if (!section) return;
  for (const node of section.querySelectorAll('.reader-map-locate')) node.tabIndex = value;
}

// 연속 스택에서 화면 밖 페이지의 수백 bbox/"원문 보기" 버튼이 Tab 순환에
// 누적되지 않게 현재 페이지의 매핑 컨트롤만 키보드 탐색에 연다. 포인터 클릭과
// 프로그램적 locate는 tabindex=-1에서도 그대로 동작한다.
// 레일 쪽은 전체 카드를 훑지 않고 직전/현재 페이지 섹션만 손댄다 — 장문서에서
// 페이지가 바뀔 때마다 O(전체 카드) 순회가 되지 않게. (새로 그린 카드는
// readerBlockCard가 같은 기준(readerRailFocusPage)으로 tabindex를 붙인다.)
export function syncReaderInteractiveTabStops() {
  for (const node of el.readerPageStage.querySelectorAll('.reader-map-box[data-block-id]')) {
    const page = Number(node.closest('.reader-page')?.dataset.page) || 0;
    node.tabIndex = page === state.readerPage ? 0 : -1;
  }
  const railPage = readerRailFocusPage();
  if (state.readerRailTabPage && state.readerRailTabPage !== railPage) {
    setRailLocateTabIndex(state.readerRailTabPage, -1);
  }
  setRailLocateTabIndex(railPage, 0);
  state.readerRailTabPage = railPage;
}

export function updateReaderProgress() {
  const total = readerTotal();
  const pane = el.readerPagePane;
  let ratio = total > 1 ? (state.readerPage - 1) / (total - 1) : 1;
  if (pane && pane.scrollHeight > pane.clientHeight) {
    ratio = pane.scrollTop / (pane.scrollHeight - pane.clientHeight);
  }
  el.readerProgressFill.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
}

export function readerScrollBehavior() {
  const reduce = typeof window !== 'undefined' && window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  return reduce ? 'auto' : 'smooth';
}

// 우리가 만든 스크롤은 그 면의 핸들러만 잠시 재운다 (핑퐁 방지). 양쪽을 하나의
// 플래그로 묶으면 사용자가 계속 굴리는 동안 따라오는 쪽이 뒤처진다 — 면별로 나눈다.
export function quietPane() { state.readerPaneQuietUntil = nowMs() + READER_SYNC_QUIET_MS; }
export function quietRail() { state.readerRailQuietUntil = nowMs() + READER_SYNC_QUIET_MS; }
export function paneQuiet() { return nowMs() < state.readerPaneQuietUntil; }
export function railQuiet() { return nowMs() < state.readerRailQuietUntil; }

// 페이지 점프. 이동 중에는 좌우 동기화를 통째로 멈춘다 — 동기화가 pane.scrollTop을
// 쓰면 진행 중인 smooth 스크롤이 그 자리에서 취소돼 엉뚱한 페이지에 멈춘다.
// 먼 거리는 즉시 이동(실제 PDF 뷰어와 같은 감각), 인접 페이지만 부드럽게.
export function scrollReaderToPage(page, behavior) {
  if (!state.readerBands.length) measureReaderBands();
  const band = state.readerBands.find((b) => b.page === page);
  if (!band) return;
  const top = Math.max(0, band.top - 2);
  const far = Math.abs(page - readerFocusAt(state.readerBands, el.readerPagePane.scrollTop).page) > 2;
  const mode = behavior === 'auto' || far ? 'auto' : behavior || 'auto';
  const life = mode === 'auto' ? 400 : 1200;
  state.readerScrollTarget = { page, top, until: nowMs() + life };
  // 스크롤 이벤트가 더 오지 않는 경우(즉시 이동·짧은 문서)에도 반드시 풀린다.
  clearTimeout(state.readerJumpTimer);
  state.readerJumpTimer = setTimeout(() => {
    state.readerJumpTimer = 0;
    if (!state.readerScrollTarget) return;
    state.readerScrollTarget = null;
    updateReaderPositionFromScroll();
  }, life + 60);
  quietPane();
  quietRail();
  el.readerPagePane.scrollTo({ top, behavior: mode });
  scrollRailToPage(page, mode);
}

// 사용자가 직접 굴리면 진행 중인 점프를 즉시 포기한다 — 입력이 항상 이긴다.
export function cancelReaderJump(ev) {
  // 프로그램적 점프 직후라도 사용자가 직접 잡은 면은 quiet 기간을 즉시 끝낸다.
  // 그렇지 않으면 짧은 휠/트랙패드 입력이 260ms 안에 끝났을 때 반대편이 영구히
  // 뒤처지고, 다음 스크롤이 있을 때까지 다시 맞춰지지 않는다.
  if (ev && ev.currentTarget === el.readerPagePane) state.readerPaneQuietUntil = 0;
  if (ev && ev.currentTarget === el.readerContent) state.readerRailQuietUntil = 0;
  if (!state.readerScrollTarget) return;
  state.readerScrollTarget = null;
  clearTimeout(state.readerJumpTimer);
  state.readerJumpTimer = 0;
}



// 점프가 아직 진행 중인가 (도착했거나 기한이 지나면 해제).
export function readerJumpActive() {
  const target = state.readerScrollTarget;
  if (!target) return false;
  const arrived = Math.abs(el.readerPagePane.scrollTop - target.top) < 6;
  if (arrived || nowMs() > target.until) {
    state.readerScrollTarget = null;
    clearTimeout(state.readerJumpTimer);
    state.readerJumpTimer = 0;
    if (arrived) {
      state.readerPage = clampReaderPage(target.page, readerTotal());
      renderReaderPage();
      hydrateReaderPages();
    }
    return false;
  }
  return true;
}

export function scrollRailToPage(page, behavior) {
  const section = state.readerRailEls.get(page);
  if (!section) return;
  const rail = el.readerContent;
  const top = section.getBoundingClientRect().top - rail.getBoundingClientRect().top + rail.scrollTop;
  rail.scrollTo({ top: Math.max(0, top - 2), behavior: behavior || 'auto' });
}

// 페이지 컨트롤(◀▶·번호 입력·썸네일·목차)의 단일 착지점 — 그 페이지로 스크롤.
export function setReaderPage(n) {
  if (!state.readerPages[readerLangKey()]) return; // 본문 로드 전 — 컨트롤 비활성 상태
  const next = clampReaderPage(n, readerTotal());
  state.readerPage = next;
  state.readerLastFocus = { page: next, fraction: 0 };
  if (state.viewerOpen) state.viewerIntent.page = next;
  renderReaderPage();
  hydrateReaderPages();
  scrollReaderToPage(next, readerScrollBehavior());
  if (state.viewerOpen) syncViewerSearch();
  saveReaderPosition();
}

/* ── 스크롤 추적 + 좌우 동기화 ────────────────────────────────────── */

// 원문 면의 스크롤 위치 → 현재 페이지/진행률 반영 (+ 필요한 창 hydrate).
export function updateReaderPositionFromScroll() {
  const pane = el.readerPagePane;
  if (!pane || !state.readerBands.length) return;
  const focus = readerFocusAt(state.readerBands, pane.scrollTop + pane.clientHeight * READER_FOCUS_RATIO);
  state.readerLastFocus = focus;
  const changed = focus.page !== state.readerPage;
  state.readerPage = focus.page;
  if (changed) {
    if (state.viewerOpen) {
      state.viewerIntent.page = focus.page;
      scheduleViewerSearchSync(); // 스크롤 한 번에 replaceState 수십 번을 막는다
    }
    hydrateReaderPages(); // 앵커 id를 옮기기 전에 현재 페이지 이미지를 먼저 붙인다
    renderReaderPage();
    saveReaderPosition();
  } else {
    updateReaderProgress();
  }
  return focus;
}

// 스크롤 중 주소창 갱신은 트레일링 디바운스 — 브라우저의 replaceState 빈도 제한에
// 걸리지 않게 하고, 뒤로가기 히스토리도 오염시키지 않는다.
export function scheduleViewerSearchSync() {
  clearTimeout(state.readerUrlTimer);
  state.readerUrlTimer = setTimeout(() => {
    state.readerUrlTimer = 0;
    if (state.viewerOpen) syncViewerSearch();
  }, 250);
}

export function onReaderPaneScroll() {
  if (state.readerScrollRaf) return;
  state.readerScrollRaf = requestAnimationFrame(() => {
    state.readerScrollRaf = 0;
    if (readerJumpActive()) { updateReaderProgress(); return; }
    const focus = updateReaderPositionFromScroll();
    if (!focus || !state.readerSync || paneQuiet()) return;
    syncRailFromSource(focus);
  });
}

export function onReaderRailScroll() {
  if (state.readerRailRaf) return;
  state.readerRailRaf = requestAnimationFrame(() => {
    state.readerRailRaf = 0;
    updateReaderRailPageFromScroll();
    captureRailAnchor(); // 개별 모드의 최신 위치 — 늦은 이미지 로드가 이 자리로 되돌린다
    if (!state.readerSync || railQuiet() || readerJumpActive()) return;
    syncSourceFromRail();
  });
}

export function updateReaderRailPageFromScroll() {
  const rail = el.readerContent;
  const layout = readerRailLayoutIndex();
  const band = readerRailBandAt(
    layout.bands, rail.scrollTop + rail.clientHeight * READER_FOCUS_RATIO,
  );
  if (!band || band.page === state.readerRailPage) return band;
  state.readerRailPage = band.page;
  syncReaderInteractiveTabStops();
  // 개별 스크롤에서는 좌측 페이지가 멈춰 있다 — 레일이 보고 있는 창을 직접
  // 로드하지 않으면 좌측 ±READER_HYDRATE_RADIUS 밖이 영원히 '불러오는 중…'으로 남는다.
  if (!state.readerSync && state.currentJobId) {
    loadReaderAlignmentWindow(state.currentJobId, state.currentLang, band.page);
  }
  return band;
}

// 원문 → 번역: 지금 보고 있는 좌표 블록의 카드를 같은 눈높이에 놓는다.
export function syncRailFromSource(focus) {
  const rail = el.readerContent;
  if (state.readerRailPage !== focus.page) {
    state.readerRailPage = focus.page;
    syncReaderInteractiveTabStops();
  }
  const cache = state.readerAlignments[readerLangKey()];
  const alignment = cache.get(focus.page);
  let top = null;
  if (alignment && alignment.blocks.length) {
    const blockId = blockAtFraction(alignment.blocks, focus.fraction);
    const card = blockId && state.readerCardEls.get(blockId);
    if (card && card.isConnected) {
      top = card.getBoundingClientRect().top - rail.getBoundingClientRect().top + rail.scrollTop
        - rail.clientHeight * READER_FOCUS_RATIO;
    }
  }
  if (top == null) {
    const section = state.readerRailEls.get(focus.page);
    if (!section) return;
    const base = section.getBoundingClientRect().top - rail.getBoundingClientRect().top + rail.scrollTop;
    top = base + section.offsetHeight * focus.fraction - rail.clientHeight * READER_FOCUS_RATIO;
  }
  const next = Math.max(0, Math.round(top));
  if (Math.abs(next - rail.scrollTop) < 4) return;
  quietRail();
  rail.scrollTop = next;
}

// 번역 → 원문: 눈높이의 카드가 가리키는 bbox를 원문 면의 같은 높이로 올린다.
export function syncSourceFromRail() {
  const rail = el.readerContent;
  const pane = el.readerPagePane;
  const focusY = rail.scrollTop + rail.clientHeight * READER_FOCUS_RATIO;
  const layout = readerRailLayoutIndex();
  const railBand = readerRailBandAt(layout.bands, focusY);
  if (!railBand) return;
  const page = railBand.page;

  // 좌표 카드가 아직 없거나 이 페이지가 흐름형이면 페이지 안 진행률로 맞춘다.
  let fraction = railBand.height > 0
    ? Math.min(1, Math.max(0, (focusY - railBand.top) / railBand.height)) : 0;
  const alignment = state.readerAlignments[readerLangKey()].get(page);
  if (alignment && alignment.blocks.length) {
    const pageEntries = readerRailCardEntries(page);
    let entry = pageEntries[0] || null;
    for (const item of pageEntries) {
      if (item.top <= focusY) entry = item;
      else break;
    }
    const block = entry && alignment.blocks.find((candidate) => candidate.id === entry.id);
    if (block) fraction = Math.min(1, Math.max(0, block.rect.top / 100));
  }
  const band = state.readerBands.find((b) => b.page === page);
  if (!band) return;
  const next = Math.max(0, Math.round(band.top + band.height * fraction - pane.clientHeight * READER_FOCUS_RATIO));
  if (Math.abs(next - pane.scrollTop) >= 4) {
    quietPane();
    pane.scrollTop = next;
  }
  const changed = page !== state.readerPage;
  state.readerPage = page;
  state.readerLastFocus = { page, fraction };
  if (changed) {
    if (state.viewerOpen) {
      state.viewerIntent.page = page;
      scheduleViewerSearchSync();
    }
    renderReaderPage();
    hydrateReaderPages();
    saveReaderPosition();
  }
}

export function setReaderSync(on) {
  state.readerSync = !!on;
  localSet(READER_SYNC_KEY, state.readerSync ? '1' : '0');
  el.readerSync.setAttribute('aria-pressed', state.readerSync ? 'true' : 'false');
  el.readerSync.classList.toggle('is-off', !state.readerSync);
  el.readerSync.textContent = state.readerSync ? '연동' : '개별';
  el.readerSync.title = state.readerSync
    ? '원문과 번역문의 스크롤을 같은 문단에 맞춰 함께 움직입니다'
    : '원문과 번역문을 따로 스크롤합니다';
  if (state.readerSync) {
    state.readerRailAnchor = null; // 연동에서는 원문 면이 기준 — 이전 앵커는 폐기
    const focus = updateReaderPositionFromScroll();
    if (focus) syncRailFromSource(focus);
  } else {
    // 개별로 바꾼 직후의 레일 페이지·정렬 창을 즉시 맞춘다 (첫 스크롤을 기다리지 않게)
    updateReaderRailPageFromScroll();
    captureRailAnchor();
  }
  syncReaderInteractiveTabStops(); // Tab 기준 페이지가 원문 ↔ 레일로 바뀐다
}

// 마지막으로 읽던 페이지를 잡별로 기억한다 — 다음에 열면 그 자리에서 시작.
export function saveReaderPosition() {
  const id = state.currentJobId;
  if (!id) return;
  clearTimeout(state.readerPosTimer);
  state.readerPosTimer = setTimeout(() => {
    state.readerPosTimer = 0;
    localSet(readerPosKey(id), String(state.readerPage));
  }, 400);
}

export function restoreReaderPosition() {
  const id = state.currentJobId;
  if (!id) return 0;
  const saved = Math.floor(Number(localGet(readerPosKey(id))));
  return Number.isFinite(saved) && saved >= 1 ? saved : 0;
}

// 창 크기/줌 변화로 페이지 높이가 바뀌면 밴드를 다시 잰다.
export function observeReaderResize() {
  if (state.readerResizeObserver || typeof ResizeObserver === 'undefined') return;
  state.readerResizeObserver = new ResizeObserver((entries) => {
    state.readerRailIndexDirty = true;
    // 오른쪽 rail 내용만 도착/재조판된 경우 원문 페이지 밴드는 변하지 않는다.
    // 그때 source 측정→anchor 복원→hydrate까지 돌리면 transient alignment 실패가
    // 즉시 같은 요청을 재귀적으로 만들 수 있다. source stage 크기가 바뀐 경우만
    // 밴드를 다시 재고, rail 단독 변화는 현재 문단 정렬만 갱신한다.
    if (entries.some((entry) => entry.target === el.readerPageStage)) scheduleReaderMeasure();
  });
  state.readerResizeObserver.observe(el.readerPageStage);
  state.readerResizeObserver.observe(el.readerContent);
}

/* ── 이미지 로드/실패 (스택 위임) ─────────────────────────────────── */

// load/error는 버블링하지 않는다 — 스테이지에서 캡처 단계로 받는다.
export function onReaderStackLoad(ev) {
  const image = ev.target;
  if (!image || image.tagName !== 'IMG' || !image.classList.contains('reader-page-image')) return;
  const section = image.parentElement;
  // 원본 PNG fallback 성공 상태는 이 hydration 창 안에서 유지한다. 이를 지우면
  // 다음 hydrate가 실패한 facsimile URL로 되돌려 404→대기→fallback을 반복한다.
  if (image.dataset.retried !== 'source') delete image.dataset.retried;
  if (section) {
    section.classList.remove('is-failed');
    const page = Number(section.dataset.page) || 0;
    if (rememberReaderPageSize(page, image.naturalWidth, image.naturalHeight)) {
      scheduleReaderMeasure();
    }
  }
}

// 페이지별 3단 폴백: (1) 1.5초 뒤 캐시버스트 재시도 → (2) 렌더 단계 원본 PNG
// (/files/pages/…, layout.json에 구멍이 나도 존재한다) → (3) 자리표시자.
// 타이머는 페이지별로 따로 잡는다 — 전역 단일 타이머면 두 장이 동시에 실패할 때
// 서로의 재시도를 지운다.
export function onReaderStackError(ev) {
  const image = ev.target;
  if (!image || image.tagName !== 'IMG' || !image.classList.contains('reader-page-image')) return;
  const url = image.dataset.url;
  if (!url) return; // 창 밖으로 나가며 src를 뗀 경우
  const section = image.parentElement;
  const page = section ? Number(section.dataset.page) || 0 : 0;
  const stage = image.dataset.retried || '';
  if (!stage) {
    image.dataset.retried = 'bust';
    const timer = setTimeout(() => {
      state.readerImgTimers.delete(page);
      if (image.dataset.url !== url) return; // 페이지/잡 전환됨
      image.src = `${url}${url.includes('?') ? '&' : '?'}r=${Date.now()}`;
    }, 1500);
    clearTimeout(state.readerImgTimers.get(page));
    state.readerImgTimers.set(page, timer);
    return;
  }
  if (stage === 'bust' && page && state.currentJobId) {
    image.dataset.retried = 'source';
    state.readerSourceImagePages.add(page);
    const fallback = livePageImageUrl(state.currentJobId, page);
    image.dataset.url = fallback;
    image.src = fallback;
    return;
  }
  if (section) section.classList.add('is-failed');
}

/* ── 줌 (좌측 스택 폭 % — localStorage 'uocr-reader-zoom' 유지) ── */
export function applyReaderZoom() {
  el.readerPageStage.style.width = `${state.readerZoom}%`;
  el.readerZoomOut.title = `페이지 축소 (현재 ${state.readerZoom}%)`;
  el.readerZoomIn.title = `페이지 확대 (현재 ${state.readerZoom}%)`;
}

export function readerZoomBy(delta) {
  const anchor = readerViewportFocus();
  state.readerZoom = Math.min(READER_ZOOM_MAX, Math.max(READER_ZOOM_MIN, state.readerZoom + delta));
  localSet(READER_ZOOM_KEY, String(state.readerZoom));
  applyReaderZoom();
  // 폭이 바뀌면 전체 높이가 바뀐다 — 같은 페이지의 같은 읽던 줄로 복원.
  remeasureReaderWithAnchor(anchor);
}

export function updateReaderResearchTools() {
  const selected = (state.readerSelection || '').trim();
  const hasSelection = !!selected;
  el.readerExplain.disabled = !hasSelection;
  el.readerHighlight.disabled = !hasSelection;
  el.readerCite.disabled = !hasSelection;
  el.readerSelection.classList.toggle('has-selection', hasSelection);
  el.readerSelection.textContent = hasSelection
    ? `“${selected.slice(0, 180)}${selected.length > 180 ? '…' : ''}”`
    : '오른쪽 텍스트에서 문장을 선택하세요.';
}

export function captureReaderSelection() {
  const selection = window.getSelection && window.getSelection();
  let text = '';
  let page = state.readerPage;
  if (selection && !selection.isCollapsed && selection.rangeCount) {
    const range = selection.getRangeAt(0);
    if (el.readerContent.contains(range.commonAncestorContainer)) {
      text = selection.toString().replace(/\s+/g, ' ').trim().slice(0, 2000);
      const start = range.startContainer.nodeType === 1
        ? range.startContainer
        : range.startContainer.parentElement;
      const section = start && start.closest && start.closest('.reader-rail-page[data-page]');
      if (section) page = Number(section.dataset.page) || page;
    }
  }
  state.readerSelection = text;
  state.readerSelectionPage = page;
  updateReaderResearchTools();
}

export function openReaderQa(prompt, page = state.readerPage) {
  // 전체 화면 reader는 나머지 탭을 inert 처리한 modal이다. 먼저 modal을 닫아야
  // QA 패널과 입력창이 다시 접근성/포커스 트리에 들어온다.
  if (state.viewerOpen) closeViewer({ restoreFocus: false });
  activateTab('qa');
  // "선택 설명"은 일반 탭 자동 프리필과 달리 명시적 사용자 동작이다. 과거에
  // QA 페이지를 수정했어도 실제 선택이 속한 페이지를 강제로 사용한다.
  prefillQaPageFromReader(page, true);
  el.qaInput.value = prompt;
  el.qaInput.focus();
}

export function highlightReaderSelection() {
  const selection = window.getSelection && window.getSelection();
  if (!selection || selection.isCollapsed || !selection.rangeCount || !state.readerSelection) return;
  const range = selection.getRangeAt(0);
  if (!el.readerContent.contains(range.commonAncestorContainer)) return;
  const mark = document.createElement('mark');
  mark.className = 'reader-highlight';
  try {
    range.surroundContents(mark);
  } catch (_) {
    const fragment = range.extractContents();
    mark.appendChild(fragment);
    range.insertNode(mark);
  }
  state.readerHighlights.push({
    page: state.readerSelectionPage,
    lang: readerLangKey(),
    text: state.readerSelection,
  });
  selection.removeAllRanges();
  state.readerSelection = '';
  state.readerSelectionPage = state.readerPage;
  updateReaderResearchTools();
}

export function saveReaderCitation() {
  if (!state.readerSelection) return;
  const page = state.readerSelectionPage;
  state.readerCitations.push({
    page,
    lang: readerLangKey(),
    text: state.readerSelection,
  });
  updateReaderResearchTools();
  showToast(`${page}페이지 인용을 세션에 저장했습니다.`);
}

// 리더 CTA [한국어로 읽기] — 메인 [한국어 번역] 버튼이 보일 때만(=번역 상태
// none/error/canceled) 함께 보인다. 프로바이더 미설정(translate_available=false)
// 이면 숨김 — 사유 안내는 다운로드 행의 기존 번역 UI가 담당한다.
export function applyReaderTranslateCta() {
  el.readerTranslateBtn.hidden = el.translateBtn.hidden || state.translateAvailable === false;
}

/* ── 한국어 HTML / 원문·한국어 대조 PDF 내보내기 (다운로드 행) ── */
// HTML은 번역 완료만, PDF는 번역 완료 ∧ 레이아웃 있음일 때 노출한다.
export function applyPdfExport() {
  const hs = translatedHtmlExportState(state.displayedStatus, state.translateState);
  el.dlDocKo.hidden = !hs.visible;
  if (hs.visible) {
    const u = state.resultUrls || {};
    setDownload(
      el.dlDocKo,
      u.documentHtml ? withLangUrl(u.documentHtml, 'ko') : null,
      `${state.currentBaseName || 'document'}.ko.html`,
    );
  }

  const ps = pdfExportState(state.displayedStatus, state.translateState, state.resultHasLayout);
  el.dlPdf.hidden = !ps.visible;
  el.viewerDlPdf.hidden = !ps.visible;
  if (ps.visible) {
    const u = state.resultUrls || {};
    setDownload(el.dlPdf, u.pdf || null, `${state.currentBaseName || 'document'}.ko.pdf`);
    setDownload(el.viewerDlPdf, u.pdf || null, `${state.currentBaseName || 'document'}.ko.pdf`);
  }
}

export async function downloadPdfWithReport(ev) {
  ev.preventDefault();
  if (state.pdfDownloadBusy || el.dlPdf.classList.contains('disabled')) return;
  const url = el.dlPdf.getAttribute('href');
  if (!url) return;
  state.pdfDownloadBusy = true;
  el.dlPdf.setAttribute('aria-busy', 'true');
  try {
    const res = await fetch(url, { headers: { Accept: 'application/pdf' } });
    if (!res.ok) {
      let detail = '';
      try { detail = (await res.json()).detail || ''; } catch (_) { /* body may be plain */ }
      throw new Error(detail || `PDF 생성 실패 (${res.status})`);
    }
    const blob = await res.blob();
    if (!blob.size || !String(blob.type || '').includes('pdf')) {
      throw new Error('서버가 올바른 PDF를 반환하지 않았습니다.');
    }
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objectUrl;
    a.download = el.dlPdf.getAttribute('download') || 'document.ko.pdf';
    a.hidden = true;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 2000);
    const report = {
      replaced: res.headers.get('X-UOCR-PDF-Replaced'),
      kept: res.headers.get('X-UOCR-PDF-Preserved'),
      relocated: res.headers.get('X-UOCR-PDF-Relocated'),
      tableCells: res.headers.get('X-UOCR-PDF-Table-Cells'),
      warnings: res.headers.get('X-UOCR-PDF-Warnings'),
      specialistKept: res.headers.get('X-UOCR-PDF-Specialist-Preserved'),
    };
    showToast(pdfReportMessage(report), Number(report.kept) || Number(report.warnings) ? 'warn' : '');
  } catch (e) {
    showToast(e && e.message ? e.message : 'PDF 다운로드에 실패했습니다.', 'error');
  } finally {
    state.pdfDownloadBusy = false;
    el.dlPdf.removeAttribute('aria-busy');
  }
}
