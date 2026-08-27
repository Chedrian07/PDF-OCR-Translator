// PDF OCR Translator — local OCR, facsimile HTML, optional Korean translation
// Vanilla ES module. No external dependencies. Same-origin /api calls.
//
// Active-job live view = synchronized 3 panes fed by one SSE token stream:
//   left   : source page image + layout boxes parsed from grounding tokens
//   middle : raw token stream (with <PAGE> dividers)
//   right  : det-block structured markdown rendered via POST /render-preview
//
// 스트림 문법 (실캡처 확정, docs/ARCHITECTURE.md §5 / frontend/tests/fixtures):
//   각 페이지는 progress(phase=ocr, current_page=p)로 먼저 "선언"된 뒤
//   토큰 스트림의 <PAGE> 마커로 시작한다. 선언 직후의 첫 마커는 재확인(no-op)이고,
//   선언 없이 만나는 마커만 +1 이다. 블록 문법: <|det|>label [x1,y1,x2,y2]<|/det|>텍스트…
//
// 테스트: npm test --prefix frontend
//   픽스처 리플레이 테스트가 아래 "Pure live-stream core" 익스포트를 직접 임포트한다.

'use strict';

// 모듈 경계는 frontend/js/ 아래에 있다 (브라우저 네이티브 ES 모듈 — 빌드 스텝 없음).
// 이 파일은 진입점이다: 부트스트랩(init)과, 테스트가 임포트하는 공개 심볼의 재노출만 담당한다.
import {
  ICON, QA_LS_EFFORT, QA_LS_PROVIDER, QA_LS_SUMMARY, QA_LS_THINKING, READER_SYNC_KEY,
  READER_ZOOM_KEY, READER_ZOOM_MAX, READER_ZOOM_MIN, READER_ZOOM_STEP, qaModelKey,
} from './js/constants.js';
import { clampQaPage, clampReaderPage, jobIdFromHash, parseViewerSearch } from './js/core.js';
import { el, grabEls, state } from './js/state.js';
import { localGet, localSet, setupTheme, showToast } from './js/ui.js';
import { loadHealth } from './js/health.js';
import {
  onPageImgError, onPageImgLoad, onPreviewScroll, onStreamScroll, pageNav, updateLeftPane,
} from './js/live.js';
import {
  armDelete, deleteJob, openJob, refreshJobs, requestCancel, showEmptyState,
} from './js/jobs.js';
import { teardownConnections } from './js/sse.js';
import { cancelTranslate, setLang, startTranslate } from './js/translate.js';
import { submitQaQuestion, updateQaProviderControls } from './js/qa.js';
import {
  applyReaderZoom, cancelReaderJump, captureReaderSelection, downloadPdfWithReport,
  highlightReaderSelection, onReaderPaneScroll, onReaderRailScroll, onReaderStackError,
  onReaderStackLoad, openReaderQa, previewReaderBlock, readerIsActive, readerTotal,
  readerViewportFocus, readerZoomBy, remeasureReaderWithAnchor, renderReaderPage,
  saveReaderCitation, setReaderActiveBlock, setReaderPage, setReaderSync,
} from './js/reader.js';
import { applyViewerPanelState, closeViewer, openViewer } from './js/viewer.js';
import { setupTabs } from './js/tabs.js';
import { handleUpload, setupDropzone } from './js/upload.js';

/* ============================================================================
 * 공개 API 재노출 — frontend/tests/*.test.mjs 가 이 경로에서 임포트한다.
 * (순수 코어는 js/core.js, 나머지는 각 기능 모듈에 있다.)
 * ========================================================================== */
export {
  PAGE_MARKER, PDF_RETRY_MAX, RETRY_AFTER_FALLBACK_S, RETRY_AFTER_MAX_S,
  TRANSLATE_KEPT_LABELS, TRANSLATE_SKIP_LABELS, alignmentBatchPlan,
  alignmentFailureIsPermanent, alignmentRect,
  armTransition, blockAtFraction, buildQaBody, buildViewerSearch, clampQaPage, clampReaderPage,
  classifyFiles, createGroundState, docLayoutIsFigureOnly, extractDocPages, fileSizeError,
  groundAnnounce, groundDrain, groundPush, healthCapabilities, incompleteTailIndex,
  jobIdFromHash, jobModelChip, livePageImageUrl, normalizeAlignmentPayload, normalizeLabel,
  overlayInKeepWindow, parseRetryAfter, parseViewerSearch, pdfExportState,
  pdfProgressLabel, pdfReportMessage, pdfRetryDelay, pickQaModels, planPreviewRender,
  progressPhaseText, providerIssue, qaProviderHint, railAnchorFrom, railAnchorTarget,
  rateLimitNotice, readerFocusAt, readerHydrationWindow,
  readerImageUrl, readerPageBands, readerRailBandAt, replayExtendsRaw, scanQuads,
  selectionSummary, splitInlineMath, splitPreviewPages, ssePromoteDelay, statusLabel,
  streamPaneTrimCount, structurePreview, summarizeIssues, syncedStreamPageNo,
  truncateRawToPage,
  translateKeptSummary, translateUiStateFor, translatedHtmlExportState, viewerThumbnailWindow,
  withLangUrl,
} from './js/core.js';
export { applyRetryLock } from './js/ui.js';

/* ============================ Init ============================ */

function init() {
  grabEls();
  setupTheme();
  setupDropzone();
  setupTabs();

  el.uploadBtn.addEventListener('click', handleUpload);
  el.streamPane.addEventListener('scroll', onStreamScroll, { passive: true });
  el.livePreview.addEventListener('scroll', onPreviewScroll, { passive: true });
  el.jobStop.addEventListener('click', requestCancel);
  el.jobDelete.addEventListener('click', () => {
    if (!state.currentJobId) return;
    // 키에 잡 id 포함 — 잡 전환 뒤 남은 무장이 다른 잡을 삭제하지 못하게 한다.
    armDelete(el.jobDelete, `header:${state.currentJobId}`, () => deleteJob(state.currentJobId));
  });

  // 번역 컨트롤
  el.translateCancel.innerHTML = ICON.x;
  el.translateBtn.addEventListener('click', startTranslate);
  el.translateCancel.addEventListener('click', cancelTranslate);
  el.dlPdf.addEventListener('click', downloadPdfWithReport);
  el.langOrig.addEventListener('click', () => setLang('orig'));
  el.langKo.addEventListener('click', () => setLang('ko'));
  el.viewerLangOrig.addEventListener('click', () => setLang('orig'));
  el.viewerLangKo.addEventListener('click', () => setLang('ko'));
  // 요약 칩 클릭 — hover title을 못 보는 환경(터치·키보드)에서도 사유를 읽을 수 있게
  for (const node of [el.translateSummary, el.viewerTranslateSummary]) {
    if (!node) continue;
    node.addEventListener('click', () => {
      const s = state.translateSummary;
      if (!s) return;
      showToast(s.detail.split('\n').join(' / '), s.tone);
    });
  }
  el.viewerOpen.addEventListener('click', () => openViewer());
  el.viewerClose.addEventListener('click', () => closeViewer());
  el.viewerToggleNav.addEventListener('click', () => {
    const anchor = readerViewportFocus() || state.readerLastFocus;
    state.viewerNavCollapsed = !state.viewerNavCollapsed;
    localSet('uocr-viewer-nav-collapsed', state.viewerNavCollapsed ? '1' : '0');
    applyViewerPanelState();
    remeasureReaderWithAnchor(anchor);
  });
  el.viewerToggleRail.addEventListener('click', () => {
    const anchor = readerViewportFocus() || state.readerLastFocus;
    state.viewerRailCollapsed = !state.viewerRailCollapsed;
    localSet('uocr-viewer-rail-collapsed', state.viewerRailCollapsed ? '1' : '0');
    applyViewerPanelState();
    remeasureReaderWithAnchor(anchor);
  });
  el.viewerDlPdf.addEventListener('click', downloadPdfWithReport);

  // 질문(Q&A) 컨트롤 — 선택은 localStorage에 기억 (공급자별 모델 포함)
  el.qaProvider.addEventListener('change', () => {
    if (!el.qaProvider.value) return; // 카탈로그 로드 전 플레이스홀더 옵션
    state.qaProvider = el.qaProvider.value;
    localSet(QA_LS_PROVIDER, state.qaProvider);
    updateQaProviderControls();
  });
  el.qaModel.addEventListener('change', () => {
    state.qaModel = el.qaModel.value;
    if (state.qaModel) localSet(qaModelKey(state.qaProvider), state.qaModel);
  });
  el.qaEffort.addEventListener('change', () => {
    state.qaEffort = el.qaEffort.value;
    localSet(QA_LS_EFFORT, state.qaEffort);
  });
  el.qaThinking.addEventListener('click', () => {
    state.qaThinking = !state.qaThinking;
    localSet(QA_LS_THINKING, String(state.qaThinking));
    updateQaProviderControls(); // summary 가용성/토글 표시 갱신
  });
  el.qaSummary.addEventListener('change', () => {
    state.qaSummary = el.qaSummary.value;
    localSet(QA_LS_SUMMARY, state.qaSummary);
  });
  el.qaPage.addEventListener('change', () => {
    state.qaPageTouched = true; // 이 잡에서는 리더 페이지 프리필로 덮어쓰지 않는다
    el.qaPage.value = String(clampQaPage(el.qaPage.value, state.qaTotalPages));
  });
  for (const btn of el.qaSuggestions) {
    btn.addEventListener('click', () => { // 추천 질문 → 입력창 채우기 (전송은 사용자가)
      el.qaInput.value = btn.textContent;
      el.qaInput.focus();
    });
  }
  el.qaForm.addEventListener('submit', (ev) => {
    ev.preventDefault();
    submitQaQuestion();
  });

  el.pagerPrev.addEventListener('click', () => pageNav(-1));
  el.pagerNext.addEventListener('click', () => pageNav(1));
  el.followChip.addEventListener('click', () => {
    state.followLive = true;
    state.viewPage = state.ground.page;
    updateLeftPane();
  });
  el.pageImg.addEventListener('load', onPageImgLoad);
  el.pageImg.addEventListener('error', onPageImgError);

  // 읽기(리더) 탭 컨트롤
  el.readerPrev.addEventListener('click', () => setReaderPage(state.readerPage - 1));
  el.readerNext.addEventListener('click', () => setReaderPage(state.readerPage + 1));
  el.readerPageInput.addEventListener('change', () => setReaderPage(el.readerPageInput.value));
  el.readerZoomOut.addEventListener('click', () => readerZoomBy(-READER_ZOOM_STEP));
  el.readerZoomIn.addEventListener('click', () => readerZoomBy(READER_ZOOM_STEP));
  el.readerFitWidth.addEventListener('click', () => {
    const anchor = readerViewportFocus();
    state.readerZoom = 100;
    localSet(READER_ZOOM_KEY, '100');
    applyReaderZoom();
    remeasureReaderWithAnchor(anchor);
  });
  el.readerSync.addEventListener('click', () => setReaderSync(!state.readerSync));
  el.readerPagePane.addEventListener('scroll', onReaderPaneScroll, { passive: true });
  el.readerContent.addEventListener('scroll', onReaderRailScroll, { passive: true });
  // 사용자의 직접 조작은 진행 중인 페이지 점프보다 항상 우선한다.
  for (const pane of [el.readerPagePane, el.readerContent]) {
    pane.addEventListener('wheel', cancelReaderJump, { passive: true });
    pane.addEventListener('touchstart', cancelReaderJump, { passive: true });
    pane.addEventListener('pointerdown', cancelReaderJump, { passive: true });
    pane.addEventListener('keydown', (ev) => {
      if (['PageUp', 'PageDown', 'Home', 'End', ' ', 'ArrowUp', 'ArrowDown'].includes(ev.key)) {
        // 브라우저 기본 키보드 스크롤도 휠/터치처럼 사용자의 직접 입력이다.
        cancelReaderJump({ currentTarget: pane });
      }
    }, true);
  }
  // load/error는 버블링하지 않는다 — 스택 컨테이너에서 캡처 단계로 받는다.
  el.readerPageStage.addEventListener('load', onReaderStackLoad, true);
  el.readerPageStage.addEventListener('error', onReaderStackError, true);
  el.readerPageStage.addEventListener('click', (ev) => {
    const block = ev.target.closest && ev.target.closest('.reader-map-box[data-block-id]');
    if (!block || !el.readerPageStage.contains(block)) return;
    setReaderActiveBlock(block.dataset.blockId, 'card');
  });
  el.readerSummary.addEventListener('click', () => openReaderQa('이 페이지를 핵심 주장, 근거, 결론으로 나눠 요약해줘.'));
  el.readerExplain.addEventListener('click', () => {
    if (state.readerSelection) {
      openReaderQa(
        `다음 선택 문장을 문맥에 맞게 쉽게 설명해줘:\n\n${state.readerSelection}`,
        state.readerSelectionPage,
      );
    }
  });
  el.readerHighlight.addEventListener('click', highlightReaderSelection);
  el.readerCite.addEventListener('click', saveReaderCitation);
  el.readerTranslateBtn.addEventListener('click', startTranslate); // 기존 번역 시작 경로에 위임
  el.readerContent.addEventListener('mouseup', captureReaderSelection);
  el.readerContent.addEventListener('keyup', captureReaderSelection);
  el.readerContent.addEventListener('click', (ev) => {
    const block = ev.target.closest && ev.target.closest('[data-block-id]');
    if (!block || !el.readerContent.contains(block)) return;
    setReaderActiveBlock(block.dataset.blockId, 'box');
  });
  el.readerContent.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    const block = ev.target.closest && ev.target.closest('[data-block-id]');
    if (!block || !el.readerContent.contains(block)) return;
    ev.preventDefault();
    setReaderActiveBlock(block.dataset.blockId, 'box');
  });
  for (const container of [el.readerContent, el.readerPageStage]) {
    container.addEventListener('pointerover', (ev) => {
      const block = ev.target.closest && ev.target.closest('[data-block-id]');
      if (!block || !container.contains(block)) return;
      const from = ev.relatedTarget && ev.relatedTarget.closest
        ? ev.relatedTarget.closest('[data-block-id]')
        : null;
      if (from === block) return;
      previewReaderBlock(block.dataset.blockId, true);
    });
    container.addEventListener('pointerout', (ev) => {
      const block = ev.target.closest && ev.target.closest('[data-block-id]');
      if (!block || !container.contains(block)) return;
      const to = ev.relatedTarget && ev.relatedTarget.closest
        ? ev.relatedTarget.closest('[data-block-id]')
        : null;
      if (to === block) return;
      previewReaderBlock(block.dataset.blockId, false);
    });
    container.addEventListener('focusin', (ev) => {
      const block = ev.target.closest && ev.target.closest('[data-block-id]');
      if (block && container.contains(block)) previewReaderBlock(block.dataset.blockId, true);
    });
    container.addEventListener('focusout', (ev) => {
      const block = ev.target.closest && ev.target.closest('[data-block-id]');
      if (block && container.contains(block)) previewReaderBlock(block.dataset.blockId, false);
    });
  }
  // 전체 화면 뷰어: Escape 닫기, 포커스 순환, 비대화형 표면에서 페이지/줌 단축키.
  window.addEventListener('keydown', (ev) => {
    if (state.viewerOpen && ev.key === 'Escape') {
      ev.preventDefault();
      closeViewer();
      return;
    }
    if (state.viewerOpen && ev.key === 'Tab') {
      const focusable = [...el.viewerRoot.querySelectorAll(
        'a[href]:not([hidden]),button:not([disabled]):not([hidden]),input:not([disabled]),[tabindex]:not([tabindex="-1"])',
      )].filter((node) => node.offsetParent !== null
        && getComputedStyle(node).visibility !== 'hidden'
        && !node.closest('[inert],[aria-hidden="true"]'));
      if (focusable.length) {
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (ev.shiftKey && document.activeElement === first) {
          ev.preventDefault();
          last.focus();
        } else if (!ev.shiftKey && document.activeElement === last) {
          ev.preventDefault();
          first.focus();
        }
      }
      return;
    }
    if (!['ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End', '+', '=', '-'].includes(ev.key)) return;
    if (ev.defaultPrevented) return; // 탭 줄 roving 포커스 등 선행 핸들러 존중
    const active = document.activeElement;
    const tag = active ? active.tagName : '';
    if (active && (active.isContentEditable ||
      ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A', 'SUMMARY'].includes(tag))) return;
    if (!readerIsActive()) return;
    // 스크롤 면에 포커스가 있으면 PageUp/PageDown/Home/End는 브라우저 기본
    // 스크롤에 맡긴다 — 연속 스크롤에서는 그쪽이 더 자연스럽다.
    const onPane = (el.readerPagePane && el.readerPagePane.contains(active)) ||
      (el.readerContent && el.readerContent.contains(active));
    if (onPane && ['PageUp', 'PageDown', 'Home', 'End'].includes(ev.key)) return;
    ev.preventDefault();
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') {
      setReaderPage(state.readerPage + (ev.key === 'ArrowRight' ? 1 : -1));
    } else if (ev.key === 'PageUp' || ev.key === 'PageDown') {
      setReaderPage(state.readerPage + (ev.key === 'PageDown' ? 1 : -1));
    } else if (ev.key === 'Home' || ev.key === 'End') {
      setReaderPage(ev.key === 'Home' ? 1 : readerTotal());
    } else {
      readerZoomBy(ev.key === '-' ? -READER_ZOOM_STEP : READER_ZOOM_STEP);
    }
  });
  // 저장된 리더 줌 복원 (60–220% 클램프)
  const savedZoom = parseInt(localGet(READER_ZOOM_KEY) || '', 10);
  if (Number.isFinite(savedZoom)) {
    state.readerZoom = Math.min(READER_ZOOM_MAX, Math.max(READER_ZOOM_MIN, savedZoom));
  }
  applyReaderZoom();
  setReaderSync(localGet(READER_SYNC_KEY) !== '0'); // 기본값 = 연동

  // 사용자가 주소창을 직접 고치거나 뒤로가기로 해시가 바뀐 경우 해당 잡을 연다.
  // 우리가 만드는 변경은 replaceState라 hashchange가 발생하지 않는다(루프 없음).
  window.addEventListener('hashchange', () => {
    const id = jobIdFromHash(location.hash);
    if (id && id !== state.currentJobId) openJob(id);
  });
  window.addEventListener('popstate', () => {
    const next = parseViewerSearch(location.search);
    state.viewerIntent = next;
    if (next.open && state.displayedStatus === 'done') {
      state.readerPage = clampReaderPage(next.page, readerTotal());
      if (next.lang === 'ko' && state.translateState === 'done') setLang('ko');
      openViewer({ restore: true });
      renderReaderPage();
    } else if (!next.open && state.viewerOpen) {
      closeViewer({ sync: false });
    }
  });

  showEmptyState();
  loadHealth();
  refreshJobs().then(() => {
    // 첫 잡 목록 수신 직후 해시의 잡 복원 — 새로고침·공유 링크 진입.
    // 404면 openJob의 기존 처리(토스트)가 동작하고 해시를 비운다.
    const id = jobIdFromHash(location.hash);
    if (id) openJob(id);
  });
  state.jobsTimer = setInterval(refreshJobs, 5000);

  window.addEventListener('beforeunload', teardownConnections);
}

// Browser bootstrap only — the module is also imported by frontend/tests/
// under Node, where no DOM exists (only the exported pure core is used).
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}
