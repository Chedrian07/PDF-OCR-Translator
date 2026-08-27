import { ssePromoteDelay, syncedStreamPageNo } from './core.js';
import { el, state } from './state.js';
import { isTerminal, parseEventData } from './ui.js';
import { apiGet } from './api.js';
import {
  appendSystemLine, applyStreamReset, drainGroundToUI, enqueueToken, flushStream,
  restoreTokenReplay,
} from './live.js';
import {
  applyProgress, hasLiveContent, refreshJobs, removeJobFromList, renderJob, setStopButton,
  showEmptyState, syncJobHash, updateHeaderChip,
} from './jobs.js';
import { renderError, renderPartialResult, renderThumbGrid } from './results.js';
import {
  applyDownloadLangs, initTranslateForJob, resetTranslateUI, teardownTranslate,
} from './translate.js';
import { initQaForJob, teardownQa } from './qa.js';
import { teardownReader } from './reader.js';
import { applyViewerIntent } from './viewer.js';
import { activateTab } from './tabs.js';

/* ============================ SSE + fallback ============================ */


export function teardownConnections() {
  if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
  if (state.fallbackTimer) { clearInterval(state.fallbackTimer); state.fallbackTimer = 0; }
  clearSsePromote(); // 잡 전환·삭제·터미널 상태에서 재승격 재시도도 함께 정리
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  state.previewDirty = false;
  state.fallbackActive = false;
  state.sseErrorCount = 0;
  teardownTranslate(); // 잡 전환·삭제·페이지 이탈 시 번역 구독도 함께 정리
  teardownQa();        // 질문 탭도 잡 단위 — 이전 잡의 대화 로그/진행 정리
  teardownReader();    // 리더 이미지 재시도 타이머 정리
}

// 최초 구독(selectJob 경로)과 폴링 강등 후 재승격 시도가 공유하는 진입점.
// 재승격 중에는 fallbackActive를 건드리지 않는다 — open 성공까지 폴링 유지.
export function startStream(id) {
  // 중복 구독 방지 — 이미 열린 연결을 덮어쓰면 그 EventSource는 참조를 잃고도
  // 서버 retry(3초) 주기로 영원히 재접속하며 token/done을 두 번씩 배달한다.
  if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
  state.sseErrorCount = 0;

  if (typeof EventSource === 'undefined') {
    appendSystemLine('이 브라우저는 실시간 스트림을 지원하지 않아 상태 폴링을 사용합니다.', 'warn');
    startFallbackPolling(id);
    return;
  }

  let es;
  try {
    es = new EventSource(`/api/jobs/${id}/events`);
  } catch (_) {
    if (!state.fallbackActive) startFallbackPolling(id);
    scheduleSsePromote(id); // 강등은 임시 — 백오프 후 다시 시도
    return;
  }
  state.es = es;

  es.addEventListener('open', () => {
    if (state.currentJobId !== id) return;
    state.sseErrorCount = 0;
    clearSsePromote(); // 재승격 성공 — 백오프 단계 리셋
    if (state.fallbackActive) stopFallbackPolling(); // 폴링 해제, 정상 복귀
    if (!state.streamConnected) {
      state.streamConnected = true;
      appendSystemLine('실시간 스트림에 연결되었습니다.');
    } else {
      // 서버의 replay 이벤트가 누적 원문으로 세 패널을 다시 동기화한다.
      // replay가 비어 있는(아직 OCR 토큰 없음) 재연결도 연결 자체는 정상이다.
      flushStream(false);
      drainGroundToUI(false);
      state.streamPageNo = syncedStreamPageNo(state.streamPageNo, state.ground);
      appendSystemLine('스트림 재연결됨 — 누적 출력을 동기화합니다.');
    }
  });

  es.addEventListener('progress', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d) applyProgress(d);
  });

  es.addEventListener('token', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d && typeof d.text === 'string') enqueueToken(d.text);
  });

  es.addEventListener('replay', (e) => {
    if (state.currentJobId !== id) return;
    restoreTokenReplay(parseEventData(e));
  });

  // 서버가 이미 보낸 출력을 폐기하고 그 페이지부터 다시 처리한다 (재시도·
  // 반복 감지 폴백·텍스트 레이어 복구·실패 플레이스홀더).
  es.addEventListener('reset', (e) => {
    if (state.currentJobId !== id) return;
    applyStreamReset(parseEventData(e));
  });

  es.addEventListener('done', (e) => {
    if (state.currentJobId !== id) return;
    onJobDone(id, parseEventData(e) || {});
  });

  es.addEventListener('error', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d) onJobError(id, d);      // server-sent job error (has JSON data)
    else handleSseConnError(id);   // transport-level error (no data)
  });
}

export function handleSseConnError(id) {
  if (state.currentJobId !== id) return;
  if (state.fallbackActive) {
    // 폴링 중의 재승격 시도가 실패 — es를 닫고(브라우저 자동 재시도 차단)
    // 다음 백오프 단계로 재시도만 예약한다. 폴링은 그대로 유지된다.
    if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
    scheduleSsePromote(id);
    return;
  }
  state.sseErrorCount += 1;
  if (state.sseErrorCount >= 2) {
    if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; }
    appendSystemLine('라이브 스트림을 사용할 수 없어 상태 폴링으로 전환했습니다 — 주기적으로 재연결을 시도합니다.', 'warn');
    startFallbackPolling(id);
    scheduleSsePromote(id); // 강등은 임시 — 백오프 후 SSE 재승격 시도
  }
}

// 폴링 강등 후 SSE 재승격 시도를 백오프(10s→20s→30s 상한)로 예약한다.
// 타이머는 teardownConnections / 터미널 폴링 / open 성공에서 정리된다.
export function scheduleSsePromote(id) {
  if (state.ssePromoteTimer) clearTimeout(state.ssePromoteTimer);
  const delay = ssePromoteDelay(state.ssePromoteAttempts);
  state.ssePromoteAttempts += 1;
  state.ssePromoteTimer = setTimeout(() => {
    state.ssePromoteTimer = 0;
    // 잡 전환·터미널(폴링 해제)·이미 복귀한 경우에는 시도하지 않는다
    if (state.currentJobId !== id || !state.fallbackActive) return;
    startStream(id);
  }, delay);
}

export function clearSsePromote() {
  if (state.ssePromoteTimer) { clearTimeout(state.ssePromoteTimer); state.ssePromoteTimer = 0; }
  state.ssePromoteAttempts = 0;
}

export function stopFallbackPolling() {
  if (state.fallbackTimer) { clearInterval(state.fallbackTimer); state.fallbackTimer = 0; }
  state.fallbackActive = false;
}

export function startFallbackPolling(id) {
  state.fallbackActive = true;
  if (state.fallbackTimer) clearInterval(state.fallbackTimer);
  state.fallbackTimer = setInterval(async () => {
    if (state.currentJobId !== id) { clearInterval(state.fallbackTimer); state.fallbackTimer = 0; return; }
    let job;
    try {
      job = await apiGet(`/api/jobs/${id}`);
    } catch (e) {
      if (e.status === 404) {
        clearInterval(state.fallbackTimer);
        state.fallbackTimer = 0;
        clearSsePromote();
        if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; } // 재승격 시도 중이던 es
        removeJobFromList(id);
        if (state.currentJobId === id) { state.currentJobId = null; showEmptyState(); syncJobHash(null); }
      }
      return;
    }
    if (state.currentJobId !== id) return;
    // SSE가 이 fetch 사이에 재승격됐다면 이 스냅샷은 이미 낡았다. 그대로 적용하면
    // groundAnnounce가 expectAnnounce를 다시 세워 다음 <PAGE> 마커를 재확인으로
    // 삼켜 버리고, 그 뒤 페이지 번호가 한 칸씩 밀린다.
    if (!state.fallbackActive && !isTerminal(job.status)) return;
    if (isTerminal(job.status)) {
      clearInterval(state.fallbackTimer);
      state.fallbackTimer = 0;
      state.fallbackActive = false;
      clearSsePromote(); // 터미널 — 재승격 재시도도 정리
      if (state.es) { try { state.es.close(); } catch (_) { /* ignore */ } state.es = null; } // 재승격 시도 중이던 es
      flushStream(true);
      drainGroundToUI(true);
      renderJob(job);
      refreshJobs();
    } else {
      applyProgress(Object.assign({}, job.progress || {}, {
        status: job.status,
        queue_position: job.queue_position, // 상세 폴링도 대기열 위치를 반영
      }));
    }
  }, 1000);
}

export async function onJobDone(id, data) {
  if (state.currentJobId !== id) return;
  flushStream(true);
  drainGroundToUI(true);
  teardownConnections();
  state.displayedStatus = 'done';
  updateHeaderChip('done');
  setStopButton('done');

  let job = null;
  try {
    job = await apiGet(`/api/jobs/${id}`);
  } catch (_) { /* fall back to event data below */ }
  if (state.currentJobId !== id) return;

  if (job && job.result) {
    renderJob(job);
  } else {
    // Minimal render from the done event payload (URLs only).
    el.progressSection.hidden = true;
    el.errorSection.hidden = true;
    el.resultSection.hidden = false;
    el.liveDetails.open = false;
    resetTranslateUI();
    const base = 'document';
    state.currentBaseName = base;
    state.readerTotalHint = 0;       // 총 페이지 미상 — 리더는 섹션 수로 폴백
    state.resultHasLayout = undefined; // 미상 → PDF 내보내기 fail-open (서버 409가 방어)
    state.resultUrls = {
      markdown: data.markdown_url,
      archive: data.archive_url,
      documentHtml: `/api/jobs/${id}/document.html`,
      pdf: `/api/jobs/${id}/pdf?lang=ko&view=dual`,
      viewerManifest: `/api/jobs/${id}/viewer-manifest`,
    };
    applyDownloadLangs();
    renderThumbGrid(el.layoutsGrid, [], '레이아웃 이미지를 불러오지 못했습니다.');
    renderThumbGrid(el.pagesGrid, [], '페이지 이미지를 불러오지 못했습니다.');
    state.previewLoaded = false;
    state.markdownLoaded = false;
    state.docLayoutLoaded = false;
    el.previewBody.innerHTML = '';
    el.doclayoutBody.innerHTML = '';
    el.mdCode.textContent = '';
    activateTab('reader'); // 완료 잡의 기본 뷰 (renderResult와 동일 규칙)
    el.viewerOpen.hidden = false;
    initTranslateForJob();
    initQaForJob(job); // job=null 허용 — 총 페이지 미상으로 폼만 연다
    applyViewerIntent();
  }
  refreshJobs();
}

export function onJobError(id, d) {
  if (state.currentJobId !== id) return;
  flushStream(true);
  drainGroundToUI(true);
  teardownConnections();
  const canceled = !!(d && d.canceled);
  state.displayedStatus = canceled ? 'canceled' : 'error';
  state.cancelRequestedFor = null;
  updateHeaderChip(state.displayedStatus);
  el.progressSection.hidden = true;
  el.errorSection.hidden = false;
  el.liveDetails.open = false;
  el.liveDetails.hidden = !hasLiveContent();
  setStopButton(state.displayedStatus);
  renderError(d && d.message, canceled);
  if (canceled) {
    // partial markdown stays available → offer the Markdown/미리보기 tabs
    el.resultSection.hidden = false;
    renderPartialResult({ job_id: id, filename: el.jobFilename.textContent || '' });
  } else {
    el.resultSection.hidden = true;
  }
  refreshJobs();
}
