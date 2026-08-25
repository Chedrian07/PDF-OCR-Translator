import { rateLimitNotice, translateKeptSummary, translateUiStateFor, withLangUrl } from './core.js';
import { el, state } from './state.js';
import {
  applyRetryLock, lockRetry, parseEventData, retryLockRemaining, safeParse, setDownload,
  showToast,
} from './ui.js';
import { apiGet } from './api.js';
import {
  applyPdfExport, applyReaderTranslateCta, clearReaderAlignmentRetryLanguage, loadReader,
} from './reader.js';
import { loadViewerManifest, renderViewerThumbnails, syncViewerSearch } from './viewer.js';
import { loadDocLayout, loadMarkdown, loadPreview } from './tabs.js';

/* ============================ Translation (한국어 번역) ============================
 * 완료된 잡 결과 화면 전용. 상태 머신:
 *   none/error/canceled → [한국어 번역] 버튼 (error/canceled는 재시도 의미)
 *   running             → 진행바 + 취소(✕), EventSource 재접속
 *   done                → [원문 | 한국어] 세그먼트 토글
 * 라이브 변환 뷰(result-section이 hidden)에는 렌더되지 않는다 — done일 때만 init.
 * ================================================================================ */


export function teardownTranslate() {
  if (state.translateEs) { try { state.translateEs.close(); } catch (_) { /* ignore */ } state.translateEs = null; }
  if (state.translatePollTimer) { clearInterval(state.translatePollTimer); state.translatePollTimer = 0; }
  state.translateSseErrors = 0;
}

// 번역 UI를 원문 기준으로 완전 초기화 (구독 정리 + 세 컨트롤 숨김 + 언어 속성 제거).
export function resetTranslateUI() {
  teardownTranslate();
  state.currentLang = 'orig';
  state.translateState = 'none';
  el.translateBtn.hidden = true;
  el.translateProgress.hidden = true;
  el.langToggle.hidden = true;
  el.viewerLangToggle.hidden = true;
  setLangSegActive('orig');
  setResultLangAttr();
  renderTranslateSummary(null);  // 이전 잡의 원문 유지 요약이 새 잡에 새지 않게
  applyReaderTranslateCta(); // 번역 상태 확인 전에는 리더 CTA도 숨김
  applyPdfExport();          // 번역 없음 → 원문·한국어 대조 PDF 내보내기 숨김
}

// health의 translate_available을 버튼에 반영 — false일 때만 비활성.
// undefined(미수신·구버전 서버)는 활성 유지(fail-open, 서버 503이 최후 방어).
// 가용성 사유의 비활성만 dataset으로 표시해, 요청 중 일시 비활성(startTranslate)을
// health 갱신이 잘못 풀어버리지 않게 한다.
export function applyTranslateAvailability() {
  const btn = el.translateBtn;
  if (state.translateAvailable === false) {
    btn.disabled = true;
    btn.title = '번역 프로바이더가 설정되지 않았습니다 (.env 설정 후 재시작)';
    btn.dataset.unavailable = '1';
  } else {
    btn.removeAttribute('title');
    if (btn.dataset.unavailable) {
      delete btn.dataset.unavailable;
      btn.disabled = false;
    }
  }
  applyReaderTranslateCta(); // health 갱신도 리더 CTA 노출에 반영
}

// 번역 결과 요약 칩 — 원문 그대로 남은 문단 수와 사유를 결과 툴바·뷰어 양쪽에 노출.
// null이면 숨김(번역 없음·리포트 없는 구버전 잡).
export function renderTranslateSummary(data) {
  const summary = translateKeptSummary(data);
  state.translateSummary = summary;
  for (const node of [el.translateSummary, el.viewerTranslateSummary]) {
    if (!node) continue;
    node.hidden = !summary;
    if (!summary) { node.textContent = ''; node.removeAttribute('title'); continue; }
    node.textContent = summary.text;
    node.title = `${summary.detail}\n(클릭하면 자세히)`;
    node.classList.toggle('is-warn', summary.tone === 'warn');
  }
}

// 사유별 집계는 report.json에만 있다 — state 조회가 이를 병합해 돌려준다.
export async function refreshTranslateSummary(id) {
  let st = null;
  try { st = await apiGet(`/api/jobs/${id}/translate/state?lang=ko`); }
  catch (_) { return; } // 부가 정보 — 실패해도 직전 요약을 그대로 둔다
  if (state.currentJobId !== id || state.translateState !== 'done') return;
  renderTranslateSummary(st);
}

/* ── 컨트롤 3종 교체 노출 ─────────────────────────────────────────────── */
export function showTranslateButton() {
  el.translateBtn.hidden = false;
  el.translateProgress.hidden = true;
  el.langToggle.hidden = true;
  el.viewerLangToggle.hidden = true;
  el.translateBtn.disabled = false;
  el.readerTranslateBtn.disabled = false; // 리더 CTA도 같은 시작 경로 — 함께 복원
  applyTranslateAvailability(); // 프로바이더 미설정이면 비활성 + 안내 title (CTA 노출도 갱신)
  // 직전 429 잠금이 아직 유효하면(잡 전환 포함) 다시 비활성 — 표시와 동작 일치
  applyRetryLock(retryLockRemaining('translateRetryAt'), [el.translateBtn, el.readerTranslateBtn]);
}
export function showTranslateProgress(current, total) {
  el.translateBtn.hidden = true;
  el.translateProgress.hidden = false;
  el.langToggle.hidden = true;
  el.viewerLangToggle.hidden = true;
  el.translateCancel.disabled = false;
  updateTranslateProgress(current, total);
  applyReaderTranslateCta(); // 번역 중에는 리더 CTA 숨김
}
export function showLangToggle() {
  el.translateBtn.hidden = true;
  el.translateProgress.hidden = true;
  el.langToggle.hidden = false;
  el.viewerLangToggle.hidden = false;
  applyReaderTranslateCta(); // 번역 완료 — 리더 CTA 숨김
}

export function updateTranslateProgress(current, total) {
  const cur = Number(current) || 0;
  const tot = Number(total) || 0;
  el.translateProgressLabel.textContent = tot > 0 ? `번역 중 ${cur}/${tot}` : '번역 중…';
  const determinate = tot > 0;
  el.translateProgressTrack.classList.toggle('indeterminate', !determinate);
  el.translateProgressFill.style.width = determinate
    ? `${Math.min(100, Math.max(0, (cur / tot) * 100))}%`
    : '';
}

export function setLangSegActive(lang) {
  const ko = lang === 'ko';
  for (const [button, active] of [
    [el.langOrig, !ko],
    [el.langKo, ko],
    [el.viewerLangOrig, !ko],
    [el.viewerLangKo, ko],
  ]) {
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
}

// 문서/레이아웃/리더 뷰 컨테이너의 lang="ko" 속성 토글 (CJK 조판 CSS 적용용).
export function setResultLangAttr() {
  const ko = state.currentLang === 'ko';
  for (const node of [el.previewBody, el.doclayoutBody, el.readerContent]) {
    if (!node) continue;
    if (ko) node.setAttribute('lang', 'ko');
    else node.removeAttribute('lang');
  }
}

// 현재 언어에 맞춰 Markdown을 다시 세팅한다. 문서 HTML은 현재
// 보기 토글과 분리해 원문/한국어 버튼을 각각 고정하므로 잘못된 언어 파일을
// 조용히 받지 않는다. 아카이브는 ko 파일이 자동 포함되므로 원본 URL 그대로 둔다.
export function applyDownloadLangs() {
  const u = state.resultUrls || {};
  const base = state.currentBaseName || 'document';
  const suffix = state.currentLang === 'ko' ? '.ko' : '';
  setDownload(el.dlMd, u.markdown ? withLangUrl(u.markdown, state.currentLang) : null, `${base}${suffix}.md`);
  setDownload(el.dlZip, u.archive || null, `${base}.md.zip`);
  setDownload(el.dlDoc, u.documentHtml || null, `${base}.html`);
  setDownload(
    el.viewerDlHtml,
    u.documentHtml ? withLangUrl(u.documentHtml, state.currentLang) : null,
    `${base}${suffix}.html`,
  );
}

// 결과 뷰 진입 시 번역 상태를 조회해 알맞은 컨트롤을 노출한다 (done 잡에서만 호출).
export async function initTranslateForJob() {
  const id = state.currentJobId;
  if (!id) return;
  let st = null;
  try {
    st = await apiGet(`/api/jobs/${id}/translate/state?lang=ko`);
  } catch (_) { /* state 엔드포인트 불가 → 버튼 노출로 폴백 */ }
  if (state.currentJobId !== id) return;
  const status = (st && st.status) || 'none';
  state.translateState = status;
  const ui = translateUiStateFor(status);
  if (ui === 'progress') {
    showTranslateProgress(st && st.current, st && st.total);
    connectTranslateEvents(id); // 진행 중이던 번역에 재접속
  } else if (ui === 'toggle') {
    showLangToggle(); // 이미 번역 완료 → 토글만 노출(원문 기본, 사용자가 선택)
    renderTranslateSummary(st); // state에 병합된 사유별 집계를 그대로 요약
  } else {
    showTranslateButton();
  }
  applyPdfExport(); // 이미 번역된 잡을 열면 이 시점에 대조 PDF 버튼이 나타난다
  if (state.viewerOpen && state.viewerIntent.lang === 'ko' && status === 'done') setLang('ko');
}

// [한국어 번역] / 리더 [한국어로 읽기] 클릭 → 번역 시작 (공용 경로).
export async function startTranslate() {
  const id = state.currentJobId;
  if (!id) return;
  const waiting = retryLockRemaining('translateRetryAt');
  if (waiting) { // 직전 429의 대기 시간이 남아 있다 — 요청을 보내지 않는다
    showToast(`요청이 많습니다 — ${waiting}초 후 다시 시도해 주세요.`, 'warn');
    return;
  }
  el.translateBtn.disabled = true;
  el.readerTranslateBtn.disabled = true; // 리더 CTA도 같은 요청 — 이중 클릭 방지
  let res = null;
  let data = null;
  try {
    res = await fetch(`/api/jobs/${id}/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ lang: 'ko', force: false }),
    });
    const text = await res.text().catch(() => '');
    data = text ? safeParse(text) : null;
  } catch (_) {
    if (state.currentJobId !== id) return;
    el.translateBtn.disabled = false;
    el.readerTranslateBtn.disabled = false;
    showToast('번역 요청 중 네트워크 오류가 발생했습니다.', 'error');
    return;
  }
  if (state.currentJobId !== id) return;

  if (!res.ok) {
    el.translateBtn.disabled = false;
    el.readerTranslateBtn.disabled = false;
    const detail = (data && typeof data.detail === 'string') ? data.detail : null;
    if (res.status === 429) {
      // 레이트리밋/동시 번역 상한 — Retry-After만큼 안내하고 버튼을 잠근다
      const notice = rateLimitNotice(res.headers.get('Retry-After'), detail);
      showToast(notice.message, 'warn');
      lockRetry('translateRetryAt', 'translateRetryTimer', notice.seconds,
        [el.translateBtn, el.readerTranslateBtn], applyTranslateAvailability);
    } else if (res.status === 503) showToast(detail || '번역 프로바이더가 설정되지 않았습니다.', 'error');
    else if (res.status === 409) showToast(detail || '아직 완료되지 않은 작업은 번역할 수 없습니다.', 'warn');
    else if (res.status === 400) showToast(detail || '지원하지 않는 번역 언어입니다.', 'error');
    else showToast(detail || `번역 요청에 실패했습니다. (${res.status})`, 'error');
    return;
  }

  // 202/200 — 이미 번역돼 있으면(done) 바로 토글, 아니면 진행 UI + 구독.
  state.translateState = 'running';
  if (data && data.status === 'done') { onTranslateDone(id); return; }
  showTranslateProgress(0, 0);
  connectTranslateEvents(id);
}

export function connectTranslateEvents(id) {
  teardownTranslate(); // 중복 구독 방지
  if (typeof EventSource === 'undefined') { startTranslatePolling(id); return; }
  let es;
  try {
    es = new EventSource(`/api/jobs/${id}/translate/events?lang=ko`);
  } catch (_) { startTranslatePolling(id); return; }
  state.translateEs = es;

  es.addEventListener('progress', (e) => {
    if (state.currentJobId !== id) return;
    state.translateSseErrors = 0;
    const d = parseEventData(e);
    if (d) { state.translateState = 'running'; showTranslateProgress(d.current, d.total); }
  });
  es.addEventListener('done', (e) => {
    if (state.currentJobId !== id) return;
    onTranslateDone(id, parseEventData(e)); // counts(원문 유지·건너뜀)를 요약에 쓴다
  });
  es.addEventListener('error', (e) => {
    if (state.currentJobId !== id) return;
    const d = parseEventData(e);
    if (d) onTranslateError(id, d);        // 서버가 보낸 번역 오류(JSON)
    else handleTranslateConnError(id);     // 전송 계층 오류(데이터 없음)
  });
}

export function handleTranslateConnError(id) {
  if (state.currentJobId !== id || !state.translateEs) return;
  state.translateSseErrors += 1;
  if (state.translateSseErrors >= 2) { teardownTranslate(); startTranslatePolling(id); }
}

// SSE 불가/불안정 시 state를 폴링해 진행/완료/오류를 반영하는 폴백.
export function startTranslatePolling(id) {
  teardownTranslate();
  state.translatePollTimer = setInterval(async () => {
    if (state.currentJobId !== id) { clearInterval(state.translatePollTimer); state.translatePollTimer = 0; return; }
    let st;
    try { st = await apiGet(`/api/jobs/${id}/translate/state?lang=ko`); }
    catch (_) { return; }
    if (state.currentJobId !== id) return;
    const status = st && st.status;
    if (status === 'running') { showTranslateProgress(st.current, st.total); return; }
    clearInterval(state.translatePollTimer); state.translatePollTimer = 0;
    if (status === 'done') onTranslateDone(id, st); // 폴백 경로의 state에도 사유별 집계가 있다
    else if (status === 'error') onTranslateError(id, { message: st.error });
    else if (status === 'canceled') onTranslateError(id, { canceled: true });
    else showTranslateButton();
  }, 1500);
}

export function onTranslateDone(id, payload) {
  if (state.currentJobId !== id) return;
  teardownTranslate();
  state.translateState = 'done';
  showLangToggle();
  applyPdfExport(); // 번역이 끝나는 즉시 원문·한국어 대조 PDF 내보내기 노출
  // done 이벤트의 counts(원문 유지 개수)를 즉시 보여 주고, 사유별 집계는 이어서 채운다.
  renderTranslateSummary(payload);
  refreshTranslateSummary(id);
  setLang('ko'); // 완료 직후 자동으로 한국어 뷰로 전환
}

export function onTranslateError(id, d) {
  if (state.currentJobId !== id) return;
  teardownTranslate();
  const canceled = !!(d && d.canceled);
  state.translateState = canceled ? 'canceled' : 'error';
  showTranslateButton(); // 버튼 복원(재시도 가능)
  applyPdfExport();
  if (canceled) showToast('번역이 취소되었습니다.', 'warn');
  else showToast((d && d.message) || '번역 중 오류가 발생했습니다.', 'error');
}

// 취소(✕) — 요청만 보내고, UI 확정은 error(canceled) 이벤트/폴링에 맡긴다.
export async function cancelTranslate() {
  const id = state.currentJobId;
  if (!id) return;
  el.translateCancel.disabled = true;
  let ok = false;
  try {
    const res = await fetch(`/api/jobs/${id}/translate/cancel?lang=ko`, { method: 'POST' });
    ok = res.ok || res.status === 404;
  } catch (_) { /* 네트워크 오류 */ }
  if (state.currentJobId !== id) return;
  if (!ok) {
    el.translateCancel.disabled = false;
    showToast('번역 취소 요청에 실패했습니다.', 'error');
  }
  // 성공: error(canceled) 이벤트 또는 state 폴링이 버튼을 복원한다.
}

// 언어 토글. 캐시를 무효화하고 현재 탭을 새 언어로 다시 로드한다.
export function setLang(lang) {
  const next = lang === 'ko' ? 'ko' : 'orig';
  setLangSegActive(next);
  if (state.viewerOpen) state.viewerIntent = { open: true, page: state.readerPage, lang: next };
  if (state.currentLang === next) {
    if (state.viewerOpen) syncViewerSearch();
    return;
  }
  clearReaderAlignmentRetryLanguage(state.currentJobId, state.currentLang);
  state.currentLang = next;
  state.readerActiveBlock = '';
  setResultLangAttr();
  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
  el.mdCode.textContent = '';
  applyDownloadLangs();
  reloadActiveResultTab();
  if (state.viewerOpen) {
    renderViewerThumbnails();
    loadViewerManifest();
    syncViewerSearch();
  }
}

// 번역본 fetch가 404/실패일 때 조용히 원문으로 되돌린다 (호출부가 재로드).
export function revertToOriginal(reason) {
  if (state.currentLang !== 'ko') return false;
  clearReaderAlignmentRetryLanguage(state.currentJobId, state.currentLang);
  state.currentLang = 'orig';
  setLangSegActive('orig');
  setResultLangAttr();
  applyDownloadLangs();
  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
  el.mdCode.textContent = '';
  showToast(reason || '번역본을 불러오지 못해 원문을 표시합니다.', 'warn');
  return true;
}

// 현재 활성 결과 탭만 다시 로드 (썸네일·질문 탭은 언어 무관 → 스킵).
// 리더는 언어별 캐시가 있으면 같은 페이지를 새 언어로 즉시 재스왑한다.
export function reloadActiveResultTab() {
  const active = el.tabs.find((t) => t.classList.contains('active'));
  const name = active ? active.dataset.tab : 'preview';
  if (name === 'preview') loadPreview();
  else if (name === 'markdown') loadMarkdown();
  else if (name === 'doclayout') loadDocLayout();
  else if (name === 'reader') loadReader();
}
