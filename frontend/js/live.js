import {
  BOX_COLORS, BOX_FALLBACK_COLOR, STREAM_PANE_MAX_NODES, STREAM_PANE_TRIM_SLACK,
} from './constants.js';
import {
  PAGE_MARKER, createGroundState, groundDrain, groundPush, livePageImageUrl, normalizeLabel,
  planPreviewRender, replayExtendsRaw, streamPaneTrimCount, syncedStreamPageNo,
  truncateRawToPage,
} from './core.js';
import { el, state } from './state.js';
import { h, typesetMath } from './ui.js';
import { postPreviewRender } from './api.js';

/* ============================ Live state ============================ */

export function resetLiveState() {
  state.liveGen += 1;
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  state.previewDirty = false;
  state.previewFails = 0;
  state.previewStopped = false;
  state.previewPageCache = [];
  state.previewPageNodes = [];
  state.previewTailNodes = [];
  state.previewTailMd = '';
  state.previewTailSep = false;
  state.previewAutoScroll = true;
  state.streamPending = '';
  state.streamPageNo = 0;
  state.streamAutoScroll = true;
  state.streamConnected = false;
  state.rawText = '';
  state.ground = createGroundState();
  state.viewPage = 1;
  state.followLive = true;
  state.pageBoxes = new Map();
  state.imgFailed = false;
  state.imgLastTry = 0;
  state.cancelRequestedFor = null;

  el.streamPane.textContent = '';
  state.streamTrimNote = null;
  el.livePreview.innerHTML = '';
  el.boxOverlay.textContent = '';
  el.pageImg.hidden = true;
  el.pageImg.removeAttribute('src');
  delete el.pageImg.dataset.url;
  el.pageNote.hidden = false;
  el.pageNote.textContent = '페이지 이미지 대기 중…';
  el.pagerLabel.textContent = '– / –';
  el.pagerPrev.disabled = true;
  el.pagerNext.disabled = true;
  el.followChip.hidden = true;
}

/* ============================ Raw stream (middle pane) ============================ */


// 상한을 넘으면 앞쪽 노드를 잘라낸다. 사용자가 위로 올려 읽는 중(autoScroll off)
// 이면 없어진 높이만큼 scrollTop을 보정해 화면이 튀지 않게 한다.
export function trimStreamPane() {
  const pane = el.streamPane;
  const drop = streamPaneTrimCount(
    pane.childNodes.length, STREAM_PANE_MAX_NODES, STREAM_PANE_TRIM_SLACK);
  if (!drop) return;
  if (!state.streamTrimNote || !state.streamTrimNote.isConnected) {
    state.streamTrimNote = h('div', {
      class: 'stream-sys warn',
      text: '앞부분 원시 출력은 화면에서 정리했습니다 — 전체 원문은 완료 후 결과 탭에서 확인하세요.',
    });
    pane.insertBefore(state.streamTrimNote, pane.firstChild);
  }
  const before = pane.scrollHeight;
  for (let k = 0; k < drop; k += 1) {
    const node = state.streamTrimNote.nextSibling;
    if (!node) break;
    pane.removeChild(node);
  }
  if (!state.streamAutoScroll) {
    pane.scrollTop = Math.max(0, pane.scrollTop - (before - pane.scrollHeight));
  }
}

export function enqueueToken(text) {
  if (!text) return;
  state.streamPending += text;
  state.rawText += text;
  groundPush(state.ground, text);
  scheduleFlush();
}

// 서버가 EventSource 최초 연결/재연결 전에 누적한 전체 token 스트림을 원자적으로
// 보내는 replay 이벤트. 기존 부분 스트림을 append하면 중복되므로 RAW/grounding/
// preview 세 축을 같은 원문으로 함께 재구축한다. 진행 스냅샷보다 새 청크 선언이
// 앞선 경우(current_page > replay의 마지막 마커)는 다음 마커를 confirmation으로 둔다.
export function restoreTokenReplay(d) {
  if (!d || typeof d.text !== 'string' || !d.text) return;
  if (d.truncated) {
    appendSystemLine('누적 출력이 복구 상한을 넘어 전체 리플레이를 적용하지 못했습니다 — 완료 결과는 보존됩니다.', 'warn');
    return;
  }

  const oldFollow = state.followLive;
  const oldView = state.viewPage;
  // replay가 지금까지 받은 원문을 그대로 이어받은 정상 재연결이면 확정 페이지의
  // 렌더 캐시/DOM을 그대로 둔다 — 매 재연결마다 전 페이지를 다시 POST하지 않는다.
  const keepPreview = replayExtendsRaw(state.rawText, d.text);
  state.liveGen += 1; // 진행 중 preview 응답 무효화
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  state.previewDirty = false;
  state.previewFails = 0;
  state.previewStopped = false;
  if (!keepPreview) { state.previewPageCache = []; state.previewPageNodes = []; }
  // 꼬리는 어느 경우든 다시 렌더한다 — 소유 노드만 걷어내고 확정 노드는 남긴다.
  for (const n of state.previewTailNodes) n.remove();
  state.previewTailNodes = [];
  state.previewTailMd = '';
  state.previewTailSep = false;
  state.streamPending = d.text;
  state.streamPageNo = 0;
  state.rawText = d.text;
  state.ground = createGroundState();
  state.pageBoxes = new Map();

  el.streamPane.textContent = '';
  state.streamTrimNote = null;
  if (keepPreview) {
    // 중단 안내(lp-note)는 재개하는 렌더와 함께 걷어낸다 — 확정 본문만 남긴다.
    for (const note of el.livePreview.querySelectorAll('.lp-note')) note.remove();
  } else {
    el.livePreview.innerHTML = '';
  }
  el.boxOverlay.textContent = '';

  groundPush(state.ground, d.text);
  flushStream(false);
  drainGroundToUI(false);

  const total = Number(d.total_pages) || 0;
  if (total > state.ground.totalPages) state.ground.totalPages = total;
  state.ground.ocrSeen = true;
  const current = Number(d.current_page) || 0;
  if (current > state.ground.page) {
    state.ground.page = total ? Math.min(current, total) : current;
    state.ground.expectAnnounce = true;
  }
  state.streamPageNo = syncedStreamPageNo(state.streamPageNo, state.ground);
  state.followLive = oldFollow;
  state.viewPage = oldFollow
    ? state.ground.page
    : Math.min(Math.max(1, oldView), Math.max(total, state.ground.page, 1));
  updateLeftPane();
  schedulePreviewRender();
  appendSystemLine('누적 OCR 출력을 복구해 실시간 뷰를 동기화했습니다.');
}

// 서버가 이미 흘려보낸 출력을 폐기하고 그 페이지부터 다시 처리한다고 알리는
// reset 이벤트(backend/app/pipeline/runner.py BrokerSink.rewind_to). 클라이언트의
// 원문은 append-only라 이 신호 없이는 폐기된 출력이 영원히 남는다:
//   · 같은 페이지 박스가 두 번 쌓이고, 마커 없는 재처리 출력이 한 페이지에 몰린다
//   · 출력 상한에서 잘린 <table>이 미리보기의 뒤 내용을 통째로 삼킨 채 굳는다
//   · RAW 패널에 같은 페이지가 두 번 흐른다
// 세 패널을 잘라낸 원문 하나로 함께 되돌린다 — replay와 같은 재구축 경로다.
export function applyStreamReset(d) {
  const from = Math.floor(Number(d && d.from_page) || 0);
  if (from < 1) return;
  flushStream(false);            // 대기 중 토큰까지 원문에 반영한 뒤 자른다
  const truncated = truncateRawToPage(state.rawText, from);
  if (truncated.length === state.rawText.length) {
    // 그 페이지의 마커를 받은 적이 없다(늦게 접속·앞부분 유실) — 자를 지점을
    // 모르면 그대로 두고 사용자에게만 알린다.
    appendSystemLine(`${from}페이지부터 재처리합니다 — 이후 출력이 중복될 수 있습니다.`, 'warn');
    return;
  }

  state.liveGen += 1;            // 진행 중 preview 응답 무효화
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  state.previewDirty = false;
  state.previewFails = 0;
  state.previewStopped = false;
  state.streamPending = '';
  state.rawText = truncated;
  state.streamAutoScroll = true;

  // ① RAW 패널: 페이지 from의 디바이더부터 뒤를 전부 걷어낸다. 디바이더가 없으면
  //    (앞부분 정리로 잘려 나갔거나 아직 그려지지 않음) 자를 지점을 모르므로
  //    남은 원문으로 통째로 다시 그린다 — 폐기분이 화면에 남는 것보다 낫다.
  const divider = el.streamPane.querySelector(`.stream-page-break[data-page="${from}"]`);
  if (divider) {
    while (divider.nextSibling) divider.nextSibling.remove();
    divider.remove();
    state.streamPageNo = from - 1;
  } else {
    el.streamPane.textContent = '';
    state.streamTrimNote = null;
    state.streamPageNo = 0;
    state.streamPending = truncated;
    flushStream(false);
  }

  // ② 미리보기: 확정 페이지 캐시/노드를 잘라낸 원문에 맞춘다.
  //    잘라낸 원문에는 마커가 from-1개 남으므로 splitPreviewPages의 확정 페이지도
  //    from-1개다(인덱스 0 = 첫 마커 앞의 빈 서두, 1..from-2 = 페이지 1..from-2).
  //    페이지 from-1은 다시 "미확정 꼬리"가 되므로 캐시에서 빼야 중복 렌더되지 않는다.
  const keep = Math.max(0, from - 1);
  for (let i = keep; i < state.previewPageNodes.length; i += 1) {
    for (const n of state.previewPageNodes[i]) n.remove();
  }
  state.previewPageCache.length = Math.min(state.previewPageCache.length, keep);
  state.previewPageNodes.length = Math.min(state.previewPageNodes.length, keep);
  for (const n of state.previewTailNodes) n.remove();
  state.previewTailNodes = [];
  state.previewTailMd = '';
  state.previewTailSep = false;
  for (const note of el.livePreview.querySelectorAll('.lp-note')) note.remove();

  // ③ 레이아웃 박스: 잘라낸 원문으로 페이지 상태머신과 박스를 다시 만든다
  const total = state.ground.totalPages;
  state.ground = createGroundState();
  state.ground.totalPages = total;
  state.ground.ocrSeen = true;
  state.pageBoxes = new Map();
  el.boxOverlay.textContent = '';
  groundPush(state.ground, truncated);
  drainGroundToUI(false);
  if (state.followLive) state.viewPage = state.ground.page;
  updateLeftPane();

  schedulePreviewRender();
  const why = (d && typeof d.reason === 'string' && d.reason) ? ` (${d.reason})` : '';
  appendSystemLine(`${from}페이지부터 다시 처리합니다 — 앞선 출력은 폐기했습니다${why}.`, 'warn');
}

export function scheduleFlush() {
  if (state.rafId) return;
  state.rafId = requestAnimationFrame(() => {
    state.rafId = 0;
    flushStream(false);
    drainGroundToUI(false);
    // pane이 방금 pending 마커를 전부 소화한 시점 — 재연결 갭 등으로 마커가
    // 유실됐다면 ground 페이지 기준으로 다음 디바이더 번호를 재동기화한다.
    state.streamPageNo = syncedStreamPageNo(state.streamPageNo, state.ground);
    schedulePreviewRender();
  });
}

// Append pending stream text, converting <PAGE> markers into page-break
// dividers. Divider k reads "페이지 k" — marker k announces page k (each
// page's stream segment BEGINS with its marker). A partial marker at the
// tail is held back (unless final) so it is never split.
export function flushStream(final) {
  const buf = state.streamPending;
  state.streamPending = '';
  if (!buf) return;

  const frag = document.createDocumentFragment();
  let i = 0;
  while (true) {
    const idx = buf.indexOf(PAGE_MARKER, i);
    if (idx === -1) break;
    if (idx > i) frag.appendChild(document.createTextNode(buf.slice(i, idx)));
    state.streamPageNo += 1;
    frag.appendChild(makePageDivider(state.streamPageNo));
    i = idx + PAGE_MARKER.length;
  }

  let rest = buf.slice(i);
  if (!final && rest) {
    // hold back the longest suffix that could be the start of "<PAGE>"
    const maxCheck = Math.min(rest.length, PAGE_MARKER.length - 1);
    for (let k = maxCheck; k > 0; k -= 1) {
      if (PAGE_MARKER.startsWith(rest.slice(rest.length - k))) {
        state.streamPending = rest.slice(rest.length - k) + state.streamPending;
        rest = rest.slice(0, rest.length - k);
        break;
      }
    }
  }
  if (rest) frag.appendChild(document.createTextNode(rest));

  if (frag.childNodes.length) {
    el.streamPane.appendChild(frag);
    trimStreamPane();
    if (state.streamAutoScroll) el.streamPane.scrollTop = el.streamPane.scrollHeight;
  }
}

export function makePageDivider(n) {
  const div = h('div', { class: 'stream-page-break' }, h('span', { class: 'spb-label', text: `페이지 ${n}` }));
  div.dataset.page = String(n); // applyStreamReset가 이 지점부터 잘라낸다
  return div;
}

export function appendSystemLine(text, kind) {
  const line = h('div', { class: 'stream-sys' + (kind ? ' ' + kind : ''), text });
  el.streamPane.appendChild(line);
  if (state.streamAutoScroll) el.streamPane.scrollTop = el.streamPane.scrollHeight;
}

export function onStreamScroll() {
  const pane = el.streamPane;
  state.streamAutoScroll = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;
}

/* ============================ Grounding → left pane ============================ */

export function drainGroundToUI(final) {
  const events = groundDrain(state.ground, final);
  if (!events.length) return;
  let pageMoved = false;
  for (const ev of events) {
    if (ev.type === 'page') pageMoved = true;
    else addBoxes(ev.page, ev.label, ev.boxes);
  }
  if (pageMoved) {
    if (state.followLive) state.viewPage = state.ground.page;
    updateLeftPane();
  }
}

export function labelColor(label) {
  return BOX_COLORS[normalizeLabel(label)] || BOX_FALLBACK_COLOR;
}

export function addBoxes(page, label, boxes) {
  let arr = state.pageBoxes.get(page);
  if (!arr) { arr = []; state.pageBoxes.set(page, arr); }
  const labeled = boxes.map((b) => ({ label, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 }));
  for (const b of labeled) arr.push(b);
  if (page === state.viewPage) {
    const frag = document.createDocumentFragment();
    for (const b of labeled) frag.appendChild(makeBoxEl(b, true));
    el.boxOverlay.appendChild(frag);
  }
}

export function makeBoxEl(b, animate) {
  const div = h('div', { class: 'gbox' + (animate ? ' gbox-in' : '') });
  div.style.left = `${(b.x1 / 999) * 100}%`;
  div.style.top = `${(b.y1 / 999) * 100}%`;
  div.style.width = `${((b.x2 - b.x1) / 999) * 100}%`;
  div.style.height = `${((b.y2 - b.y1) / 999) * 100}%`;
  div.style.setProperty('--gbox-c', labelColor(b.label));
  div.appendChild(h('span', { class: 'gbox-label', text: b.label }));
  return div;
}

export function pageImageUrl(id, n) {
  // 라이브 패널은 렌더 단계 산출물을 직접 본다 (진행 중 페이지도 즉시 제공).
  // readerImageUrl(/page/{n})은 병합 완료 페이지 전용이라 여기서 쓰면 영구 404.
  return livePageImageUrl(id, n);
}

export function updateLeftPane() {
  const id = state.currentJobId;
  if (!id) return;
  const g = state.ground;
  const total = Math.max(g.totalPages || 0, g.page, 1);
  if (state.viewPage > total) state.viewPage = total;

  el.pagerLabel.textContent = `${state.viewPage} / ${total}`;
  el.pagerPrev.disabled = state.viewPage <= 1;
  el.pagerNext.disabled = state.viewPage >= total;
  el.followChip.hidden = state.followLive;

  // render 단계의 progress는 페이지 수를 먼저 알리지만 pages/*.png 쓰기는 아직
  // 진행 중일 수 있다. 첫 OCR 선언은 렌더 완료 뒤에 오므로 그 전에는 URL 자체를
  // 붙이지 않아 정상적인 준비 시간을 404/콘솔 오류로 만들지 않는다.
  if (!g.ocrSeen) {
    el.pageImg.hidden = true;
    el.pageNote.hidden = false;
    el.pageNote.textContent = '페이지 이미지 준비 중…';
    renderOverlay();
    return;
  }

  const url = pageImageUrl(id, state.viewPage);
  if (el.pageImg.dataset.url !== url) {
    el.pageImg.dataset.url = url;
    state.imgFailed = false;
    el.pageImg.src = url; // visibility settled by the load/error handlers
  }
  renderOverlay();
}

export function renderOverlay() {
  el.boxOverlay.textContent = '';
  const boxes = state.pageBoxes.get(state.viewPage);
  if (!boxes || !boxes.length) return;
  const frag = document.createDocumentFragment();
  for (const b of boxes) frag.appendChild(makeBoxEl(b, false));
  el.boxOverlay.appendChild(frag);
}

export function pageNav(dir) {
  const g = state.ground;
  const total = Math.max(g.totalPages || 0, g.page, 1);
  const next = Math.min(total, Math.max(1, state.viewPage + dir));
  if (next === state.viewPage) return;
  state.viewPage = next;
  state.followLive = next === g.page; // paging away disables follow; reaching the live page re-enables
  updateLeftPane();
}

export function onPageImgLoad() {
  state.imgFailed = false;
  el.pageImg.hidden = false;
  el.pageNote.hidden = true;
}

export function onPageImgError() {
  if (!el.pageImg.dataset.url) return; // src was cleared on reset
  state.imgFailed = true;
  el.pageImg.hidden = true;
  el.pageNote.hidden = false;
  el.pageNote.textContent = '페이지 이미지 준비 중…';
}

// Page PNGs appear once the render phase finishes; retry quietly on progress.
export function retryPageImageIfNeeded() {
  if (!state.imgFailed) return;
  const url = el.pageImg.dataset.url;
  if (!url) return;
  const now = Date.now();
  if (now - state.imgLastTry < 1500) return;
  state.imgLastTry = now;
  el.pageImg.src = `${url}?r=${now}`; // cache-bust the failed attempt
}

/* ============================ Live rendered preview (right pane) ============================ */

export function schedulePreviewRender() {
  if (state.previewStopped) return;
  state.previewDirty = true;
  if (state.previewTimer || state.previewInFlight) return;
  state.previewTimer = setTimeout(runPreviewRender, 600);
}

export function maybeReschedulePreview() {
  if (state.previewDirty && state.currentJobId && !state.previewStopped &&
      !state.previewTimer && !state.previewInFlight) {
    state.previewTimer = setTimeout(runPreviewRender, state.previewFails >= 4 ? 3000 : 600);
  }
}


// Trusted server-rendered fragment(/html과 동일 렌더러)를 pane 끝에 붙이고
// 붙인 노드 목록을 돌려준다. withSep이면 페이지 경계 hr을 그룹 선두에 포함.
// 타이포셋은 새로 붙인 노드로만 제한 — 기존 확정 노드는 재타이포셋하지 않는다.
export function appendPreviewFragment(html, withSep) {
  const nodes = [];
  if (withSep) nodes.push(h('hr'));
  const tpl = document.createElement('template');
  tpl.innerHTML = html;
  nodes.push(...tpl.content.childNodes);
  for (const n of nodes) {
    el.livePreview.appendChild(n);
    if (n.nodeType === 1) typesetMath(n);
  }
  return nodes;
}

// 413(서버 2MB 상한) 또는 연속 실패 시 라이브 프리뷰를 중단하고 원인에 맞는
// 한 줄 안내를 남긴다(일시 장애 중단에 '문서가 커서'라고 표시하지 않도록).
// 잡 완료 후 결과 탭(/html) 렌더는 이와 무관하게 기존 경로로 동작한다.
export function stopLivePreview(message) {
  state.previewStopped = true;
  state.previewDirty = false;
  clearTimeout(state.previewTimer);
  state.previewTimer = 0;
  el.livePreview.appendChild(h('div', {
    class: 'lp-note',
    text: message || '라이브 미리보기를 중단했습니다 — 완료 후 결과 탭에서 확인하세요',
  }));
}

// Throttled, latest-wins (queue of 1): at most one cycle in flight; tokens
// arriving mid-flight mark it dirty and exactly one follow-up is scheduled.
// 증분 렌더: 확정 페이지는 최초 1회만 POST해 HTML을 캐시하고, 이후에는
// 미확정 꼬리만 재전송한다 — 누적 전체 재전송(O(n²)·2MB 413 루프)을 피한다.
export async function runPreviewRender() {
  state.previewTimer = 0;
  if (state.previewInFlight || !state.previewDirty || state.previewStopped) return;
  const id = state.currentJobId;
  if (!id) { state.previewDirty = false; return; }
  const gen = state.liveGen;
  state.previewDirty = false;

  const plan = planPreviewRender(
    state.rawText, state.previewPageCache, state.previewTailMd, state.previewTailSep);
  if (!plan.newPages.length && !plan.tailChanged) { maybeReschedulePreview(); return; }

  state.previewInFlight = true;
  let failStatus = -1; // -1 = 실패 없음
  const pageHtmls = [];
  for (const p of plan.newPages) {
    if (!p.md) { pageHtmls.push(''); continue; }
    const r = await postPreviewRender(id, p.md);
    if (state.currentJobId !== id || state.liveGen !== gen) { // 잡 전환 가드
      state.previewInFlight = false;
      maybeReschedulePreview();
      return;
    }
    if (r.html == null) { failStatus = r.status; break; }
    pageHtmls.push(r.html);
  }
  let tailHtml = '';
  if (failStatus < 0 && plan.tailChanged && plan.tailMd) {
    const r = await postPreviewRender(id, plan.tailMd);
    if (state.currentJobId !== id || state.liveGen !== gen) { // 잡 전환 가드
      state.previewInFlight = false;
      maybeReschedulePreview();
      return;
    }
    if (r.html == null) failStatus = r.status;
    else tailHtml = r.html;
  }
  state.previewInFlight = false;

  if (failStatus >= 0) {
    state.previewFails += 1;
    state.previewDirty = true; // 전송하지 못한 조각은 다음 사이클에 재시도
    if (failStatus === 413) {
      stopLivePreview('문서가 커서 라이브 미리보기를 중단했습니다 — 완료 후 결과 탭에서 확인하세요');
    } else if (state.previewFails >= 5) {
      stopLivePreview('라이브 미리보기 렌더가 계속 실패해 중단했습니다 — 완료 후 결과 탭에서 확인하세요');
    } else {
      maybeReschedulePreview();
    }
    return;
  }
  state.previewFails = 0;

  // DOM 증분 적용: 확정 페이지 노드는 유지하고 꼬리 노드만 이동/교체한다.
  const oldTail = state.previewTailNodes;
  for (const n of oldTail) n.remove();
  state.previewTailNodes = [];
  plan.newPages.forEach((p, i) => {
    state.previewPageCache.push(pageHtmls[i]); // p.idx === 캐시 길이 (순서 보장)
    // 노드 목록도 같은 인덱스로 남긴다 — reset(재처리)이 그 페이지들만 걷어낸다
    state.previewPageNodes.push(pageHtmls[i] ? appendPreviewFragment(pageHtmls[i], p.sep) : []);
  });
  if (plan.tailChanged) {
    state.previewTailMd = plan.tailMd;
    state.previewTailSep = plan.tailSep;
    if (tailHtml) state.previewTailNodes = appendPreviewFragment(tailHtml, plan.tailSep);
  } else {
    // 꼬리 내용은 그대로인데 앞에 확정 페이지가 생긴 경우 — 같은 노드를 재부착
    for (const n of oldTail) el.livePreview.appendChild(n);
    state.previewTailNodes = oldTail;
  }
  if (state.previewAutoScroll) el.livePreview.scrollTop = el.livePreview.scrollHeight;
  maybeReschedulePreview();
}

export function onPreviewScroll() {
  const pane = el.livePreview;
  state.previewAutoScroll = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;
}
