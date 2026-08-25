import { ICON, readerPosKey } from './constants.js';
import {
  armTransition, clampReaderPage, fmtTime, groundAnnounce, jobModelChip, parseViewerSearch,
  progressPhaseText, statusLabel,
} from './core.js';
import { armTimers, el, state } from './state.js';
import { h, isTerminal, localGet, localRemove, showToast } from './ui.js';
import { apiDelete, apiGet } from './api.js';
import {
  drainGroundToUI, flushStream, renderOverlay, resetLiveState, retryPageImageIfNeeded,
  updateLeftPane,
} from './live.js';
import { startStream, teardownConnections } from './sse.js';
import { renderError, renderPartialResult, renderResult } from './results.js';
import { readerTotal, resetReaderForJob } from './reader.js';
import { closeViewer, openViewer } from './viewer.js';

/* ============================ Job history ============================ */

export async function refreshJobs() {
  let data;
  try {
    data = await apiGet('/api/jobs');
  } catch (_) {
    return; // keep last known list on transient failure
  }
  const jobs = (data && Array.isArray(data.jobs)) ? data.jobs : [];
  state.jobs = jobs.slice(0, 50);
  renderJobList();

  if (state.currentJobId) {
    const open = state.jobs.find((j) => j.job_id === state.currentJobId);
    if (open) {
      noteQueuePosition(open.status, open.queue_position);
      updateHeaderChip(open.status);
      // queued 동안은 SSE progress가 없어 이 5초 목록 폴링이 유일한 대기열 위치
      // 갱신원이다 — 진행 영역의 '대기중 · N번째' 문구도 여기서 함께 갱신한다.
      if (open.status === 'queued' && state.displayedStatus === 'queued') {
        updateProgress(open.progress || {}, 'queued');
      }
      // openJob이 해시 잡을 최초 fetch하는 동안 displayedStatus는 null이다. 이를
      // running→terminal 전이로 오인해 syncOpenJob을 겹쳐 실행하면 같은 결과를
      // 두 번 렌더하고 reader retry/cache를 중간 teardown한다. 실제로 한 번이라도
      // 상태를 표시한 실행 중 잡만 목록 폴링으로 terminal 승격한다.
      if (state.displayedStatus != null
          && isTerminal(open.status) && !isTerminal(state.displayedStatus)) {
        syncOpenJob();
      }
    }
  }
}

export function renderJobList() {
  const list = el.jobList;
  list.textContent = '';
  if (!state.jobs.length) {
    el.jobListEmpty.hidden = false;
    return;
  }
  el.jobListEmpty.hidden = true;
  for (const job of state.jobs) list.appendChild(jobListItem(job));
}

export function jobListItem(job) {
  const status = job.status || 'queued';
  const active = job.job_id === state.currentJobId;
  const fname = job.filename || '(이름 없음)';

  const name = h('span', { class: 'ji-name', text: fname, title: job.filename || '' });
  const chip = h('span', { class: `chip chip-${status}`, text: statusLabel(job) });
  const time = h('span', { class: 'ji-time muted', text: fmtTime(job.created_at) });
  const sub = h('span', { class: 'ji-sub' }, chip, time);

  // 잡 열기·삭제를 형제 버튼으로 분리 — role="button" li 안에 버튼을 중첩하면
  // 스크린리더가 내부 삭제 버튼에 진입할 수 없다(중첩 인터랙티브 컨트롤 금지).
  const open = h('button', { class: 'ji-open', type: 'button' },
    h('span', { class: 'ji-main' }, name, sub));
  open.addEventListener('click', () => openJob(job.job_id));

  // 완료된 잡은 목록에서 한 번에 논문 뷰어로 — 잡 열기 → 뷰어 열기 2단계를 없앤다.
  let read = null;
  if (status === 'done') {
    read = h('button', {
      class: 'ji-read icon-btn-sm', type: 'button',
      'aria-label': `"${fname}" 논문 뷰어로 열기`, title: '논문 뷰어로 열기', html: ICON.read,
    });
    read.addEventListener('click', () => openJobInViewer(job.job_id));
  }

  const del = h('button', {
    class: 'ji-del icon-btn-sm', type: 'button',
    'aria-label': `"${fname}" 삭제`, title: '삭제', html: ICON.x,
  });
  del.addEventListener('click', () => armDelete(del, job.job_id, () => deleteJob(job.job_id)));
  // 재렌더가 무장(armed) 상태를 파괴하지 않도록 살아있는 무장을 새 버튼에 복원.
  // 만료 타이머가 최신 버튼을 해제하도록 참조도 교체한다.
  const arm = armTimers.get(job.job_id);
  if (arm) {
    arm.btn = del;
    del.dataset.baseTitle = del.title;
    del.classList.add('armed');
    del.title = '한 번 더 클릭하면 삭제됩니다';
  }

  const item = h('li', { class: `job-item${active ? ' active' : ''}` }, open);
  if (read) item.appendChild(read);
  item.appendChild(del);
  return item;
}

// 목록에서 바로 논문 뷰어로. 이미 열려 있는 잡이면 곧장 열고, 아니면 뷰어
// 의사(intent)를 심어 두고 잡을 연다 — renderResult의 applyViewerIntent가 받는다.
export function openJobInViewer(id) {
  const saved = Math.floor(Number(localGet(readerPosKey(id))));
  const page = Number.isFinite(saved) && saved >= 1 ? saved : 1;
  const lang = state.viewerIntent && state.viewerIntent.lang === 'ko' ? 'ko' : 'orig';
  const intent = { open: true, page, lang };
  if (id === state.currentJobId && state.displayedStatus === 'done') {
    state.viewerIntent = intent;
    state.readerPage = clampReaderPage(page, readerTotal());
    openViewer();
    return;
  }
  openJob(id, { viewer: intent });
}

export function disarmDeleteBtn(btn) {
  btn.classList.remove('armed');
  btn.title = btn.dataset.baseTitle || '삭제';
}

export function armDelete(btn, key, onConfirm) {
  const { confirm, clearKeys } = armTransition(
    Array.from(armTimers, ([k, e]) => [k, e.btn]), key, btn);
  for (const k of clearKeys) {
    const e = armTimers.get(k);
    if (e) clearTimeout(e.t);
    armTimers.delete(k);
  }
  if (confirm) {
    disarmDeleteBtn(btn);
    onConfirm();
    return;
  }
  if (!btn.dataset.baseTitle) btn.dataset.baseTitle = btn.title || '삭제';
  btn.classList.add('armed');
  btn.title = '한 번 더 클릭하면 삭제됩니다';
  const t = setTimeout(() => {
    const e = armTimers.get(key);
    armTimers.delete(key);
    if (e) disarmDeleteBtn(e.btn); // 재렌더로 교체됐어도 최신 버튼을 해제
  }, 2600);
  armTimers.set(key, { t, btn });
}

export function removeJobFromList(id) {
  state.jobs = state.jobs.filter((j) => j.job_id !== id);
  renderJobList();
}

export function upsertJob(job) {
  state.jobs = state.jobs.filter((j) => j.job_id !== job.job_id);
  state.jobs.unshift(job);
  state.jobs = state.jobs.slice(0, 50);
  renderJobList();
}

export async function deleteJob(id) {
  try {
    await apiDelete(`/api/jobs/${id}`);
  } catch (e) {
    if (e.status !== 404) {
      showToast('삭제에 실패했습니다.', 'error');
      return;
    }
    // 404 → already gone; fall through to local cleanup
  }
  removeJobFromList(id);
  localRemove(readerPosKey(id)); // 이어읽기 위치도 함께 정리 (localStorage 누수 방지)
  if (state.currentJobId === id) {
    teardownConnections();
    state.currentJobId = null;
    state.displayedStatus = null;
    state.displayedPhase = null;
    showEmptyState();
    syncJobHash(null); // 삭제된 잡을 가리키는 해시 정리
  }
  refreshJobs();
}

/* ============================ View switching ============================ */

export function showEmptyState() {
  closeViewer({ sync: false, restoreFocus: false });
  el.jobView.hidden = true;
  el.emptyState.hidden = false;
}

export function showJobView() {
  el.emptyState.hidden = true;
  el.jobView.hidden = false;
}

export function updateHeaderChip(status) {
  el.jobChip.className = `chip chip-${status}`;
  el.jobChip.textContent = statusLabel({ status, queue_position: state.queuePos });
}

// 잡 JSON/진행 페이로드의 대기열 위치를 상태에 흡수. queued가 아니면 해제하고,
// queued인데 필드가 없으면(SSE 스냅샷·구버전 서버) 마지막 값을 유지한다 —
// 계약상 필드 부재는 "기존 표시 그대로"가 안전 폴백이다.
export function noteQueuePosition(status, pos) {
  if (status !== 'queued') state.queuePos = null;
  else if (Number.isInteger(pos) && pos >= 1) state.queuePos = pos;
}

/* ============================ location.hash 잡 복원 ============================ */


// 현재 잡을 주소창 해시에 반영 — 새로고침 복원·영속 링크용. replaceState라
// 히스토리 스택을 오염시키지 않고 hashchange도 발생하지 않는다(자기 변경 루프
// 없음). id=null이면 해시 제거 — 잡 삭제·404로 현재 잡이 사라진 경우.
export function syncJobHash(id) {
  try {
    if (id) history.replaceState(null, '', '#' + id);
    else if (location.hash) history.replaceState(null, '', location.pathname + location.search);
  } catch (_) { /* ignore */ }
}

/* ============================ Open / render a job ============================ */

// options.viewer를 주면 주소창 대신 그 의사(intent)로 연다 — 목록의 "바로 읽기"처럼
// 잡을 열자마자 뷰어까지 이어 가는 경로용. 없으면 기존대로 주소창에서 복원한다.
export async function openJob(id, options = {}) {
  if (!id || id === state.currentJobId) return;
  // A→B→A처럼 같은 잡으로 되돌아오면 id 비교만으로는 재진입을 못 걸러낸다.
  // 세대 번호로 "가장 마지막 openJob"만 스냅샷을 적용하고 스트림을 연다.
  const gen = ++state.openGen;

  closeViewer({ sync: false, restoreFocus: false });
  state.viewerIntent = options.viewer || (typeof location !== 'undefined'
    ? parseViewerSearch(location.search)
    : { open: false, page: 1, lang: 'orig' });
  teardownConnections();
  state.currentJobId = id;
  // 잡 전환 시 이전 잡의 헤더 삭제 무장 잔상 제거 — 기능상 armTransition이 키
  // 불일치로 confirm을 거부하지만, armed 시각 표시가 남으면 거짓 안내가 된다.
  for (const [k, e] of armTimers) {
    if (k.startsWith('header:') && k !== `header:${id}`) {
      clearTimeout(e.t);
      armTimers.delete(k);
      disarmDeleteBtn(e.btn);
    }
  }
  state.displayedStatus = null;
  state.displayedPhase = null;
  state.queuePos = null; // 이전 잡의 대기열 위치가 새 잡 칩에 새지 않도록
  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  state.currentLang = 'orig'; // 잡이 바뀌면 언어 선택 초기화
  state.resultHasLayout = undefined;
  resetReaderForJob(); // 리더 페이지·언어별 캐시·질문 프리필 가드 초기화
  resetLiveState();
  showJobView();
  renderJobList(); // refresh active highlight

  let job;
  try {
    job = await apiGet(`/api/jobs/${id}`);
  } catch (e) {
    if (state.currentJobId !== id || state.openGen !== gen) return;
    if (e.status === 404) {
      showToast('해당 작업을 찾을 수 없습니다.', 'warn');
      removeJobFromList(id);
      syncJobHash(null); // 사라진 잡을 가리키는 해시(공유 링크 등) 정리
    } else {
      showToast('작업 정보를 불러오지 못했습니다.', 'error');
    }
    state.currentJobId = null;
    showEmptyState();
    return;
  }
  // 늦게 도착한 stale 스냅샷이 새 UI에 주입되지 않게 세대까지 확인한다.
  if (state.currentJobId !== id || state.openGen !== gen) return; // user switched away during await

  syncJobHash(id); // 성공 경로에서만 해시 동기화 — 새로고침 복원·링크 공유
  renderJob(job);
  if (job.status === 'queued' || job.status === 'running') startStream(id);
}

// Re-render the currently open job without tearing down the live panes.
export async function syncOpenJob() {
  const id = state.currentJobId;
  if (!id) return;
  let job;
  try {
    job = await apiGet(`/api/jobs/${id}`);
  } catch (_) {
    return;
  }
  if (state.currentJobId !== id) return;
  if (isTerminal(job.status)) {
    flushStream(true);
    drainGroundToUI(true);
    teardownConnections();
  }
  renderJob(job);
}

export function renderJob(job) {
  state.displayedStatus = job.status;
  noteQueuePosition(job.status, job.queue_position);

  el.jobFilename.textContent = job.filename || '(이름 없음)';
  el.jobFilename.title = job.filename || '';
  el.viewerFilename.textContent = job.filename || '논문';
  el.jobTime.textContent = job.created_at ? fmtTime(job.created_at) : '';
  updateHeaderChip(job.status);

  // 잡의 엔진/모델 메타 칩 — 완료 후에도 어떤 모델로 변환했는지 확인 가능.
  // 구버전 잡(필드 없음)은 칩 자체를 숨긴다.
  state.currentJobEngine = typeof job.engine === 'string' ? job.engine : undefined;
  const modelChip = jobModelChip(job);
  if (el.jobModel) {
    if (modelChip) {
      el.jobModel.textContent = modelChip.text;
      el.jobModel.title = modelChip.title;
      el.jobModel.hidden = false;
    } else {
      el.jobModel.hidden = true;
    }
  }

  const running = job.status === 'queued' || job.status === 'running';
  const done = job.status === 'done';
  const canceled = job.status === 'canceled';
  const failed = job.status === 'error';
  // 모델 로딩 대기 단계는 아직 페이지가 렌더되지 않아 라이브 뷰(페이지 이미지)를
  // 열면 404가 쏟아진다 — 진행바만 보여준다.
  const loadingPhase = running && (job.progress || {}).phase === 'loading';

  el.progressSection.hidden = !running;
  el.resultSection.hidden = !(done || canceled);
  el.errorSection.hidden = !(failed || canceled);
  el.liveDetails.hidden = (loadingPhase || !running) && !hasLiveContent();
  el.liveDetails.open = running && !loadingPhase;
  setStopButton(job.status);

  if (running) {
    const p = job.progress || {};
    state.displayedPhase = p.phase;
    updateProgress(p, job.status);
    if (!loadingPhase) {
      // A status snapshot goes through the same announce machine as SSE
      // progress (phase-gated: an OCR snapshot seeds the page when opening a
      // job mid-OCR; render/merge snapshots must not pin it).
      const r = groundAnnounce(state.ground, p.phase, p.current_page, p.total_pages);
      if (r.firstOcr && state.pageBoxes.size > 0) state.pageBoxes = new Map();
      if (state.followLive) state.viewPage = state.ground.page;
      updateLeftPane();
    }
  }
  if (done) renderResult(job);
  if (canceled) {
    renderError(job.error, true);
    if (job.result) renderResult(job);
    else renderPartialResult(job);
  }
  if (failed) renderError(job.error, false);
}

export function hasLiveContent() {
  return state.rawText.length > 0 || state.pageBoxes.size > 0 || el.streamPane.childNodes.length > 0;
}

/* ============================ Progress ============================ */

// The progress BAR consumes every phase (render progress is real progress);
// page tracking for the left pane is delegated to groundAnnounce, which
// filters to phase==="ocr".
export function updateProgress(p, status) {
  const queued = status === 'queued';
  // 모델 로딩 대기(note)는 진행 단계가 아직 미정이라 queued와 동일하게 스피너/불확정
  const note = (p && typeof p.note === 'string' && p.note) ? p.note : '';
  const total = Number(p.total_pages) || 0;
  const cur = Number(p.current_page) || 0;
  const totalChunks = Number(p.total_chunks) || 0;
  const chunk = Number(p.chunk) || 0;

  // queued 문구는 헤더/목록 칩과 동일 조합('대기중 · N번째')으로 통일
  el.progressPhase.textContent = progressPhaseText(
    p, status, statusLabel({ status: 'queued', queue_position: state.queuePos }),
  );
  el.progressSpinner.hidden = !queued && !note;

  const determinate = !queued && !note && total > 0;
  el.progressTrack.classList.toggle('indeterminate', !determinate);
  if (determinate) {
    const pct = Math.min(100, Math.max(0, (cur / total) * 100));
    el.progressFill.style.width = `${pct}%`;
    el.progressCount.textContent = `${cur} / ${total} 페이지`;
  } else {
    el.progressFill.style.width = '';
    el.progressCount.textContent = '';
  }

  el.progressChunk.textContent = (!queued && totalChunks > 0) ? `청크 ${chunk} / ${totalChunks}` : '';
}

// SSE / poll progress payloads are flat objects that include "status".
export function applyProgress(d) {
  const status = d.status || state.displayedStatus;
  const wasRunning = state.displayedStatus === 'queued' || state.displayedStatus === 'running';
  const prevPhase = state.displayedPhase;
  state.displayedStatus = status;
  state.displayedPhase = d.phase;
  noteQueuePosition(status, d.queue_position);
  updateHeaderChip(status);

  const running = status === 'queued' || status === 'running';
  el.progressSection.hidden = !running;
  if (!running) return;

  el.resultSection.hidden = true;
  el.errorSection.hidden = true;
  setStopButton(status);
  updateProgress(d, status);

  // 모델 로딩 대기 단계에서는 아직 페이지가 없으니 라이브 3-패널을 열지 않는다 —
  // 없는 페이지 이미지를 요청해 404가 쏟아지는 것을 막는다(라이브 내용이 이미
  // 쌓여 있으면 유지). 실제 render/ocr로 진입하면 아래 경로로 넘어간다.
  if (d.phase === 'loading') {
    el.liveDetails.hidden = !hasLiveContent();
    return;
  }

  el.liveDetails.hidden = false;
  // 로딩/미표시에서 실제 처리 단계로 처음 진입할 때 라이브 뷰를 연다 (수동 접힘 존중)
  if (!wasRunning || prevPhase === 'loading') el.liveDetails.open = true;

  // Drain buffered markers/boxes FIRST so content preceding this announcement
  // stays attributed to its own page, then apply the announcement.
  drainGroundToUI(false);
  const r = groundAnnounce(state.ground, d.phase, d.current_page, d.total_pages);
  if (r.firstOcr && state.pageBoxes.size > 0) {
    state.pageBoxes = new Map(); // stale pre-OCR boxes (rerun leftovers)
    renderOverlay();
  }
  if (r.pageChanged || r.firstOcr || r.totalChanged) {
    if (state.followLive) state.viewPage = state.ground.page;
    updateLeftPane();
  }
  retryPageImageIfNeeded();
}

/* ============================ Cancel (STOP) ============================ */

export function setStopButton(status) {
  const running = status === 'queued' || status === 'running';
  el.jobStop.hidden = !running;
  if (!running) return;
  const canceling = state.cancelRequestedFor === state.currentJobId;
  el.jobStop.disabled = canceling;
  el.jobStopLabel.textContent = canceling ? '취소 중…' : '정지';
}

export async function requestCancel() {
  const id = state.currentJobId;
  if (!id) return;
  const status = state.displayedStatus;
  if (status !== 'queued' && status !== 'running') return;

  state.cancelRequestedFor = id;
  setStopButton(status);

  let ok = false;
  let gone = false;
  try {
    const res = await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
    ok = res.ok;
    gone = res.status === 404;
  } catch (_) { /* network error */ }

  if (state.currentJobId !== id) return;
  if (gone) {
    removeJobFromList(id);
    teardownConnections();
    state.currentJobId = null;
    state.displayedStatus = null;
    state.displayedPhase = null;
    showEmptyState();
    syncJobHash(null); // 404로 사라진 잡 — 해시 정리
    showToast('해당 작업을 찾을 수 없습니다.', 'warn');
    return;
  }
  if (!ok) {
    state.cancelRequestedFor = null;
    setStopButton(state.displayedStatus);
    showToast('취소 요청에 실패했습니다.', 'error');
  }
  // success: the SSE error event (canceled:true) or status polling finalizes the UI
}
