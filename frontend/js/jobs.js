import { ICON, readerPosKey } from './constants.js';
import {
  armTransition, clampReaderPage, fmtTime, groundAnnounce, jobModelChip, jobNotices,
  jobRowSignature, parseViewerSearch, progressPhaseText, statusLabel, warningSegments,
} from './core.js';
import { armTimers, el, state } from './state.js';
import { h, isTerminal, localGet, localRemove, safeParse, showToast } from './ui.js';
import { POLL_TIMEOUT_MS, apiDelete, apiGet } from './api.js';
import {
  drainGroundToUI, flushStream, renderOverlay, resetLiveState, retryPageImageIfNeeded,
  updateLeftPane,
} from './live.js';
import { onJobError, startStream, teardownConnections } from './sse.js';
import { renderError, renderPartialResult, renderResult } from './results.js';
import {
  readerIsActive, readerLangKey, readerTotal, resetReaderForJob, setReaderPage,
} from './reader.js';
import { closeViewer, openViewer } from './viewer.js';
import { activateTab } from './tabs.js';
import { forgetReaderNotes } from './notes.js';

/* ============================ Job history ============================ */

// 목록 갱신은 한 번에 하나만 돈다. 느린 서버에서 5초 틱이 겹치면 늦게 도착한 옛
// 목록이 새 목록을 덮고 syncOpenJob이 중복 실행된다(frontend-12). 진행 중에 들어온
// 명시적 갱신(삭제·업로드·완료 후)은 끝난 뒤 한 번 더 돌고, 주기 폴링(pollJobs)은
// 진행 중이면 그냥 건너뛴다 — 응답이 느릴수록 요청이 쌓이지 않는다.
let jobsRefresh = null;
let jobsRefreshAgain = false;

export function refreshJobs() {
  if (jobsRefresh) {
    jobsRefreshAgain = true;
    return jobsRefresh;
  }
  jobsRefresh = (async () => {
    try {
      do {
        jobsRefreshAgain = false;
        await refreshJobsOnce();
      } while (jobsRefreshAgain);
    } finally {
      jobsRefresh = null;
    }
  })();
  return jobsRefresh;
}

// 5초 주기 폴링 진입점 — 이전 갱신이 아직 응답을 기다리면 이번 틱은 건너뛴다.
export function pollJobs() {
  return jobsRefresh || refreshJobs();
}

async function refreshJobsOnce() {
  let data;
  try {
    data = await apiGet('/api/jobs', { timeoutMs: POLL_TIMEOUT_MS });
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

// 키(job_id) 기반 증분 렌더 — 기존 <li>를 재사용하고 바뀐 줄만 제자리에서 고친다.
// 5초 폴링마다 목록을 통째로 다시 만들면 목록 안의 키보드 포커스가 body로 날아가
// 2단계 삭제의 두 번째 Enter가 허공에 가고, 스크린리더 가상 커서도 초기화된다.
export function renderJobList() {
  const list = el.jobList;
  const rows = state.jobs.map((job) => ({
    job, sig: jobRowSignature(job, job.job_id === state.currentJobId),
  }));
  el.jobListEmpty.hidden = rows.length > 0;
  const items = [...list.children];
  // 같은 순서·같은 서명 — 폴링 대부분은 DOM을 전혀 건드리지 않고 끝난다.
  if (items.length === rows.length && rows.every((row, i) =>
    items[i].dataset.jobId === row.job.job_id && items[i].dataset.sig === row.sig)) return;

  const focus = captureListFocus(list, items);
  const byId = new Map(items.map((item) => [item.dataset.jobId, item]));
  const wanted = new Set(rows.map((row) => row.job.job_id));
  // 곧 지워질 줄은 기준점에서 건너뛴다 — 남는 줄을 불필요하게 옮기지 않게.
  const skipStale = (node) => {
    let cur = node;
    while (cur && !wanted.has(cur.dataset.jobId)) cur = cur.nextElementSibling;
    return cur;
  };
  let cursor = skipStale(list.firstElementChild);
  for (const { job, sig } of rows) {
    let item = byId.get(job.job_id);
    if (item) {
      byId.delete(job.job_id);
      if (item.dataset.sig !== sig) updateJobListItem(item, job);
    } else {
      item = jobListItem(job);
    }
    item.dataset.sig = sig;
    if (item === cursor) cursor = skipStale(cursor.nextElementSibling);
    else list.insertBefore(item, cursor);
  }
  for (const stale of byId.values()) stale.remove();
  restoreListFocus(list, focus);
}

// 목록 안 포커스를 (잡, 컨트롤 종류, 위치)로 기억한다. 줄이 옮겨지거나 지워지면
// 브라우저가 포커스를 body로 보내므로, 렌더 뒤 같은 잡의 같은 컨트롤로 돌려준다.
function captureListFocus(list, items) {
  const active = typeof document !== 'undefined' ? document.activeElement : null;
  if (!active || active === list || !list.contains(active)) return null;
  const item = active.closest('.job-item');
  if (!item) return null;
  const role = ['ji-read', 'ji-del'].find((cls) => active.classList.contains(cls)) || 'ji-open';
  return { node: active, jobId: item.dataset.jobId, role, index: items.indexOf(item) };
}

function restoreListFocus(list, focus) {
  if (!focus) return;
  if (focus.node.isConnected && document.activeElement === focus.node) return;
  const items = [...list.children];
  let item = items.find((li) => li.dataset.jobId === focus.jobId);
  let target = item && item.querySelector(`.${focus.role}`);
  if (!target && !item && items.length) {
    // 그 잡이 목록에서 사라졌다(삭제) — 같은 자리(없으면 마지막) 줄로 포커스를 넘긴다.
    item = items[Math.min(Math.max(0, focus.index), items.length - 1)];
  }
  if (!target && item) target = item.querySelector('.ji-open');
  if (target) target.focus({ preventScroll: true });
}

function jobReadButton(job, fname) {
  const read = h('button', {
    class: 'ji-read icon-btn-sm', type: 'button',
    'aria-label': `"${fname}" 논문 뷰어로 열기`, title: '논문 뷰어로 열기', html: ICON.read,
  });
  read.addEventListener('click', () => openJobInViewer(job.job_id));
  return read;
}

export function jobListItem(job) {
  const status = job.status || 'queued';
  const active = job.job_id === state.currentJobId;
  const fname = job.filename || '(이름 없음)';

  const name = h('span', { class: 'ji-name', text: fname, title: job.filename || '' });
  const chip = h('span', { class: `chip chip-${status}`, text: statusLabel(job) });
  const time = h('span', { class: 'ji-time muted', text: fmtTime(job.created_at) });
  const sub = h('span', { class: 'ji-sub' }, chip, time);
  const warn = jobListWarnBadge(jobNotices(job).warnings.length);
  if (warn) sub.appendChild(warn);

  // 잡 열기·삭제를 형제 버튼으로 분리 — role="button" li 안에 버튼을 중첩하면
  // 스크린리더가 내부 삭제 버튼에 진입할 수 없다(중첩 인터랙티브 컨트롤 금지).
  const open = h('button', { class: 'ji-open', type: 'button' },
    h('span', { class: 'ji-main' }, name, sub));
  open.addEventListener('click', () => openJob(job.job_id));

  // 완료된 잡은 목록에서 한 번에 논문 뷰어로 — 잡 열기 → 뷰어 열기 2단계를 없앤다.
  const read = status === 'done' ? jobReadButton(job, fname) : null;

  const del = h('button', {
    class: 'ji-del icon-btn-sm', type: 'button',
    'aria-label': `"${fname}" 삭제`, title: '삭제', html: ICON.x,
  });
  del.addEventListener('click', () => armDelete(del, job.job_id, () => deleteJob(job.job_id)));
  // 새로 만든 줄에도 살아있는 무장(armed)을 복원한다. 만료 타이머가 최신 버튼을
  // 해제하도록 참조도 교체한다. (재사용되는 줄은 같은 버튼이라 그대로 유지된다.)
  const arm = armTimers.get(job.job_id);
  if (arm) {
    arm.btn = del;
    del.dataset.baseTitle = del.title;
    del.classList.add('armed');
    del.title = '한 번 더 클릭하면 삭제됩니다';
  }

  const item = h('li', { class: `job-item${active ? ' active' : ''}` }, open);
  item.dataset.jobId = job.job_id;
  if (read) item.appendChild(read);
  item.appendChild(del);
  item._row = { name, chip, time, read, del, sub, warn };
  return item;
}

// 목록 줄의 품질 경고 표시 — 결과를 열기 전에도 저품질 변환을 알아볼 수 있게.
function jobListWarnBadge(count) {
  if (!count) return null;
  return h('span', { class: 'ji-warn', text: `주의 ${count}`, title: `변환 품질 경고 ${count}건` });
}

// 재사용하는 줄의 바뀐 필드만 고친다 — 버튼 노드는 그대로라 포커스·무장이 유지된다.
export function updateJobListItem(item, job) {
  const row = item._row;
  const status = job.status || 'queued';
  const fname = job.filename || '(이름 없음)';
  item.classList.toggle('active', job.job_id === state.currentJobId);
  row.name.textContent = fname;
  row.name.title = job.filename || '';
  row.chip.className = `chip chip-${status}`;
  row.chip.textContent = statusLabel(job);
  row.time.textContent = fmtTime(job.created_at);
  const warnCount = jobNotices(job).warnings.length;
  if (row.warn) row.warn.remove();
  row.warn = jobListWarnBadge(warnCount);
  if (row.warn) row.sub.appendChild(row.warn);
  row.del.setAttribute('aria-label', `"${fname}" 삭제`);
  if (status === 'done' && !row.read) {
    row.read = jobReadButton(job, fname);
    item.insertBefore(row.read, row.del);
  } else if (status !== 'done' && row.read) {
    row.read.remove();
    row.read = null;
  } else if (row.read) {
    row.read.setAttribute('aria-label', `"${fname}" 논문 뷰어로 열기`);
  }
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

// 이 탭이 DELETE를 보내 응답을 기다리는 잡 — 서버가 같은 삭제를 SSE({deleted:true})로도
// 알리므로, 그 이벤트가 먼저 와도 "다른 곳에서 삭제됨" 안내를 띄우지 않는다.
const deletingJobs = new Set();

export async function deleteJob(id) {
  deletingJobs.add(id);
  try {
    await apiDelete(`/api/jobs/${id}`);
  } catch (e) {
    if (e.status !== 404) {
      showToast('삭제에 실패했습니다.', 'error');
      return;
    }
    // 404 → already gone; fall through to local cleanup
  } finally {
    deletingJobs.delete(id);
  }
  closeDeletedJob(id);
  refreshJobs();
}

// 잡이 서버에서 지워졌다(이 탭의 삭제, 또는 다른 탭·API의 삭제를 알린 SSE {deleted:true}).
// 목록·이어 읽기 위치·인용 메모를 정리하고, 열려 있던 화면(라이브 뷰·결과·전체 화면 뷰어)과
// 구독(SSE·폴링·번역·질문)을 닫는다. 예전에는 다른 곳의 삭제가 '취소됨' 화면으로 그려져
// 이미 없는 부분 결과를 계속 요청했다. remote면(SSE로 안 경우) 이유를 한 줄 알린다.
// (404처럼 서버 쪽 사정으로 안 보이는 경우는 일시적일 수 있어 이 경로를 쓰지 않는다 —
// 인용·하이라이트는 남겨 두고 보관 잡 수 상한이 결국 정리한다.)
export function closeDeletedJob(id, options = {}) {
  const wasOpen = state.currentJobId === id;
  removeJobFromList(id);
  localRemove(readerPosKey(id)); // 이어읽기 위치도 함께 정리 (localStorage 누수 방지)
  forgetReaderNotes(id);         // 지운 잡의 인용·하이라이트도 함께 지운다
  if (wasOpen) {
    teardownConnections();
    state.currentJobId = null;
    state.displayedStatus = null;
    state.displayedPhase = null;
    state.cancelRequestedFor = null;
    showEmptyState();
    syncJobHash(null); // 삭제된 잡을 가리키는 해시 정리
  }
  if (options.remote && wasOpen && !deletingJobs.has(id)) {
    showToast('열려 있던 작업이 삭제되었습니다.', 'warn');
  }
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
  state.jobWarningsOpen = false;
  renderJobWarnings(null); // 이전 잡의 경고가 새 잡 헤더에 남지 않게
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

  renderJobWarnings(job);

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

/* ============================ 잡 품질 경고 ============================ */
// 헤더 칩('주의 N건' — 경고 없이 참고만 있으면 '참고 N건')과 펼침 목록. 서버는 실패
// 플레이스홀더·텍스트 레이어 복구·충실도 예산 소진·페이지 경계 불일치를 warnings에
// 남기는데, 예전에는 어디에도 보이지 않아 저품질 결과가 초록 '완료'로만 보였다.
export function renderJobWarnings(job) {
  const chip = el.jobWarningsChip;
  if (!chip) return;
  const { warnings, notices } = jobNotices(job);
  if (!warnings.length && !notices.length) {
    chip.hidden = true;
    chip.textContent = '';
    el.jobWarningsList.textContent = '';
    el.jobNoticesList.textContent = '';
    el.jobNotices.hidden = true;
    applyJobWarningsOpen();
    return;
  }
  // 'N페이지' 링크는 리더가 있는 완료 잡에서만 — 그 밖에는 평문으로 둔다.
  const linkPages = !!job && job.status === 'done';
  chip.hidden = false;
  chip.textContent = warnings.length ? `주의 ${warnings.length}건` : `참고 ${notices.length}건`;
  chip.classList.toggle('chip-warn', warnings.length > 0);
  chip.classList.toggle('chip-note', warnings.length === 0);
  chip.title = warnings.length
    ? '변환 품질 경고가 있습니다 — 눌러서 목록 보기'
    : '변환 참고 사항이 있습니다 — 눌러서 목록 보기';
  fillJobNoteList(el.jobWarningsList, warnings, linkPages);
  fillJobNoteList(el.jobNoticesList, notices, linkPages);
  el.jobNotices.hidden = notices.length === 0;
  applyJobWarningsOpen();
}

export function toggleJobWarnings() {
  state.jobWarningsOpen = !state.jobWarningsOpen;
  applyJobWarningsOpen();
}

function applyJobWarningsOpen() {
  const open = state.jobWarningsOpen && !el.jobWarningsChip.hidden;
  el.jobWarnings.hidden = !open;
  el.jobWarningsChip.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function fillJobNoteList(list, items, linkPages) {
  list.textContent = '';
  for (const text of items) {
    const item = h('li', null);
    for (const seg of warningSegments(text)) {
      if (seg.type === 'page' && linkPages) {
        const link = h('button', {
          class: 'warning-page-link', type: 'button', text: seg.value,
          title: `${seg.page}페이지를 읽기 탭에서 열기`,
        });
        link.addEventListener('click', () => openWarningPage(seg.page));
        item.appendChild(link);
      } else {
        item.appendChild(document.createTextNode(seg.value));
      }
    }
    list.appendChild(item);
  }
}

// 경고가 가리키는 페이지를 리더에서 연다. 본문이 아직 없으면 착지 페이지만 정해 두고
// 리더 로더(renderReaderDocument)가 그 페이지로 스크롤하게 한다.
export function openWarningPage(page) {
  if (state.displayedStatus !== 'done') return;
  const target = clampReaderPage(page, readerTotal());
  if (!readerIsActive()) {
    state.readerPage = target;
    activateTab('reader');
  }
  if (state.readerPages[readerLangKey()]) setReaderPage(target);
  else state.readerPage = target;
  if (el.readerPagePane) el.readerPagePane.focus({ preventScroll: false });
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
  let reported = ''; // 202 본문의 status — 'canceling' | 'canceled' | 이미 끝난 잡의 터미널 상태
  try {
    const res = await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
    ok = res.ok;
    gone = res.status === 404;
    if (ok) {
      const data = safeParse(await res.text());
      if (data && typeof data.status === 'string') reported = data.status;
    }
  } catch (_) { /* network error (본문 읽기 실패는 status 미상 — SSE·폴링이 마감한다) */ }

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
    return;
  }
  // 'canceling'(실행 중·모델 로딩 대기): 러너가 다음 확인 지점에서 마감하고 종료 SSE
  // (canceled:true)나 상태 폴링이 화면을 마감한다. 'canceled'(워커가 아직 맡지 않은 대기 잡을
  // 서버가 즉시 마감)나 이미 끝난 잡의 터미널 상태면 그 이벤트를 기다리지 않고 지금 그린다 —
  // 폴링 강등·재연결 사이라 종료 이벤트를 놓친 화면이 '취소 중…'에 머물지 않게.
  if (isTerminal(reported) && !isTerminal(state.displayedStatus)) {
    await finalizeTerminalJob(id, reported);
  }
}

// 서버가 끝났다고 답한 열린 잡을 마감한다 — 상세를 다시 받아 결과·오류·취소 화면을 그린다
// (조회가 실패해도 취소는 응답대로 취소 화면으로). 그사이 SSE가 먼저 마감했으면 손대지 않는다.
async function finalizeTerminalJob(id, reported) {
  let job = null;
  try {
    job = await apiGet(`/api/jobs/${id}`, { timeoutMs: POLL_TIMEOUT_MS });
  } catch (e) {
    if (e && e.status === 404 && state.currentJobId === id) {
      closeDeletedJob(id, { remote: true }); // 취소 직후 삭제됨
      return;
    }
  }
  if (state.currentJobId !== id || isTerminal(state.displayedStatus)) return;
  if (job && isTerminal(job.status)) {
    flushStream(true);
    drainGroundToUI(true);
    teardownConnections(); // SSE·폴링이 같은 마감을 다시 그리지 않게 먼저 닫는다
    state.cancelRequestedFor = null;
    renderJob(job);
    refreshJobs();
  } else if (reported === 'canceled') {
    onJobError(id, { canceled: true });
  }
}
