import {
  QA_EFFORTS, QA_LS_EFFORT, QA_LS_PROVIDER, QA_LS_SUMMARY, QA_LS_THINKING, QA_SUMMARIES,
  qaModelKey,
} from './constants.js';
import { buildQaBody, clampQaPage, pickQaModels, qaProviderHint, rateLimitNotice } from './core.js';
import { el, state } from './state.js';
import {
  applyRetryLock, h, localGet, localSet, lockRetry, retryLockRemaining, safeParse, showToast,
} from './ui.js';
import { apiGet } from './api.js';

/* ============================ 질문 (Q&A) ============================
 * 완료된 잡의 페이지 텍스트에 대해 선택한 LLM(OpenAI Responses/Chat·Ollama)에게
 * 질문하는 결과 탭. 공급자/모델/effort/thinking/summary 선택은 localStorage에
 * 기억하고, 카탈로그(/api/providers)는 탭 최초 활성화 시 1회만 불러온다.
 * 미완료(취소 포함) 잡에는 플레이스홀더만 노출 — 폼 자체가 숨겨진다.
 * ================================================================== */


// 잡 전환·삭제 공통 정리 훅 (teardownConnections에서 호출) — 이전 잡의 대화
// 로그를 비우고 진행 중 요청의 응답을 무효화한다. 컨트롤/카탈로그는 잡과
// 무관한 사용자 설정이므로 유지한다.
export function teardownQa() {
  state.qaGen += 1; // 진행 중이던 질문 응답은 도착해도 무시된다
  state.qaBusy = false;
  state.qaTotalPages = 0;
  if (!el.qaLog) return; // grabEls 이전(이론상) 방어
  el.qaLog.textContent = '';
  setQaBusyUI(false);
  showQaPlaceholder(); // 다음 done 잡의 initQaForJob이 폼을 다시 연다
}

export function showQaPlaceholder() {
  el.qaPlaceholder.hidden = false;
  el.qaBodyWrap.hidden = true;
}

// 완료(done) 잡의 결과 렌더에서 호출 — 폼 노출 + 페이지 입력 기본/상한 설정.
// 총 페이지 수는 결과의 원본 페이지 목록(pages) 우선, 없으면 진행 스냅샷.
export function initQaForJob(job) {
  const r = (job && job.result) || {};
  const total = (Array.isArray(r.pages) && r.pages.length) ||
    Number(((job || {}).progress || {}).total_pages) || 0;
  state.qaTotalPages = total;
  el.qaPlaceholder.hidden = true;
  el.qaBodyWrap.hidden = false;
  el.qaPage.value = '1';
  if (total >= 1) el.qaPage.max = String(total);
  else el.qaPage.removeAttribute('max');
}

// 질문 탭 활성화 진입점 — 카탈로그 미로드 시에만 fetch (실패 시 다음 활성화에 재시도).
export function initQaTab() {
  if (state.qaCatalog || state.qaCatalogLoading) return;
  loadQaCatalog();
}

export function setQaCatalogNote(text) {
  el.qaProvider.textContent = '';
  el.qaProvider.appendChild(h('option', { value: '', text }));
  el.qaModel.textContent = '';
  el.qaModel.appendChild(h('option', { value: '', text: '—' }));
}

export async function loadQaCatalog() {
  state.qaCatalogLoading = true;
  setQaCatalogNote('공급자 확인 중…');
  let catalog = null;
  try {
    catalog = await apiGet('/api/providers');
  } catch (_) { /* 아래 공통 실패 처리 */ }
  state.qaCatalogLoading = false;
  if (!catalog || !Array.isArray(catalog.providers) || !catalog.providers.length) {
    setQaCatalogNote('공급자 정보를 불러오지 못했습니다');
    showToast('LLM 공급자 정보를 불러오지 못했습니다. 탭을 다시 열면 재시도합니다.', 'error');
    return;
  }
  state.qaCatalog = catalog;

  // 저장된 선택 복원 — 카탈로그에 없으면 서버 기본값 → 첫 공급자 순으로 폴백
  const ids = catalog.providers.map((p) => p && p.id);
  const savedProvider = localGet(QA_LS_PROVIDER);
  state.qaProvider = ids.includes(savedProvider) ? savedProvider
    : (ids.includes(catalog.default_provider) ? catalog.default_provider : (ids[0] || ''));
  localSet(QA_LS_PROVIDER, state.qaProvider);

  const savedEffort = localGet(QA_LS_EFFORT);
  state.qaEffort = QA_EFFORTS.includes(savedEffort) ? savedEffort
    : (QA_EFFORTS.includes(catalog.default_reasoning_effort) ? catalog.default_reasoning_effort : 'default');
  state.qaThinking = localGet(QA_LS_THINKING) !== 'false'; // 기본 켜짐
  const savedSummary = localGet(QA_LS_SUMMARY);
  state.qaSummary = QA_SUMMARIES.includes(savedSummary) ? savedSummary : 'none';
  el.qaEffort.value = state.qaEffort;

  el.qaProvider.textContent = '';
  for (const p of catalog.providers) {
    if (!p || !p.id) continue;
    el.qaProvider.appendChild(h('option', {
      value: p.id,
      text: p.available ? String(p.label || p.id) : `${p.label || p.id} (설정 필요)`,
    }));
  }
  el.qaProvider.value = state.qaProvider;
  updateQaProviderControls();
}

// 공급자 선택에 맞춰 모델 목록/summary 가용성/thinking 표시를 갱신.
// (Localight updateProviderControls 이식 — 공급자별 저장 모델 복원 포함)
export function updateQaProviderControls() {
  const picked = pickQaModels(state.qaCatalog, state.qaProvider);

  el.qaModel.textContent = '';
  const names = picked.models.length ? picked.models : [''];
  for (const name of names) {
    el.qaModel.appendChild(h('option', { value: name, text: name || '모델 없음' }));
  }
  const savedModel = localGet(qaModelKey(state.qaProvider)) || '';
  state.qaModel = picked.models.includes(savedModel) ? savedModel : picked.defaultModel;
  el.qaModel.value = state.qaModel;
  if (el.qaModel.value !== state.qaModel) { // default_model이 목록에 없는 방어
    state.qaModel = names[0] || '';
    el.qaModel.value = state.qaModel;
  }
  if (state.qaModel) localSet(qaModelKey(state.qaProvider), state.qaModel);

  // summary는 OpenAI Responses(supports_reasoning_summary) + Thinking 켜짐에서만 의미 있음
  const summaryOn = picked.supportsSummary && state.qaThinking;
  el.qaSummary.disabled = !summaryOn;
  el.qaSummaryField.classList.toggle('unsupported', !summaryOn);
  el.qaSummary.value = summaryOn ? state.qaSummary : 'none';

  el.qaThinking.classList.toggle('active', state.qaThinking);
  el.qaThinking.setAttribute('aria-pressed', state.qaThinking ? 'true' : 'false');
  el.qaThinking.innerHTML = '<i></i>Thinking ' + (state.qaThinking ? 'ON' : 'OFF');
}

export function qaProviderLabel(providerId) {
  const providers = (state.qaCatalog && state.qaCatalog.providers) || [];
  const p = providers.find((x) => x && x.id === providerId);
  return (p && p.label) || providerId || '모델';
}

// 대화 로그에 말풍선 추가. kind: 'user' | 'assistant' (+ ' loading'/' error')
export function appendQaMessage(kind, text) {
  const node = h('div', { class: `qa-msg ${kind}`, text });
  el.qaLog.appendChild(node);
  el.qaLog.scrollTop = el.qaLog.scrollHeight;
  return node;
}

export function setQaMessageError(node, message) {
  node.classList.remove('loading');
  node.classList.add('error');
  node.textContent = message;
  el.qaLog.scrollTop = el.qaLog.scrollHeight;
}

// 응답 도착 — 로딩 말풍선을 답변 + 메타 한 줄(공급자 · 모델 · effort)로 교체,
// reasoning summary가 있으면 접힌 <details>로 덧붙인다.
export function renderQaAnswer(node, d) {
  node.classList.remove('loading');
  node.textContent = String(d.answer || '');
  const meta = [qaProviderLabel(d.provider || state.qaProvider), d.model, d.reasoning_effort]
    .filter((x) => typeof x === 'string' && x)
    .join(' · ');
  if (meta) node.appendChild(h('div', { class: 'qa-meta', text: meta }));
  if (d.reasoning_summary) {
    node.appendChild(h('details', { class: 'qa-summary' },
      h('summary', { text: 'Reasoning summary' }),
      h('p', { text: String(d.reasoning_summary) })));
  }
  el.qaLog.scrollTop = el.qaLog.scrollHeight;
}

export function setQaBusyUI(on) {
  el.qaInput.disabled = on;
  el.qaSend.disabled = on;
  for (const b of el.qaSuggestions) b.disabled = on;
  // 전송 종료로 폼을 되살릴 때도 직전 429 잠금이 남아 있으면 비활성을 유지한다.
  if (!on) applyRetryLock(retryLockRemaining('qaRetryAt'), [el.qaSend, ...el.qaSuggestions]);
}

export async function submitQaQuestion() {
  const id = state.currentJobId;
  if (!id || state.qaBusy) return;
  const waiting = retryLockRemaining('qaRetryAt');
  if (waiting) { // 직전 429의 대기 시간이 남아 있다 — Enter 전송도 여기서 막힌다
    showToast(`요청이 많습니다 — ${waiting}초 후 다시 시도해 주세요.`, 'warn');
    return;
  }
  const question = (el.qaInput.value || '').trim();
  if (!question) return;

  if (!state.qaCatalog) {
    // 카탈로그 없이는 공급자 가드도 요청 구성도 못 한다 — 재시도 유도
    showToast('LLM 공급자 정보를 아직 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.', 'warn');
    initQaTab();
    return;
  }
  const picked = pickQaModels(state.qaCatalog, state.qaProvider);
  if (!picked.available) {
    // Localight식 클라이언트 가드 — 서버 503 대신 설정 방법을 바로 안내
    const hint = qaProviderHint(state.qaProvider);
    showToast(hint, 'warn');
    appendQaMessage('assistant error', hint);
    return;
  }

  const page = clampQaPage(el.qaPage.value, state.qaTotalPages);
  el.qaPage.value = String(page);
  const body = buildQaBody({
    question,
    page,
    provider: state.qaProvider,
    model: state.qaModel,
    effort: state.qaEffort,
    summary: state.qaSummary,
    thinking: state.qaThinking,
  });

  const gen = state.qaGen;
  state.qaBusy = true;
  setQaBusyUI(true);
  appendQaMessage('user', question);
  const loading = appendQaMessage('assistant loading',
    `${qaProviderLabel(state.qaProvider)} — ${page}페이지를 읽고 답변을 생성하는 중…`);
  el.qaInput.value = '';

  let res = null;
  let data = null;
  try {
    res = await fetch(`/api/jobs/${id}/qa`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
    });
    const text = await res.text().catch(() => '');
    data = text ? safeParse(text) : null;
  } catch (_) {
    if (state.currentJobId !== id || state.qaGen !== gen) return; // 잡 전환 가드
    state.qaBusy = false;
    setQaBusyUI(false);
    setQaMessageError(loading, '네트워크 오류로 질문을 보내지 못했습니다. 다시 시도해 주세요.');
    return;
  }
  if (state.currentJobId !== id || state.qaGen !== gen) return; // 잡 전환 가드
  state.qaBusy = false;
  setQaBusyUI(false);

  if (!res.ok) {
    // 409(미완료)/422(페이지·빈 텍스트)/503(LLM 오류)의 한국어 detail을 그대로 노출
    const detail = (data && typeof data.detail === 'string') ? data.detail : null;
    if (res.status === 429) {
      // 질문 레이트리밋/동시 실행 상한 — 남은 대기 시간을 알리고 전송을 잠근다
      const notice = rateLimitNotice(res.headers.get('Retry-After'), detail);
      setQaMessageError(loading, notice.message);
      showToast(notice.message, 'warn');
      lockRetry('qaRetryAt', 'qaRetryTimer', notice.seconds,
        [el.qaSend, ...el.qaSuggestions], () => setQaBusyUI(state.qaBusy));
      return;
    }
    let msg = detail;
    if (!msg) {
      if (res.status === 503) msg = 'LLM 공급자를 사용할 수 없습니다. 서버 설정을 확인해 주세요.';
      else if (res.status === 422) msg = '해당 페이지에서 질문에 사용할 텍스트를 찾지 못했습니다.';
      else if (res.status === 409) msg = '변환이 완료된 작업에서만 질문할 수 있습니다.';
      else msg = `질문 요청에 실패했습니다. (${res.status})`;
    }
    setQaMessageError(loading, msg);
    return;
  }

  renderQaAnswer(loading, data || {});
}

// 질문 탭 활성화 시 리더의 현재 페이지를 페이지 입력에 프리필 — 사용자가 이 잡에서
// 직접 값을 바꾼 적이 없을 때만 (qaPageTouched는 잡 전환 시 리셋).
export function prefillQaPageFromReader(page = state.readerPage, force = false) {
  if (state.qaPageTouched && !force) return;
  el.qaPage.value = String(clampQaPage(page, state.qaTotalPages));
}
