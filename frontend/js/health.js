import { ICON } from './constants.js';
import {
  HEALTH_POLL_FAST_MS, healthCapabilities, healthPollDelay, healthStatus, providerIssue,
} from './core.js';
import { el, state } from './state.js';
import { h } from './ui.js';
import { POLL_TIMEOUT_MS, apiGet } from './api.js';
import { applyTranslateAvailability } from './translate.js';

/* ============================ Health ============================ */

export function shortenGpu(name) {
  return String(name || '')
    .replace(/^NVIDIA\s+GeForce\s+/i, '').replace(/^NVIDIA\s+/i, '')
    .replace(/^Apple\s+/i, '').trim();
}

// health는 정상일 때도 주기적으로 다시 묻는다 — 예전에는 한 번 정상이면 다시 묻지 않아
// 나중에 죽은 sidecar·워커가 배지로 드러나지 않았고, 프리로드 실패는 '로딩 중'으로만
// 끝없이 보였다(frontend-8). 탭이 숨겨져 있으면 멈추고, 다시 보이면 바로 묻는다.
export async function loadHealth() {
  clearTimeout(state.healthTimer);
  state.healthTimer = 0;
  if (state.healthInFlight) return; // 진행 중인 조회가 끝나며 다음 조회를 예약한다
  state.healthInFlight = true;
  let data;
  try {
    data = await apiGet('/api/health', { timeoutMs: POLL_TIMEOUT_MS });
  } catch (_) {
    renderHealthError();
    scheduleHealth(HEALTH_POLL_FAST_MS);
    return;
  } finally {
    state.healthInFlight = false;
  }
  renderHealth(data || {});
  scheduleHealth(healthPollDelay(data));
}

function scheduleHealth(ms) {
  clearTimeout(state.healthTimer);
  state.healthTimer = 0;
  if (typeof document !== 'undefined' && document.hidden) return; // 보이면 다시 묻는다
  state.healthTimer = setTimeout(loadHealth, ms);
}

// 탭 가시성에 맞춰 health 폴링을 멈추고 다시 켠다 (init에서 1회 연결).
export function setupHealthPolling() {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      clearTimeout(state.healthTimer);
      state.healthTimer = 0;
    } else {
      loadHealth();
    }
  });
}

export function renderHealth(d) {
  // 업로드 사전 검증·번역 버튼 가용성이 소비하는 계약 필드 보관.
  // 구버전 서버 응답(필드 부재)은 undefined 유지 — 두 소비처 모두 fail-open.
  state.maxUploadMb = typeof d.max_upload_mb === 'number' ? d.max_upload_mb : undefined;
  state.translateAvailable = typeof d.translate_available === 'boolean' ? d.translate_available : undefined;
  // 모델 로드 여부 — 업로드 영역의 "로딩 중" 안내 표시에 사용 (필드 부재는 로드된 것으로 간주)
  state.modelLoaded = d.model_loaded === false ? false : true;
  const status = healthStatus(d);
  state.modelLoadError = status.loadError;
  applyTranslateAvailability(); // 잡 뷰가 열려 있는 동안의 health 갱신도 버튼에 반영
  applyModelLoadingNotice();    // 모델 로딩 중(또는 로드 실패)이면 업로드 영역에 안내

  // 엔진 capability (신규 계약 — 필드 부재는 undefined = 기존 UI 그대로)
  const hc = healthCapabilities(d);
  state.healthEngine = hc.engine;
  state.streamGranularity = hc.streamGranularity;
  state.layoutCapability = hc.layoutCapability;
  applyStreamModeChip();

  const c = el.healthBadges;
  c.textContent = '';

  const modelId = d.model_id || 'baidu/Unlimited-OCR';
  const modelTitle = modelId +
    (d.model_revision ? ` @ ${String(d.model_revision).slice(0, 8)}` : '') +
    (d.provider ? ` · ${d.provider}` : '');
  c.appendChild(h('span', { class: 'badge badge-model', title: modelTitle },
    h('span', { class: 'badge-ico', html: ICON.chip }),
    h('span', { text: modelId }),
  ));

  const isCuda = d.device === 'cuda';
  const isMetal = d.device === 'metal';
  const devName = isCuda ? 'CUDA' : (isMetal ? 'Metal' : (d.device === 'cpu' ? 'CPU' : String(d.device || '?').toUpperCase()));
  let devText = devName;
  if ((isCuda || isMetal) && d.gpu_name) {
    const short = shortenGpu(d.gpu_name);
    if (short) devText = `${devName} · ${short}`;
  }
  const devTitle = `디바이스: ${devName}` +
    (d.gpu_name ? ` (${d.gpu_name})` : '') +
    ` · dtype: ${d.dtype || '-'} · 네이티브 연산: ${d.native_ops ? 'on' : 'off'}`;
  const devClass = isCuda ? 'is-cuda' : (isMetal ? 'is-metal' : 'is-cpu');
  c.appendChild(h('span', { class: `badge badge-device ${devClass}`, title: devTitle },
    h('span', { class: 'badge-dot' }),
    h('span', { text: devText }),
  ));

  if (d.engine === 'fake') {
    c.appendChild(h('span', {
      class: 'badge badge-warn',
      title: '실제 모델 대신 데모용 가짜 엔진이 실행 중입니다.',
    }, 'FAKE 엔진'));
  }

  // sidecar provider 상태 — 죽어 있으면 명확한 배지 (메인 앱 health는 200이어도)
  const pIssue = providerIssue(d);
  if (pIssue) {
    c.appendChild(h('span', {
      class: 'badge badge-error',
      title: '엔진 서버(sidecar)에 연결할 수 없습니다: ' + pIssue,
    }, '엔진 서버 연결 안 됨'));
  }

  // 페이지 단위 스트리밍 엔진 안내 (Unlimited의 토큰 스트리밍과 구분)
  if (state.streamGranularity === 'page') {
    c.appendChild(h('span', {
      class: 'badge badge-page-stream',
      title: '이 엔진은 페이지가 완료될 때마다 결과를 일괄 표시합니다 (토큰 스트리밍 아님)',
    }, '페이지 단위'));
  }

  if (status.loadError) {
    c.appendChild(h('span', {
      class: 'badge badge-error',
      title: `모델을 불러오지 못했습니다: ${status.loadError}\n작업을 올리면 변환 시작 때 다시 시도하지만 같은 이유로 실패할 수 있습니다.`,
    }, '모델 로드 실패'));
  } else if (status.loading) {
    c.appendChild(h('span', {
      class: 'badge badge-loading',
      title: '모델을 메모리에 로딩하는 중입니다. 첫 작업에서 시간이 걸릴 수 있습니다.',
    }, h('span', { class: 'spinner spinner-xs' }), h('span', { text: '모델 로딩 중…' })));
  }

  // 워커 스레드가 죽으면 잡이 영원히 queued로 남는다 — 운영자가 바로 알아야 한다.
  if (status.workerDead) {
    c.appendChild(h('span', {
      class: 'badge badge-error',
      title: '변환 작업자가 멈췄습니다 — 대기 중인 작업이 처리되지 않습니다. 서버를 재시작해 주세요.',
    }, '작업 처리기 중지됨'));
  }
}

export function renderHealthError() {
  el.healthBadges.textContent = '';
  el.healthBadges.appendChild(h('span', {
    class: 'badge badge-error',
    title: '서버 상태를 확인할 수 없습니다. 자동으로 재시도합니다.',
  }, '서버 연결 실패'));
}

// 페이지 단위 스트리밍 엔진이면 라이브 뷰 요약에 안내 칩 표시
export function applyStreamModeChip() {
  if (!el.streamModeChip) return;
  el.streamModeChip.hidden = state.streamGranularity !== 'page';
}

// 모델 로딩 중이면 업로드 영역에 안내 배너 — 업로드가 "실패"가 아니라 "대기"임을 알린다.
// 프리로드가 실패했으면 같은 자리에 사유를 보인다 — '로딩이 끝나는 대로'는 거짓 안내다.
export function applyModelLoadingNotice() {
  if (!el.uploadModelNotice) return;
  el.uploadModelNotice.hidden = state.modelLoaded !== false;
  const failed = !!state.modelLoadError;
  el.uploadModelNotice.classList.toggle('is-error', failed);
  el.uploadModelNotice.setAttribute('role', failed ? 'alert' : 'status');
  if (el.uploadModelNoticeSpinner) el.uploadModelNoticeSpinner.hidden = failed;
  if (el.uploadModelNoticeText) {
    el.uploadModelNoticeText.textContent = failed
      ? `모델을 불러오지 못했습니다 (${state.modelLoadError}). 업로드하면 변환 시작 때 다시 시도하지만 같은 이유로 실패할 수 있습니다 — 서버 로그를 확인해 주세요.`
      : '모델을 불러오는 중입니다. 지금 업로드하면 로딩이 끝나는 대로 자동으로 변환됩니다.';
  }
}
