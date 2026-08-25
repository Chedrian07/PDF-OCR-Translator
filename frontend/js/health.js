import { ICON } from './constants.js';
import { healthCapabilities, providerIssue } from './core.js';
import { el, state } from './state.js';
import { h } from './ui.js';
import { apiGet } from './api.js';
import { applyTranslateAvailability } from './translate.js';

/* ============================ Health ============================ */

export function shortenGpu(name) {
  return String(name || '')
    .replace(/^NVIDIA\s+GeForce\s+/i, '').replace(/^NVIDIA\s+/i, '')
    .replace(/^Apple\s+/i, '').trim();
}

export async function loadHealth() {
  clearTimeout(state.healthTimer);
  let data;
  try {
    data = await apiGet('/api/health');
  } catch (_) {
    renderHealthError();
    state.healthTimer = setTimeout(loadHealth, 10000);
    return;
  }
  renderHealth(data || {});
  if (data && data.model_loaded === false) {
    state.healthTimer = setTimeout(loadHealth, 10000);
  }
}

export function renderHealth(d) {
  // 업로드 사전 검증·번역 버튼 가용성이 소비하는 계약 필드 보관.
  // 구버전 서버 응답(필드 부재)은 undefined 유지 — 두 소비처 모두 fail-open.
  state.maxUploadMb = typeof d.max_upload_mb === 'number' ? d.max_upload_mb : undefined;
  state.translateAvailable = typeof d.translate_available === 'boolean' ? d.translate_available : undefined;
  // 모델 로드 여부 — 업로드 영역의 "로딩 중" 안내 표시에 사용 (필드 부재는 로드된 것으로 간주)
  state.modelLoaded = d.model_loaded === false ? false : true;
  applyTranslateAvailability(); // 잡 뷰가 열려 있는 동안의 health 갱신도 버튼에 반영
  applyModelLoadingNotice();    // 모델 로딩 중이면 업로드 영역에 안내

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

  if (d.model_loaded === false) {
    c.appendChild(h('span', {
      class: 'badge badge-loading',
      title: '모델을 메모리에 로딩하는 중입니다. 첫 작업에서 시간이 걸릴 수 있습니다.',
    }, h('span', { class: 'spinner spinner-xs' }), h('span', { text: '모델 로딩 중…' })));
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
export function applyModelLoadingNotice() {
  if (!el.uploadModelNotice) return;
  el.uploadModelNotice.hidden = state.modelLoaded !== false;
}
