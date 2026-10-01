import { BUSY_RETRY_MAX, busyRetryDelay } from './core.js';
import { safeParse } from './ui.js';

// 바쁨 재시도의 대기 함수 — 테스트가 실제 시간을 기다리지 않도록 바꿔 끼울 수 있다.
export const busyRetryClock = {
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
};

// 파생 산출물(/layout·/html·/markdown 등) 텍스트 GET. 503(빌드·예열 중)이면
// Retry-After를 지켜 BUSY_RETRY_MAX회까지 다시 묻고, 기다릴 때마다
// onWait(초, 시도 번호, 상한)으로 진행을 알린다. isCurrent()가 거짓이 되면(잡·언어
// 전환) 즉시 포기한다. 반환 {status, text}: status 0 = 네트워크 오류, text는 2xx일
// 때만 문자열이다. 실패의 의미 판정은 호출부가 langFetchVerdict로 한다.
export async function fetchTextWithBusyRetry(url, options = {}) {
  const { accept = '*/*', onWait, isCurrent, maxAttempts = BUSY_RETRY_MAX } = options;
  for (let attempt = 0; ; attempt += 1) {
    let res;
    try {
      res = await fetch(url, { headers: { Accept: accept } });
    } catch (_) {
      return { status: 0, text: null };
    }
    if (res.ok) {
      try {
        return { status: res.status, text: await res.text() };
      } catch (_) {
        return { status: 0, text: null }; // 본문 전송 중 끊김 — 네트워크 오류와 같다
      }
    }
    const wait = busyRetryDelay(res.status, res.headers.get('Retry-After'), attempt, maxAttempts);
    if (!wait || (isCurrent && !isCurrent())) return { status: res.status, text: null };
    if (onWait) onWait(wait, attempt + 1, maxAttempts);
    await busyRetryClock.sleep(wait * 1000);
    if (isCurrent && !isCurrent()) return { status: res.status, text: null };
  }
}

export async function apiGet(path) {
  const res = await fetch(path, { headers: { Accept: 'application/json' } });
  const text = await res.text().catch(() => '');
  const data = text ? safeParse(text) : null;
  if (!res.ok) {
    const msg = (data && typeof data.detail === 'string') ? data.detail : `요청 실패 (${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

export async function apiDelete(path) {
  const res = await fetch(path, { method: 'DELETE' });
  if (!res.ok) {
    const err = new Error(`삭제 실패 (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return true;
}

// 라이브 프리뷰 렌더 요청 상한 — 서버는 2MB 본문을 ~0.2초에 렌더한다. 이보다
// 훨씬 큰 여유를 두되 무한 대기는 막는다: 응답 없이 매달리면 previewInFlight가
// 영원히 참으로 남아 오른쪽 패널이 잡이 끝날 때까지 멈춰 버린다.
const PREVIEW_TIMEOUT_MS = 20000;

// POST 한 번 — 성공 시 {html}, HTTP 실패 시 {status}, 네트워크 오류·시간 초과 시 {status: 0}.
export async function postPreviewRender(id, body) {
  try {
    const res = await fetch(`/api/jobs/${id}/render-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body,
      signal: (typeof AbortSignal !== 'undefined' && AbortSignal.timeout)
        ? AbortSignal.timeout(PREVIEW_TIMEOUT_MS)
        : undefined,
    });
    if (res.ok) return { html: await res.text() };
    return { status: res.status };
  } catch (_) {
    return { status: 0 }; // network error → retried on the next schedule
  }
}

// XHR 업로드 — fetch에는 업로드 진행 이벤트가 없어 진행률 표시용으로만 XHR을
// 쓴다. 응답은 fetch 경로와 같은 의미의 {status, text}로 통일하고, 전송 실패
// (네트워크 오류)만 reject한다. HTTP 오류 상태는 resolve — 호출부가 분기한다.
export function uploadWithProgress(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    if (xhr.upload) {
      xhr.upload.addEventListener('progress', (e) => {
        onProgress(e.lengthComputable && e.total > 0 ? e.loaded / e.total : null);
      });
    }
    xhr.addEventListener('load', () => resolve({ status: xhr.status, text: xhr.responseText || '' }));
    xhr.addEventListener('error', () => reject(new Error('network error')));
    xhr.addEventListener('abort', () => reject(new Error('aborted')));
    xhr.send(form);
  });
}
