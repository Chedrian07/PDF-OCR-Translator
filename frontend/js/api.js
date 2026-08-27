import { safeParse } from './ui.js';

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
