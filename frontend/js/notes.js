import { READER_NOTES_MAX_JOBS, READER_NOTES_PREFIX, readerNotesKey } from './constants.js';
import { normalizeReaderNotes, readerNotesPruneKeys } from './core.js';
import { localGet, localRemove, safeParse } from './ui.js';

/* ============================ 리더 인용·하이라이트 저장소 ============================
 * 잡별 localStorage 키(uocr-reader-notes-<잡 id>)에 {v, updated, items}로 남긴다.
 * 저장 실패(용량 초과·사생활 모드)는 false로 알려 호출부가 사용자에게 말하게 한다 —
 * "저장했다"는 토스트를 띄워 놓고 실제로는 버리는 일이 없게(frontend-6). 용량 초과는 먼저
 * 가장 오래 손대지 않은 다른 잡의 메모를 지우고 다시 시도한다(storeReaderNotes).
 * ================================================================================ */

export function loadReaderNotes(jobId) {
  if (!jobId) return [];
  const raw = localGet(readerNotesKey(jobId));
  return raw ? normalizeReaderNotes(safeParse(raw)) : [];
}

export function saveReaderNotes(jobId, items) {
  return storeReaderNotes(jobId, items).ok;
}

// 저장 공간 부족(쿼터)인가 — 브라우저마다 이름·코드가 다르다.
function isQuotaError(err) {
  return !!err && (err.name === 'QuotaExceededError' || err.name === 'NS_ERROR_DOM_QUOTA_REACHED'
    || err.code === 22 || err.code === 1014);
}

// 이 잡(keepKey)을 뺀 메모 키 — 가장 오래 손대지 않은 잡부터.
function evictionOrder(keepKey) {
  const entries = [];
  for (let i = 0; i < localStorage.length; i += 1) {
    const key = localStorage.key(i);
    if (!key || key === keepKey || !key.startsWith(READER_NOTES_PREFIX)) continue;
    const data = safeParse(localStorage.getItem(key) || '');
    entries.push({ key, updated: (data && Number(data.updated)) || 0 });
  }
  return entries.sort((a, b) => a.updated - b.updated).map((entry) => entry.key);
}

// 저장하고 {ok, evicted}를 돌려준다. 용량이 모자라면(QuotaExceededError) 이 잡을 뺀 가장 오래
// 손대지 않은 잡의 메모부터 지우고 다시 시도한다 — 보관 상한(잡 50 × 메모 200 × 2000자)은
// 오리진 쿼터(약 5M자)보다 커서, 예전에는 한 번 차면 정리 없이 모든 저장이 계속 실패했다
// (frontend-8). evicted는 지운 다른 잡 수다(호출부가 사용자에게 알린다). 쿼터가 아닌 실패
// (사생활 보호 모드·저장소 차단)는 아무것도 지우지 않고 실패로 끝낸다.
export function storeReaderNotes(jobId, items) {
  if (!jobId) return { ok: false, evicted: 0 };
  const key = readerNotesKey(jobId);
  const notes = normalizeReaderNotes(items);
  if (!notes.length) {
    try {
      localStorage.removeItem(key);
      return { ok: true, evicted: 0 };
    } catch (_) {
      return { ok: false, evicted: 0 };
    }
  }
  const value = JSON.stringify({ v: 1, updated: Date.now(), items: notes });
  let evicted = 0;
  let victims = null;
  for (;;) {
    try {
      localStorage.setItem(key, value);
      break;
    } catch (err) {
      if (!isQuotaError(err)) return { ok: false, evicted };
      try {
        if (!victims) victims = evictionOrder(key);
        const victim = victims.shift();
        if (!victim) return { ok: false, evicted };
        localStorage.removeItem(victim);
        evicted += 1;
      } catch (_) {
        return { ok: false, evicted };
      }
    }
  }
  pruneReaderNotes(key);
  return { ok: true, evicted };
}

// 잡 삭제·소멸(404) 때 그 잡의 메모를 함께 지운다 (이어읽기 위치와 같은 규칙).
export function forgetReaderNotes(jobId) {
  if (jobId) localRemove(readerNotesKey(jobId));
}

// 보관 잡 수 상한 — 서버 TTL로 사라진 잡의 메모가 끝없이 쌓이지 않게, 가장 오래 손대지
// 않은 잡부터 지운다. 키 수가 상한 이하이면 값을 읽지도 않는다.
function pruneReaderNotes(keepKey) {
  try {
    const keys = [];
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (key && key.startsWith(READER_NOTES_PREFIX)) keys.push(key);
    }
    if (keys.length <= READER_NOTES_MAX_JOBS) return;
    const entries = keys.map((key) => {
      const data = safeParse(localStorage.getItem(key) || '');
      return { key, updated: (data && Number(data.updated)) || 0 };
    });
    for (const key of readerNotesPruneKeys(entries, keepKey, READER_NOTES_MAX_JOBS)) {
      localStorage.removeItem(key);
    }
  } catch (_) { /* 저장소 접근 불가 — 정리는 다음 저장 때 다시 시도한다 */ }
}
