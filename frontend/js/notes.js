import { READER_NOTES_MAX_JOBS, READER_NOTES_PREFIX, readerNotesKey } from './constants.js';
import { normalizeReaderNotes, readerNotesPruneKeys } from './core.js';
import { localGet, localRemove, safeParse } from './ui.js';

/* ============================ 리더 인용·하이라이트 저장소 ============================
 * 잡별 localStorage 키(uocr-reader-notes-<잡 id>)에 {v, updated, items}로 남긴다.
 * 저장 실패(용량 초과·사생활 모드)는 false로 알려 호출부가 사용자에게 말하게 한다 —
 * "저장했다"는 토스트를 띄워 놓고 실제로는 버리는 일이 없게(frontend-6).
 * ================================================================================ */

export function loadReaderNotes(jobId) {
  if (!jobId) return [];
  const raw = localGet(readerNotesKey(jobId));
  return raw ? normalizeReaderNotes(safeParse(raw)) : [];
}

export function saveReaderNotes(jobId, items) {
  if (!jobId) return false;
  const key = readerNotesKey(jobId);
  const notes = normalizeReaderNotes(items);
  try {
    if (!notes.length) {
      localStorage.removeItem(key);
      return true;
    }
    localStorage.setItem(key, JSON.stringify({ v: 1, updated: Date.now(), items: notes }));
  } catch (_) {
    return false;
  }
  pruneReaderNotes(key);
  return true;
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
