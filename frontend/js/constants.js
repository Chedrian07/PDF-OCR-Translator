/* ============================ UI constants ============================ */

export const PHASE_LABELS = { loading: '모델 로딩 대기', render: '렌더링', ocr: 'OCR', merge: '병합' };

export const STATUS_LABELS = {
  queued: '대기중',
  running: '변환중',
  done: '완료',
  error: '오류',
  canceled: '취소됨',
};

export const THEME_KEY = 'uocr-theme';

export const BOX_COLORS = {
  title: '#e5484d',
  text: '#4662d9',
  image: '#2f9e6e',
  table: '#8e4ec6',
  formula: '#d97706',
  equation: '#d97706',
  page_number: '#8b8d98',
  footnote: '#8b8d98',
  header: '#8b8d98',
  footer: '#8b8d98',
};
export const BOX_FALLBACK_COLOR = '#6b7280';

export const ICON = {
  moon: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
  sun: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/></svg>',
  x: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  chip: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
  docLayout: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 9v12"/></svg>',
  read: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 5h6a3 3 0 0 1 3 3v11a2.5 2.5 0 0 0-2.5-2.5H3z"/><path d="M21 5h-6a3 3 0 0 0-3 3v11a2.5 2.5 0 0 1 2.5-2.5H21z"/></svg>',
};

// KaTeX 렌더 옵션 — 앱의 모든 katex.render 호출부(ui.js typesetMath, reader.js
// mathTextNodes)가 이 함수 하나로 옵션을 만든다. 수식 TeX는 업로드 PDF(OCR·텍스트
// 레이어)에서 오는 신뢰할 수 없는 입력이다.
//  · maxSize: \rule{2000em}{2000em}·\hspace{-300em} 같은 사용자 지정 크기를 묶는다
//    (기본값 Infinity — 수만 px 박스나 화면 밖 오프셋이 생긴다).
//  · maxExpand: 매크로 확장 상한(KaTeX 기본 1000)을 명시해 무한 매크로 루프를 막는다.
//  · trust·strict도 명시한다 — 오염된 Object.prototype이 기본값을 덮어 \href 등이
//    살아나는 경로(GHSA-238p-pmpm-9mq7)를 버전과 무관하게 끊는다. strict 'ignore'는
//    유니코드 수식마다 쌓이던 콘솔 경고도 없앤다.
export const KATEX_MAX_SIZE_EM = 10;
export const KATEX_MAX_EXPAND = 1000;
// 조판 결과의 style 길이 상한(em) — 정상 수식은 30줄 aligned도 약 63em이다. 넘으면 원문 TeX로.
export const KATEX_MAX_BOX_EM = 100;
export function katexOptions(displayMode) {
  return {
    displayMode: !!displayMode,
    throwOnError: false,
    maxSize: KATEX_MAX_SIZE_EM,
    maxExpand: KATEX_MAX_EXPAND,
    strict: 'ignore',
    trust: false,
  };
}

export const STREAM_PANE_MAX_NODES = 3000;  // 원시 pane DOM 상한 (장시간 OCR 메모리 방어)
// 라이브 미리보기 백로그(중간에 연 잡의 확정 페이지들)를 동시에 렌더 요청할 개수 —
// RTT마다 한 장씩 보내면 VPN에서 수백 쪽이 수십 초 걸린다. 서버 부담은 작게 유지한다.
export const PREVIEW_RENDER_CONCURRENCY = 3;
export const STREAM_PANE_TRIM_SLACK = 600;  // 한 번에 덜어내는 여유분 — 매 프레임 삭제 방지

export const QA_LS_PROVIDER = 'uocr-qa-provider';
export const QA_LS_EFFORT = 'uocr-qa-effort';
export const QA_LS_THINKING = 'uocr-qa-thinking';
export const QA_LS_SUMMARY = 'uocr-qa-summary';
export const qaModelKey = (providerId) => `uocr-qa-model-${providerId}`;
// index.html의 select 옵션과 동일한 닫힌 집합 — 저장값 복원 시 검증용
export const QA_EFFORTS = ['default', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
export const QA_SUMMARIES = ['none', 'auto', 'concise', 'detailed'];

export const READER_ZOOM_KEY = 'uocr-reader-zoom';
export const READER_ZOOM_MIN = 60;
export const READER_ZOOM_MAX = 220;
export const READER_ZOOM_STEP = 10;
export const READER_SYNC_KEY = 'uocr-reader-sync';
export const READER_HYDRATE_RADIUS = 2;    // 이미지/정렬을 실제로 붙이는 현재 페이지 ± 범위
export const READER_KEEP_RADIUS = 6;       // 이 밖의 페이지 이미지는 src를 떼어 메모리를 돌려준다
export const READER_FOCUS_RATIO = 0.28;    // "지금 읽는 줄"로 보는 뷰포트 높이 비율
export const READER_SYNC_QUIET_MS = 260;   // 프로그램적 스크롤 뒤 반대편 핸들러를 쉬게 하는 시간
export const READER_DEFAULT_RATIO = 1700 / 2200; // 페이지 크기 미상일 때 자리표시자 종횡비
export const READER_ALIGNMENT_COOLDOWN_MS = 30_000; // 제한 재시도 소진 뒤 사용자 재진입 전 휴지기
export const readerPosKey = (jobId) => `uocr-reader-pos-${jobId}`;
// 리더의 인용·하이라이트 — 잡별 localStorage에 남긴다(잡 삭제 시 함께 정리).
// 한 잡 READER_NOTES_MAX개, 보관 잡 READER_NOTES_MAX_JOBS개(가장 오래 손대지 않은 잡부터
// 정리)로 묶어 저장 공간이 끝없이 자라지 않게 한다.
export const READER_NOTES_PREFIX = 'uocr-reader-notes-';
export const readerNotesKey = (jobId) => `${READER_NOTES_PREFIX}${jobId}`;
export const READER_NOTES_MAX = 200;
export const READER_NOTES_MAX_JOBS = 50;
export const READER_NOTE_MAX_CHARS = 2000;
