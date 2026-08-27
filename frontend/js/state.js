import { createGroundState } from './core.js';

/* ============================ State ============================ */

export const state = {
  jobs: [],
  currentJobId: null,
  displayedStatus: null,
  displayedPhase: null,   // 마지막 진행 phase — loading→render 전이에서 라이브 뷰 오픈 판정
  queuePos: null, // 열린 잡의 마지막 대기열 위치 — queued가 아니게 되면 해제
  selectedFiles: [], // 다중 선택 지원 — 검증을 통과한 파일들만 담긴다
  uploading: false, // 업로드 루프 재진입 가드 — 진행 중 새 선택이 버튼을 되살리지 않게
  // /api/health 스냅샷 — 필드 부재·미수신 시 undefined (검증·비활성은 fail-open)
  maxUploadMb: undefined,
  translateAvailable: undefined,
  // 엔진 capability 스냅샷 (신규 health 계약 — 구버전 서버는 undefined 유지)
  healthEngine: undefined,        // 현재 활성 엔진 이름 ('unlimited'|'ovisocr2'|…)
  streamGranularity: undefined,   // 'token' | 'page'
  layoutCapability: undefined,    // 'full' | 'figure_only' | 'none'
  modelLoaded: true,              // 모델 로드 여부 — false면 업로드 영역에 로딩 안내
  currentJobEngine: undefined,    // 열린 잡의 engine 메타 (구 잡은 undefined)
  // raw stream pane
  streamPending: '',
  streamPageNo: 0, // markers seen by the raw pane — divider k reads "페이지 k"
  streamAutoScroll: true,
  streamConnected: false,
  streamTrimNote: null,   // 원시 pane 앞부분 잘라내기 안내 줄 (상한 초과 시 1회 삽입)
  rafId: 0,
  // accumulated raw model output (for the preview structurer)
  rawText: '',
  // grounding state machine (pure core) + left-pane view state
  ground: createGroundState(),
  viewPage: 1,          // page shown in the left pane
  followLive: true,
  pageBoxes: new Map(), // pageNo -> [{label,x1,y1,x2,y2}]
  imgFailed: false,
  imgLastTry: 0,
  // live rendered preview (right pane)
  liveGen: 0,
  previewDirty: false,
  previewTimer: 0,
  previewInFlight: false,
  previewFails: 0,
  previewStopped: false,   // 413/연속 실패로 라이브 프리뷰 중단됨
  previewPageCache: [],    // 확정 페이지 렌더 HTML 캐시 (인덱스 = 확정 페이지 순번)
  previewPageNodes: [],    // 확정 페이지별 DOM 노드 (reset으로 되돌릴 때 제거 대상)
  previewTailNodes: [],    // 현재 꼬리 렌더가 소유한 DOM 노드들 (선행 hr 포함)
  previewTailMd: '',       // 마지막으로 렌더된 꼬리 markdown
  previewTailSep: false,   // 마지막 꼬리 렌더의 선행 hr 유무
  previewAutoScroll: true,
  // cancel
  cancelRequestedFor: null,
  // sse / fallback
  es: null,
  sseErrorCount: 0,
  fallbackActive: false,
  fallbackTimer: 0,
  ssePromoteTimer: 0,     // 폴링 강등 후 SSE 재승격 재시도 타이머
  ssePromoteAttempts: 0,  // 재승격 백오프 단계 — 성공(open)·teardown 시 0으로
  // result tab caches
  previewLoaded: false,
  docLayoutLoaded: false,
  markdownLoaded: false,
  // translation (완료된 잡 결과 화면 전용 — 잡 전환 시 초기화)
  currentLang: 'orig',      // 'orig' | 'ko' — 현재 결과 뷰 언어
  translateState: 'none',   // none|running|done|error|canceled
  translateSummary: null,   // 원문 유지/건너뜀 요약 (translateKeptSummary 결과, 없으면 null)
  translateEs: null,        // 번역 진행 EventSource
  translatePollTimer: 0,    // SSE 불가/실패 시 state 폴링 폴백
  translateSseErrors: 0,
  resultUrls: null,         // { markdown, archive, documentHtml, pdf } — 다운로드 빌드용
  currentBaseName: 'document',
  resultHasLayout: undefined, // r.has_layout(+figure_only 판정 반영) — PDF 내보내기 가드
  // 읽기(리더) 탭 (완료 잡 기본 뷰 — 잡 전환 시 resetReaderForJob)
  readerPage: 1,            // 현재 리더 페이지 (1-base)
  readerTotalHint: 0,       // 잡의 총 페이지 힌트 (result.pages → progress.total_pages, 0 = 미상)
  readerPages: { orig: null, ko: null }, // 언어별 페이지 섹션 캐시 [{page, html}]
  readerOutline: { orig: null, ko: null }, // layout title 블록 기반 목차
  readerAlignments: { orig: new Map(), ko: new Map() }, // page -> 정규화된 bbox/본문 연결
  readerAlignmentPending: new Set(), // `${job}:${lang}:${page}` 페이지별 중복 fetch 방지
  readerAlignmentRetryTimers: new Map(), // page key -> transient 오류 제한 재시도 timer
  readerAlignmentRetryCounts: new Map(), // page key -> 현재 창의 transient 재시도 횟수
  readerAlignmentBackoff: new Set(), // 예약된 재시도 전까지 같은 페이지의 우회 fetch 차단
  readerActiveBlock: '',    // 클릭으로 고정한 현재 대응 블록 id
  readerSelection: '',      // 현재 페이지 텍스트에서 선택한 문장
  readerSelectionPage: 1,   // 선택이 실제로 속한 오른쪽 레일 페이지(sync off에서도 정확)
  readerHighlights: [],     // 세션 내 하이라이트 [{page,lang,text}]
  readerCitations: [],      // 세션 내 인용 [{page,lang,text}]
  readerZoom: 100,          // PDF 페이지 이미지 폭 % (60–220, localStorage 'uocr-reader-zoom')
  readerImgTimers: new Map(), // page -> 이미지 재시도 타이머 (페이지별로 독립)
  readerSourceImagePages: new Set(), // primary 실패 뒤 원본 PNG가 성공한 페이지
  // 연속 스크롤 리더 — 전체 페이지를 한 면에 쌓고 스크롤로 읽는다
  readerBands: [],          // [{page, top, height}] — pane.scrollTop 좌표계의 페이지 밴드
  readerLastFocus: null,    // 숨은 reader 탭에서도 복원할 마지막 {page,fraction}
  readerRailPage: 1,        // 개별 스크롤 모드에서 실제 보이는 번역 레일 페이지
  readerRailAnchor: null,   // 개별 모드의 마지막 레일 앵커 {page,offset} — 늦은 레이아웃 변화 복원용
  readerPageEls: new Map(), // page -> .reader-page 섹션
  readerRailEls: new Map(), // page -> .reader-rail-page 섹션
  readerCardEls: new Map(), // block id -> .reader-map-card (원문→번역 동기화용)
  readerRailBands: [],      // [{page,top,height}] — 레일 페이지 위치 (dirty 때만 실측)
  readerRailIndexByPage: new Map(), // page -> [{id,top}] — 역동기화 페이지별 색인(요청 시)
  readerRailIndexDirty: true,
  readerRailTabPage: 0,     // 레일에서 Tab 순환에 열려 있는 페이지 (0 = 없음)
  readerPageSizes: new Map(), // page -> {w,h} (서버 배치 또는 이미지 natural 크기)
  readerPageSizeSeed: null,   // 이 문서에서 처음 확인한 페이지 크기 (미상 페이지 기본값)
  readerStackKey: '',       // 원문 스택 서명 (잡|총페이지)
  readerRailKey: '',        // 레일 스택 서명 (잡|언어|총페이지)
  readerThumbSignature: '', // 썸네일 창 서명 — 바뀔 때만 다시 만든다
  readerOutlineSignature: '',
  readerSync: true,         // 좌우 스크롤 연동 (localStorage 'uocr-reader-sync')
  readerPaneQuietUntil: 0,  // 원문 면의 프로그램적 스크롤 억제 만료 시각
  readerRailQuietUntil: 0,  // 번역 레일의 프로그램적 스크롤 억제 만료 시각
  readerScrollRaf: 0,
  readerRailRaf: 0,
  readerMeasureRaf: 0,
  readerAnchorRaf: 0,       // 줌/패널 변경의 앵커 복원 rAF (잡 전환 때 취소)
  readerMeasureAnchor: null, // 리사이즈 전 의미 위치 {page,fraction}
  readerPosTimer: 0,        // 읽던 페이지 저장 디바운스
  readerUrlTimer: 0,        // 주소창(?page=) 갱신 디바운스
  readerScrollTarget: null, // 진행 중인 페이지 점프 {page, top, until}
  readerJumpTimer: 0,       // 점프 해제 보장 타이머
  readerResizeObserver: null,
  viewerOpen: false,        // 전체 화면 3열 논문 뷰어 표시 상태
  viewerIntent: { open: false, page: 1, lang: 'orig' }, // 주소창에서 복원할 초기 상태
  viewerManifest: null,     // viewer bootstrap/capability snapshot
  viewerNavCollapsed: false,
  viewerRailCollapsed: false,
  viewerReturnFocus: null,  // 닫을 때 원래 진입 버튼으로 포커스 복귀
  qaPageTouched: false,     // 이 잡에서 사용자가 질문 페이지를 직접 수정했는지 (리더 프리필 가드)
  // 질문(Q&A — 완료된 잡 전용 탭. 잡 전환 시 teardownQa가 로그/진행을 초기화)
  qaCatalog: null,          // /api/providers 스냅샷 — 질문 탭 최초 활성화 시 1회 로드
  qaCatalogLoading: false,
  qaProvider: '',           // 선택된 공급자 id (localStorage 복원)
  qaModel: '',              // 공급자별 선택 모델 (localStorage 'uocr-qa-model-<id>')
  qaEffort: 'default',      // reasoning effort ('default' = API 기본값)
  qaThinking: true,
  qaSummary: 'none',        // reasoning summary (openai-responses + thinking에서만 전송)
  qaTotalPages: 0,          // 열린 잡의 총 페이지 수 — 페이지 입력 상한 (0 = 미상)
  qaBusy: false,            // 질문 전송 중 — 폼 비활성
  qaGen: 0,                 // 잡 전환 시 증가 — 늦게 도착한 질문 응답 무시
  openGen: 0,               // 잡 열기 재진입 — 늦게 도착한 스냅샷/중복 구독 무시
  // timers
  jobsTimer: 0,
  healthTimer: 0,
  toastTimer: 0,
  // 429(상한 초과) 뒤 재시도 가능 시각(ms) + 버튼 재활성 타이머 — 연타 방지
  qaRetryAt: 0,
  qaRetryTimer: 0,
  translateRetryAt: 0,
  translateRetryTimer: 0,
  pdfDownloadBusy: false,
};

// 2단계 삭제 확인의 무장(armed) 상태: key → { t: 만료 타이머, btn: 현재 버튼 }.
// key는 목록 항목이면 잡 id, 헤더 버튼이면 'header:<잡 id>' — DOM 버튼이 아니라
// 키로 관리해 5초 주기 재렌더(renderJobList)에도 무장이 유지된다.
export const armTimers = new Map();

/* ============================ DOM refs ============================ */

export const el = {};
export const EL_IDS = {
  healthBadges: 'health-badges',
  themeToggle: 'theme-toggle',
  dropzone: 'dropzone',
  fileInput: 'file-input',
  fileInfo: 'file-info',
  fileName: 'file-name',
  fileSize: 'file-size',
  fileClear: 'file-clear',
  uploadError: 'upload-error',
  uploadModelNotice: 'upload-model-notice',
  uploadBtn: 'upload-btn',
  uploadProgress: 'upload-progress',
  uploadProgressFill: 'upload-progress-fill',
  dpiInput: 'dpi-input',
  jobList: 'job-list',
  jobListEmpty: 'job-list-empty',
  emptyState: 'empty-state',
  jobView: 'job-view',
  jobChip: 'job-status-chip',
  jobFilename: 'job-filename',
  jobTime: 'job-time',
  jobModel: 'job-model',
  streamModeChip: 'stream-mode-chip',
  jobStop: 'job-stop',
  jobStopLabel: 'job-stop-label',
  jobDelete: 'job-delete',
  progressSection: 'progress-section',
  progressPhase: 'progress-phase',
  progressSpinner: 'progress-spinner',
  progressCount: 'progress-count',
  progressTrack: 'progress-track',
  progressFill: 'progress-fill',
  progressChunk: 'progress-chunk',
  liveDetails: 'live-details',
  streamPane: 'stream-pane',
  pageImg: 'page-img',
  boxOverlay: 'box-overlay',
  pageNote: 'page-note',
  pagerPrev: 'pager-prev',
  pagerNext: 'pager-next',
  pagerLabel: 'pager-label',
  followChip: 'follow-chip',
  livePreview: 'live-preview',
  errorSection: 'error-section',
  errorTitle: 'error-title',
  errorMessage: 'error-message',
  errorHint: 'error-hint',
  resultSection: 'result-section',
  dlMd: 'dl-md',
  dlZip: 'dl-zip',
  dlDoc: 'dl-doc',
  dlDocKo: 'dl-doc-ko',
  dlPdf: 'dl-pdf',
  viewerRoot: 'production-viewer',
  viewerOpen: 'viewer-open',
  viewerClose: 'viewer-close',
  viewerFilename: 'viewer-filename',
  viewerToggleNav: 'viewer-toggle-nav',
  viewerToggleRail: 'viewer-toggle-rail',
  viewerNavigation: 'viewer-navigation',
  viewerTranslation: 'viewer-translation',
  viewerThumbnails: 'viewer-thumbnails',
  viewerLangToggle: 'viewer-lang-toggle',
  viewerLangOrig: 'viewer-lang-orig',
  viewerLangKo: 'viewer-lang-ko',
  viewerDlHtml: 'viewer-dl-html',
  viewerDlPdf: 'viewer-dl-pdf',
  readerPrev: 'reader-prev',
  readerNext: 'reader-next',
  readerPageInput: 'reader-page',
  readerTotal: 'reader-total',
  readerZoomOut: 'reader-zoom-out',
  readerZoomIn: 'reader-zoom-in',
  readerFitWidth: 'reader-fit-width',
  readerTranslateBtn: 'reader-translate-btn',
  readerPagePane: 'reader-page-pane',
  readerPageStage: 'reader-page-stage',
  readerSync: 'reader-sync',
  readerProgressFill: 'reader-progress-fill',
  readerVisualLabel: 'reader-visual-label',
  readerRailTitle: 'reader-rail-title',
  readerMapStatus: 'reader-map-status',
  readerSummary: 'reader-summary',
  readerExplain: 'reader-explain',
  readerHighlight: 'reader-highlight',
  readerCite: 'reader-cite',
  readerSelection: 'reader-selection',
  readerOutline: 'reader-outline',
  readerActivity: 'reader-activity',
  readerContent: 'reader-content',
  translateSummary: 'translate-summary',
  viewerTranslateSummary: 'viewer-translate-summary',
  translateBtn: 'translate-btn',
  translateProgress: 'translate-progress',
  translateProgressLabel: 'translate-progress-label',
  translateProgressTrack: 'translate-progress-track',
  translateProgressFill: 'translate-progress-fill',
  translateCancel: 'translate-cancel',
  langToggle: 'lang-toggle',
  langOrig: 'lang-orig',
  langKo: 'lang-ko',
  previewBody: 'preview-body',
  doclayoutBody: 'doclayout-body',
  mdCode: 'md-code',
  copyMd: 'copy-md',
  layoutsGrid: 'layouts-grid',
  pagesGrid: 'pages-grid',
  qaPlaceholder: 'qa-placeholder',
  qaBodyWrap: 'qa-body',
  qaProvider: 'qa-provider',
  qaModel: 'qa-model',
  qaEffort: 'qa-effort',
  qaThinking: 'qa-thinking',
  qaSummary: 'qa-summary',
  qaSummaryField: 'qa-summary-field',
  qaPage: 'qa-page',
  qaLog: 'qa-log',
  qaForm: 'qa-form',
  qaInput: 'qa-input',
  qaSend: 'qa-send',
  toast: 'toast',
};

export function grabEls() {
  for (const key of Object.keys(EL_IDS)) el[key] = document.getElementById(EL_IDS[key]);
  el.tabs = Array.from(document.querySelectorAll('.tab'));
  el.panels = Array.from(document.querySelectorAll('.tab-panel'));
  el.modeRadios = Array.from(document.querySelectorAll('input[name="mode"]'));
  el.qaSuggestions = Array.from(document.querySelectorAll('.qa-suggestion'));
}
