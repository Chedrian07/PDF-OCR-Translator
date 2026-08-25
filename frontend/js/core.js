import {
  PHASE_LABELS, STATUS_LABELS, READER_FOCUS_RATIO, READER_KEEP_RADIUS,
} from './constants.js';

/* ============================================================================
 * Pure live-stream core — exported for frontend/tests/, no DOM access.
 * ========================================================================== */

export const PAGE_MARKER = '<PAGE>';
// literals whose partial prefix at a chunk boundary must be held back
export const MARKER_LITERALS = ['<PAGE>', '<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>'];
export const IMAGE_BLOCK = '> 🖼 그림 감지됨';
// noise labels dropped from the reading-view preview
export const DROP_LABELS = new Set(['page_number', 'header', 'footer', 'footnote']);

export function normalizeLabel(label) {
  return String(label || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
}

export function scanQuads(payload) {
  const nums = String(payload).match(/\d+/g);
  if (!nums) return [];
  const quads = [];
  for (let i = 0; i + 3 < nums.length; i += 4) {
    quads.push([Number(nums[i]), Number(nums[i + 1]), Number(nums[i + 2]), Number(nums[i + 3])]);
  }
  return quads;
}

export const clampCoord = (v) => Math.max(0, Math.min(999, Number(v) || 0));

// Index from which `s` may contain an incomplete grounding structure / marker.
// Returns s.length when the whole string is safe to consume. `cap` guards
// against holding back forever on a malformed block that never closes.
export function incompleteTailIndex(s, cap) {
  const n = s.length;
  let cut = n;

  // an opened ref block that has not seen its closing <|/det|> yet
  const lastRef = s.lastIndexOf('<|ref|>');
  if (lastRef !== -1 && s.indexOf('<|/det|>', lastRef) === -1) cut = Math.min(cut, lastRef);

  // an opened det that has not closed yet
  const lastDet = s.lastIndexOf('<|det|>');
  if (lastDet !== -1 && s.indexOf('<|/det|>', lastDet + 7) === -1) cut = Math.min(cut, lastDet);

  // an unterminated special token: "<|" with no "|>" after it
  const lastPipe = s.lastIndexOf('<|');
  if (lastPipe !== -1 && s.indexOf('|>', lastPipe + 2) === -1) cut = Math.min(cut, lastPipe);

  // a partial literal prefix at the very tail (e.g. "<PA", "<|de", "<|/re")
  for (let k = Math.min(7, n); k > 0; k -= 1) {
    const tail = s.slice(n - k);
    let isPrefix = false;
    for (const mk of MARKER_LITERALS) {
      if (mk.length > k && mk.startsWith(tail)) { isPrefix = true; break; }
    }
    if (isPrefix) { cut = Math.min(cut, n - k); break; }
  }

  if (cap && n - cut > cap) return n; // stale/malformed opener: stop holding back
  return cut;
}

// Marker/page state machine + grounding buffer. `page` is the page currently
// being parsed — every box attaches to it.
export function createGroundState() {
  return {
    buf: '',
    page: 1,
    // Job start: page 1 counts as pre-announced, so the very first <PAGE>
    // marker of the stream is consumed as its confirmation (no advance).
    expectAnnounce: true,
    ocrSeen: false,
    markerCount: 0,
    totalPages: 0,
  };
}

// Apply one progress event to the state machine.
// Only phase==="ocr" may drive page tracking: the render phase emits
// current_page=1..N in quick succession while rasterizing (before any token
// exists) and merge walks the pages again — adopting either would pin the
// page at N and pile every box onto the last page.
export function groundAnnounce(g, phase, currentPage, totalPages) {
  const out = { firstOcr: false, pageChanged: false, totalChanged: false };
  const total = Number(totalPages) || 0;
  if (total > g.totalPages) { g.totalPages = total; out.totalChanged = true; }
  if (phase !== 'ocr') return out;

  if (!g.ocrSeen) {
    g.ocrSeen = true;
    out.firstOcr = true;
    if (g.page !== 1) { g.page = 1; out.pageChanged = true; } // stale pre-OCR advancement guard
  }
  // The next <PAGE> marker is the start-of-page confirmation of this
  // announcement — it must not advance the page again.
  g.expectAnnounce = true;
  const cur = Number(currentPage) || 0;
  const target = g.totalPages ? Math.min(cur, g.totalPages) : cur;
  if (target > g.page) { g.page = target; out.pageChanged = true; } // never backwards
  return out;
}

export function groundPush(g, text) {
  if (text) g.buf += text;
}

// Drain the grounding buffer: emit COMPLETE det/ref matches and apply <PAGE>
// markers in positional order, then consume up to the last complete match.
// The remainder is kept only from the first potentially-incomplete structure
// onward (see incompleteTailIndex), so matches split across SSE chunk
// boundaries are parsed exactly once, after they fully assemble.
// Returns events: {type:'page', page} | {type:'boxes', page, label, boxes:[{x1,y1,x2,y2}]}
export function groundDrain(g, final) {
  const out = [];
  const buf = g.buf;
  if (!buf) return out;

  const events = [];
  let m;
  // inline dets: <|det|>label [x1,y1,x2,y2]<|/det|>
  const reDet = /<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*<\|\/det\|>/g;
  while ((m = reDet.exec(buf)) !== null) {
    events.push({
      start: m.index,
      end: reDet.lastIndex,
      label: m[1],
      quads: [[Number(m[2]), Number(m[3]), Number(m[4]), Number(m[5])]],
    });
  }
  // ref blocks: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2],...]<|/det|>
  const reRef = /<\|ref\|>([^<]{1,40})<\|\/ref\|><\|det\|>(\[\[?[\d,\s\[\]]*\]\]?)<\|\/det\|>/g;
  while ((m = reRef.exec(buf)) !== null) {
    events.push({ start: m.index, end: reRef.lastIndex, label: m[1].trim(), quads: scanQuads(m[2]) });
  }
  // page markers
  let idx = -1;
  while ((idx = buf.indexOf(PAGE_MARKER, idx + 1)) !== -1) {
    events.push({ start: idx, end: idx + PAGE_MARKER.length, page: true });
  }

  events.sort((a, b) => a.start - b.start);
  let pos = 0;
  for (const ev of events) {
    if (ev.start < pos) continue;
    if (ev.page) {
      g.markerCount += 1;
      if (g.expectAnnounce) {
        g.expectAnnounce = false; // start-of-page confirmation of the announced page
      } else {
        const next = g.totalPages ? Math.min(g.page + 1, g.totalPages) : g.page + 1;
        if (next > g.page) {
          g.page = next;
          out.push({ type: 'page', page: g.page });
        }
      }
    } else {
      const boxes = [];
      for (const q of ev.quads) {
        const x1 = clampCoord(q[0]), y1 = clampCoord(q[1]), x2 = clampCoord(q[2]), y2 = clampCoord(q[3]);
        if (x2 <= x1 || y2 <= y1) continue; // degenerate box
        boxes.push({ x1, y1, x2, y2 });
      }
      if (boxes.length) out.push({ type: 'boxes', page: g.page, label: ev.label, boxes });
    }
    pos = ev.end;
  }

  if (final) { g.buf = ''; return out; }
  const rest = buf.slice(pos);
  g.buf = rest.slice(incompleteTailIndex(rest, 1200));
  return out;
}

// Build STRUCTURED markdown from the raw stream for the live preview pane.
// The model carries structure only in det labels — a flat cleanup collapses
// everything into run-on paragraphs. Instead each det block becomes its own
// markdown block: title → "## ", image → placeholder blockquote, page
// furniture → dropped, everything else (text / raw <table> html / LaTeX) →
// its own paragraph. <PAGE> → "---" separator. Blank lines between blocks.
export function structurePreview(raw, final) {
  let s = raw;
  if (!final) s = s.slice(0, incompleteTailIndex(s, 2000));
  if (!s) return '';

  // structural tokens, position-ordered
  const toks = [];
  let m;
  const reDet = /<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*<\|\/det\|>/g;
  while ((m = reDet.exec(s)) !== null) toks.push({ start: m.index, end: reDet.lastIndex, label: m[1] });
  const reRef = /<\|ref\|>([^<]{1,40})<\|\/ref\|><\|det\|>\[\[?[\d,\s\[\]]*\]\]?<\|\/det\|>/g;
  while ((m = reRef.exec(s)) !== null) toks.push({ start: m.index, end: reRef.lastIndex, label: m[1].trim(), ref: true });
  let idx = -1;
  while ((idx = s.indexOf(PAGE_MARKER, idx + 1)) !== -1) {
    toks.push({ start: idx, end: idx + PAGE_MARKER.length, page: true });
  }
  toks.sort((a, b) => a.start - b.start);

  const parts = [];
  const pushSep = () => {
    if (parts.length && parts[parts.length - 1] !== '---') parts.push('---'); // no leading/duplicate hr
  };
  const pushBlock = (label, text) => {
    const key = normalizeLabel(label);
    if (DROP_LABELS.has(key)) return;
    if (key === 'image') { parts.push(IMAGE_BLOCK); return; }
    const body = String(text).replace(/<\|[^|>]{0,64}\|>/g, '').trim(); // strip stray specials
    if (!body) return;
    if (key === 'title') parts.push('## ' + body.replace(/\s*\n+\s*/g, ' '));
    else parts.push(body); // text / table(raw html) / equation(LaTeX literal) / unknown
  };

  let pos = 0;
  let currentLabel = null; // det label owning the text that follows it
  for (const t of toks) {
    if (t.start < pos) continue; // overlap safety
    pushBlock(currentLabel, s.slice(pos, t.start));
    if (t.page) {
      pushSep();
      currentLabel = null;
    } else if (normalizeLabel(t.label) === 'image') {
      pushBlock('image', '');
      currentLabel = null;
    } else if (t.ref) {
      currentLabel = null; // non-image ref: grounding only, no reading content
    } else {
      currentLabel = t.label;
    }
    pos = t.end;
  }
  pushBlock(currentLabel, s.slice(pos));

  if (final && parts.length && parts[parts.length - 1] === '---') parts.pop();
  return parts.join('\n\n');
}

// ── 라이브 프리뷰 증분 분할 (순수 — frontend/tests/에서 직접 검증) ──────────
// raw를 <PAGE> 마커 기준으로 "확정 페이지(뒤에 새 페이지가 시작된 세그먼트)"와
// "미확정 꼬리"로 나눈다. 마커가 조각나 도착하면(<PA + GE>) 완성되기 전까지
// 꼬리에 남는다. raw는 잡 안에서 append-only라 확정 세그먼트는 불변 → 캐시 가능.
export function splitPreviewPages(raw) {
  const pages = [];
  let pos = 0;
  let idx;
  while ((idx = raw.indexOf(PAGE_MARKER, pos)) !== -1) {
    pages.push(raw.slice(pos, idx));
    pos = idx + PAGE_MARKER.length;
  }
  return { pages, tail: raw.slice(pos) };
}

// 이번 사이클에 렌더해야 할 조각 계산: 캐시에 없는 확정 페이지들의 markdown과
// 꼬리 markdown. sep은 "앞에 렌더된 내용이 있으면 페이지 경계 hr을 붙여라" —
// 전체 텍스트 structurePreview의 pushSep(선두/중복 hr 억제)과 동치다.
// cachedHtmls: 확정 페이지별 렌더 HTML 캐시 (빈 문자열 = 내용 없는 페이지).
export function planPreviewRender(raw, cachedHtmls, lastTailMd, lastTailSep) {
  const { pages, tail } = splitPreviewPages(raw);
  let hasBefore = cachedHtmls.some((html) => !!html);
  const newPages = [];
  for (let i = cachedHtmls.length; i < pages.length; i += 1) {
    const md = structurePreview(pages[i], true); // 확정 세그먼트는 완결 — 홀드백 불필요
    newPages.push({ idx: i, md, sep: !!md && hasBefore });
    if (md) hasBefore = true;
  }
  const tailMd = structurePreview(tail, false);
  const tailSep = !!tailMd && hasBefore;
  // 꼬리 md가 같아도 sep이 바뀌면(앞에 내용 있는 페이지가 확정) 재렌더 대상
  const tailChanged = tailMd !== lastTailMd || tailSep !== !!lastTailSep;
  return { newPages, tailMd, tailSep, tailChanged };
}

// ── SSE 폴링 강등 → 재승격 백오프 (순수 — frontend/tests/에서 직접 검증) ──────
// 강등 후 attempt번째(0부터) 재시도까지 기다릴 지연: 10초 → 20초 → 30초 상한.
export function ssePromoteDelay(attempt) {
  return Math.min(30000, 10000 * ((Number(attempt) || 0) + 1));
}

// raw pane 디바이더 번호(streamPageNo)를 ground 상태머신의 페이지로 재동기화.
// 디바이더 k는 "페이지 k"이고 마커 k가 페이지 k를 시작하므로, streamPageNo는
// "다음 마커가 시작할 페이지 - 1"이어야 한다: 선언 대기 중(expectAnnounce)이면
// 다음 마커는 g.page의 시작 확인이라 g.page-1, 아니면 g.page+1을 시작하라 g.page.
// 재연결 갭으로 마커가 유실돼도 ground.page는 progress 선언(폴링 포함)으로
// 따라가므로 이 보정으로 이후 디바이더 번호가 복구된다. 절대 뒤로 가지 않는다.
export function syncedStreamPageNo(streamPageNo, g) {
  const target = g.expectAnnounce ? g.page - 1 : g.page;
  return Math.max(Number(streamPageNo) || 0, target);
}

// 원시 토큰 pane에서 이번에 걷어낼 앞쪽 노드 수. 수백 페이지 OCR은 rAF마다
// 텍스트 노드를 하나씩 붙여 pane이 무한히 자란다 — 상한을 넘으면 slack만큼
// 넉넉히 잘라 매 프레임 O(n) 삭제가 반복되지 않게 한다.
// 최소 한 노드는 남긴다(잘라내기 안내 줄 자리).
export function streamPaneTrimCount(childCount, max, slack) {
  const count = Math.max(0, Math.floor(Number(childCount)) || 0);
  const cap = Math.max(1, Math.floor(Number(max)) || 1);
  if (count <= cap) return 0;
  const extra = Math.max(0, Math.floor(Number(slack)) || 0);
  return Math.min(count - 1, count - cap + extra);
}

// replay(재연결) 텍스트가 지금까지 받은 원문을 그대로 이어받았는지. 참이면
// 확정 페이지의 렌더 캐시/DOM이 그대로 유효하다 — 버리면 재연결마다 확정
// 페이지 전부를 /render-preview로 다시 POST한다(수백 요청).
export function replayExtendsRaw(prevRaw, replayText) {
  return !!prevRaw && typeof replayText === 'string' && replayText.startsWith(prevRaw);
}

/* ── 질문(Q&A) 순수 코어 (DOM 없음 — frontend/tests/에서 직접 검증) ──────────
 * 완료된 잡의 페이지 텍스트에 대해 선택한 LLM 공급자에게 질문하는 탭의
 * 요청 본문/공급자 카탈로그/안내 문구 로직. POST /api/jobs/{id}/qa 계약:
 *   {question, page, provider, model, reasoning_effort, reasoning_summary, thinking}
 */

// POST /qa 요청 본문 빌드. reasoning_summary는 OpenAI Responses 형식 + Thinking
// 켜짐일 때만 선택값을 보내고, 그 외에는 항상 'none'(다른 공급자는 미지원).
// effort는 그대로 전달('default' = API 기본값), 빈 모델은 null(서버 기본 모델).
export function buildQaBody(s) {
  const provider = (s && s.provider) || '';
  const thinking = !!(s && s.thinking);
  return {
    question: String((s && s.question) || '').trim(),
    page: Number(s && s.page) || 1,
    provider,
    model: (s && s.model) ? s.model : null,
    reasoning_effort: (s && s.effort) || 'default',
    reasoning_summary: (provider === 'openai-responses' && thinking)
      ? ((s && s.summary) || 'none')
      : 'none',
    thinking,
  };
}

// 공급자 미설정(available=false) 시 사용자 안내 문구 (클라이언트 측 가드용).
export function qaProviderHint(provider) {
  const id = String(provider || '');
  if (id.startsWith('openai-')) {
    // 질문(Q&A) 전용 키는 번역용 OPENAI_API_KEY와 분리돼 있다 — 번역 키는 임의
    // 게이트웨이를 가리킬 수 있어 api.openai.com으로 재사용하면 자격증명이 샌다.
    return 'LLM_OPENAI_API_KEY가 설정되지 않았습니다. 서버 환경(.env)에 키를 추가한 뒤 다시 시도해 주세요.';
  }
  if (id === 'ollama') {
    return '로컬 Ollama를 사용할 수 없습니다. ollama serve 실행 후 ollama pull qwen3:8b 로 모델을 준비해 주세요.';
  }
  return '선택한 LLM 공급자를 사용할 수 없습니다. 서버 설정을 확인해 주세요.';
}

/* ── 429 (남용 방어 상한) 안내 ────────────────────────────────────────────
 * 서버는 QA·번역의 잡/IP 레이트리밋과 동시 실행 상한을 초과하면
 * 429 + Retry-After 헤더 + {"detail": "…"} 본문으로 거절한다(backend/app/api.py).
 * generic 실패 토스트로는 "왜 막혔는지·언제 되는지"를 알 수 없으므로 전용 문구로 바꾼다.
 * ==================================================================== */

export const RETRY_AFTER_FALLBACK_S = 30; // 헤더가 없거나 파싱 불가일 때
export const RETRY_AFTER_MAX_S = 300;     // 상한 — 거대/악의적 값으로 UI가 잠기지 않게

// Retry-After 파싱: RFC 9110에 따라 delta-seconds(정수) 또는 HTTP-date 둘 다 온다.
// 파싱 실패는 기본값으로, 음수·과거 시각은 1초로, 거대값은 상한으로 강등한다.
export function parseRetryAfter(value, nowMs = Date.now()) {
  if (value === null || value === undefined) return RETRY_AFTER_FALLBACK_S;
  const raw = String(value).trim();
  if (!raw) return RETRY_AFTER_FALLBACK_S;
  let secs;
  if (/^[+-]?\d+$/.test(raw)) {
    secs = Number(raw);
  } else {
    const at = Date.parse(raw);
    if (!Number.isFinite(at)) return RETRY_AFTER_FALLBACK_S;
    const base = Number(nowMs);
    secs = (at - (Number.isFinite(base) ? base : Date.now())) / 1000;
  }
  if (!Number.isFinite(secs)) return RETRY_AFTER_FALLBACK_S;
  if (secs < 1) return 1; // 0·음수(이미 지난 시각) — 연타 방지로 최소 1초는 기다린다
  return Math.min(Math.ceil(secs), RETRY_AFTER_MAX_S);
}

// 429 응답(헤더 + detail)에서 대기 초와 한국어 안내 문구를 만든다.
// 서버 detail의 앞부분("요청이 너무 잦습니다" 등)만 살리고 뒤의 모호한
// "잠시 후" 절은 구체적인 초로 바꾼다.
export function rateLimitNotice(retryAfter, detail, nowMs = Date.now()) {
  const seconds = parseRetryAfter(retryAfter, nowMs);
  // detail은 문자열일 때만 신뢰한다 (FastAPI가 객체 detail을 줄 수도 있다)
  const head = (typeof detail === 'string' ? detail : '').split('—')[0].trim();
  return {
    seconds,
    message: `${head || '요청이 많습니다'} — ${seconds}초 후 다시 시도해 주세요.`,
  };
}

// /api/providers 카탈로그에서 공급자 하나의 모델 목록/기본 모델/가용성 추출.
// 모델 목록이 비어 있으면 default_model 하나로 폴백(선택지 유지), 카탈로그에
// 없는 공급자·비정상 응답은 전부 빈 값/false — 소비처는 가드 문구로 처리.
export function pickQaModels(catalog, providerId) {
  const providers = (catalog && Array.isArray(catalog.providers)) ? catalog.providers : [];
  const p = providers.find((x) => x && x.id === providerId) || null;
  const listed = (p && Array.isArray(p.models))
    ? p.models.filter((m) => typeof m === 'string' && m)
    : [];
  const models = listed.length ? listed.slice()
    : ((p && p.default_model) ? [p.default_model] : []);
  return {
    models,
    defaultModel: (p && p.default_model) || models[0] || '',
    available: !!(p && p.available),
    supportsSummary: !!(p && p.supports_reasoning_summary),
  };
}

// 질문 대상 페이지 번호 보정: 1 이상, 총 페이지 수를 알면 그 이하로.
// totalPages 미상(0·비수치)이면 하한만 적용 — 서버 422가 최후 방어.
export function clampQaPage(value, totalPages) {
  let v = Math.floor(Number(value));
  if (!Number.isFinite(v) || v < 1) v = 1;
  const total = Math.floor(Number(totalPages));
  if (Number.isFinite(total) && total >= 1 && v > total) v = total;
  return v;
}

// 전체 화면 논문 뷰어의 공유 가능한 query 계약. 잡 식별자는 기존 hash가 담당하고
// query는 뷰어 표시 상태만 담당하므로 서로 독립적으로 갱신할 수 있다.
export function parseViewerSearch(search) {
  const params = new URLSearchParams(String(search || '').replace(/^\?/, ''));
  const rawPage = Math.floor(Number(params.get('page')));
  return {
    open: params.get('viewer') === '1',
    page: Number.isFinite(rawPage) && rawPage >= 1 ? rawPage : 1,
    lang: params.get('lang') === 'ko' ? 'ko' : 'orig',
  };
}

export function buildViewerSearch(search, viewer) {
  const params = new URLSearchParams(String(search || '').replace(/^\?/, ''));
  params.delete('viewer');
  params.delete('page');
  params.delete('lang');
  if (viewer && viewer.open) {
    const page = Math.max(1, Math.floor(Number(viewer.page)) || 1);
    params.set('viewer', '1');
    params.set('page', String(page));
    params.set('lang', viewer.lang === 'ko' ? 'ko' : 'orig');
  }
  const result = params.toString();
  return result ? `?${result}` : '';
}

// 긴 논문에서도 모든 페이지 이미지를 한꺼번에 다운로드하지 않는다. 현재 페이지
// 주변과 양 끝만 미리보기로 유지해 탐색성은 보존하고 네트워크/메모리는 제한한다.
export function viewerThumbnailWindow(total, current, radius = 2) {
  const count = Math.max(0, Math.floor(Number(total)) || 0);
  if (!count) return [];
  const active = Math.min(count, Math.max(1, Math.floor(Number(current)) || 1));
  const span = Math.max(0, Math.floor(Number(radius)) || 0);
  const pages = new Set([1, count]);
  for (let page = active - span; page <= active + span; page += 1) {
    if (page >= 1 && page <= count) pages.add(page);
  }
  return [...pages].sort((a, b) => a - b);
}

/* ── 읽기(리더) 탭 + PDF 내보내기 순수 코어 (DOM 없음 — frontend/tests/에서 직접 검증) ──
 * 리더 탭은 /html 응답을 페이지 섹션 단위로 나눠 전 페이지 연속 rail로 보여준다.
 * PDF 내보내기는 GET /api/jobs/{id}/pdf?lang=ko&view=dual 계약(200 pdf | 400 | 404 | 409)을
 * 전제로, 노출 여부만 클라이언트에서 판정한다.
 */

// /html 응답을 서버의 안정 래퍼 <section class="doc-page" data-page="N">…</section>
// 단위로 분해한다 (비중첩 — 정규식 분해로 충분). 속성 순서·따옴표·추가 클래스
// 변형을 허용하고, 섹션이 하나도 없으면(구버전 렌더) 전체 HTML 한 장 폴백.
// 반환: [{page:number, html:string(섹션 전체 마크업)}] — 래퍼를 포함해 돌려줘
// .doc-page 스타일 훅(페이지 번호 꼬리표 등)이 리더에서도 그대로 살아 있다.
export function extractDocPages(html) {
  const s = String(html == null ? '' : html);
  const out = [];
  const re = /<section\b([^>]*)>([\s\S]*?)<\/section\s*>/gi;
  let m;
  while ((m = re.exec(s)) !== null) {
    const attrs = m[1] || '';
    const cm = attrs.match(/\bclass\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))/i);
    const cls = cm ? (cm[1] != null ? cm[1] : (cm[2] != null ? cm[2] : (cm[3] || ''))) : '';
    if ((' ' + cls + ' ').indexOf(' doc-page ') === -1) continue; // 다른 섹션은 무시
    const pm = attrs.match(/\bdata-page\s*=\s*(?:"(\d+)"|'(\d+)'|(\d+))/i);
    const page = pm ? Number(pm[1] || pm[2] || pm[3]) : out.length + 1; // 번호 부재 → 등장 순번
    out.push({ page, html: m[0] });
  }
  if (!out.length) return [{ page: 1, html: s }];
  return out;
}

// 리더 페이지 번호 보정 — clampQaPage와 동일 규칙(1 이상, total을 알면 그 이하,
// total 미상(0·비수치)이면 하한만). 별도 이름으로 노출해 소비처를 명확히 한다.
export function clampReaderPage(n, total) {
  return clampQaPage(n, total);
}

// 리더 왼쪽은 번역 언어와 무관하게 원문 PDF를 좌표 기준면으로 고정한다.
// 한국어는 오른쪽 alignment rail에 표시하므로 bbox가 가리키는 대상이 바뀌지 않는다.
export function readerImageUrl(jobId, page, _lang = 'orig') {
  let n = Math.floor(Number(page));
  if (!Number.isFinite(n) || n < 1) n = 1;
  return `/api/jobs/${jobId}/page/${n}`;
}

// 라이브 뷰는 렌더 단계 산출물(pages/)을 직접 본다 — /page/{n}은 layout.json에
// 병합 완료된 페이지만 제공하므로, 항상 'OCR 진행 중인 페이지'를 따라가는
// follow-live가 그 URL을 쓰면 영구 404가 된다. 렌더 단계는 첫 OCR 선언 전에
// 전 페이지 PNG를 만든다. updateLeftPane은 ground.ocrSeen 뒤에만 이 URL을 붙여
// render 시작 직후의 아직 없는 파일을 요청하지 않는다.
// (readerImageUrl은 완료 잡 리더/뷰어 전용 — 페이지 보정 규칙은 동일하게 유지)
export function livePageImageUrl(jobId, page) {
  let n = Math.floor(Number(page));
  if (!Number.isFinite(n) || n < 1) n = 1;
  return `/api/jobs/${jobId}/files/pages/page_${String(n).padStart(4, '0')}.png`;
}

// 서버 alignment 응답의 bbox(0..bbox_space)를 퍼센트 좌표로 변환한다.
// 잘못된 값은 null로 제거해 DOM style 주입과 화면 밖 오버레이를 함께 차단한다.
export function alignmentRect(bbox, bboxSpace = 1000) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  const space = Number(bboxSpace);
  if (!Number.isFinite(space) || space <= 0) return null;
  const values = bbox.map(Number);
  if (!values.every(Number.isFinite)) return null;
  const [rawX1, rawY1, rawX2, rawY2] = values;
  const clamp = (v) => Math.min(space, Math.max(0, v));
  const x1 = clamp(rawX1);
  const y1 = clamp(rawY1);
  const x2 = clamp(rawX2);
  const y2 = clamp(rawY2);
  if (x2 <= x1 || y2 <= y1) return null;
  const pct = (v) => Math.round(((v / space) * 100) * 1_000_000) / 1_000_000;
  return {
    left: pct(x1),
    top: pct(y1),
    width: pct(x2 - x1),
    height: pct(y2 - y1),
  };
}

export function normalizeAlignmentPayload(payload) {
  const data = payload && typeof payload === 'object' ? payload : {};
  const space = Number(data.bbox_space) || 1000;
  const blocks = [];
  for (const raw of Array.isArray(data.blocks) ? data.blocks : []) {
    if (!raw || typeof raw !== 'object') continue;
    const rect = alignmentRect(raw.bbox, space);
    const id = String(raw.id || '');
    const source = String(raw.source || '').trim();
    const target = String(raw.target || '').trim();
    if (!id || !rect || (!source && !target)) continue;
    blocks.push({
      id,
      index: Number.isFinite(Number(raw.index)) ? Number(raw.index) : blocks.length,
      type: String(raw.type || 'text'),
      source,
      target: target || source,
      translated: raw.translated === true,
      rect,
    });
  }
  return {
    page: Math.max(1, Math.floor(Number(data.page) || 1)),
    lang: data.lang === 'ko' ? 'ko' : 'orig',
    blocks,
  };
}

/* ── 연속 스크롤 리더 순수 코어 (DOM 없음 — frontend/tests/에서 직접 검증) ──
 * 리더는 모든 페이지를 세로로 이어 붙인 하나의 스크롤 면이다. 아래 함수들은
 * "어느 페이지를 보고 있는가 / 어디까지 이미지를 붙일 것인가 / 어떤 좌표 블록이
 * 화면 중앙인가"를 순수 계산으로 분리해, 스크롤 핸들러가 DOM 측정 결과만
 * 넘겨받아 결정할 수 있게 한다.
 */

// 페이지 높이 배열 → 누적 밴드 [{page, top, height}]. gap은 페이지 사이 여백(px).
export function readerPageBands(heights, gap = 0) {
  const g = Number.isFinite(Number(gap)) ? Math.max(0, Number(gap)) : 0;
  const bands = [];
  let top = 0;
  for (let i = 0; i < (Array.isArray(heights) ? heights.length : 0); i += 1) {
    const raw = Number(heights[i]);
    const height = Number.isFinite(raw) && raw > 0 ? raw : 0;
    bands.push({ page: i + 1, top, height });
    top += height + g;
  }
  return bands;
}

// 스크롤 오프셋이 놓인 페이지와 그 페이지 안에서의 진행률(0~1).
// 페이지 사이 여백에 걸리면 직전 페이지의 끝(1)으로 본다 — 경계에서 번호가
// 앞뒤로 튀지 않게 하는 쪽이 읽는 사람 기준에 맞다.
export function readerFocusAt(bands, offset) {
  if (!Array.isArray(bands) || !bands.length) return { page: 1, fraction: 0 };
  const y = Number.isFinite(Number(offset)) ? Number(offset) : 0;
  if (y <= bands[0].top) return { page: bands[0].page, fraction: 0 };
  for (let i = 0; i < bands.length; i += 1) {
    const band = bands[i];
    if (y < band.top + band.height) {
      const fraction = band.height > 0 ? (y - band.top) / band.height : 0;
      return { page: band.page, fraction: Math.min(1, Math.max(0, fraction)) };
    }
    const next = bands[i + 1];
    if (next && y < next.top) return { page: band.page, fraction: 1 }; // 페이지 사이 여백
  }
  return { page: bands[bands.length - 1].page, fraction: 1 };
}

// 실제 이미지를 붙여 둘 페이지 구간 (현재 ± radius, 1..total로 클램프).
// 구간 밖은 종횡비만 잡아 둔 자리표시자로 남겨 메모리/네트워크를 제한한다.
export function readerHydrationWindow(total, current, radius = 2) {
  const count = Math.max(0, Math.floor(Number(total)) || 0);
  if (!count) return { start: 1, end: 0 };
  const active = Math.min(count, Math.max(1, Math.floor(Number(current)) || 1));
  const span = Math.max(0, Math.floor(Number(radius)) || 0);
  return {
    start: Math.max(1, active - span),
    end: Math.min(count, active + span),
  };
}

// 아직 없는 정렬(alignment) 페이지를 최소 횟수의 배치 GET으로 덮는 계획.
// loaded는 이미 캐시된 페이지 번호 집합(Set 또는 배열). 서버 상한은 16.
export function alignmentBatchPlan(total, current, radius, loaded, maxBatch = 16) {
  const { start, end } = readerHydrationWindow(total, current, radius);
  if (end < start) return [];
  const have = loaded instanceof Set ? loaded : new Set(Array.isArray(loaded) ? loaded : []);
  const cap = Math.min(16, Math.max(1, Math.floor(Number(maxBatch)) || 1));
  const plan = [];
  let runStart = 0;
  let runEnd = 0;
  const flush = () => {
    if (!runStart) return;
    for (let s = runStart; s <= runEnd; s += cap) {
      plan.push({ start: s, limit: Math.min(cap, runEnd - s + 1) });
    }
    runStart = 0;
    runEnd = 0;
  };
  for (let page = start; page <= end; page += 1) {
    if (have.has(page)) { flush(); continue; }
    if (!runStart) runStart = page;
    runEnd = page;
  }
  flush();
  return plan;
}

// 정렬 GET 실패가 "이 잡에서는 영영 안 되는" 실패인지 판정한다.
// 서버는 레이아웃 대응이 깨진 잡에 409(블록/페이지/좌표 불일치), 없는 페이지에
// 404, 잘못된 페이지 번호에 422를 준다 — 전부 재시도해도 결과가 같다. 이런 상태를
// transient로 오해하면 백오프 재시도 + '원문 좌표 연결 중…' 안내가 반복된다.
// 408/429는 다시 해볼 값이 있고, 5xx·네트워크 오류(0)도 일시 실패로 본다.
export function alignmentFailureIsPermanent(status) {
  const code = Math.floor(Number(status)) || 0;
  if (code === 408 || code === 429) return false;
  return code >= 400 && code < 500;
}

// 페이지 안 진행률(0~1)에 대응하는 좌표 블록 id — 그 지점을 지나온 마지막 블록.
export function blockAtFraction(blocks, fraction) {
  if (!Array.isArray(blocks) || !blocks.length) return '';
  const pct = Math.min(100, Math.max(0, (Number(fraction) || 0) * 100));
  let best = null;
  let bestTop = -Infinity;
  let first = null;
  let firstTop = Infinity;
  for (const block of blocks) {
    const rect = block && block.rect;
    const top = rect && Number(rect.top);
    if (!Number.isFinite(top)) continue;
    if (top < firstTop) { first = block; firstTop = top; }
    if (top <= pct && top >= bestTop) { best = block; bestTop = top; }
  }
  // 레이아웃 블록은 읽기 순서이며 y좌표 정렬을 보장하지 않는다(다단 문서·세로
  // 머리말). 따라서 첫 큰 y를 만났다고 중단하지 않고, 현재 눈높이를 지나온 블록
  // 중 가장 가까운 것을 고른다. 문서 맨 위에서는 y가 가장 작은 블록으로 폴백.
  return ((best || first) && (best || first).id) || '';
}

// OCR 본문의 수식 구간을 분해한다. 서버 text_with_math_html과 같은 계약 —
// `\( … \)`는 인라인, `\[ … \]`는 디스플레이. 그 외 문자는 그대로 텍스트다.
// (레이아웃 블록 content는 마크다운이 아니라 원문 표기 그대로 온다.)
export function splitInlineMath(text) {
  const s = String(text == null ? '' : text);
  const out = [];
  const re = /\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g;
  let pos = 0;
  let m;
  while ((m = re.exec(s)) !== null) {
    if (m.index > pos) out.push({ type: 'text', value: s.slice(pos, m.index) });
    const display = m[1] != null;
    const tex = String(display ? m[1] : m[2]).trim();
    if (tex) out.push({ type: 'math', value: tex, display });
    pos = m.index + m[0].length;
  }
  if (pos < s.length) out.push({ type: 'text', value: s.slice(pos) });
  if (!out.length) out.push({ type: 'text', value: '' });
  return out;
}

// 원문·한국어 대조 PDF 내보내기 버튼 노출 판정.
// 보임 ⇔ 잡 done ∧ 번역 done ∧ hasLayout !== false (필드 부재는 fail-open —
// 서버 409가 최후 방어). 번역이 없어 숨겨질 때는 그대로 숨김 유지 — 다음 행동
// 안내는 다운로드 행의 번역 UI가 이미 담당한다.
export function pdfExportState(jobStatus, koStatus, hasLayout) {
  if (jobStatus !== 'done') return { visible: false, reason: 'job-not-done' };
  if (koStatus !== 'done') return { visible: false, reason: 'translation-missing' };
  if (hasLayout === false) return { visible: false, reason: 'no-layout' };
  return { visible: true, reason: '' };
}

// HTML(한국어)는 좌표 레이아웃이 없어도 번역 완료만 되면 내보낼 수 있다.
// PDF와 조건을 분리해 figure_only 엔진에서도 명시적인 한국어 HTML을 제공한다.
export function translatedHtmlExportState(jobStatus, koStatus) {
  if (jobStatus !== 'done') return { visible: false, reason: 'job-not-done' };
  if (koStatus !== 'done') return { visible: false, reason: 'translation-missing' };
  return { visible: true, reason: '' };
}

// PDF 응답의 숫자형 생성 리포트 → 사용자에게 보여 줄 짧은 무손실 요약.
// 서버는 비ASCII 경고 본문을 헤더에 싣지 않고 개수만 보내므로 문서 내용이
// 프록시 로그에 새지 않는다.
export function pdfReportMessage(report) {
  const n = (v) => {
    const x = Math.floor(Number(v));
    return Number.isFinite(x) && x >= 0 ? x : 0;
  };
  const r = report || {};
  const replaced = n(r.replaced);
  const kept = n(r.kept);
  const relocated = n(r.relocated);
  const tableCells = n(r.tableCells);
  const warnings = n(r.warnings);
  const specialistKept = n(r.specialistKept);
  const details = [`번역 ${replaced}개 블록`];
  if (tableCells) details.push(`표 ${tableCells}개 셀`);
  if (relocated) details.push(`충돌 없이 ${relocated}개 재배치`);
  if (kept) details.push(`원문 ${kept}개 보존`);
  if (specialistKept) details.push(`전문 조판 ${specialistKept}개 원형 보존`);
  if (warnings) details.push(`주의 ${warnings}건`);
  return `PDF 생성 완료: ${details.join(' · ')}`;
}

/* ── 번역 결과: "왜 이 문단이 원문 그대로인가" 요약 ────────────────────────
 * 서버는 translations/{lang}/report.json에 사유별 집계를 남기고, GET
 * /jobs/{id}/translate/state 가 skip_reasons·kept_reasons·reference_rule을 병합해
 * 준다(backend/app/api.py: _TRANSLATE_REASON_KEYS). SSE done 이벤트에는 counts
 * (total·translated·cached·skipped·kept_original)가 실린다. 둘 중 무엇이 오든
 * 같은 요약으로 접는다 — 사용자가 "번역이 안 된 문단"의 개수와 이유를 알 수 있게.
 */
export const TRANSLATE_KEPT_LABELS = {
  'gate-rejected': '번역 품질 기준 미달',
  'placeholder-mismatch': '수식·태그 자리표시자 불일치',
  'empty-output': '빈 응답',
  'api-rejected': 'API 거절',
  'degenerate-output': '같은 출력 반복(축퇴)',
  unknown: '원인 미상',
};
export const TRANSLATE_SKIP_LABELS = {
  references: '참고문헌',
  'already-korean': '이미 한국어',
  'non-linguistic': '수식·기호만',
  identifier: '식별자·코드',
};

// {사유: 개수} → [[사유, 개수]] 내림차순. 비정상 값·0은 버린다.
export function reasonPairs(raw) {
  if (!raw || typeof raw !== 'object') return [];
  const pairs = [];
  for (const key of Object.keys(raw)) {
    const n = Math.floor(Number(raw[key]));
    if (Number.isFinite(n) && n > 0) pairs.push([String(key), n]);
  }
  pairs.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return pairs;
}

export function reasonText(pairs, labels) {
  return pairs.map(([key, n]) => `${labels[key] || key} ${n}`).join(' · ');
}

// translate/state(또는 done 이벤트) 페이로드 → 표시용 요약.
// 근거가 하나도 없으면(구버전 잡·리포트 없음) null — 호출자는 칩을 숨긴다.
export function translateKeptSummary(data) {
  const d = data && typeof data === 'object' ? data : null;
  if (!d) return null;
  const counts = d.counts && typeof d.counts === 'object' ? d.counts : null;
  const keptPairs = reasonPairs(d.kept_reasons);
  const skipPairs = reasonPairs(d.skip_reasons);
  const hasReasons = !!(d.kept_reasons && typeof d.kept_reasons === 'object')
    || !!(d.skip_reasons && typeof d.skip_reasons === 'object');
  if (!counts && !hasReasons) return null;
  const num = (v) => {
    const n = Math.floor(Number(v));
    return Number.isFinite(n) && n > 0 ? n : 0;
  };
  const sum = (pairs) => pairs.reduce((acc, [, n]) => acc + n, 0);
  // 사유별 합과 총계 중 큰 값 — 한쪽만 오는 페이로드에서도 개수를 잃지 않는다.
  const kept = Math.max(num(counts && counts.kept_original), sum(keptPairs));
  const skipped = Math.max(num(counts && counts.skipped), sum(skipPairs));
  const total = num(counts && counts.total) || num(d.total);

  let text;
  if (kept && skipped) text = `원문 유지 ${kept} · 건너뜀 ${skipped}`;
  else if (kept) text = `원문 유지 ${kept}`;
  else if (skipped) text = `건너뜀 ${skipped}`;
  else text = '전부 번역됨';

  const lines = [];
  if (kept) {
    const why = reasonText(keptPairs, TRANSLATE_KEPT_LABELS);
    lines.push(`번역이 실패해 원문 그대로 남은 문단 ${kept}개${why ? ` — ${why}` : ''}`);
  }
  if (skipped) {
    const why = reasonText(skipPairs, TRANSLATE_SKIP_LABELS);
    lines.push(`의도적으로 번역하지 않은 문단 ${skipped}개${why ? ` — ${why}` : ''}`);
  }
  if (!kept && !skipped) lines.push('모든 문단이 한국어로 교체되었습니다.');
  if (total) lines.push(`총 ${total}개 문단`);
  return { kept, skipped, total, text, detail: lines.join('\n'), tone: kept ? 'warn' : '' };
}

// 진행 페이로드 → 진행바 좌측 문구 (순수 — frontend/tests/에서 검증).
// note(모델 로딩 대기 등)가 있으면 최우선, 없으면 queued는 대기열 문구, 그 외 phase 라벨.
export function progressPhaseText(p, status, queuePosLabel) {
  if (p && typeof p.note === 'string' && p.note) return p.note;
  if (status === 'queued') return queuePosLabel || '대기중';
  return (p && PHASE_LABELS[p.phase]) || '처리 중';
}

// 잡 상태 라벨 (순수 — frontend/tests/에서 직접 검증). queued 잡에 선택 필드
// queue_position(1-base, 큐 앞의 queued 잡 수 + 1)이 있으면 '대기중 · N번째'.
// 필드 부재(구버전 서버·SSE 스냅샷)·비정상 값은 기존 라벨 그대로 — 안전 폴백.
export function statusLabel(job) {
  const status = (job && job.status) || 'queued';
  const base = STATUS_LABELS[status] || status;
  if (status === 'queued') {
    const pos = job && job.queue_position;
    if (Number.isInteger(pos) && pos >= 1) return `${base} · ${pos}번째`;
  }
  return base;
}

export function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  try {
    return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false });
  } catch (_) {
    const p = (n) => String(n).padStart(2, '0');
    return `${p(d.getHours())}:${p(d.getMinutes())}`;
  }
}

export function fmtBytes(n) {
  if (!Number.isFinite(n) || n < 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = n, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  const decimals = i > 0 && v < 10 ? 1 : 0;
  return `${v.toFixed(decimals)} ${units[i]}`;
}

// health 응답에서 엔진 capability를 추출하는 순수 코어 (tests에서 직접 검증).
// 구버전 서버(capabilities 부재)는 모두 undefined — 소비처는 기존 UI 그대로 동작.
export function healthCapabilities(d) {
  const caps = (d && d.capabilities && typeof d.capabilities === 'object') ? d.capabilities : {};
  return {
    engine: (d && typeof d.engine === 'string') ? d.engine : undefined,
    streamGranularity: typeof caps.stream_granularity === 'string' ? caps.stream_granularity : undefined,
    layoutCapability: typeof caps.layout === 'string' ? caps.layout : undefined,
  };
}

// sidecar provider 장애 요약 (순수). 문제 없으면 null.
// 메인 앱 health가 200이어도 provider가 죽어 있으면 사용자에게 알린다.
export function providerIssue(d) {
  if (!d || d.provider !== 'local-sidecar') return null;
  const ph = d.provider_health;
  if (!ph || typeof ph !== 'object' || ph.status === 'ok') return null;
  return String(ph.error || ph.status || '알 수 없음');
}

// 잡 헤더 모델 칩 데이터 (순수). 구버전 잡(메타 없음)은 null → 칩 숨김.
export function jobModelChip(job) {
  if (!job) return null;
  const text = job.model_id || job.engine;
  if (!text) return null;
  const title = (job.model_id || '') +
    (job.model_revision ? ` @ ${String(job.model_revision).slice(0, 8)}` : '') +
    (job.engine ? ` · engine: ${job.engine}` : '') +
    (job.provider ? ` · ${job.provider}` : '');
  return { text, title };
}

// figure_only 엔진(본문 텍스트에 bbox 없음 → 레이아웃 재구성이 빈 흰 페이지)인지 판정 (순수).
// **그 잡을 실제로 변환한 엔진**이 현재 엔진일 때만 확신한다 — 엔진 메타가 없는 구 잡
// (이 기능 이전 변환 = 항상 full layout)이나 다른 엔진의 잡은 제외한다(거짓 판단보다 무표시).
export function docLayoutIsFigureOnly(layoutCapability, jobEngine, healthEngine) {
  return layoutCapability === 'figure_only' && !!jobEngine && jobEngine === healthEngine;
}

// 2단계 삭제 확인의 클릭 전이 (순수 — 테스트 대상). entries는 [key, owner] 쌍의
// 이터러블(owner = 물리 버튼), key는 이번 클릭의 키, owner는 클릭된 버튼.
// 클릭된 key가 이미 무장돼 있으면 confirm(실삭제), 아니면 같은 owner에 남은
// 옛 무장(잡 전환 뒤의 헤더 버튼 등)을 회수(clearKeys)하고 새로 무장한다.
export function armTransition(entries, key, owner) {
  const clearKeys = [];
  let confirm = false;
  for (const [k, o] of entries) {
    if (k === key) confirm = true;
    else if (o === owner) clearKeys.push(k);
  }
  if (confirm) clearKeys.push(key);
  return { confirm, clearKeys };
}

// location.hash → 잡 id (순수 — frontend/tests/에서 직접 검증).
// '#abc' → 'abc'. 빈 해시·잡 id에 쓰이지 않는 문자가 섞인 이상값은 null.
export function jobIdFromHash(hash) {
  const id = String(hash || '').replace(/^#/, '');
  return /^[\w-]+$/.test(id) ? id : null;
}

// URL 언어 파라미터 빌더 (순수 — 테스트 대상). ko가 아니면 원본 URL 그대로.
export function withLangUrl(url, lang) {
  if (lang !== 'ko' || !url) return url;
  return url + (url.indexOf('?') === -1 ? '?' : '&') + 'lang=ko';
}

// translate/state·POST 응답의 status → 노출할 UI (순수 — 테스트 대상).
export function translateUiStateFor(status) {
  if (status === 'running') return 'progress';
  if (status === 'done') return 'toggle';
  return 'button'; // none | error | canceled | 미지의 값 → 버튼(재시도)
}

// 좌측 스테이지에 bbox 오버레이를 붙여 둘 페이지인가 (순수 — tests/에서 검증).
// hydrateReaderPages의 keep 창과 같은 기준 — 걷어내는 쪽과 그리는 쪽이 어긋나면
// 창 밖 페이지 오버레이가 걷히지 않고 쌓인다.
export function overlayInKeepWindow(page, currentPage, total, radius = READER_KEEP_RADIUS) {
  const keep = readerHydrationWindow(total, currentPage, radius);
  return page >= keep.start && page <= keep.end;
}

export function readerRailBandAt(bands, offset) {
  if (!bands || !bands.length) return null;
  let lo = 0;
  let hi = bands.length - 1;
  let found = bands[0];
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bands[mid].top <= offset) {
      found = bands[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

// ── 레일 스크롤 앵커 (순수 — frontend/tests/에서 직접 검증) ──────────────────
// .reader-content는 overflow-anchor: none이라 지금 보고 있는 밴드 "위쪽" 페이지가
// 뒤늦게 채워지면 화면이 그만큼 아래로 밀린다. 연동(sync on)에서는 원문 면 눈높이가
// 되잡아 주지만 개별(sync off)에서는 잡아 줄 기준이 없다 — 렌더 직전에 밴드 상대
// 오프셋을 기록해 두었다가 렌더 후 같은 오프셋으로 복원한다.
export function railAnchorFrom(bands, scrollTop, viewportHeight, ratio = READER_FOCUS_RATIO) {
  const band = readerRailBandAt(bands, (Number(scrollTop) || 0) + (Number(viewportHeight) || 0) * ratio);
  if (!band) return null;
  return { page: band.page, offset: (Number(scrollTop) || 0) - band.top };
}

// 새 밴드 배열에서 앵커가 가리키던 스크롤 위치. 그 페이지가 사라졌으면 null.
export function railAnchorTarget(bands, anchor) {
  if (!anchor || !bands || !bands.length) return null;
  const band = bands.find((candidate) => candidate.page === anchor.page);
  if (!band) return null;
  return Math.max(0, Math.round(band.top + anchor.offset));
}

// 파일 크기 사전 검증 (순수 — frontend/tests/에서 직접 검증). 상한 초과면 안내
// 문구, 통과면 null. limitMb 미수신(undefined 등 비정상)이면 검증을 생략한다
// — 서버 413이 최후 방어.
export function fileSizeError(sizeBytes, limitMb) {
  const limit = Number(limitMb);
  const size = Number(sizeBytes);
  if (!Number.isFinite(limit) || limit <= 0 || !Number.isFinite(size)) return null;
  if (size <= limit * 1024 * 1024) return null;
  // 표시 반올림이 상한과 같아지는 경계(상한+1바이트 → '100 MB — 상한 100MB')의
  // 자기모순 문구를 피한다 — 반올림 표시가 상한을 명확히 넘을 때만 크기를 병기.
  const sizeMb = size / (1024 * 1024);
  return sizeMb - limit >= 0.05
    ? `파일이 너무 큽니다 (${fmtBytes(size)} — 서버 상한 ${limit}MB)`
    : `파일이 너무 큽니다 (서버 상한 ${limit}MB 초과)`;
}

// 다중 선택 검증 분류 (순수 — frontend/tests/에서 직접 검증). validate(file)는
// 오류 문구 또는 null을 반환. 유효 파일과 '건너뜀' 대상(파일명+사유)으로 나눈다.
export function classifyFiles(files, validate) {
  const valid = [];
  const skipped = [];
  for (const f of Array.from(files || [])) {
    const reason = validate(f);
    if (reason) skipped.push({ file: f, name: (f && f.name) || '(이름 없음)', reason });
    else valid.push(f);
  }
  return { valid, skipped };
}

// file-info 표시 문구 (순수 — 테스트 대상). 1개면 기존 단일 표시(이름·크기),
// 여러 개면 'N개 파일 · 총 X' + title에 파일명 나열. 빈 선택은 null.
export function selectionSummary(files) {
  const list = Array.from(files || []);
  if (!list.length) return null;
  if (list.length === 1) {
    return { name: list[0].name, size: fmtBytes(Number(list[0].size) || 0), title: list[0].name };
  }
  const total = list.reduce((sum, f) => sum + (Number(f.size) || 0), 0);
  return {
    name: `${list.length}개 파일`,
    size: `총 ${fmtBytes(total)}`,
    title: list.map((f) => f.name).join(', '),
  };
}

// '첫 건 + 나머지 개수' 요약 (순수 — 테스트 대상). entries: [{name, reason}].
// prefix 예: '건너뜀' | '업로드 실패'. 빈 배열이면 null.
export function summarizeIssues(prefix, entries) {
  if (!entries || !entries.length) return null;
  const first = `${prefix}: ${entries[0].name} — ${entries[0].reason}`;
  return entries.length === 1 ? first : `${first} 외 ${entries.length - 1}건`;
}
