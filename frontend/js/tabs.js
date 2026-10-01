import { ICON } from './constants.js';
import { busyWaitMessage, docLayoutIsFigureOnly, langFetchVerdict, withLangUrl } from './core.js';
import { el, state } from './state.js';
import { h, setTrustedHtml, showToast, typesetMath } from './ui.js';
import { fetchTextWithBusyRetry } from './api.js';
import { revertToOriginal, setLang } from './translate.js';
import { initQaTab, prefillQaPageFromReader } from './qa.js';
import { loadReader, readerViewportFocus } from './reader.js';

/* ============================ Tabs ============================ */

export function activateTab(name) {
  // 전체화면 닫기/패널 변경의 anchor-rAF가 대기 중이면 DOM은 이미 새 폭인데 bands는
  // 아직 이전 폭이다. 그 짧은 창에서 재측정한 값을 lastFocus에 덮지 않는다.
  if (name !== 'reader' && !state.readerAnchorRaf) {
    const focus = readerViewportFocus();
    if (focus) state.readerLastFocus = focus;
  }
  el.tabs.forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', on ? 'true' : 'false');
    t.tabIndex = on ? 0 : -1;
  });
  el.panels.forEach((p) => { p.hidden = p.dataset.panel !== name; });
  if (name === 'reader') loadReader();
  else if (name === 'preview') loadPreview();
  else if (name === 'markdown') loadMarkdown();
  else if (name === 'doclayout') loadDocLayout();
  else if (name === 'qa') { prefillQaPageFromReader(); initQaTab(); }
}

// figure_only 엔진(OvisOCR2·PaddleOCR-VL 등)은 본문 텍스트에 좌표가 없어 서버 레이아웃
// 재구성이 "빈 흰 페이지 + 그림 사각형 몇 개"로 나온다 — 사용자에겐 변환이 깨진 것처럼
// 보인다. 오해를 주는 캔버스 대신, 전체 내용은 미리보기/Markdown에 있고 그림 위치는
// 감지 박스 탭에 있음을 분명히 안내하는 카드를 그린다.
export function renderFigureOnlyDocLayout() {
  el.doclayoutBody.textContent = '';
  const goPreview = h('button', { class: 'btn btn-primary btn-small', type: 'button' }, '미리보기로 이동');
  goPreview.addEventListener('click', () => activateTab('preview'));
  el.doclayoutBody.appendChild(h('div', { class: 'doclayout-figonly' },
    h('div', { class: 'df-icon', html: ICON.docLayout }),
    h('h3', { class: 'df-title', text: '이 엔진은 텍스트 배치 좌표를 제공하지 않습니다' }),
    h('p', { class: 'df-lead', text: '현재 OCR 엔진은 문서를 흐름 텍스트로 재구성합니다. '
      + '페이지 위 정확한 좌표로 텍스트를 재배치하는 레이아웃 뷰는 Unlimited 엔진에서만 제공됩니다 — '
      + '변환된 내용이 사라진 것이 아닙니다.' }),
    h('ul', { class: 'df-list' },
      h('li', null, h('strong', { text: '전체 내용' }), ' — “미리보기” · “Markdown” 탭에 텍스트·표·수식이 모두 있습니다.'),
      h('li', null, h('strong', { text: '그림·표 위치' }), ' — “감지 박스” 탭에서 원본 페이지 위에 표시됩니다.'),
    ),
    goPreview,
  ));
}

/* ── 결과 탭 로드 공통 ─────────────────────────────────────────────────
 * 계약: 503 + Retry-After(빌드·예열 중)는 제한 재시도하며 진행을 보여 주고,
 * 404/409(번역본 없음)일 때만 원문으로 폴백한다. 그 밖의 실패(네트워크·5xx)는
 * 일시 장애다 — 언어 선택을 그대로 두고 다시 시도 버튼을 띄운다(frontend-9).
 */

// 진행 중인 탭 로드 — (탭, 잡, 언어, 잡 열기 세대)별 1건. 바쁨 재시도로 수십 초를
// 기다리는 동안 탭을 다시 눌러도 같은 요청을 겹쳐 보내지 않는다.
const tabLoads = new Set();

function loadContext() {
  return { id: state.currentJobId, lang: state.currentLang, gen: state.openGen };
}

function isCurrentLoad(ctx) {
  return state.currentJobId === ctx.id && state.currentLang === ctx.lang
    && state.openGen === ctx.gen;
}

async function guardedTabLoad(kind, ctx, run) {
  const key = `${kind}|${ctx.id}|${ctx.lang}|${ctx.gen}`;
  if (tabLoads.has(key)) return;
  tabLoads.add(key);
  try {
    await run();
  } finally {
    tabLoads.delete(key);
  }
}

// 패널 내용을 안내 한 줄(로딩·바쁨 대기·확정 부재)로 바꾼다.
function setPanelNote(container, text) {
  container.textContent = '';
  container.appendChild(h('p', { class: 'muted', role: 'status', text }));
}

// 일시 실패 안내 — 언어는 유지하고 [다시 시도](+ 한국어면 [원문 보기])를 준다.
function renderLoadFailure(container, message, retry, ko) {
  container.textContent = '';
  const again = h('button', { class: 'btn btn-small', type: 'button', text: '다시 시도' });
  again.addEventListener('click', () => retry());
  const actions = h('div', { class: 'load-failure-actions' }, again);
  if (ko) {
    const original = h('button', { class: 'btn btn-small btn-ghost', type: 'button', text: '원문 보기' });
    original.addEventListener('click', () => setLang('orig'));
    actions.appendChild(original);
  }
  container.appendChild(h('div', { class: 'load-failure', role: 'alert' },
    h('p', { class: 'muted', text: message }), actions));
}

function busyNotifier(ctx, label, show) {
  return (seconds, attempt, max) => {
    if (isCurrentLoad(ctx)) show(busyWaitMessage(label, seconds, attempt, max));
  };
}

export async function loadDocLayout() {
  if (state.docLayoutLoaded) return;
  const ctx = loadContext();
  if (!ctx.id) return;
  // figure_only 엔진은 캔버스가 비어 "흰 바탕에 그림만" 나온다 — 캔버스를 아예 그리지 않고
  // 안내 카드로 대체(전체 내용은 미리보기/Markdown, 그림 위치는 감지 박스로 유도).
  if (docLayoutIsFigureOnly(state.layoutCapability, state.currentJobEngine, state.healthEngine)) {
    state.docLayoutLoaded = true;
    renderFigureOnlyDocLayout();
    return;
  }
  await guardedTabLoad('doclayout', ctx, async () => {
    const ko = ctx.lang === 'ko';
    setPanelNote(el.doclayoutBody, '레이아웃을 불러오는 중…');
    const r = await fetchTextWithBusyRetry(withLangUrl(`/api/jobs/${ctx.id}/layout`, ctx.lang), {
      accept: 'text/html',
      isCurrent: () => isCurrentLoad(ctx),
      onWait: busyNotifier(ctx, ko ? '한국어 레이아웃' : '레이아웃',
        (text) => setPanelNote(el.doclayoutBody, text)),
    });
    if (!isCurrentLoad(ctx)) return; // 잡/언어 전환 → 최신 로더에 위임
    const missing = langFetchVerdict(r.status) === 'missing';
    if (missing && ko && revertToOriginal('한국어 레이아웃이 없어 원문을 표시합니다.')) {
      loadDocLayout();
      return;
    }
    if (missing) {
      state.docLayoutLoaded = true; // 404는 재시도해도 같음
      setPanelNote(el.doclayoutBody, '이 작업에는 레이아웃 데이터가 없습니다 (이 기능 추가 이전에 변환된 결과).');
      return;
    }
    if (r.text == null) {
      if (!ko && state.resultHasLayout === false) {
        setPanelNote(el.doclayoutBody,
          '이 작업은 레이아웃 기능 이전에 변환되어 레이아웃 데이터가 없습니다 — PDF를 다시 변환하면 생깁니다.');
        return;
      }
      renderLoadFailure(el.doclayoutBody, ko
        ? '한국어 레이아웃을 불러오지 못했습니다 — 잠시 후 다시 시도해 주세요.'
        : '레이아웃 뷰를 불러오지 못했습니다.', loadDocLayout, ko);
      return;
    }
    state.docLayoutLoaded = true;
    // Trusted server-rendered fragment (pipeline/layout.py — 텍스트 전부 이스케이프됨).
    // 번역본은 루트에 lang="ko"가 붙어 오지만, 컨테이너에도 setResultLangAttr로 반영해 둔다.
    // 붙이기 전에 외부 이미지를 막고, 전 페이지 PNG(쪽당 ~350KB)를 한꺼번에 받지 않도록
    // loading=lazy를 단다 — 300쪽이면 수백 MB가 한 번에 큐에 올라 연결을 잡아먹었다.
    setTrustedHtml(el.doclayoutBody, r.text, { lazyImages: true });
    typesetMath(el.doclayoutBody);
    if (window.uocrFitLayout) window.uocrFitLayout(el.doclayoutBody);
  });
}

export async function loadPreview() {
  if (state.previewLoaded) return;
  const ctx = loadContext();
  if (!ctx.id) return;
  await guardedTabLoad('preview', ctx, async () => {
    const ko = ctx.lang === 'ko';
    setPanelNote(el.previewBody, '미리보기를 불러오는 중…');
    const r = await fetchTextWithBusyRetry(withLangUrl(`/api/jobs/${ctx.id}/html`, ctx.lang), {
      accept: 'text/html',
      isCurrent: () => isCurrentLoad(ctx),
      onWait: busyNotifier(ctx, ko ? '한국어 미리보기' : '미리보기',
        (text) => setPanelNote(el.previewBody, text)),
    });
    if (!isCurrentLoad(ctx)) return;
    if (r.text == null) {
      if (langFetchVerdict(r.status) === 'missing' && ko
          && revertToOriginal('한국어 번역본이 없어 원문 미리보기를 표시합니다.')) {
        loadPreview();
        return;
      }
      renderLoadFailure(el.previewBody, ko
        ? '한국어 미리보기를 불러오지 못했습니다 — 잠시 후 다시 시도해 주세요.'
        : '미리보기를 불러오지 못했습니다.', loadPreview, ko);
      return;
    }
    state.previewLoaded = true;
    // Trusted server-rendered fragment (/html, same renderer as /render-preview) —
    // 외부 이미지 src는 붙이기 전에 막는다(frontend-3).
    setTrustedHtml(el.previewBody, r.text);
    typesetMath(el.previewBody);
  });
}

export async function loadMarkdown() {
  if (state.markdownLoaded) return;
  const ctx = loadContext();
  if (!ctx.id) return;
  await guardedTabLoad('markdown', ctx, async () => {
    const ko = ctx.lang === 'ko';
    el.mdCode.textContent = '불러오는 중…';
    const r = await fetchTextWithBusyRetry(withLangUrl(`/api/jobs/${ctx.id}/markdown`, ctx.lang), {
      accept: 'text/markdown',
      isCurrent: () => isCurrentLoad(ctx),
      onWait: busyNotifier(ctx, ko ? '한국어 Markdown' : 'Markdown',
        (text) => { el.mdCode.textContent = text; }),
    });
    if (!isCurrentLoad(ctx)) return;
    if (r.text == null) {
      if (langFetchVerdict(r.status) === 'missing' && ko
          && revertToOriginal('한국어 Markdown이 없어 원문을 표시합니다.')) {
        loadMarkdown();
        return;
      }
      // <pre><code> 안에는 버튼을 두지 않는다 — 탭을 다시 누르면 같은 로더가 재시도한다.
      el.mdCode.textContent = ko
        ? '한국어 Markdown을 불러오지 못했습니다 — 탭을 다시 누르면 다시 시도합니다.'
        : 'Markdown을 불러오지 못했습니다 — 탭을 다시 누르면 다시 시도합니다.';
      return;
    }
    state.markdownLoaded = true;
    el.mdCode.textContent = r.text;
  });
}

/* ============================ Tabs / result wiring ============================ */

export function setupTabs() {
  el.tabs.forEach((t) => {
    t.addEventListener('click', () => activateTab(t.dataset.tab));
  });
  // basic roving-tabindex keyboard nav
  const tablist = el.tabs.length ? el.tabs[0].parentElement : null;
  if (tablist) {
    tablist.addEventListener('keydown', (ev) => {
      if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(ev.key)) return;
      const idx = el.tabs.findIndex((t) => t.classList.contains('active'));
      if (idx === -1) return;
      let next;
      if (ev.key === 'Home') next = el.tabs[0];
      else if (ev.key === 'End') next = el.tabs[el.tabs.length - 1];
      else {
        const dir = ev.key === 'ArrowRight' ? 1 : -1;
        next = el.tabs[(idx + dir + el.tabs.length) % el.tabs.length];
      }
      ev.preventDefault();
      activateTab(next.dataset.tab);
      next.focus();
    });
  }

  el.copyMd.addEventListener('click', async () => {
    const text = el.mdCode.textContent || '';
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        // navigator.clipboard는 secure context(HTTPS·localhost) 전용 — http://IP
        // 접속(VPN 배포 기본)에서는 undefined다. 사용자 제스처 하에서는 insecure
        // context에서도 동작하는 execCommand 경로로 폴백한다.
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand('copy');
        ta.remove();
        if (!ok) throw new Error('no clipboard');
      }
      el.copyMd.textContent = '복사됨';
      el.copyMd.classList.add('copied');
      setTimeout(() => { el.copyMd.textContent = '복사'; el.copyMd.classList.remove('copied'); }, 1600);
    } catch (_) {
      showToast('클립보드 복사에 실패했습니다. (HTTPS가 아닌 접속에서는 브라우저가 복사를 제한할 수 있습니다)', 'error');
    }
  });
}
