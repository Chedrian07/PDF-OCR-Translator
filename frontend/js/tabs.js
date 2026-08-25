import { ICON } from './constants.js';
import { docLayoutIsFigureOnly, withLangUrl } from './core.js';
import { el, state } from './state.js';
import { h, showToast, typesetMath } from './ui.js';
import { revertToOriginal } from './translate.js';
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

export async function loadDocLayout() {
  if (state.docLayoutLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  // figure_only 엔진은 캔버스가 비어 "흰 바탕에 그림만" 나온다 — 캔버스를 아예 그리지 않고
  // 안내 카드로 대체(전체 내용은 미리보기/Markdown, 그림 위치는 감지 박스로 유도).
  if (docLayoutIsFigureOnly(state.layoutCapability, state.currentJobEngine, state.healthEngine)) {
    state.docLayoutLoaded = true;
    renderFigureOnlyDocLayout();
    return;
  }
  const lang = state.currentLang; // 응답 도착 시점에 언어가 바뀌었는지 판별용
  el.doclayoutBody.textContent = '';
  el.doclayoutBody.appendChild(h('p', { class: 'muted', text: '레이아웃을 불러오는 중…' }));
  let html = null;
  let missing = false;
  try {
    const res = await fetch(withLangUrl(`/api/jobs/${id}/layout`, lang), { headers: { Accept: 'text/html' } });
    if (res.status === 404) missing = true;
    else if (res.ok) html = await res.text();
  } catch (_) { /* 아래 공통 실패 처리 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return; // 잡/언어 전환 → 최신 로더에 위임
  // 한국어 뷰에서 번역본을 못 받으면(404·실패) 조용히 원문으로 폴백 + 토스트.
  if ((missing || html == null) && lang === 'ko' &&
      revertToOriginal('한국어 레이아웃을 불러오지 못해 원문을 표시합니다.')) {
    loadDocLayout();
    return;
  }
  el.doclayoutBody.textContent = '';
  if (missing) {
    state.docLayoutLoaded = true; // 404는 재시도해도 같음
    el.doclayoutBody.appendChild(h('p', {
      class: 'muted',
      text: '이 작업에는 레이아웃 데이터가 없습니다 (이 기능 추가 이전에 변환된 결과).',
    }));
    return;
  }
  if (html == null) {
    const noLayout = state.resultHasLayout === false;
    el.doclayoutBody.appendChild(h('p', {
      class: 'muted',
      text: noLayout
        ? '이 작업은 레이아웃 기능 이전에 변환되어 레이아웃 데이터가 없습니다 — PDF를 다시 변환하면 생깁니다.'
        : '레이아웃 뷰를 불러오지 못했습니다.',
    }));
    return;
  }
  state.docLayoutLoaded = true;
  // Trusted server-rendered fragment (pipeline/layout.py — 텍스트 전부 이스케이프됨).
  // 번역본은 루트에 lang="ko"가 붙어 오지만, 컨테이너에도 setResultLangAttr로 반영해 둔다.
  el.doclayoutBody.innerHTML = html;
  typesetMath(el.doclayoutBody);
  if (window.uocrFitLayout) window.uocrFitLayout(el.doclayoutBody);
}

export async function loadPreview() {
  if (state.previewLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  const lang = state.currentLang;
  el.previewBody.textContent = '';
  el.previewBody.appendChild(h('p', { class: 'muted', text: '미리보기를 불러오는 중…' }));
  let html = null;
  try {
    const res = await fetch(withLangUrl(`/api/jobs/${id}/html`, lang), { headers: { Accept: 'text/html' } });
    if (res.ok) html = await res.text();
  } catch (_) { /* 아래 공통 실패 처리 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return;
  if (html == null) {
    // 한국어 뷰에서 번역본을 못 받으면 조용히 원문으로 폴백.
    if (lang === 'ko' && revertToOriginal('한국어 미리보기를 불러오지 못해 원문을 표시합니다.')) { loadPreview(); return; }
    el.previewBody.textContent = '';
    el.previewBody.appendChild(h('p', { class: 'muted', text: '미리보기를 불러오지 못했습니다.' }));
    return;
  }
  state.previewLoaded = true;
  // Trusted server-rendered fragment (/html, same renderer as /render-preview).
  el.previewBody.innerHTML = html;
  typesetMath(el.previewBody);
}

export async function loadMarkdown() {
  if (state.markdownLoaded) return;
  const id = state.currentJobId;
  if (!id) return;
  const lang = state.currentLang;
  el.mdCode.textContent = '불러오는 중…';
  let text = null;
  try {
    const res = await fetch(withLangUrl(`/api/jobs/${id}/markdown`, lang), { headers: { Accept: 'text/markdown' } });
    if (res.ok) text = await res.text();
  } catch (_) { /* 아래 공통 실패 처리 */ }
  if (state.currentJobId !== id || state.currentLang !== lang) return;
  if (text == null) {
    if (lang === 'ko' && revertToOriginal('한국어 Markdown을 불러오지 못해 원문을 표시합니다.')) { loadMarkdown(); return; }
    el.mdCode.textContent = 'Markdown을 불러오지 못했습니다.';
    return;
  }
  state.markdownLoaded = true;
  el.mdCode.textContent = text;
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
