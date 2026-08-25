import { ICON, THEME_KEY } from './constants.js';
import { RETRY_AFTER_FALLBACK_S, RETRY_AFTER_MAX_S } from './core.js';
import { el, state } from './state.js';

/* ============================ Utilities ============================ */

export function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const key of Object.keys(attrs)) {
      const val = attrs[key];
      if (val == null || val === false) continue;
      if (key === 'class') node.className = val;
      else if (key === 'text') node.textContent = val;
      else if (key === 'html') node.innerHTML = val; // only used with trusted literal SVG strings
      else node.setAttribute(key, val === true ? '' : val);
    }
  }
  for (const child of children) {
    if (child == null || child === false) continue;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

export function safeParse(text) {
  try { return JSON.parse(text); } catch (_) { return null; }
}

export function localGet(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}
export function localSet(key, val) {
  try { localStorage.setItem(key, val); } catch (_) { /* ignore */ }
}
export function localRemove(key) {
  try { localStorage.removeItem(key); } catch (_) { /* ignore */ }
}

export function isTerminal(status) {
  return status === 'done' || status === 'error' || status === 'canceled';
}

export function showToast(message, kind) {
  el.toast.textContent = message;
  el.toast.className = 'toast' + (kind ? ' ' + kind : '');
  el.toast.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { el.toast.hidden = true; }, 3600);
}

// 429 뒤 재시도 가능 시각까지 버튼을 잠근다 — 사용자가 연타로 상한을 더 악화시키지
// 않게. atKey에 만료 시각을 남겨, 다른 경로(폼 submit·재렌더로 복구된 버튼)에서도
// 요청 자체를 막을 수 있게 한다.
export function lockRetry(atKey, timerKey, seconds, buttons, onRelease) {
  const wait = Math.max(1, Math.min(Number(seconds) || RETRY_AFTER_FALLBACK_S, RETRY_AFTER_MAX_S));
  const targets = buttons.filter(Boolean);
  state[atKey] = Date.now() + wait * 1000;
  clearTimeout(state[timerKey]);
  for (const b of targets) b.disabled = true;
  state[timerKey] = setTimeout(() => {
    state[timerKey] = 0;
    state[atKey] = 0;
    for (const b of targets) b.disabled = false;
    // 해제 시점의 다른 비활성 사유(프로바이더 미설정·질문 전송 중)를 다시 입힌다.
    if (typeof onRelease === 'function') onRelease();
  }, wait * 1000);
}

// 잠금이 아직 유효하면 남은 초, 아니면 0.
export function retryLockRemaining(atKey) {
  const until = Number(state[atKey]) || 0;
  const left = until - Date.now();
  return left > 0 ? Math.ceil(left / 1000) : 0;
}

// 잠금이 살아 있는 동안 버튼을 다시 비활성으로 되돌린다 (순수 — tests/에서 검증).
// 잠금은 잡이 아니라 클라이언트(IP) 단위라 잡을 바꿔도 유효한데, 잡 전환·재렌더는
// 버튼을 되살린다 — 그대로 두면 눌리기만 하고 요청은 나가지 않는 버튼이 된다.
export function applyRetryLock(remainingSeconds, buttons) {
  if (!(Number(remainingSeconds) > 0)) return false;
  for (const b of buttons || []) { if (b) b.disabled = true; }
  return true;
}

/* ============================ Theme ============================ */

export function resolvedTheme() {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

export function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const label = theme === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환';
  el.themeToggle.innerHTML = theme === 'dark' ? ICON.sun : ICON.moon;
  el.themeToggle.setAttribute('aria-label', label);
  el.themeToggle.title = label;
}

export function setupTheme() {
  applyTheme(resolvedTheme()); // sync icon with the value set by the inline bootstrap
  el.themeToggle.addEventListener('click', () => {
    const next = resolvedTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localSet(THEME_KEY, next);
  });
  try {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', (e) => {
      const stored = localGet(THEME_KEY);
      if (stored !== 'light' && stored !== 'dark') applyTheme(e.matches ? 'dark' : 'light');
    });
  } catch (_) { /* ignore */ }
}

/* ── KaTeX 타이포셋 (로컬 벤더 — vendor/katex) ────────────────────────── */
// 서버 렌더러(render.py)가 tex를 이스케이프해 .math-inline/.math-display로
// 내보낸다. KaTeX 미로드(자산 누락 등) 시에는 raw LaTeX 텍스트가 그대로
// 보이는 그레이스풀 폴백.
export function typesetMath(root) {
  if (!window.katex || !root) return;
  // root 자신이 수식 블록일 수 있다 (증분 프리뷰는 최상위 노드 단위로 붙인다)
  const targets = root.matches && root.matches('.math-inline, .math-display')
    ? [root, ...root.querySelectorAll('.math-inline, .math-display')]
    : root.querySelectorAll('.math-inline, .math-display');
  targets.forEach((elm) => {
    if (elm.dataset.mathDone) return;
    const tex = elm.textContent;
    try {
      window.katex.render(tex, elm, {
        displayMode: elm.classList.contains('math-display'),
        throwOnError: false,
      });
      elm.dataset.mathDone = '1';
    } catch (_) { /* 렌더 불가 tex는 원문 유지 */ }
  });
}

export function parseEventData(e) {
  if (!e || e.data == null) return null; // connection errors have no data
  return safeParse(e.data);
}

export function setDownload(anchor, url, downloadName) {
  if (url) {
    anchor.href = url;
    anchor.setAttribute('download', downloadName);
    anchor.classList.remove('disabled');
    anchor.removeAttribute('aria-disabled');
  } else {
    anchor.removeAttribute('href');
    anchor.classList.add('disabled');
    anchor.setAttribute('aria-disabled', 'true');
  }
}

export function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
