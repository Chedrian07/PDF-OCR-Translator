// 실브라우저 E2E — 실행 중인 백엔드가 필요하다 (단위 테스트 러너에 포함되지 않음).
//
//   E2E_BASE_URL=http://127.0.0.1:8002 npm run test:e2e   (기본 8000)
//
// 검증 플로우 (엔진 불문 — health capability로 분기):
//   1) 업로드 → 변환 완료 → 프로덕션 뷰어 열기/닫기·3열·페이지 탐색
//   2) 미리보기에 텍스트·표·이미지·KaTeX 수식 렌더
//   3) HTML 다운로드(document.html) — 자립형(base64 이미지·서버 참조 없음)
//   4) 레이아웃 탭 — figure_only 엔진이면 안내 카드, full이면 캔버스
//   5) Markdown 탭 본문 존재, 다크 테마 렌더
// 실패 시 exit 1. 스크린샷은 shots/(git 무시)에 남는다.
import { chromium } from 'playwright';
import { mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.E2E_BASE_URL || 'http://127.0.0.1:8000';
// 썸네일/키보드 페이지 이동은 최소 2페이지가 필요하다. 저장소의 경량 제품
// 샘플은 표·그림·수식 검증도 그대로 만족한다.
const PDF = process.env.E2E_PDF || path.resolve(HERE, '../../../sample/sample.pdf');
const OUT = path.join(HERE, 'shots');
const TIMEOUT_S = Number(process.env.E2E_TIMEOUT_S || 300); // 콜드 모델 로딩 감안
const VERIFY_MOCK_LLM = process.env.E2E_VERIFY_MOCK_LLM === '1';
mkdirSync(OUT, { recursive: true });

const failures = [];
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(name);
}

async function readerSemanticFocus() {
  return page.evaluate(() => {
    const pane = document.getElementById('reader-page-pane');
    const current = Number(document.getElementById('reader-page')?.value) || 1;
    const section = document.querySelector(`.reader-page[data-page="${current}"]`);
    if (!pane || !section || !section.offsetHeight) return { page: current, fraction: 0 };
    const paneRect = pane.getBoundingClientRect();
    const rect = section.getBoundingClientRect();
    const top = rect.top - paneRect.top + pane.scrollTop;
    const focusY = pane.scrollTop + pane.clientHeight * 0.28;
    return {
      page: current,
      fraction: Math.min(1, Math.max(0, (focusY - top) / rect.height)),
    };
  });
}

// ── 0) 백엔드 프리플라이트 ──────────────────────────────────────────────
let health;
try {
  health = await (await fetch(`${BASE}/api/health`)).json();
} catch {
  console.error(`백엔드에 연결할 수 없습니다: ${BASE} — 서버를 먼저 띄우세요 (docker compose up …)`);
  process.exit(1);
}
const layoutCap = health.capabilities && health.capabilities.layout;
console.log(`engine=${health.engine || 'unlimited'} layout=${layoutCap || 'full'} model_loaded=${health.model_loaded}`);

const browser = await chromium.launch();
const errors = [];
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('response', (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${r.url()}`); });

// ── 1) 업로드 → 완료 대기 ───────────────────────────────────────────────
await page.goto(BASE, { waitUntil: 'networkidle' });
await page.setInputFiles('#file-input', PDF);
await page.waitForTimeout(300);
check('업로드 버튼 활성화', await page.evaluate(() => !document.getElementById('upload-btn').disabled));
await page.click('#upload-btn');

let done = false;
const t0 = Date.now();
while ((Date.now() - t0) / 1000 < TIMEOUT_S) {
  await page.waitForTimeout(2000);
  const s = await page.evaluate(() => ({
    result: !document.getElementById('result-section').hidden,
    error: !document.getElementById('error-section').hidden
      && document.getElementById('progress-section').hidden,
  }));
  if (s.result) { done = true; break; }
  if (s.error) break;
}
check('변환 완료', done, `${Math.round((Date.now() - t0) / 1000)}s`);
if (!done) { await page.screenshot({ path: path.join(OUT, 'fail-not-done.png') }); }

// ── 1.5) 프로덕션 뷰어 — 명시적 진입 + 3열 제품 구조 ────────────────────
await page.waitForSelector('#viewer-open:not([hidden])');
check('프로덕션 뷰어: 완료 직후 닫힌 상태', await page.evaluate(() => {
  const viewer = document.getElementById('production-viewer');
  return !!viewer && !viewer.classList.contains('is-open')
    && !document.body.classList.contains('viewer-mode');
}));
await page.click('#viewer-open');
await page.waitForSelector('#production-viewer.is-open');
await page.waitForFunction(() =>
  document.querySelectorAll('#production-viewer [data-viewer-page]').length >= 2
  && Number(document.getElementById('reader-total')?.textContent || 0) >= 2);
const viewerStructure = await page.evaluate(() => {
  const root = document.getElementById('production-viewer');
  const visible = (selector) => {
    const node = root?.querySelector(selector);
    return !!node && !node.hidden && getComputedStyle(node).display !== 'none';
  };
  return {
    open: !!root?.classList.contains('is-open'),
    bodyMode: document.body.classList.contains('viewer-mode'),
    nav: visible('.viewer-column-nav'),
    source: visible('.viewer-column-source'),
    translation: visible('.viewer-column-translation'),
    thumbnails: root?.querySelectorAll('[data-viewer-page]').length || 0,
    backgroundInert: document.querySelector('.sidebar')?.inert === true
      && document.querySelector('.sidebar')?.getAttribute('aria-hidden') === 'true',
  };
});
check('프로덕션 뷰어: 명시적 열기 + body 모드', viewerStructure.open && viewerStructure.bodyMode,
  JSON.stringify(viewerStructure));
check('프로덕션 뷰어: 배경 앱 포커스·접근성 격리', viewerStructure.backgroundInert,
  JSON.stringify(viewerStructure));
check('프로덕션 뷰어: navigator/source/translation 3열',
  viewerStructure.nav && viewerStructure.source && viewerStructure.translation,
  JSON.stringify(viewerStructure));
check('프로덕션 뷰어: 다중 페이지 썸네일', viewerStructure.thumbnails >= 2,
  JSON.stringify(viewerStructure));

// 열린 뷰어 안에서만 원문/본문/좌표 매핑을 로드한다.
await page.waitForTimeout(1200);
// 연속 스크롤 리더 — 카드/박스 대조는 "현재 페이지" 범위로 본다 (전 페이지가
// 한 DOM에 쌓여 있으므로 전역 비교는 다페이지 문서에서 성립하지 않는다).
const reader = await page.evaluate(() => {
  const n = document.getElementById('reader-page')?.value || '1';
  return {
    textLen: (document.getElementById('reader-content')?.innerText || '').length,
    img: !!document.getElementById('reader-image')?.getAttribute('src'),
    total: document.getElementById('reader-total')?.textContent || '',
    stackPages: document.querySelectorAll('#reader-page-stage .reader-page').length,
    railPages: document.querySelectorAll('#reader-content .reader-rail-page').length,
    hydrated: [...document.querySelectorAll('.reader-page-image')]
      .filter((i) => i.getAttribute('src')).length,
    cards: document.querySelectorAll(`.reader-rail-page[data-page="${n}"] .reader-map-card`).length,
    boxes: document.querySelectorAll(`.reader-page[data-page="${n}"] .reader-map-box`).length,
  };
});
check('프로덕션 뷰어: 연속 본문 rail 렌더', reader.textLen > 20);
check('프로덕션 뷰어: 원문 페이지 이미지 로드', reader.img);
check('프로덕션 뷰어: 다중 페이지 결과', Number(reader.total) >= 2, JSON.stringify(reader));
check('연속 스크롤: 전체 페이지가 한 스크롤 면에 쌓임',
  reader.stackPages === Number(reader.total) && reader.railPages === Number(reader.total),
  JSON.stringify(reader));
check('연속 스크롤: 이미지는 현재 창만 붙는다(지연 로드)',
  reader.hydrated > 0 && reader.hydrated <= Math.min(Number(reader.total), 7),
  `hydrated=${reader.hydrated}/${reader.total}`);
check('연속 스크롤: bbox 버튼 컨테이너가 접근성 트리에서 숨겨지지 않음',
  await page.evaluate(() => {
    const overlay = document.getElementById('reader-map-overlay');
    return overlay?.getAttribute('role') === 'group' && !overlay.hasAttribute('aria-hidden');
  }));
if (layoutCap !== 'figure_only') {
  check('프로덕션 뷰어: 원문 bbox와 텍스트 블록 1:1 (현재 페이지)',
    reader.cards > 0 && reader.cards === reader.boxes, JSON.stringify(reader));
  await page.locator('#reader-content .reader-map-card').first().click();
  check('프로덕션 뷰어: 번역/본문 클릭 → 원문 bbox 활성', await page.evaluate(() => {
    const card = document.querySelector('#reader-content .reader-map-card.is-active');
    const box = document.querySelector('#reader-page-stage .reader-map-box.is-active');
    return !!card && !!box && card.dataset.blockId === box.dataset.blockId;
  }));
}

await page.click('#production-viewer [data-viewer-page="2"]');
await page.waitForFunction(() =>
  document.getElementById('reader-page')?.value === '2'
  && /\/page\/2(?:[?#]|$)/.test(document.getElementById('reader-image')?.src || ''));
check('프로덕션 뷰어: 썸네일 클릭으로 페이지 2 이동', await page.evaluate(() =>
  document.getElementById('reader-page')?.value === '2'));

await page.keyboard.press('ArrowLeft');
await page.waitForFunction(() => document.getElementById('reader-page')?.value === '1');
check('프로덕션 뷰어: ArrowLeft 이전 페이지', true);
await page.keyboard.press('ArrowRight');
await page.waitForFunction(() => document.getElementById('reader-page')?.value === '2');
check('프로덕션 뷰어: ArrowRight 다음 페이지', true);

/* ── 연속 스크롤: 스크롤만으로 페이지가 넘어가고 번역 레일이 따라온다 ── */
let readerRailStyle = '';
{
  const total = Number(await page.$eval('#reader-total', (e) => e.textContent)) || 1;
  // 2페이지짜리 경량 fixture는 번역문 전체가 레일 높이 안에 들어갈 수 있다.
  // 이 블록에서만 레일을 작게 만들어 양방향 "스크롤" 계약을 실제로 운동시킨다.
  readerRailStyle = await page.$eval('#reader-content', (rail) => rail.getAttribute('style') || '');
  await page.$eval('#reader-content', (rail) => {
    rail.style.flex = '0 0 180px';
    rail.style.height = '180px';
    rail.style.minHeight = '0';
  });
  check('연속 스크롤: 양방향 연동 테스트 레일이 스크롤 가능', await page.evaluate(() => {
    const rail = document.getElementById('reader-content');
    return rail.scrollHeight > rail.clientHeight;
  }));
  await page.$eval('#reader-page-pane', (pane) => { pane.scrollTop = 0; });
  await page.waitForFunction(() => document.getElementById('reader-page')?.value === '1');
  await page.waitForTimeout(350); // 직전 키보드 페이지 점프의 quiet 기간 해제

  // 원문 면을 끝까지 스크롤 → 페이지 번호가 따라 올라간다 (버튼 조작 없음)
  await page.$eval('#reader-page-pane', (pane) => { pane.scrollTop = pane.scrollHeight; });
  await page.waitForFunction((n) =>
    Number(document.getElementById('reader-page')?.value) === n, total, { timeout: 15000 });
  check('연속 스크롤: 스크롤만으로 마지막 페이지 도달', true, `total=${total}`);
  check('연속 스크롤: 마지막에서 다음 버튼 비활성',
    await page.evaluate(() => document.getElementById('reader-next')?.disabled === true));

  // 번역 레일도 함께 내려와 있어야 한다 (좌우 동기화)
  check('연속 스크롤: 번역 레일이 원문을 따라옴',
    await page.evaluate(() => document.getElementById('reader-content').scrollTop > 0));

  // 위로 되돌리기
  await page.$eval('#reader-page-pane', (pane) => { pane.scrollTop = 0; });
  await page.waitForFunction(() => document.getElementById('reader-page')?.value === '1');
  check('연속 스크롤: 맨 위로 되돌리면 1페이지', true);

  // 페이지 번호 입력 → 실제로 그 페이지가 뷰포트 안으로 들어온다
  await page.fill('#reader-page', String(total));
  await page.press('#reader-page', 'Enter');
  await page.waitForFunction((n) => {
    const pane = document.getElementById('reader-page-pane').getBoundingClientRect();
    const section = document.querySelector(`.reader-page[data-page="${n}"]`);
    if (!section) return false;
    const rect = section.getBoundingClientRect();
    return rect.bottom > pane.top && rect.top < pane.bottom;
  }, total, { timeout: 15000 });
  check('연속 스크롤: 페이지 번호 입력 → 해당 페이지로 스크롤', true);

  // 연동 끄기 → 원문을 움직여도 레일은 제자리
  await page.click('#reader-sync');
  await page.$eval('#reader-page-pane', (pane) => { pane.scrollTop = 0; });
  await page.waitForTimeout(500);
  const railBefore = await page.evaluate(() => document.getElementById('reader-content').scrollTop);
  await page.$eval('#reader-page-pane', (pane) => { pane.scrollTop = pane.scrollHeight; });
  await page.waitForTimeout(500);
  const railAfter = await page.evaluate(() => document.getElementById('reader-content').scrollTop);
  check('연속 스크롤: 연동 해제 시 레일 고정', railAfter === railBefore, `${railBefore} → ${railAfter}`);
  if (layoutCap !== 'figure_only') {
    check('연속 스크롤: 개별 모드의 보이는 rail 만 키보드 탐색', await page.evaluate(() => {
      const rail = document.getElementById('reader-content');
      const y = rail.getBoundingClientRect().top + rail.clientHeight * 0.28;
      const visible = [...rail.querySelectorAll('.reader-rail-page')].find((section) => {
        const rect = section.getBoundingClientRect();
        return y >= rect.top && y < rect.bottom;
      });
      const visiblePage = Number(visible?.dataset.page) || 0;
      const openPages = [...rail.querySelectorAll('.reader-map-locate[tabindex="0"]')]
        .map((node) => Number(node.closest('.reader-map-card')?.dataset.page) || 0);
      return visiblePage > 0 && openPages.length > 0
        && openPages.every((n) => n === visiblePage);
    }));
  }

  await page.click('#reader-sync'); // 기본(연동) 상태로 복원
  await page.$eval('#reader-page-pane', (pane) => { pane.scrollTop = 0; });
  await page.waitForFunction(() => document.getElementById('reader-page')?.value === '1');

  // 번역 레일을 직접 굴려도 원문이 같은 페이지로 따라온다(역방향 동기화).
  await page.waitForTimeout(350); // 연동 재활성화의 programmatic-scroll quiet 해제
  await page.$eval('#reader-content', (rail) => { rail.scrollTop = rail.scrollHeight; });
  await page.waitForFunction((n) =>
    Number(document.getElementById('reader-page')?.value) === n, total, { timeout: 15000 });
  check('연속 스크롤: 번역 레일 → 원문 역방향 연동', true, `total=${total}`);
  await page.waitForTimeout(700); // URL 250ms + 이어읽기 400ms debounce
  const persisted = await page.evaluate((n) => {
    const params = new URLSearchParams(location.search);
    const id = location.hash.replace(/^#/, '');
    return {
      urlPage: Number(params.get('page')),
      viewer: params.get('viewer'),
      saved: Number(localStorage.getItem(`uocr-reader-pos-${id}`)),
      expected: n,
    };
  }, total);
  check('연속 스크롤: 역방향 이동을 URL·이어읽기에 저장',
    persisted.viewer === '1' && persisted.urlPage === total && persisted.saved === total,
    JSON.stringify(persisted));
}

// 줌·패널 접기로 source 폭/높이가 바뀌어도 페이지 번호뿐 아니라
// 실제로 읽던 줄(페이지 내 fraction)을 보존한다.
const resizeAnchorPage = 1;
await page.fill('#reader-page', String(resizeAnchorPage));
await page.press('#reader-page', 'Enter');
await page.waitForFunction((n) =>
  Number(document.getElementById('reader-page')?.value) === n, resizeAnchorPage);
await page.waitForTimeout(400);
await page.$eval('#reader-page-pane', (pane, target) => {
  const section = document.querySelector(`.reader-page[data-page="${target.page}"]`);
  const paneRect = pane.getBoundingClientRect();
  const rect = section.getBoundingClientRect();
  const top = rect.top - paneRect.top + pane.scrollTop;
  pane.scrollTop = Math.max(0, top + rect.height * target.fraction - pane.clientHeight * 0.28);
}, { page: resizeAnchorPage, fraction: 0.48 });
await page.waitForTimeout(350);
const semanticAnchor = await readerSemanticFocus();

await page.click('#reader-zoom-in');
await page.waitForTimeout(450);
const afterZoom = await readerSemanticFocus();
check('프로덕션 뷰어: 줌 변경 뒤 읽던 줄 보존',
  afterZoom.page === semanticAnchor.page
    && Math.abs(afterZoom.fraction - semanticAnchor.fraction) < 0.02,
  `${JSON.stringify(semanticAnchor)} → ${JSON.stringify(afterZoom)}`);

await page.click('#reader-fit-width');
await page.waitForTimeout(450);
const afterFit = await readerSemanticFocus();
check('프로덕션 뷰어: 너비 맞춤 뒤 읽던 줄 보존',
  afterFit.page === semanticAnchor.page
    && Math.abs(afterFit.fraction - semanticAnchor.fraction) < 0.02,
  `${JSON.stringify(semanticAnchor)} → ${JSON.stringify(afterFit)}`);

await page.click('#viewer-toggle-nav');
await page.waitForTimeout(450);
check('프로덕션 뷰어: navigator 접힘 상태', await page.evaluate(() => {
  const root = document.getElementById('production-viewer');
  const toggle = document.getElementById('viewer-toggle-nav');
  return root?.classList.contains('nav-collapsed')
    && toggle?.getAttribute('aria-pressed') === 'false';
}));
check('프로덕션 뷰어: navigator 접기 뒤 읽던 페이지 보존',
  Number(await page.inputValue('#reader-page')) === resizeAnchorPage);
const afterNav = await readerSemanticFocus();
check('프로덕션 뷰어: navigator 접기 뒤 읽던 줄 보존',
  afterNav.page === semanticAnchor.page
    && Math.abs(afterNav.fraction - semanticAnchor.fraction) < 0.02,
  `${JSON.stringify(semanticAnchor)} → ${JSON.stringify(afterNav)}`);
await page.click('#viewer-toggle-rail');
await page.waitForTimeout(450);
check('프로덕션 뷰어: translation rail 접힘 상태', await page.evaluate(() => {
  const root = document.getElementById('production-viewer');
  const toggle = document.getElementById('viewer-toggle-rail');
  return root?.classList.contains('rail-collapsed')
    && toggle?.getAttribute('aria-pressed') === 'false';
}));
check('프로덕션 뷰어: translation rail 접기 뒤 읽던 페이지 보존',
  Number(await page.inputValue('#reader-page')) === resizeAnchorPage);
const afterRail = await readerSemanticFocus();
check('프로덕션 뷰어: translation rail 접기 뒤 읽던 줄 보존',
  afterRail.page === semanticAnchor.page
    && Math.abs(afterRail.fraction - semanticAnchor.fraction) < 0.02,
  `${JSON.stringify(semanticAnchor)} → ${JSON.stringify(afterRail)}`);
// 후속 검증과 스크린샷은 전체 3열 상태로 남긴다.
await page.click('#viewer-toggle-nav');
await page.waitForTimeout(350);
await page.click('#viewer-toggle-rail');
await page.waitForTimeout(350);
await page.$eval('#reader-content', (rail, style) => {
  if (style) rail.setAttribute('style', style);
  else rail.removeAttribute('style');
}, readerRailStyle);
await page.screenshot({ path: path.join(OUT, 'reader.png') });

// 전체화면 toolbar에서 Q&A로 바로 이동: modal/inert를 먼저 해제하고
// 현재 페이지를 보존한다. 이전에는 hidden QA에 포커스를 보내 무반응처럼 보였다.
const beforeQaJump = await readerSemanticFocus();
await page.click('.reader-tools-wrap > summary');
await page.click('#reader-summary');
await page.waitForFunction(() => {
  const qa = document.querySelector('.tab-panel[data-panel="qa"]');
  return qa && !qa.hidden && !document.body.classList.contains('viewer-mode');
});
const qaJump = await page.evaluate(() => {
  const qa = document.querySelector('.tab-panel[data-panel="qa"]');
  return {
    viewerOpen: document.getElementById('production-viewer')?.classList.contains('is-open'),
    bodyMode: document.body.classList.contains('viewer-mode'),
    inert: qa?.inert === true,
    page: Number(document.getElementById('qa-page')?.value),
    prompt: document.getElementById('qa-input')?.value || '',
    focused: document.activeElement === document.getElementById('qa-input'),
  };
});
check('프로덕션 뷰어: 페이지 요약 → Q&A modal·inert 정상 해제',
  !qaJump.viewerOpen && !qaJump.bodyMode && !qaJump.inert && qaJump.focused,
  JSON.stringify(qaJump));
check('프로덕션 뷰어: 페이지 요약 Q&A에 현재 페이지·프롬프트 전달',
  qaJump.page === beforeQaJump.page && qaJump.prompt.includes('핵심 주장'),
  JSON.stringify(qaJump));

// reader가 hidden인 Q&A 탭에서 다시 전체화면을 열어도 읽던 줄로 복원.
await page.click('#viewer-open');
await page.waitForSelector('#production-viewer.is-open');
await page.waitForTimeout(450);
const afterHiddenOpen = await readerSemanticFocus();
check('프로덕션 뷰어: 다른 탭에서 재진입해도 읽던 줄 복원',
  afterHiddenOpen.page === beforeQaJump.page
    && Math.abs(afterHiddenOpen.fraction - beforeQaJump.fraction) < 0.02,
  `${JSON.stringify(beforeQaJump)} → ${JSON.stringify(afterHiddenOpen)}`);

await page.keyboard.press('Escape');
await page.waitForFunction(() => !document.getElementById('production-viewer')?.classList.contains('is-open'));
check('프로덕션 뷰어: Escape 닫기 + body 모드 해제', await page.evaluate(() =>
  !document.body.classList.contains('viewer-mode')
  && document.querySelector('.sidebar')?.inert === false
  && !document.querySelector('.sidebar')?.hasAttribute('aria-hidden')));

await page.click('#viewer-open');
await page.waitForSelector('#production-viewer.is-open');
await page.click('#viewer-close');
await page.waitForFunction(() => !document.getElementById('production-viewer')?.classList.contains('is-open'));
check('프로덕션 뷰어: 닫기 버튼', await page.evaluate(() =>
  !document.body.classList.contains('viewer-mode')));

// ── 2) 미리보기 렌더 ────────────────────────────────────────────────────
await page.click('button[data-tab="preview"]'); // 읽기 탭이 기본이므로 명시 전환
await page.waitForTimeout(1200); // KaTeX typeset 여유
const preview = await page.evaluate(() => {
  const b = document.getElementById('preview-body');
  return {
    p: b.querySelectorAll('p').length,
    table: b.querySelectorAll('table').length,
    img: b.querySelectorAll('img').length,
    katex: b.querySelectorAll('.katex').length,
    textLen: b.innerText.length,
  };
});
check('미리보기: 문단 렌더', preview.p >= 3 && preview.textLen > 100, JSON.stringify(preview));
check('미리보기: 표 렌더', preview.table >= 1);
check('미리보기: 이미지 렌더', preview.img >= 1);
check('미리보기: KaTeX 수식 조판', preview.katex >= 1);
await page.screenshot({ path: path.join(OUT, 'preview.png') });

// ── 3) HTML 다운로드 (document.html) — 자립형 검증 ──────────────────────
const dlDoc = await page.evaluate(() => {
  const a = document.getElementById('dl-doc');
  return { href: a.getAttribute('href'), disabled: a.classList.contains('disabled'), hidden: a.hidden };
});
check('HTML 다운로드 버튼 활성', !!dlDoc.href && !dlDoc.disabled && !dlDoc.hidden, JSON.stringify(dlDoc));
if (dlDoc.href) {
  const doc = await (await fetch(new URL(dlDoc.href, BASE))).text();
check('document.html: doctype', doc.startsWith('<!doctype html>'));
check('document.html: PDF 페이지 구조 포함', doc.length > 1000 && doc.includes('layout-page-image'));
check('document.html: 페이지 PNG base64 인라인', doc.includes('data:image/png;base64,'));
check('document.html: 서버 참조 없음(자립형)', !doc.includes('/api/jobs/'));
check('document.html: 검색용 텍스트 레이어', doc.includes('facsimile-text-block'));
}

// ── 4) 레이아웃 탭 — capability에 따라 카드 or 캔버스 ────────────────────
await page.click('button[data-tab="doclayout"]');
await page.waitForTimeout(800);
const layout = await page.evaluate(() => ({
  card: !!document.querySelector('#doclayout-body .doclayout-figonly'),
  canvas: !!document.querySelector('#doclayout-body .layout-canvas'),
}));
if (layoutCap === 'figure_only') {
  check('레이아웃 탭: figure_only 안내 카드(캔버스 없음)', layout.card && !layout.canvas, JSON.stringify(layout));
} else {
  check('레이아웃 탭: 좌표 캔버스', layout.canvas && !layout.card, JSON.stringify(layout));
}
check('중복 레이아웃 HTML 버튼 제거', await page.evaluate(() => !document.getElementById('dl-layout')));
await page.screenshot({ path: path.join(OUT, 'layout-tab.png') });

// ── 5) Markdown 탭 + 다크 테마 ──────────────────────────────────────────
await page.click('button[data-tab="markdown"]');
await page.waitForTimeout(500);
check('Markdown 탭 본문', await page.evaluate(() => document.getElementById('md-code').innerText.length > 100));

const jobHash = await page.evaluate(() => location.hash);
const dctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'dark' });
const dpage = await dctx.newPage();
await dpage.goto(`${BASE}/${jobHash}`, { waitUntil: 'networkidle' });
await dpage.waitForTimeout(800);
await dpage.click('button[data-tab="preview"]'); // 기본은 읽기 탭 — 미리보기로 전환
await dpage.waitForTimeout(1200);
const dark = await dpage.evaluate(() => {
  const b = document.getElementById('preview-body');
  const p = b.querySelector('p');
  return { p: b.querySelectorAll('p').length, color: p ? getComputedStyle(p).color : null };
});
check('다크 테마: 미리보기 렌더', dark.p >= 3 && !!dark.color, JSON.stringify(dark));
await dpage.screenshot({ path: path.join(OUT, 'dark-preview.png') });
await dctx.close();

// ── 6) 선택 확장: mock-provider 번역 → PDF 다운로드 + 페이지 Q&A ─────────
// mock-full-flow.e2e.mjs가 로컬 번역/OpenAI Responses mock과 FakeEngine을 띄운
// 경우에만 실행한다. 실 API·과금·외부 전송 없이 브라우저의 마지막 제품 경로를 고정.
if (VERIFY_MOCK_LLM) {
  await page.waitForSelector('#translate-btn:not([hidden])');
  await page.click('#translate-btn');
  await page.waitForFunction(() => {
    const pdf = document.getElementById('dl-pdf');
    const html = document.getElementById('dl-doc-ko');
    const ko = document.getElementById('lang-ko');
    return pdf && !pdf.hidden && html && !html.hidden && ko && !ko.parentElement.hidden;
  }, null, { timeout: 60_000 });
  check('mock 번역: 완료 후 한국어 토글', await page.evaluate(() => !document.getElementById('lang-toggle').hidden));
  check('mock 번역: 한국어 HTML 버튼 노출', await page.evaluate(() => {
    const link = document.getElementById('dl-doc-ko');
    return !link.hidden && link.getAttribute('href')?.includes('lang=ko')
      && link.getAttribute('download')?.endsWith('.ko.html');
  }));
  check('mock 번역: 대조 PDF 버튼 노출', await page.evaluate(() => {
    const link = document.getElementById('dl-pdf');
    return !link.hidden && link.textContent.includes('원문·한국어')
      && link.getAttribute('href')?.includes('view=dual');
  }));
  await page.click('#viewer-open');
  await page.waitForSelector('#production-viewer.is-open');
  await page.waitForFunction(() =>
    document.querySelectorAll('#reader-content .reader-map-card').length > 0);
  check('mock 번역 뷰어: 오른쪽 한국어 rail + 왼쪽 원문 고정', await page.evaluate(() => {
    const text = document.getElementById('reader-content')?.innerText || '';
    const image = document.getElementById('reader-image')?.getAttribute('src') || '';
    return /[가-힣]/.test(text) && !image.includes('lang=ko');
  }));
  check('mock 번역 뷰어: source/translation 블록 ID 1:1 (현재 페이지)', await page.evaluate(() => {
    const n = document.getElementById('reader-page')?.value || '1';
    const ids = (selector) => [...document.querySelectorAll(selector)]
      .map((node) => node.dataset.blockId).filter(Boolean).sort();
    const source = ids(`.reader-page[data-page="${n}"] .reader-map-box`);
    const translated = ids(`.reader-rail-page[data-page="${n}"] .reader-map-card`);
    return source.length > 0 && source.length === new Set(source).size
      && translated.length === new Set(translated).size
      && JSON.stringify(source) === JSON.stringify(translated);
  }));
  await page.click('#viewer-close');
  await page.waitForFunction(() => !document.getElementById('production-viewer')?.classList.contains('is-open'));

  const htmlDownloadPromise = page.waitForEvent('download');
  await page.click('#dl-doc-ko');
  const htmlDownload = await htmlDownloadPromise;
  const htmlPath = await htmlDownload.path();
  const htmlText = htmlPath ? readFileSync(htmlPath, 'utf8') : '';
  check('mock 한국어 HTML: 브라우저 다운로드 완료',
    htmlDownload.suggestedFilename().endsWith('.ko.html'));
  check('mock 한국어 HTML: lang·한글 번역 본문 포함',
    htmlText.includes('<html lang="ko">') && /[가-힣]/.test(htmlText));

  const downloadPromise = page.waitForEvent('download');
  await page.click('#dl-pdf');
  const download = await downloadPromise;
  const downloadedPath = await download.path();
  const first = downloadedPath ? readFileSync(downloadedPath).subarray(0, 5).toString('ascii') : '';
  check('mock PDF: 브라우저 다운로드 완료', download.suggestedFilename().endsWith('.ko.pdf'));
  check('mock PDF: 실제 PDF 바이트', first === '%PDF-');
  check('mock PDF: 생성 리포트 토스트', await page.evaluate(() =>
    (document.getElementById('toast')?.textContent || '').includes('PDF 생성 완료')));

  await page.click('button[data-tab="qa"]');
  await page.waitForFunction(() => document.getElementById('qa-provider')?.value === 'openai-responses');
  await page.fill('#qa-input', '이 페이지의 핵심을 알려줘');
  await page.click('#qa-send');
  await page.waitForFunction(() => {
    const replies = [...document.querySelectorAll('#qa-log .qa-msg.assistant:not(.loading):not(.error)')];
    return replies.some((node) => node.textContent.includes('모의 Q&A 응답'));
  }, null, { timeout: 30_000 });
  check('mock Q&A: OpenAI Responses 공급자 선택',
    await page.inputValue('#qa-provider') === 'openai-responses');
  check('mock Q&A: 브라우저 답변 렌더', await page.evaluate(() =>
    document.getElementById('qa-log').innerText.includes('모의 Q&A 응답')));
  await page.screenshot({ path: path.join(OUT, 'mock-translation-qa.png') });

  // 정렬 API만 일시 장애여도 본문 자체는 /html 폴백으로 읽을 수 있어야 하며,
  // 재시도는 0.8/1.6초 두 번으로 제한되어 빠른 5xx 요청 루프가 생기면 안 된다.
  const failureCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  let failedBatchCalls = 0;
  let failedSingleCalls = 0;
  let failureHtmlCalls = 0;
  const failedRequestDetails = [];
  const failureStartedAt = Date.now();
  await failureCtx.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    const batchAlignment = url.pathname.endsWith('/viewer/pages')
      && (url.searchParams.get('include') || '').split(',').includes('alignment');
    const singleAlignment = url.pathname.endsWith('/alignment');
    if (url.pathname.endsWith('/html')) failureHtmlCalls += 1;
    if (batchAlignment || singleAlignment) {
      failedRequestDetails.push(`${Date.now() - failureStartedAt}ms ${batchAlignment ? 'batch' : 'single'} ${url.search}`);
      if (batchAlignment) failedBatchCalls += 1;
      else failedSingleCalls += 1;
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'e2e temporary alignment failure' }),
      });
      return;
    }
    await route.continue();
  });
  const failurePage = await failureCtx.newPage();
  await failurePage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
  await failurePage.waitForSelector('.reader-rail-retry-note', { timeout: 15_000 });
  const fallbackRail = await failurePage.evaluate(() => ({
    note: document.querySelector('.reader-rail-retry-note')?.textContent || '',
    noteRole: document.querySelector('.reader-rail-retry-note')?.getAttribute('role') || '',
    noteLive: document.querySelector('.reader-rail-retry-note')?.getAttribute('aria-live') || '',
    textLen: (document.getElementById('reader-content')?.innerText || '').length,
    pending: document.querySelectorAll('.reader-rail-pending').length,
  }));
  check('정렬 API 일시 장애: 좌표 없이도 본문 rail 즉시 표시',
    fallbackRail.note.includes('본문은 표시했습니다') && fallbackRail.textLen > 100
      && fallbackRail.noteRole === 'status' && fallbackRail.noteLive === 'polite',
    JSON.stringify(fallbackRail));
  await failurePage.waitForTimeout(3_800);
  // 상한 형태로 단정한다 — 이 검사의 목적은 "재시도가 유한하다"(무한 5xx 루프가
  // 없다)이지 정확한 횟수가 아니다. 하이드레이션 트리거가 러너 지터로 2회 겹치면
  // 창(3회 시도)도 2벌이 되므로 상한을 6/12로 둔다. 회귀(스크롤마다 무한 재요청)는
  // 3.8초 안에 수십~수백 회가 되어 이 상한을 확실히 넘는다. 소진 뒤 정지는
  // 아래 quiet window가 본다. 하한 1은 재시도 경로 자체가 사라지는 회귀를 막는다.
  check('정렬 API 일시 장애: bounded backoff — 재시도 횟수가 상한 안',
    failedBatchCalls >= 1 && failedBatchCalls <= 6
      && failedSingleCalls >= 1 && failedSingleCalls <= 12,
    `batch=${failedBatchCalls}, single=${failedSingleCalls}, html=${failureHtmlCalls} | ${failedRequestDetails.join(' | ')}`);
  const callsAtExhaustion = failedBatchCalls + failedSingleCalls;
  await failurePage.waitForTimeout(800);
  check('정렬 API 일시 장애: 재시도 소진 뒤 quiet window 유지',
    failedBatchCalls + failedSingleCalls === callsAtExhaustion,
    `calls=${failedBatchCalls + failedSingleCalls}`);
  await failureCtx.close();

  // transient가 풀리면 flow 안내를 걷고 원문 bbox ↔ 카드 정렬 모드로 복귀한다.
  const recoveryCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  let recoveryBatchCalls = 0;
  let recoverySingleCalls = 0;
  const recoveryBatchAt = [];
  await recoveryCtx.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    const batchAlignment = url.pathname.endsWith('/viewer/pages')
      && (url.searchParams.get('include') || '').split(',').includes('alignment');
    const singleAlignment = url.pathname.endsWith('/alignment');
    if (batchAlignment) {
      recoveryBatchCalls += 1;
      recoveryBatchAt.push(Date.now());
      if (recoveryBatchCalls === 1) {
        await route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
        return;
      }
    } else if (singleAlignment && recoveryBatchCalls === 1) {
      recoverySingleCalls += 1;
      await route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
      return;
    }
    await route.continue();
  });
  const recoveryPage = await recoveryCtx.newPage();
  await recoveryPage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
  await recoveryPage.waitForSelector('.reader-rail-retry-note', { timeout: 15_000 });
  await recoveryPage.waitForSelector('.reader-map-card', { timeout: 15_000 });
  const recovered = await recoveryPage.evaluate(() => ({
    note: !!document.querySelector('.reader-rail-retry-note'),
    cards: document.querySelectorAll('.reader-rail-page[data-page="1"] .reader-map-card').length,
    boxes: document.querySelectorAll('.reader-page[data-page="1"] .reader-map-box').length,
  }));
  // 여기서 보는 계약은 "503 한 번 뒤 재시도가 성공하면 안내를 걷고 정렬 모드로
  // 돌아온다"이다. 재시도 간격 단정은 넣지 않는다 — 하이드레이션 트리거가 겹치면
  // 2번째 배치가 backoff가 아니라 중복 최초 요청일 수 있어 간격이 20ms로도 잡힌다
  // (실측). 무한 루프 방지는 위 bounded backoff·quiet window가 담당하고,
  // 여기서는 호출 수 상한만 함께 본다. 간격은 진단용으로 detail에만 남긴다.
  const recoveryGap = recoveryBatchAt.length > 1 ? recoveryBatchAt[1] - recoveryBatchAt[0] : -1;
  check('정렬 API 일시 장애: 재시도에서 정렬 카드·bbox로 회복',
    !recovered.note && recovered.cards > 0 && recovered.cards === recovered.boxes
      && recoveryBatchCalls >= 2 && recoveryBatchCalls <= 4 && recoverySingleCalls >= 1,
    `${JSON.stringify(recovered)}, batch=${recoveryBatchCalls}, single=${recoverySingleCalls}, gap=${recoveryGap}ms`);
  await recoveryCtx.close();
}

check('프로덕션 뷰어 포함 콘솔 에러/HTTP 4xx·5xx 없음',
  errors.length === 0, errors.slice(0, 5).join(' | '));
await browser.close();

console.log(failures.length ? `\n${failures.length}개 실패` : '\n전부 통과');
process.exit(failures.length ? 1 : 0);
