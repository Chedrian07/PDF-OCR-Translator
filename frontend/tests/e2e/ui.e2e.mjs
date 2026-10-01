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
import { fileURLToPath, pathToFileURL } from 'node:url';

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

  /* 개별 모드에서 "레일을 직접 굴리는" 경로 — 좌측 면이 멈춰 있는 동안 레일이
     스스로 정렬 창을 로드한다. 이 경로에는 자동 검증이 없었다.
     좌측 면은 문서 끝에 세워 둔 채 레일만 문서 중앙으로 보낸다 — 좌측 하이드레이션
     창(±2)이 절대 덮지 않는 페이지라, 레일이 스스로 로드하지 않으면 영원히
     '불러오는 중…'으로 남는다. 동시에 좌측 오버레이 누적도 여기서만 드러난다. */
  await page.$eval('#reader-page-pane', (pane) => { pane.scrollTop = 0; }); // 좌측은 문서 앞
  await page.waitForTimeout(500); // 하이드레이션 창(1쪽 기준)이 자리 잡을 때까지
  const paneBeforeRail = await page.evaluate(
    () => document.getElementById('reader-page-pane').scrollTop);
  await page.$eval('#reader-content', (rail) => { rail.scrollTop = Math.round(rail.scrollHeight / 2); });
  await page.waitForTimeout(900); // rAF + 정렬 배치 GET
  const railOnly = await page.evaluate(() => {
    const rail = document.getElementById('reader-content');
    const y = rail.getBoundingClientRect().top + rail.clientHeight * 0.28;
    const visible = [...rail.querySelectorAll('.reader-rail-page')].find((section) => {
      const rect = section.getBoundingClientRect();
      return y >= rect.top && y < rect.bottom;
    });
    const current = Number(document.getElementById('reader-page')?.value) || 1;
    const stage = [...document.querySelectorAll('#reader-page-stage .reader-page[data-page]')];
    return {
      pane: document.getElementById('reader-page-pane').scrollTop,
      railScroll: rail.scrollTop,
      visiblePage: Number(visible?.dataset.page) || 0,
      // 레일이 보고 있는 페이지가 여전히 자리표시자면 정렬 로드가 안 걸린 것
      visiblePending: !!visible?.querySelector('.reader-rail-pending'),
      visibleTextLen: (visible?.innerText || '').trim().length,
      tabPages: [...rail.querySelectorAll('.reader-map-locate[tabindex="0"]')]
        .map((node) => Number(node.closest('.reader-map-card')?.dataset.page) || 0),
      current,
      // 좌측 스테이지의 bbox 오버레이는 좌측 현재 페이지의 keep 창(±6) 안에만 남아야
      // 한다 — 레일 스크롤이 먼 페이지 정렬을 계속 불러오는 동안 걷어내는 경로가
      // 돌지 않으면 여기서 무제한으로 쌓인다.
      boxes: document.querySelectorAll('#reader-page-stage .reader-map-box').length,
      boxPages: stage
        .filter((section) => section.querySelector('.reader-map-box'))
        .map((section) => Number(section.dataset.page)),
      boxPagesOutsideKeep: stage
        .filter((section) => section.querySelector('.reader-map-box'))
        .map((section) => Number(section.dataset.page))
        .filter((n) => Math.abs(n - current) > 6),
    };
  });
  check('개별 모드 레일 스크롤: 좌측 원문 면은 움직이지 않는다',
    railOnly.pane === paneBeforeRail && railOnly.railScroll > 0,
    `pane ${paneBeforeRail} → ${railOnly.pane}, rail=${railOnly.railScroll}`);
  check('개별 모드 레일 스크롤: 보고 있는 레일 페이지의 본문이 채워진다',
    railOnly.visiblePage > 0 && !railOnly.visiblePending && railOnly.visibleTextLen > 0,
    JSON.stringify(railOnly));
  check('개별 모드 레일 스크롤: 좌측 bbox 오버레이가 keep 창 밖에 쌓이지 않는다',
    railOnly.boxPagesOutsideKeep.length === 0,
    `current=${railOnly.current}, rail=${railOnly.visiblePage}, boxes=${railOnly.boxes}, `
    + `pages=[${railOnly.boxPages}], outside=[${railOnly.boxPagesOutsideKeep}]`);
  if (layoutCap !== 'figure_only') {
    check('개별 모드 레일 스크롤: Tab 순환이 레일이 보는 페이지로 옮겨간다',
      railOnly.tabPages.length > 0 && railOnly.tabPages.every((n) => n === railOnly.visiblePage),
      JSON.stringify({ visiblePage: railOnly.visiblePage, tabPages: railOnly.tabPages }));
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
  // 긴 문서에서 전 페이지 PNG를 한꺼번에 받지 않는다(frontend-7).
  const lazyPages = await page.evaluate(() => {
    const imgs = [...document.querySelectorAll('#doclayout-body img')];
    return { count: imgs.length, lazy: imgs.filter((i) => i.getAttribute('loading') === 'lazy').length };
  });
  check('레이아웃 탭: 페이지 이미지는 loading=lazy', lazyPages.count > 0 && lazyPages.lazy === lazyPages.count,
    JSON.stringify(lazyPages));
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
// 테마 부트스트랩은 CSP(script-src 'self') 아래에서 외부 파일(theme-init.js)로 돈다.
check('다크 테마: CSP 아래 테마 부트스트랩이 첫 페인트 전에 적용',
  await dpage.evaluate(() => document.documentElement.getAttribute('data-theme') === 'dark'
    && !!document.querySelector('meta[http-equiv="Content-Security-Policy"]')));
await dpage.screenshot({ path: path.join(OUT, 'dark-preview.png') });
await dctx.close();

// ── 5.5) 안전·접근성·연구 도구 회귀 (frontend lane) ──────────────────────
// 각 시나리오는 별도 컨텍스트에서 돈다 — 주 페이지의 콘솔/HTTP 오류 수집을 오염시키지 않게.
const jobId = jobHash.replace(/^#/, '');
const freshContext = (options = {}) => browser.newContext({ viewport: { width: 1280, height: 900 }, ...options });

// (a) 문서 속 외부 이미지(열람 추적 비컨·LAN 주소)는 요청되지 않고 자리표시로 바뀐다(frontend-3).
{
  const imgCtx = await freshContext();
  const beaconHits = [];
  imgCtx.on('request', (r) => {
    if (/tracker\.example|192\.168\.0\.1/.test(r.url())) beaconHits.push(r.url());
  });
  await imgCtx.route((url) => url.pathname === `/api/jobs/${jobId}/html`, async (route) => {
    const res = await route.fetch();
    const body = await res.text();
    await route.fulfill({
      response: res,
      body: body.replace('</section>',
        '<p><img src="https://tracker.example/p.png?doc=42" alt="beacon">'
        + '<img src="http://192.168.0.1/cgi-bin/ping.gif"></p></section>'),
    });
  });
  const imgPage = await imgCtx.newPage();
  await imgPage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
  await imgPage.waitForSelector('#result-section:not([hidden])', { timeout: 20_000 });
  await imgPage.click('button[data-tab="preview"]');
  await imgPage.waitForSelector('#preview-body .blocked-image', { timeout: 15_000 }).catch(() => {});
  const blocked = await imgPage.evaluate(() => ({
    placeholders: document.querySelectorAll('#preview-body .blocked-image').length,
    externalImgs: [...document.querySelectorAll('img')]
      .filter((i) => /tracker\.example|192\.168/.test(i.getAttribute('src') || '')).length,
    label: document.querySelector('#preview-body .blocked-image')?.textContent || '',
  }));
  // CSP 자체도 막는가 — 스크립트가 직접 만든 외부 이미지는 img-src 위반으로 거절된다.
  const violated = await imgPage.evaluate(() => new Promise((resolve) => {
    document.addEventListener('securitypolicyviolation', (e) => resolve(e.violatedDirective), { once: true });
    const probe = new Image();
    probe.src = 'https://tracker.example/csp-probe.png';
    setTimeout(() => resolve(''), 3000);
  }));
  check('외부 이미지: 미리보기에 붙기 전에 자리표시로 바뀌고 요청이 없다',
    blocked.placeholders >= 2 && blocked.externalImgs === 0
      && beaconHits.filter((u) => !u.includes('csp-probe')).length === 0,
    JSON.stringify({ blocked, beaconHits }));
  check('CSP: 스크립트가 만든 외부 이미지도 img-src로 거절된다', /img-src/.test(violated), violated);
  await imgCtx.close();
}

// (b) 5초 목록 폴링이 목록을 다시 그려도 키보드 포커스·2단계 삭제 무장이 유지된다(frontend-4).
{
  const kept = await page.evaluate(async () => {
    const { state } = await import('/js/state.js');
    const { refreshJobs, renderJobList } = await import('/js/jobs.js');
    const del = document.querySelector('#job-list .job-item.active .ji-del')
      || document.querySelector('#job-list .ji-del');
    del.focus();
    del.click(); // 1단계 — 무장
    // 다른 잡이 맨 위에 생겼다가(업로드) 사라지는(삭제) 두 번의 증분 렌더 + 실제 목록 갱신
    state.jobs = [{ job_id: 'e2e-ghost', filename: 'ghost.pdf', status: 'queued',
      created_at: new Date().toISOString() }, ...state.jobs];
    renderJobList();
    await refreshJobs();
    return {
      sameFocus: document.activeElement === del,
      connected: del.isConnected,
      armed: del.classList.contains('armed'),
      ghost: !!document.querySelector('#job-list [data-job-id="e2e-ghost"]'),
    };
  });
  check('작업 목록: 증분 렌더 뒤에도 포커스·삭제 무장 유지',
    kept.sameFocus && kept.connected && kept.armed && !kept.ghost, JSON.stringify(kept));
  await page.waitForTimeout(2800); // 무장 만료 — 실수로 지우지 않게
  check('작업 목록: 2단계 삭제 무장은 시간이 지나면 풀린다',
    await page.evaluate(() => !document.querySelector('#job-list .ji-del.armed')));
}

// (b2) '더 보기': 최신 50건 뒤의 기록을 before 커서로 이어 받는다 — 서버가 has_more를 알려야
//      버튼이 보인다(api-jobs-7). 이 하네스의 잡은 몇 개뿐이라 목록 API를 51건짜리 가짜 목록으로
//      대신한다(api.list_jobs와 같은 limit·before·has_more·total 규칙).
{
  const moreCtx = await freshContext();
  const fakeJobs = Array.from({ length: 51 }, (_, i) => ({
    job_id: `e2e-list-${String(i).padStart(2, '0')}`, filename: `list-${i}.pdf`, status: 'done',
    created_at: new Date(Date.UTC(2026, 0, 1) - i * 60_000).toISOString(),
  }));
  const listCalls = [];
  await moreCtx.route((url) => url.pathname === '/api/jobs', async (route) => {
    if (route.request().method() !== 'GET') { await route.continue(); return; }
    const url = new URL(route.request().url());
    listCalls.push(url.search);
    const limit = Number(url.searchParams.get('limit') || 50);
    const before = url.searchParams.get('before');
    const rest = before ? fakeJobs.slice(fakeJobs.findIndex((j) => j.job_id === before) + 1) : fakeJobs;
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      jobs: rest.slice(0, limit), has_more: rest.length > limit, total: fakeJobs.length,
    }) });
  });
  const morePage = await moreCtx.newPage();
  await morePage.goto(BASE, { waitUntil: 'networkidle' });
  await morePage.waitForSelector('#job-list-more:not([hidden])', { timeout: 10_000 }).catch(() => {});
  const beforeMore = await morePage.evaluate(() => ({
    label: document.getElementById('job-list-more')?.textContent || '',
    rows: document.querySelectorAll('#job-list .job-item').length,
  }));
  await morePage.click('#job-list-more', { timeout: 10_000 }).catch(() => {});
  await morePage.waitForSelector('#job-list [data-job-id="e2e-list-50"]', { timeout: 10_000 }).catch(() => {});
  const afterMore = await morePage.evaluate(() => ({
    rows: document.querySelectorAll('#job-list .job-item').length,
    last: document.querySelector('#job-list .job-item:last-child')?.dataset.jobId || '',
    hidden: document.getElementById('job-list-more')?.hidden,
  }));
  check("작업 목록 '더 보기': before 커서로 다음 쪽을 이어 붙이고 끝이면 버튼을 숨긴다",
    beforeMore.label === '더 보기 (50/51)' && beforeMore.rows === 50
      && afterMore.rows === 51 && afterMore.last === 'e2e-list-50' && afterMore.hidden === true
      && listCalls.includes('?limit=50&before=e2e-list-49'),
    JSON.stringify({ beforeMore, afterMore, listCalls }));
  await moreCtx.close();
}

// (c) 잡 품질 경고: '주의 N건' 칩 → 펼침 목록, 'N페이지'는 리더 이동 링크, 목록 줄 표시(frontend-2).
{
  const warnCtx = await freshContext();
  const injected = [
    '2페이지: single OCR 실패 후 PDF 내장 텍스트 레이어로 복구 (이미지·정밀 레이아웃 제외; RuntimeError: e2e)',
  ];
  await warnCtx.route((url) => url.pathname === '/api/jobs' || url.pathname === `/api/jobs/${jobId}`,
    async (route) => {
      if (route.request().method() !== 'GET') { await route.continue(); return; }
      const res = await route.fetch();
      const data = await res.json();
      if (Array.isArray(data.jobs)) {
        for (const j of data.jobs) if (j.job_id === jobId) j.warnings = injected;
      } else {
        data.warnings = injected;
        data.notices = ['e2e 참고: 페이지 단위로 처리했습니다'];
      }
      await route.fulfill({ response: res, json: data });
    });
  const warnPage = await warnCtx.newPage();
  await warnPage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
  await warnPage.waitForSelector('#job-warnings-chip:not([hidden])', { timeout: 20_000 });
  const chip = await warnPage.evaluate(() => ({
    text: document.getElementById('job-warnings-chip').textContent,
    expanded: document.getElementById('job-warnings-chip').getAttribute('aria-expanded'),
    panelHidden: document.getElementById('job-warnings').hidden,
    listBadge: document.querySelector(`#job-list .job-item.active .ji-warn`)?.textContent || '',
  }));
  check('품질 경고: 헤더 "주의 N건" 칩(접힌 상태)과 목록 줄 표시',
    chip.text === '주의 1건' && chip.expanded === 'false' && chip.panelHidden && chip.listBadge === '주의 1',
    JSON.stringify(chip));
  await warnPage.click('#job-warnings-chip');
  const panel = await warnPage.evaluate(() => ({
    hidden: document.getElementById('job-warnings').hidden,
    items: document.querySelectorAll('#job-warnings-list li').length,
    notices: document.querySelectorAll('#job-notices-list li').length,
    noticesHidden: document.getElementById('job-notices').hidden,
    link: document.querySelector('#job-warnings-list .warning-page-link')?.textContent || '',
  }));
  check('품질 경고: 펼친 목록에 경고·참고를 따로 보인다',
    !panel.hidden && panel.items === 1 && panel.notices === 1 && !panel.noticesHidden && panel.link === '2페이지',
    JSON.stringify(panel));
  await warnPage.screenshot({ path: path.join(OUT, 'job-warnings.png') });
  await warnPage.waitForSelector('#reader-content .reader-rail-page', { timeout: 15_000 });
  await warnPage.click('#job-warnings-list .warning-page-link');
  await warnPage.waitForFunction(() => document.getElementById('reader-page')?.value === '2', null, { timeout: 10_000 })
    .catch(() => {});
  check('품질 경고: "2페이지" 링크가 리더 2페이지로 이동',
    await warnPage.evaluate(() => document.getElementById('reader-page')?.value === '2'));
  await warnCtx.close();
}

// (d) 인용·하이라이트: 카드 경계를 넘는 하이라이트가 DOM을 복제하지 않고, 탭 전환·새로고침
//     뒤에도 남으며, 목록·Markdown 내보내기·삭제가 동작한다(frontend-5/6/10).
if (layoutCap !== 'figure_only') {
  const noteCtx = await freshContext({ acceptDownloads: true });
  const notePage = await noteCtx.newPage();
  await notePage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
  await notePage.waitForSelector('#reader-content .reader-rail-page[data-page="1"] .reader-map-card .reader-map-target',
    { timeout: 20_000 });
  const railShape = () => notePage.evaluate(() => {
    const ids = [...document.querySelectorAll('#reader-content [data-block-id]')].map((n) => n.dataset.blockId);
    return {
      sections: document.querySelectorAll('#reader-content .reader-rail-page').length,
      cards: ids.length,
      duplicates: ids.length - new Set(ids).size,
      marks: document.querySelectorAll('#reader-content mark.reader-highlight').length,
    };
  });
  const before = await railShape();
  const selectAcross = (crossCards) => notePage.evaluate((cross) => {
    const textNode = (root) => {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode: (n) => (n.data.trim() && !n.parentElement.closest('.katex')
          ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP),
      });
      return walker.nextNode();
    };
    const targets = [...document.querySelectorAll(
      '#reader-content .reader-rail-page[data-page="1"] .reader-map-card .reader-map-target')];
    const range = document.createRange();
    if (cross) {
      const a = textNode(targets[0]);
      const b = textNode(targets[targets.length > 1 ? 1 : 0]);
      range.setStart(a, Math.min(2, Math.max(0, a.length - 1)));
      range.setEnd(b, b !== a ? Math.min(6, b.length) : a.length);
    } else {
      range.selectNodeContents(targets[0]); // 카드 하나 전체 — 인용
    }
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.getElementById('reader-content').dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    return { cards: targets.length, text: document.getElementById('reader-selection').textContent };
  }, crossCards);
  const picked = await selectAcross(true);
  await notePage.click('.reader-tools-wrap > summary');
  await notePage.click('#reader-highlight');
  const afterHighlight = await railShape();
  const stored = () => notePage.evaluate((id) => {
    const raw = localStorage.getItem(`uocr-reader-notes-${id}`);
    return raw ? JSON.parse(raw).items : [];
  }, jobId);
  const storedAfterHighlight = await stored();
  check('하이라이트: 카드 경계를 넘어도 섹션·카드·data-block-id를 복제하지 않는다',
    afterHighlight.sections === before.sections && afterHighlight.cards === before.cards
      && afterHighlight.duplicates === 0 && afterHighlight.marks >= (picked.cards > 1 ? 2 : 1),
    JSON.stringify({ before, afterHighlight, picked }));
  check('하이라이트: 잡별 저장소와 목록에 남는다',
    storedAfterHighlight.length === 1 && storedAfterHighlight[0].kind === 'highlight'
      && await notePage.evaluate(() => document.querySelectorAll('#reader-notes-list li').length === 1),
    JSON.stringify(storedAfterHighlight));
  await notePage.click('button[data-tab="qa"]');
  await notePage.click('button[data-tab="reader"]');
  await notePage.waitForTimeout(300);
  const afterTabs = await railShape();
  check('하이라이트: 질문 탭을 다녀와도 레일을 다시 그리지 않아 남는다',
    afterTabs.marks === afterHighlight.marks, JSON.stringify(afterTabs));

  await selectAcross(false);
  await notePage.click('#reader-cite');
  await notePage.waitForFunction(() => document.querySelectorAll('#reader-notes-list li').length === 2,
    null, { timeout: 5_000 }).catch(() => {});
  const citeToast = await notePage.evaluate(() => document.getElementById('toast')?.textContent || '');
  check('인용 저장: 목록에 쌓이고 정직한 안내(내보낼 수 있음)를 보인다',
    (await stored()).length === 2 && /인용을 저장했습니다/.test(citeToast), citeToast);
  await notePage.screenshot({ path: path.join(OUT, 'reader-notes.png') });

  const download = notePage.waitForEvent('download');
  await notePage.click('#reader-notes-export');
  const file = await download;
  const md = readFileSync(await file.path(), 'utf8');
  check('인용·하이라이트: Markdown 파일로 내보낸다',
    file.suggestedFilename().endsWith('.notes.md') && md.startsWith('# ')
      && md.includes('## 인용 (1)') && md.includes('## 하이라이트 (1)'), md.slice(0, 200));

  await notePage.reload({ waitUntil: 'domcontentloaded' });
  await notePage.waitForSelector('#reader-content .reader-map-card', { timeout: 20_000 });
  await notePage.waitForFunction(() => document.querySelectorAll('#reader-content mark.reader-highlight').length > 0,
    null, { timeout: 10_000 }).catch(() => {});
  const reloaded = await notePage.evaluate(() => ({
    notes: document.querySelectorAll('#reader-notes-list li').length,
    badge: document.getElementById('reader-notes-badge')?.textContent || '',
    marks: document.querySelectorAll('#reader-content mark.reader-highlight').length,
  }));
  check('새로고침 뒤에도 목록과 하이라이트가 그대로', reloaded.notes === 2 && reloaded.badge === '저장 2'
    && reloaded.marks >= 1, JSON.stringify(reloaded));

  await notePage.click('.reader-tools-wrap > summary');
  const deleteButtons = notePage.locator('#reader-notes-list li .reader-note-del');
  for (let i = 0; i < 5 && await deleteButtons.count(); i += 1) await deleteButtons.first().click();
  const cleared = await notePage.evaluate((id) => ({
    storage: localStorage.getItem(`uocr-reader-notes-${id}`),
    marks: document.querySelectorAll('#reader-content mark.reader-highlight').length,
    empty: !document.getElementById('reader-notes-empty').hidden,
  }), jobId);
  check('삭제: 저장값·하이라이트·목록이 함께 지워진다',
    cleared.storage === null && cleared.marks === 0 && cleared.empty, JSON.stringify(cleared));
  await noteCtx.close();
}

// (e) health: 프리로드 실패·워커 중지를 '로딩 중'과 구분해 보인다(frontend-8).
{
  const healthCtx = await freshContext();
  await healthCtx.route((url) => url.pathname === '/api/health', (route) => route.fulfill({
    status: 200, contentType: 'application/json',
    body: JSON.stringify({
      ...health, model_loaded: false, worker_alive: false,
      model_load_error: 'E2E: 모델 가중치를 찾을 수 없습니다',
    }),
  }));
  const healthPage = await healthCtx.newPage();
  await healthPage.goto(BASE, { waitUntil: 'domcontentloaded' });
  await healthPage.waitForSelector('#health-badges .badge', { timeout: 10_000 });
  const shown = await healthPage.evaluate(() => ({
    badges: [...document.querySelectorAll('#health-badges .badge')].map((b) => b.textContent),
    notice: !document.getElementById('upload-model-notice').hidden,
    noticeError: document.getElementById('upload-model-notice').classList.contains('is-error'),
    noticeText: document.getElementById('upload-model-notice-text').textContent,
  }));
  await healthPage.screenshot({ path: path.join(OUT, 'health-failure.png') });
  check('health: 로드 실패·작업 처리기 중지 배지와 사유가 담긴 업로드 안내',
    shown.badges.includes('모델 로드 실패') && shown.badges.includes('작업 처리기 중지됨')
      && !shown.badges.some((b) => b.includes('로딩 중')) && shown.notice && shown.noticeError
      && shown.noticeText.includes('모델 가중치를 찾을 수 없습니다'), JSON.stringify(shown));
  await healthCtx.close();
}

// (f) SSE 첫 연결이 비-200(프록시 502)이면 기다리지 않고 바로 상태 폴링으로 강등한다(frontend-1).
//     새 PDF를 하나 더 변환하므로 즉시 끝나는 FakeEngine 하네스에서만 돈다(실 모델은 수 분).
if (health.engine === 'fake') {
  const sseCtx = await freshContext();
  let uploadedId = '';
  let fakedRunning = false;
  await sseCtx.route((url) => /\/api\/jobs\/[^/]+\/events$/.test(url.pathname), (route) => route.fulfill({
    status: 502, contentType: 'text/plain', body: 'bad gateway (e2e)',
  }));
  await sseCtx.route((url) => /^\/api\/jobs\/[^/]+$/.test(url.pathname), async (route) => {
    const res = await route.fetch();
    const data = await res.json().catch(() => null);
    // 새로 올린 잡의 첫 상세 조회만 '실행 중'으로 보여 실시간 스트림 경로를 강제한다
    // (FakeEngine은 즉시 끝난다). 업로드 응답 id를 기다리면 경합이 생겨 기존 잡이 아닌지로 고른다.
    if (data && route.request().method() === 'GET' && !fakedRunning
        && data.job_id && data.job_id !== jobId) {
      fakedRunning = true;
      uploadedId = data.job_id;
      await route.fulfill({ response: res, json: {
        ...data, status: 'running', result: null,
        progress: { phase: 'ocr', current_page: 1, total_pages: 2 },
      } });
      return;
    }
    await route.fulfill({ response: res });
  });
  const ssePage = await sseCtx.newPage();
  await ssePage.goto(BASE, { waitUntil: 'networkidle' });
  await ssePage.setInputFiles('#file-input', PDF);
  await ssePage.waitForTimeout(300);
  await ssePage.click('#upload-btn');
  const degraded = await ssePage.waitForFunction(
    () => /상태 폴링으로 전환/.test(document.getElementById('stream-pane')?.textContent || ''),
    null, { timeout: 4_000 }).then(() => true).catch(() => false);
  await ssePage.waitForSelector('#result-section:not([hidden])', { timeout: 30_000 }).catch(() => {});
  const finished = await ssePage.evaluate(() => !document.getElementById('result-section').hidden);
  check('SSE 첫 연결 502: 바로 폴링으로 강등해 결과까지 이어진다', degraded && finished && fakedRunning,
    JSON.stringify({ degraded, finished, fakedRunning, uploadedId }));
  await sseCtx.close();
}

// ── 6) 선택 확장: mock-provider 번역 → PDF 다운로드 + 페이지 Q&A ─────────
// mock-full-flow.e2e.mjs가 로컬 번역/OpenAI Responses mock과 FakeEngine을 띄운
// 경우에만 실행한다. 실 API·과금·외부 전송 없이 브라우저의 마지막 제품 경로를 고정.
if (VERIFY_MOCK_LLM) {
  await page.waitForSelector('#translate-btn:not([hidden])');
  await page.click('button[data-tab="reader"]');
  await page.waitForSelector('#reader-translate-btn:not([hidden])');
  let translatePosts = 0;
  let markTranslateStarted;
  let releaseTranslate;
  const translateStarted = new Promise((resolve) => { markTranslateStarted = resolve; });
  const translateGate = new Promise((resolve) => { releaseTranslate = resolve; });
  const translateRoute = async (route) => {
    if (route.request().method() !== 'POST') { await route.continue(); return; }
    translatePosts += 1;
    if (translatePosts === 1) {
      markTranslateStarted();
      await translateGate;
    }
    await route.continue();
  };
  await page.route('**/api/jobs/*/translate', translateRoute);
  // 번역 참고 사항: translate/state가 report.json의 warnings를 함께 준다(p2-w1). 실서버 응답에
  // 문장 하나를 덧붙여 결과 화면의 흐린 접이식 목록을 확인한다(번역이 끝난 state에만).
  const E2E_TRANSLATE_NOTE = 'E2E 참고: 기존 캐시 0건 적중 — 유닛 전량을 새로 번역했습니다';
  const stateRoute = async (route) => {
    const res = await route.fetch();
    const data = await res.json().catch(() => null);
    if (!data || data.status !== 'done') { await route.fulfill({ response: res }); return; }
    const warnings = Array.isArray(data.warnings) ? data.warnings : [];
    await route.fulfill({ response: res, json: { ...data, warnings: [...warnings, E2E_TRANSLATE_NOTE] } });
  };
  await page.route('**/api/jobs/*/translate/state*', stateRoute);
  let available = false;
  const healthRoute = (route) => route.fulfill({
    status: 200, contentType: 'application/json',
    body: JSON.stringify({ ...health, translate_available: available }),
  });
  await page.route('**/api/health', healthRoute);
  try {
    await page.click('#translate-btn');
    await translateStarted;
    // 실제 health 로더로 프로바이더 false→true 전이를 재현한다. POST를 기다리는
    // 중에도 가용성 갱신이 메인/리더 버튼 잠금을 풀면 중복 번역이 시작될 수 있다.
    await page.evaluate(async () => { const { loadHealth } = await import('/js/health.js'); await loadHealth(); });
    available = true;
    await page.evaluate(async () => { const { loadHealth } = await import('/js/health.js'); await loadHealth(); });
    const pendingControls = await page.evaluate(() => ({
      main: document.getElementById('translate-btn').disabled,
      reader: document.getElementById('reader-translate-btn').disabled,
    }));
    // 강제 pointer 클릭은 Playwright의 disabled 대기만 건너뛴다. 브라우저는 여전히
    // disabled 버튼의 click을 억제해야 한다 — 실제 사용자 입력 경로를 그대로 탄다.
    await page.click('#reader-translate-btn', { force: true });
    await page.click('#translate-btn', { force: true });
    check('mock 번역: health 갱신 중 메인·리더 버튼 잠금 + 중복 POST 차단',
      pendingControls.main && pendingControls.reader && translatePosts === 1,
      `${JSON.stringify(pendingControls)}, posts=${translatePosts}`);
  } finally {
    releaseTranslate();
    await page.unroute('**/api/health', healthRoute);
  }
  await page.waitForFunction(() => {
    const pdf = document.getElementById('dl-pdf');
    const html = document.getElementById('dl-doc-ko');
    const ko = document.getElementById('lang-ko');
    return pdf && !pdf.hidden && html && !html.hidden && ko && !ko.parentElement.hidden;
  }, null, { timeout: 60_000 });
  await page.unroute('**/api/jobs/*/translate', translateRoute);
  check('mock 번역: 번역 완료까지 시작 요청은 한 번', translatePosts === 1, `posts=${translatePosts}`);
  check('mock 번역: 완료 후 한국어 토글', await page.evaluate(() => !document.getElementById('lang-toggle').hidden));
  // 원문 그대로 남은 문단의 개수·사유를 사용자가 볼 수 있어야 한다(서버 report.json →
  // translate/state 병합). 사유별 집계가 도착하면 title에 근거가 들어온다.
  await page.waitForFunction(() => {
    const chip = document.getElementById('translate-summary');
    return chip && !chip.hidden && (chip.title || '').includes('문단');
  }, null, { timeout: 15_000 }).catch(() => {});
  const keptSummary = await page.evaluate(() => {
    const chip = document.getElementById('translate-summary');
    return { hidden: chip.hidden, text: (chip.textContent || '').trim(), title: chip.title || '' };
  });
  check('mock 번역: 원문 유지/건너뜀 요약을 사용자에게 노출',
    !keptSummary.hidden && keptSummary.text.length > 0 && keptSummary.title.includes('문단'),
    JSON.stringify(keptSummary));
  await page.waitForFunction(() => !document.getElementById('translate-warnings')?.hidden,
    null, { timeout: 15_000 }).catch(() => {});
  const translateNotes = await page.evaluate(() => {
    const box = document.getElementById('translate-warnings');
    const visibleBefore = !!box && !box.hidden && box.getBoundingClientRect().height > 0;
    const openBefore = !!box && box.open;
    document.getElementById('translate-warnings-summary')?.click(); // 펼친다
    const items = [...document.querySelectorAll('#translate-warnings-list li')].map((li) => li.textContent);
    return {
      visibleBefore, openBefore, open: !!box && box.open, items,
      summary: document.getElementById('translate-warnings-summary')?.textContent || '',
      color: items.length ? getComputedStyle(document.querySelector('#translate-warnings-list li')).color : '',
      title: document.getElementById('translate-summary')?.title || '',
    };
  });
  check('번역 참고 사항: translate/state warnings가 번역 요약 아래 흐린 접이식 목록으로 보인다',
    translateNotes.visibleBefore && !translateNotes.openBefore && translateNotes.open
      && translateNotes.items.includes(E2E_TRANSLATE_NOTE) && /번역 참고 사항 \d+건/.test(translateNotes.summary)
      && translateNotes.title.includes('번역 참고 사항'),
    JSON.stringify(translateNotes));
  await page.unroute('**/api/jobs/*/translate/state*', stateRoute);
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
  /* 한국어 보기의 일시 503(예열 빌드·내보내기 대기열)은 "번역본 없음"이 아니다 — 진행을
     보이며 Retry-After만큼 기다렸다 다시 묻고, 전역 언어를 원문으로 되돌리지 않는다
     (frontend-9 · gap1-metal-real-e2e-7). 리더(/html)와 레이아웃 탭(/layout) 둘 다 본다. */
  {
    const busyCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const busyHits = { html: 0, layout: 0 };
    const busyOnce = (kind) => async (route) => {
      busyHits[kind] += 1;
      if (busyHits[kind] === 1) {
        await route.fulfill({
          status: 503, contentType: 'application/json', headers: { 'Retry-After': '1' },
          body: JSON.stringify({ detail: 'PDF 내보내기 대기열이 가득 찼습니다 — 잠시 후 다시 시도하세요' }),
        });
        return;
      }
      await route.continue();
    };
    await busyCtx.route((url) => url.pathname === `/api/jobs/${jobId}/html`
      && url.searchParams.get('lang') === 'ko', busyOnce('html'));
    await busyCtx.route((url) => url.pathname === `/api/jobs/${jobId}/layout`
      && url.searchParams.get('lang') === 'ko', busyOnce('layout'));
    const busyPage = await busyCtx.newPage();
    await busyPage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
    await busyPage.waitForSelector('#lang-toggle:not([hidden])', { timeout: 20_000 });
    await busyPage.click('#lang-ko');
    const readerWaited = await busyPage.waitForFunction(
      () => /한국어 본문 준비 중… 1초 뒤 다시 시도합니다 \(1\/4\)/
        .test(document.getElementById('reader-content')?.textContent || ''),
      null, { timeout: 10_000 }).then(() => true).catch(() => false);
    await busyPage.waitForSelector('#reader-content .reader-rail-page', { timeout: 20_000 });
    await busyPage.click('button[data-tab="doclayout"]');
    const layoutWaited = await busyPage.waitForFunction(
      () => /한국어 레이아웃 준비 중/.test(document.getElementById('doclayout-body')?.textContent || ''),
      null, { timeout: 10_000 }).then(() => true).catch(() => false);
    await busyPage.waitForSelector('#doclayout-body .layout-canvas', { timeout: 30_000 }).catch(() => {});
    const busyState = await busyPage.evaluate(() => ({
      ko: document.getElementById('lang-ko').getAttribute('aria-pressed'),
      canvas: !!document.querySelector('#doclayout-body .layout-canvas'),
      toast: document.getElementById('toast')?.textContent || '',
    }));
    check('한국어 보기 503: 진행을 보이며 재시도하고 한국어를 유지한다',
      readerWaited && layoutWaited && busyState.ko === 'true' && busyState.canvas
        && !/원문을 표시합니다/.test(busyState.toast) && busyHits.html >= 2 && busyHits.layout >= 2,
      JSON.stringify({ readerWaited, layoutWaited, busyState, busyHits }));
    await busyCtx.close();
  }

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
  // 영어 문단을 만나는 곳은 뷰어다 — 요약은 뷰어 툴바에서도 보여야 한다.
  check('mock 번역 뷰어: 원문 유지/건너뜀 요약을 뷰어에서도 노출', await page.evaluate(() => {
    const chip = document.getElementById('viewer-translate-summary');
    return !!chip && !chip.hidden && getComputedStyle(chip).display !== 'none'
      && (chip.textContent || '').trim().length > 0;
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
  // 내려받은 standalone HTML은 서버 CSP 헤더 없이 디스크에서 열린다 — 파일 안의 meta CSP가
  // 바깥 출처를 막고도 수식(KaTeX·data: 폰트)과 페이지 이미지(data:)는 그대로 그려야 한다.
  if (htmlPath) {
    // download.path()는 확장자 없는 임시 파일이라 file://로 열면 평문으로 보인다 — .html로 남긴다.
    const standalonePath = path.join(OUT, 'standalone.ko.html');
    await htmlDownload.saveAs(standalonePath);
    const fileCtx = await freshContext();
    const filePage = await fileCtx.newPage();
    const fileHits = [];
    filePage.on('requestfinished', (r) => { if (!/^(file|data):/.test(r.url())) fileHits.push(r.url()); });
    await filePage.addInitScript(() => {
      window.__cspViolations = [];
      document.addEventListener('securitypolicyviolation', (e) => {
        window.__cspViolations.push(`${e.violatedDirective} ${e.blockedURI}`);
      });
    });
    await filePage.goto(pathToFileURL(standalonePath).href, { waitUntil: 'load' });
    const standalone = await filePage.evaluate(async () => {
      const imgs = [...document.images];
      await Promise.all(imgs.map((img) => img.decode().catch(() => {})));
      const before = window.__cspViolations.slice();
      const beacon = new Image();
      beacon.src = 'http://127.0.0.1:9/e2e-standalone-beacon.png';
      document.body.appendChild(beacon);
      await new Promise((resolve) => setTimeout(resolve, 300));
      return {
        csp: document.querySelector('meta[http-equiv="Content-Security-Policy"]')?.content || '',
        referrer: document.querySelector('meta[name="referrer"]')?.content || '',
        katex: document.querySelectorAll('.katex').length,
        math: document.querySelectorAll('.math-inline,.math-display').length,
        imgs: imgs.length,
        loaded: imgs.filter((img) => img.complete && img.naturalWidth > 0).length,
        before,
        after: window.__cspViolations.slice(before.length),
      };
    });
    check('standalone HTML: meta CSP 아래에서도 수식·페이지 이미지를 그대로 그린다',
      /default-src 'none'/.test(standalone.csp) && standalone.referrer === 'no-referrer'
        && standalone.math > 0 && standalone.katex >= standalone.math
        && standalone.imgs > 0 && standalone.loaded === standalone.imgs && standalone.before.length === 0,
      JSON.stringify(standalone));
    check('standalone HTML: 본문에 끼어든 바깥 이미지는 CSP가 막고 요청도 나가지 않는다',
      standalone.after.some((v) => v.startsWith('img-src') && v.includes('e2e-standalone-beacon'))
        && fileHits.length === 0,
      JSON.stringify({ after: standalone.after, fileHits }));
    await fileCtx.close();
  }

  // PDF 생성 리포트: 헤더에는 숫자만 실린다 — 원문 보존 사유·스캔 원문 지움·주의 문장은 같은
  // 빌드의 JSON 리포트(GET /pdf/report, p2-w1)에서 온다. report.py report() 모양 그대로 돌려준다.
  const pdfReportRoute = (route) => route.fulfill({
    status: 200, contentType: 'application/json',
    body: JSON.stringify({
      job_id: jobId, lang: 'ko', format_version: 10, replaced: 9, kept: 4, relocated: 0,
      table_cells_replaced: 0, listing_lines_replaced: 2, raster_blocks_erased: 3,
      specialist_kept: {}, kept_reasons: { listing_line_unaligned: 3, unchanged: 1 }, warning_count: 2,
      warnings: [
        'p1: 블록 2의 원문이 이미지 픽셀이라 지우지 못함 — 번역이 이미지 속 글자와 겹쳐 보일 수 있음',
        'p2: 블록 1의 3줄 교체 생략(원문 줄 위치 정렬 실패) — 그 줄만 원문 보존',
      ],
    }),
  });
  await page.route('**/api/jobs/*/pdf/report*', pdfReportRoute);
  const downloadPromise = page.waitForEvent('download');
  await page.click('#dl-pdf');
  const download = await downloadPromise;
  const downloadedPath = await download.path();
  const first = downloadedPath ? readFileSync(downloadedPath).subarray(0, 5).toString('ascii') : '';
  check('mock PDF: 브라우저 다운로드 완료', download.suggestedFilename().endsWith('.ko.pdf'));
  check('mock PDF: 실제 PDF 바이트', first === '%PDF-');
  check('mock PDF: 생성 리포트 토스트', await page.evaluate(() =>
    (document.getElementById('toast')?.textContent || '').includes('PDF 생성 완료')));
  await page.waitForFunction(() => !document.getElementById('pdf-report')?.hidden,
    null, { timeout: 10_000 }).catch(() => {});
  const pdfReport = await page.evaluate(() => {
    document.getElementById('pdf-report-summary')?.click();
    return {
      hidden: document.getElementById('pdf-report')?.hidden,
      summary: document.getElementById('pdf-report-summary')?.textContent || '',
      items: [...document.querySelectorAll('#pdf-report-list li')].map((li) => li.textContent),
      warn: document.querySelectorAll('#pdf-report-list li.is-warn').length,
      toast: document.getElementById('toast')?.textContent || '',
    };
  });
  check('PDF 생성 리포트: 스캔 원문 지움·줄 정렬 실패 보존·새 주의 문장을 보인다',
    pdfReport.hidden === false && pdfReport.summary.includes('주의 2건') && pdfReport.warn === 2
      && pdfReport.items.some((t) => t.includes('스캔(이미지) 원문 3개 블록'))
      && pdfReport.items.some((t) => t.includes('원문 줄 위치 정렬 실패(그 줄만 원문) 3'))
      && pdfReport.items.some((t) => t.startsWith('1페이지:') && t.includes('이미지 픽셀이라 지우지 못함'))
      && pdfReport.toast.includes('스캔 원문 3개 블록 지움'),
    JSON.stringify(pdfReport));
  await page.unroute('**/api/jobs/*/pdf/report*', pdfReportRoute);

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
  const failedCalls = []; // {at, batch, limit} — 간격/밀도 단정용 (상한 형태)
  const failureStartedAt = Date.now();
  await failureCtx.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    const batchAlignment = url.pathname.endsWith('/viewer/pages')
      && (url.searchParams.get('include') || '').split(',').includes('alignment');
    const singleAlignment = url.pathname.endsWith('/alignment');
    if (url.pathname.endsWith('/html')) failureHtmlCalls += 1;
    if (batchAlignment || singleAlignment) {
      failedRequestDetails.push(`${Date.now() - failureStartedAt}ms ${batchAlignment ? 'batch' : 'single'} ${url.search}`);
      failedCalls.push({
        at: Date.now() - failureStartedAt,
        batch: batchAlignment,
        limit: Number(url.searchParams.get('limit')) || 0,
      });
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
  // 횟수 상한만으로는 "지연 없이 즉시 두 번 재시도"(backoff가 0으로 붕괴)를 잡지 못한다.
  // 간격 자체를 상한/하한 형태로 본다 — 정확 일치는 러너 지터로 깨지므로 쓰지 않는다.
  //  · span(첫 호출 → 마지막 호출)이 0.7초 이상  ⇒ 재시도가 실제로 지연됐다(0.8s 타이머).
  //  · span이 3.5초 이하                         ⇒ 재시도가 유한 시간에 끝났다.
  //  · 가장 붐비는 400ms 창의 호출 수가 상한 이하 ⇒ busy-loop가 아니다. 한 창은 배치 1 +
  //    단건 limit개이고, 하이드레이션 트리거가 겹치면 두 벌이 올 수 있어 3배까지 허용한다.
  const stamps = failedCalls.map((c) => c.at);
  const span = stamps.length > 1 ? stamps[stamps.length - 1] - stamps[0] : 0;
  const batchLimit = Math.max(1, ...failedCalls.map((c) => (c.batch ? c.limit : 0)));
  const burstCap = 3 * (1 + batchLimit);
  const densest = stamps.reduce(
    (max, t) => Math.max(max, stamps.filter((x) => x >= t && x < t + 400).length), 0);
  check('정렬 API 일시 장애: bounded backoff — 재시도 간격이 실제로 벌어진다',
    span >= 700 && span <= 3500 && densest <= burstCap,
    `span=${span}ms, densest(400ms)=${densest}/${burstCap}, calls=${stamps.length}`);
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

  /* 429(상한 초과) 잠금은 잡이 아니라 클라이언트 단위다 — 잡을 바꿔도 서버는 계속
     429를 준다. 그런데 잡 전환은 번역 버튼을 되살렸다: 눌리기만 하고 요청은 나가지
     않는 버튼. 표시 상태가 잠금과 일치하는지 본다. (mock 하네스에서만 — 새 잡을
     하나 더 변환해야 잡 전환 경로를 탈 수 있다.) */
  const lockCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  let lockTranslatePosts = 0;
  await lockCtx.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === 'POST' && url.pathname.endsWith('/translate')) {
      lockTranslatePosts += 1;
      await route.fulfill({
        status: 429,
        contentType: 'application/json',
        headers: { 'Retry-After': '30' },
        body: JSON.stringify({ detail: '요청이 너무 잦습니다 — 잠시 후 다시 시도하세요' }),
      });
      return;
    }
    await route.continue();
  });
  const lockPage = await lockCtx.newPage();
  await lockPage.goto(BASE, { waitUntil: 'networkidle' });
  await lockPage.setInputFiles('#file-input', PDF);
  await lockPage.waitForTimeout(300);
  await lockPage.click('#upload-btn');
  await lockPage.waitForSelector('#translate-btn:not([hidden])', { timeout: 120_000 });
  const lockedHash = await lockPage.evaluate(() => location.hash);
  await lockPage.click('#translate-btn');
  await lockPage.waitForFunction(
    () => (document.getElementById('toast')?.textContent || '').includes('30초 후'),
    null, { timeout: 15_000 });
  const lockedAfter429 = await lockPage.evaluate(
    () => document.getElementById('translate-btn').disabled);
  // 다른 잡(이미 번역된 잡)으로 갔다가 돌아온다 — 해시 전환 = openJob 경로(새로고침 아님)
  await lockPage.evaluate((h) => { location.hash = h; }, jobHash);
  await lockPage.waitForSelector('#lang-toggle:not([hidden])', { timeout: 20_000 });
  await lockPage.evaluate((h) => { location.hash = h; }, lockedHash);
  await lockPage.waitForSelector('#translate-btn:not([hidden])', { timeout: 20_000 });
  await lockPage.waitForTimeout(500);
  const lockedAfterSwitch = await lockPage.evaluate(() => ({
    disabled: document.getElementById('translate-btn').disabled,
    readerCta: document.getElementById('reader-translate-btn').disabled,
  }));
  check('429 잠금: 잡을 바꿨다 돌아와도 번역 버튼이 잠금과 같은 상태로 남는다',
    lockedAfter429 === true && lockedAfterSwitch.disabled === true
      && lockedAfterSwitch.readerCta === true && lockTranslatePosts === 1,
    `after429=${lockedAfter429}, ${JSON.stringify(lockedAfterSwitch)}, posts=${lockTranslatePosts}`);
  await lockCtx.close();
}

/* 개별(sync off) 모드에서 레일만 굴릴 때 좌측 스테이지의 bbox 오버레이가 무제한
   쌓이지 않는가. 정렬 캐시가 비어 있는 새 세션에서만 드러나므로 별도 컨텍스트로 본다.
   좌측 면은 1쪽에 세워 두고 레일만 문서 끝까지 보낸다 — 뒤늦게 도착하는 먼 페이지
   정렬이 좌측 keep 창(±6) 밖까지 버튼을 붙이면, 좌측이 움직이지 않는 동안에는
   걷어내는 경로(hydrateReaderPages)가 돌지 않아 그대로 누적된다(Tab 순환·히트 테스트 저하). */
if (layoutCap !== 'figure_only') {
  const railCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const railPage = await railCtx.newPage();
  await railPage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
  await railPage.waitForSelector('#reader-content .reader-map-card', { timeout: 20_000 });
  await railPage.$eval('#reader-content', (rail) => {
    rail.style.flex = '0 0 180px';
    rail.style.height = '180px';
    rail.style.minHeight = '0';
  });
  await railPage.click('#reader-sync'); // 개별 모드 — 좌측 면은 이제 따라오지 않는다
  await railPage.waitForTimeout(200);
  for (let step = 1; step <= 4; step += 1) {
    // 문서 끝까지 나눠 굴린다 — 스크롤마다 새 정렬 창이 도착한다.
    await railPage.$eval('#reader-content', (rail, ratio) => {
      rail.scrollTop = Math.round((rail.scrollHeight - rail.clientHeight) * ratio);
    }, step / 4);
    await railPage.waitForTimeout(700);
  }
  const railGrowth = await railPage.evaluate(() => {
    const current = Number(document.getElementById('reader-page')?.value) || 1;
    const stage = [...document.querySelectorAll('#reader-page-stage .reader-page[data-page]')];
    const railSections = [...document.querySelectorAll('#reader-content .reader-rail-page[data-page]')];
    const boxPages = stage
      .filter((section) => section.querySelector('.reader-map-box'))
      .map((section) => Number(section.dataset.page));
    return {
      current,
      total: stage.length,
      pending: railSections
        .filter((section) => section.querySelector('.reader-rail-pending'))
        .map((section) => Number(section.dataset.page)),
      filled: railSections
        .filter((section) => !section.querySelector('.reader-rail-pending'))
        .map((section) => Number(section.dataset.page)),
      boxes: document.querySelectorAll('#reader-page-stage .reader-map-box').length,
      boxPages,
      outside: boxPages.filter((n) => Math.abs(n - current) > 6),
    };
  });
  // 좌측 면은 1쪽에 멈춰 있으므로 좌측 하이드레이션 창은 문서 앞부분만 덮는다.
  // 레일이 스스로 창을 로드하지 않으면 뒷부분은 영원히 '불러오는 중…'으로 남는다.
  check('개별 모드 레일 스크롤: 좌측이 멈춰 있어도 레일이 스스로 문서 끝까지 로드한다',
    railGrowth.current === 1 && railGrowth.pending.length === 0
      && railGrowth.filled.includes(railGrowth.total),
    JSON.stringify({ current: railGrowth.current, total: railGrowth.total, pending: railGrowth.pending }));
  check('개별 모드 레일 스크롤: 좌측 면이 멈춰 있어도 bbox 오버레이가 keep 창 안에만 남는다',
    railGrowth.current === 1 && railGrowth.outside.length === 0,
    `total=${railGrowth.total}, boxes=${railGrowth.boxes}, pages=[${railGrowth.boxPages}]`);
  await railCtx.close();
}

/* 레일 흐름형 본문(정렬 좌표 없이 /html로 그린 본문)의 그림은 뒤늦게 로드되며
   섹션 높이를 바꾼다. 연동 모드는 원문 면 눈높이가 되잡아 주지만 개별 모드에는
   기준이 없어 읽던 문단이 그림 높이만큼 아래로 밀린다 — 마지막 레일 앵커로 되돌린다.
   정렬 API를 계속 503으로 막아 흐름형 본문을 강제하고, 본문 그림만 늦게 준다. */
{
  const jobId = jobHash.replace(/^#/, '');
  let pagePng = null;
  try {
    const res = await fetch(`${BASE}/api/jobs/${jobId}/page/1`);
    if (res.ok) pagePng = Buffer.from(await res.arrayBuffer()); // 실제 페이지 PNG = 충분히 큰 그림
  } catch { /* 아래에서 건너뛴다 */ }
  const lateCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await lateCtx.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    const alignment = url.pathname.endsWith('/alignment')
      || (url.pathname.endsWith('/viewer/pages')
        && (url.searchParams.get('include') || '').split(',').includes('alignment'));
    if (alignment) {
      await route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
      return;
    }
    if (pagePng && url.pathname.includes('/files/images/')) {
      await new Promise((resolve) => setTimeout(resolve, 2_500)); // 본문 렌더보다 늦게 도착
      await route.fulfill({ status: 200, contentType: 'image/png', body: pagePng });
      return;
    }
    await route.continue();
  });
  const latePage = await lateCtx.newPage();
  const railFocus = () => latePage.evaluate(() => {
    const rail = document.getElementById('reader-content');
    const y = rail.scrollTop + rail.clientHeight * 0.28;
    const origin = rail.getBoundingClientRect().top - rail.scrollTop;
    let found = null;
    for (const section of rail.querySelectorAll('.reader-rail-page')) {
      const top = section.getBoundingClientRect().top - origin;
      if (top <= y) found = { page: Number(section.dataset.page), top: Math.round(top), offset: Math.round(y - top) };
    }
    return { ...found, scrollTop: rail.scrollTop, height: rail.scrollHeight };
  });
  await latePage.goto(`${BASE}/${jobHash}`, { waitUntil: 'domcontentloaded' });
  await latePage.waitForSelector('.reader-rail-retry-note', { timeout: 20_000 });
  await latePage.$eval('#reader-content', (rail) => {
    rail.style.flex = '0 0 180px';
    rail.style.height = '180px';
    rail.style.minHeight = '0';
  });
  await latePage.click('#reader-sync'); // 개별 모드
  await latePage.$eval('#reader-content', (rail) => {
    rail.scrollTop = Math.round((rail.scrollHeight - rail.clientHeight) * 0.8);
  });
  await latePage.waitForTimeout(400);
  const beforeLate = await railFocus();
  const loaded = await latePage.waitForFunction(() =>
    [...document.querySelectorAll('#reader-content img')].some((i) => i.complete && i.naturalHeight > 0),
  null, { timeout: 20_000 }).then(() => true).catch(() => false);
  await latePage.waitForTimeout(500);
  const afterLate = await railFocus();
  // 그림이 위쪽에서 자라난 경우에만(밴드 top이 내려간 경우) 밀림을 볼 수 있다.
  const grewAbove = loaded && afterLate.top > beforeLate.top + 8;
  check('개별 모드: 레일 본문 그림이 늦게 로드돼도 읽던 자리가 밀리지 않는다',
    !grewAbove || (afterLate.page === beforeLate.page
      && Math.abs(afterLate.offset - beforeLate.offset) <= 24),
    `loaded=${loaded}, grewAbove=${grewAbove}, ${JSON.stringify(beforeLate)} → ${JSON.stringify(afterLate)}`);
  await lateCtx.close();
}

check('프로덕션 뷰어 포함 콘솔 에러/HTTP 4xx·5xx 없음',
  errors.length === 0, errors.slice(0, 5).join(' | '));
await browser.close();

console.log(failures.length ? `\n${failures.length}개 실패` : '\n전부 통과');
process.exit(failures.length ? 1 : 0);
