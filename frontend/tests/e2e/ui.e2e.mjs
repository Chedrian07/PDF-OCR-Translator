// 실브라우저 E2E — 실행 중인 백엔드가 필요하다 (단위 테스트 러너에 포함되지 않음).
//
//   E2E_BASE_URL=http://127.0.0.1:8002 npm run test:e2e   (기본 8000)
//
// 검증 플로우 (엔진 불문 — health capability로 분기):
//   1) 업로드 → 변환 완료 → 미리보기에 텍스트·표·이미지·KaTeX 수식 렌더
//   2) HTML 다운로드(document.html) — 자립형(base64 이미지·서버 참조 없음)
//   3) 레이아웃 탭 — figure_only 엔진이면 안내 카드, full이면 캔버스
//   4) Markdown 탭 본문 존재, 다크 테마 렌더
// 실패 시 exit 1. 스크린샷은 shots/(git 무시)에 남는다.
import { chromium } from 'playwright';
import { mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.E2E_BASE_URL || 'http://127.0.0.1:8000';
const PDF = path.join(HERE, 'fixtures', 'sample.pdf');
const OUT = path.join(HERE, 'shots');
const TIMEOUT_S = Number(process.env.E2E_TIMEOUT_S || 300); // 콜드 모델 로딩 감안
const VERIFY_MOCK_LLM = process.env.E2E_VERIFY_MOCK_LLM === '1';
mkdirSync(OUT, { recursive: true });

const failures = [];
function check(name, ok, detail = '') {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(name);
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

// ── 1.5) 읽기 탭 — 완료된 잡의 기본 뷰 ──────────────────────────────────
await page.waitForTimeout(1200);
const reader = await page.evaluate(() => ({
  active: !!document.querySelector('button[data-tab="reader"].active'),
  textLen: (document.getElementById('reader-content')?.innerText || '').length,
  img: !!document.getElementById('reader-image')?.getAttribute('src'),
  total: document.getElementById('reader-total')?.textContent || '',
  cards: document.querySelectorAll('#reader-content .reader-map-card').length,
  boxes: document.querySelectorAll('#reader-map-overlay .reader-map-box').length,
}));
check('읽기 탭: 완료 시 기본 활성', reader.active, JSON.stringify(reader));
check('읽기 탭: 현재 페이지 본문 렌더', reader.textLen > 20);
check('읽기 탭: 페이지 이미지 로드', reader.img);
if (layoutCap !== 'figure_only') {
  check('읽기 탭: 원문 bbox와 텍스트 블록 수 일치',
    reader.cards > 0 && reader.cards === reader.boxes, JSON.stringify(reader));
  await page.locator('#reader-content .reader-map-card').first().click();
  check('읽기 탭: 번역/본문 클릭 → 원문 bbox 활성', await page.evaluate(() => {
    const card = document.querySelector('#reader-content .reader-map-card.is-active');
    const box = document.querySelector('#reader-map-overlay .reader-map-box.is-active');
    return !!card && !!box && card.dataset.blockId === box.dataset.blockId;
  }));
}
await page.screenshot({ path: path.join(OUT, 'reader.png') });

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
  check('mock 번역: PDF 버튼 노출', await page.evaluate(() => !document.getElementById('dl-pdf').hidden));
  await page.click('button[data-tab="reader"]');
  await page.waitForFunction(() =>
    document.querySelectorAll('#reader-content .reader-map-card').length > 0);
  check('mock 번역: 오른쪽 한국어 rail + 왼쪽 원문 고정', await page.evaluate(() => {
    const text = document.getElementById('reader-content')?.innerText || '';
    const image = document.getElementById('reader-image')?.getAttribute('src') || '';
    return /[가-힣]/.test(text) && !image.includes('lang=ko');
  }));

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
}

check('콘솔 에러/4xx 없음', errors.length === 0, errors.slice(0, 5).join(' | '));
await browser.close();

console.log(failures.length ? `\n${failures.length}개 실패` : '\n전부 통과');
process.exit(failures.length ? 1 : 0);
