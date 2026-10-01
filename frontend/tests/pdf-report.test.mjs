// PDF 생성 리포트 — 대조 PDF 다운로드 뒤 원문 보존 사유·스캔 원문 픽셀 지움·주의 문장을 보인다.
//
//   node --test frontend/tests/pdf-report.test.mjs
//
// 서버 계약(backend/app/pipeline/pdf_export/report.py PdfExportResult.report()):
//   {format_version, replaced, kept, relocated, table_cells_replaced, listing_lines_replaced,
//    raster_blocks_erased, specialist_kept, kept_reasons, warning_count, warnings(앞 50건)}
// 다운로드 응답 헤더(X-UOCR-PDF-*)에는 숫자만 실린다 — 사유·문장은 같은 빌드의 JSON 리포트
// (GET /api/jobs/{id}/pdf/report?lang=)에서 온다. 리포트가 없는 서버(404)면 헤더 요약 그대로다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  PDF_REPORT_MAX_WARNINGS, pdfKeptReasonLabel, pdfReportDetails, pdfReportMessage, pdfReportUrl,
} from '../js/core.js';
import { EL_IDS, el, state } from '../js/state.js';
import { applyPdfExport, downloadPdfWithReport } from '../js/reader.js';
import { installFakeDom, mount } from './helpers/fake-dom.mjs';

const RASTER_WARNING = 'p2: 블록 3의 원문이 이미지 픽셀이라 지우지 못함 — 번역이 이미지 속 글자와 겹쳐 보일 수 있음';
const UNALIGNED_WARNING = 'p4: 블록 1의 3줄 교체 생략(원문 줄 위치 정렬 실패) — 그 줄만 원문 보존';

// report.py report()가 만드는 그대로의 모양
const REPORT = {
  format_version: 10,
  replaced: 12,
  kept: 4,
  relocated: 1,
  table_cells_replaced: 3,
  listing_lines_replaced: 5,
  raster_blocks_erased: 2,
  specialist_kept: { figure_text: 1, reference: 2 },
  kept_reasons: { listing_line_unaligned: 3, unchanged: 1 },
  warning_count: 2,
  warnings: [RASTER_WARNING, UNALIGNED_WARNING],
};

test('pdfReportDetails: 스캔 원문 지움·줄 정렬 실패 보존·새 주의 문장을 보인다', () => {
  const d = pdfReportDetails(REPORT);
  assert.equal(d.message,
    'PDF 생성 완료: 번역 12개 블록 · 표 3개 셀 · 스캔 원문 2개 블록 지움 · 충돌 없이 1개 재배치'
    + ' · 원문 4개 보존 · 전문 조판 3개 원형 보존 · 주의 2건');
  assert.equal(d.tone, 'warn');
  assert.deepEqual(d.lines, [
    '스캔(이미지) 원문 2개 블록은 픽셀을 바탕색으로 덮고 번역을 넣었습니다',
    '목록·코드 5줄은 원문 줄 위치에 맞춰 조판했습니다',
    '원문 보존 사유 — 원문 줄 위치 정렬 실패(그 줄만 원문) 3 · 번역문이 원문과 같음 1',
  ]);
  assert.deepEqual(d.warnings, [
    '2페이지: 블록 3의 원문이 이미지 픽셀이라 지우지 못함 — 번역이 이미지 속 글자와 겹쳐 보일 수 있음',
    '4페이지: 블록 1의 3줄 교체 생략(원문 줄 위치 정렬 실패) — 그 줄만 원문 보존',
  ]);
  assert.equal(d.moreWarnings, 0);
});

test('pdfReportDetails: 깨끗한 내보내기는 상세 없이 요약만, 비정상 값은 0으로', () => {
  const clean = pdfReportDetails({ format_version: 10, replaced: 7, kept: 0, kept_reasons: {}, warnings: [] });
  assert.equal(clean.message, 'PDF 생성 완료: 번역 7개 블록');
  assert.equal(clean.tone, '');
  assert.deepEqual(clean.lines, []);
  assert.deepEqual(clean.warnings, []);
  const junk = pdfReportDetails({
    replaced: -3, kept: 'x', raster_blocks_erased: NaN, kept_reasons: 'nope',
    specialist_kept: { a: -1, b: '2' }, warnings: [null, 7, '', { message: 'p1: 메시지' }],
  });
  assert.equal(junk.message, 'PDF 생성 완료: 번역 0개 블록 · 전문 조판 2개 원형 보존 · 주의 1건');
  assert.deepEqual(junk.warnings, ['1페이지: 메시지']);
  assert.deepEqual(pdfReportDetails(null).lines, []);
});

test('pdfReportDetails: 서버가 앞 50건만 실은 경고의 나머지 수를 알린다', () => {
  const warnings = Array.from({ length: PDF_REPORT_MAX_WARNINGS }, (_, i) => `p${i + 1}: 경고`);
  const d = pdfReportDetails({ kept: 0, warning_count: 73, warnings });
  assert.equal(d.warnings.length, PDF_REPORT_MAX_WARNINGS);
  assert.equal(d.moreWarnings, 23);
  assert.match(d.message, /주의 73건/);
});

test('pdfKeptReasonLabel: 새 사유·접두 사유·모르는 사유', () => {
  assert.equal(pdfKeptReasonLabel('listing_line_unaligned'), '원문 줄 위치 정렬 실패(그 줄만 원문)');
  assert.equal(pdfKeptReasonLabel('preserve_type:ref_text'), '원문 유지 대상(참고문헌)');
  assert.equal(pdfKeptReasonLabel('preserve_type:sidebar'), '원문 유지 대상(sidebar)');
  assert.equal(pdfKeptReasonLabel('preserved:code'), '번역 단계에서 보존(code)');
  assert.equal(pdfKeptReasonLabel('future_reason'), 'future_reason', '개수를 잃지 않게 키 그대로');
  assert.equal(pdfKeptReasonLabel('toString'), 'toString', '프로토타입 이름에 속지 않는다');
});

test('pdfReportUrl·pdfReportMessage: 같은 잡·언어의 리포트 주소, 헤더 요약의 스캔 지움', () => {
  assert.equal(pdfReportUrl('/api/jobs/j_abc/pdf?lang=ko&view=dual'), '/api/jobs/j_abc/pdf/report?lang=ko');
  assert.equal(pdfReportUrl('/api/jobs/j_abc/pdf'), '/api/jobs/j_abc/pdf/report?lang=ko');
  assert.equal(pdfReportUrl('https://evil.example/api/jobs/x/pdf'), null);
  assert.equal(pdfReportUrl(null), null);
  assert.equal(pdfReportMessage({ replaced: 3, rasterErased: '2' }),
    'PDF 생성 완료: 번역 3개 블록 · 스캔 원문 2개 블록 지움');
});

/* ---------------- 런타임: 다운로드 뒤 리포트 ---------------- */

const flush = () => new Promise((resolve) => setImmediate(resolve));

function setup(t, { reportStatus = 200 } = {}) {
  const doc = installFakeDom(t);
  const savedState = { ...state };
  const savedEls = { ...el };
  const tags = { pdfReport: 'details', pdfReportSummary: 'summary', pdfReportList: 'ul', dlPdf: 'a' };
  for (const key of Object.keys(EL_IDS)) el[key] = mount(doc, tags[key] || 'div', EL_IDS[key]);
  el.pdfReport.hidden = true; // index.html의 초기 상태(hidden)
  el.dlPdf.setAttribute('href', '/api/jobs/job-a/pdf?lang=ko&view=dual');
  el.dlPdf.setAttribute('download', 'paper.ko.pdf');
  Object.assign(state, {
    currentJobId: 'job-a', displayedStatus: 'done', translateState: 'done', resultHasLayout: true,
    resultUrls: { pdf: '/api/jobs/job-a/pdf?lang=ko&view=dual' }, currentBaseName: 'paper',
    pdfDownloadBusy: false, toastTimer: 0,
  });
  t.mock.method(URL, 'createObjectURL', () => 'blob:job-a');
  t.mock.method(URL, 'revokeObjectURL', () => {});
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    calls.push(String(url));
    if (String(url).startsWith('/api/jobs/job-a/pdf?')) {
      return new Response('%PDF-1.4\n', {
        headers: {
          'Content-Type': 'application/pdf', 'X-UOCR-PDF-Replaced': '12',
          'X-UOCR-PDF-Preserved': '4', 'X-UOCR-PDF-Warnings': '2',
        },
      });
    }
    if (String(url) === '/api/jobs/job-a/pdf/report?lang=ko' && reportStatus === 200) {
      return new Response(JSON.stringify({ job_id: 'job-a', lang: 'ko', ...REPORT }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response('{"detail":"Not Found"}', { status: 404 });
  });
  t.after(() => {
    clearTimeout(state.toastTimer);
    Object.assign(state, savedState);
    Object.assign(el, savedEls);
  });
  return { doc, calls };
}

const listText = () => el.pdfReportList.children.map((li) => li.textContent);

test('대조 PDF를 받으면 같은 빌드의 리포트로 상세 목록과 전체 요약을 보인다', async (t) => {
  const { calls } = setup(t);
  await downloadPdfWithReport({ preventDefault() {}, currentTarget: el.dlPdf });
  for (let i = 0; i < 10; i += 1) await flush();
  assert.deepEqual(calls, ['/api/jobs/job-a/pdf?lang=ko&view=dual', '/api/jobs/job-a/pdf/report?lang=ko']);
  assert.equal(el.toast.textContent, pdfReportDetails(REPORT).message, '헤더 요약을 전체 요약으로 바꾼다');
  assert.equal(el.pdfReport.hidden, false);
  assert.equal(el.pdfReportSummary.textContent, 'PDF 생성 리포트 · 주의 2건');
  const text = listText();
  assert.ok(text.some((line) => line.includes('스캔(이미지) 원문 2개 블록')), text.join('\n'));
  assert.ok(text.some((line) => line.includes('원문 줄 위치 정렬 실패(그 줄만 원문) 3')), text.join('\n'));
  assert.ok(text.includes('2페이지: 블록 3의 원문이 이미지 픽셀이라 지우지 못함 — 번역이 이미지 속 글자와 겹쳐 보일 수 있음'));
  assert.ok(text.includes('4페이지: 블록 1의 3줄 교체 생략(원문 줄 위치 정렬 실패) — 그 줄만 원문 보존'));
  const warnItems = el.pdfReportList.children.filter((li) => li.classList.contains('is-warn'));
  assert.equal(warnItems.length, 2, '주의 문장은 경고색');

  // 잡 전환·번역 초기화로 대조 PDF 내보내기가 숨겨지면 이전 빌드의 리포트도 지운다
  state.translateState = 'none';
  applyPdfExport();
  assert.equal(el.pdfReport.hidden, true);
  assert.deepEqual(listText(), []);
});

test('리포트가 없는 서버(404)면 헤더 요약 토스트만 남고 상세 목록은 숨긴다', async (t) => {
  setup(t, { reportStatus: 404 });
  await downloadPdfWithReport({ preventDefault() {}, currentTarget: el.dlPdf });
  for (let i = 0; i < 10; i += 1) await flush();
  assert.equal(el.toast.textContent, 'PDF 생성 완료: 번역 12개 블록 · 원문 4개 보존 · 주의 2건');
  assert.equal(el.pdfReport.hidden, true);
});

test('리포트를 기다리는 사이 다른 잡으로 바꾸면 그리지 않는다', async (t) => {
  setup(t);
  await downloadPdfWithReport({ preventDefault() {}, currentTarget: el.dlPdf });
  state.currentJobId = 'job-b';
  for (let i = 0; i < 10; i += 1) await flush();
  assert.equal(el.pdfReport.hidden, true);
  assert.doesNotMatch(el.toast.textContent, /스캔 원문/);
});
