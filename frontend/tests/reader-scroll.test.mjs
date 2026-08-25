// 연속 스크롤 리더의 순수 코어 — DOM 없이 Node에서 직접 검증한다.
//
//   node --test frontend/tests/reader-scroll.test.mjs
//   cd frontend && npm test
//
// 여기서 지키는 계약:
//  · 페이지 밴드 ↔ 스크롤 오프셋 왕복이 어긋나지 않는다 (페이지 번호가 진실을 말한다).
//  · 프리페치 계획이 서버 배치 계약(start≥1, 1≤limit≤16)을 절대 위반하지 않는다.
//  · 좌표 블록 역산이 경계에서 튀지 않는다.
//  · 레일 수식 분해가 서버 text_with_math_html과 같은 구분자 규칙을 따른다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  readerPageBands,
  readerFocusAt,
  readerHydrationWindow,
  alignmentBatchPlan,
  alignmentFailureIsPermanent,
  blockAtFraction,
  splitInlineMath,
  readerRailBandAt,
  railAnchorFrom,
  railAnchorTarget,
} from '../app.js';

/* ---------------- readerPageBands ---------------- */

test('readerPageBands: 높이 배열을 누적 밴드로 바꾼다', () => {
  assert.deepEqual(readerPageBands([100, 200], 10), [
    { page: 1, top: 0, height: 100 },
    { page: 2, top: 110, height: 200 },
  ]);
});

test('readerPageBands: gap 0과 빈 입력', () => {
  assert.deepEqual(readerPageBands([50, 50]), [
    { page: 1, top: 0, height: 50 },
    { page: 2, top: 50, height: 50 },
  ]);
  assert.deepEqual(readerPageBands([]), []);
  assert.deepEqual(readerPageBands(null), []);
});

test('readerPageBands: 비정상 높이는 0으로 방어 (NaN·음수)', () => {
  assert.deepEqual(readerPageBands([NaN, -20, 30], 5), [
    { page: 1, top: 0, height: 0 },
    { page: 2, top: 5, height: 0 },
    { page: 3, top: 10, height: 30 },
  ]);
});

/* ---------------- readerFocusAt ---------------- */

const BANDS = readerPageBands([100, 100, 100], 10); // top 0 / 110 / 220

test('readerFocusAt: 페이지와 페이지 내 진행률을 돌려준다', () => {
  assert.deepEqual(readerFocusAt(BANDS, 0), { page: 1, fraction: 0 });
  assert.deepEqual(readerFocusAt(BANDS, 50), { page: 1, fraction: 0.5 });
  assert.deepEqual(readerFocusAt(BANDS, 110), { page: 2, fraction: 0 });
  assert.deepEqual(readerFocusAt(BANDS, 160), { page: 2, fraction: 0.5 });
});

test('readerFocusAt: 위/아래 바깥과 페이지 사이 여백', () => {
  assert.deepEqual(readerFocusAt(BANDS, -40), { page: 1, fraction: 0 });
  // 100~110은 페이지 사이 여백 — 직전 페이지의 끝으로 본다(번호가 튀지 않게)
  assert.deepEqual(readerFocusAt(BANDS, 105), { page: 1, fraction: 1 });
  assert.deepEqual(readerFocusAt(BANDS, 99999), { page: 3, fraction: 1 });
  assert.deepEqual(readerFocusAt([], 10), { page: 1, fraction: 0 });
});

test('readerFocusAt: 각 페이지의 처음과 끝이 그 페이지로 돌아온다 (왕복 무모순)', () => {
  const bands = readerPageBands([120, 300, 80, 240], 18);
  for (const band of bands) {
    assert.equal(readerFocusAt(bands, band.top).page, band.page);
    assert.equal(readerFocusAt(bands, band.top + band.height - 1).page, band.page);
  }
});

/* ---------------- readerHydrationWindow ---------------- */

test('readerHydrationWindow: 현재 페이지 ± radius를 문서 범위로 클램프', () => {
  assert.deepEqual(readerHydrationWindow(10, 1, 2), { start: 1, end: 3 });
  assert.deepEqual(readerHydrationWindow(10, 5, 2), { start: 3, end: 7 });
  assert.deepEqual(readerHydrationWindow(10, 10, 2), { start: 8, end: 10 });
  assert.deepEqual(readerHydrationWindow(10, 5, 0), { start: 5, end: 5 });
});

test('readerHydrationWindow: 총 페이지 미상이면 아무것도 붙이지 않는다', () => {
  const empty = readerHydrationWindow(0, 1, 2);
  assert.ok(empty.end < empty.start, JSON.stringify(empty));
});

/* ---------------- alignmentBatchPlan ---------------- */

test('alignmentBatchPlan: 미로드 구간만 배치로 덮는다', () => {
  assert.deepEqual(alignmentBatchPlan(100, 50, 2, new Set()), [{ start: 48, limit: 5 }]);
  // 가운데가 이미 캐시돼 있으면 연속 구간이 쪼개진다
  assert.deepEqual(alignmentBatchPlan(100, 50, 2, new Set([49, 50])), [
    { start: 48, limit: 1 },
    { start: 51, limit: 2 },
  ]);
  assert.deepEqual(alignmentBatchPlan(100, 50, 2, new Set([48, 49, 50, 51, 52])), []);
});

test('alignmentBatchPlan: 서버 계약(start≥1, 1≤limit≤16)을 절대 넘지 않는다', () => {
  for (const [total, current, radius, max] of [
    [200, 100, 40, 100], [200, 1, 40, 16], [200, 200, 40, 999], [3, 2, 10, 16],
  ]) {
    const plan = alignmentBatchPlan(total, current, radius, new Set(), max);
    assert.ok(plan.length > 0);
    for (const batch of plan) {
      assert.ok(batch.start >= 1, `start=${batch.start}`);
      assert.ok(batch.limit >= 1 && batch.limit <= 16, `limit=${batch.limit}`);
      assert.ok(batch.start + batch.limit - 1 <= total, JSON.stringify(batch));
    }
  }
});

test('alignmentBatchPlan: 문서보다 큰 창을 줘도 페이지 밖을 요청하지 않는다', () => {
  const plan = alignmentBatchPlan(3, 1, 10, new Set());
  assert.deepEqual(plan, [{ start: 1, limit: 3 }]);
});

/* ---------------- blockAtFraction ---------------- */

const BLOCKS = [
  { id: 'p1-b0', rect: { top: 0 } },
  { id: 'p1-b1', rect: { top: 30 } },
  { id: 'p1-b2', rect: { top: 70 } },
];

test('blockAtFraction: 그 지점을 지나온 마지막 블록', () => {
  assert.equal(blockAtFraction(BLOCKS, 0), 'p1-b0');
  assert.equal(blockAtFraction(BLOCKS, 0.29), 'p1-b0');
  assert.equal(blockAtFraction(BLOCKS, 0.30), 'p1-b1');
  assert.equal(blockAtFraction(BLOCKS, 0.99), 'p1-b2');
});

test('blockAtFraction: 범위 밖 진행률은 클램프, 블록 없으면 빈 문자열', () => {
  assert.equal(blockAtFraction(BLOCKS, -1), 'p1-b0');
  assert.equal(blockAtFraction(BLOCKS, 4), 'p1-b2');
  assert.equal(blockAtFraction([], 0.5), '');
  assert.equal(blockAtFraction(null, 0.5), '');
});

test('blockAtFraction: 다단·세로 머리말처럼 y 순서가 뒤섞여도 가장 가까운 블록', () => {
  const unsorted = [
    { id: 'vertical-margin', rect: { top: 30 } },
    { id: 'main-heading', rect: { top: 10 } },
    { id: 'main-body', rect: { top: 45 } },
  ];
  assert.equal(blockAtFraction(unsorted, 0), 'main-heading');
  assert.equal(blockAtFraction(unsorted, 0.20), 'main-heading');
  assert.equal(blockAtFraction(unsorted, 0.35), 'vertical-margin');
  assert.equal(blockAtFraction(unsorted, 0.80), 'main-body');
});

/* ---------------- splitInlineMath ---------------- */

test('splitInlineMath: 인라인 \\( … \\)를 수식 조각으로 분리', () => {
  assert.deepEqual(splitInlineMath('Weizhe Yuan \\( ^{1,2} \\) Meta'), [
    { type: 'text', value: 'Weizhe Yuan ' },
    { type: 'math', value: '^{1,2}', display: false },
    { type: 'text', value: ' Meta' },
  ]);
});

test('splitInlineMath: 디스플레이 \\[ … \\]는 display=true', () => {
  const parts = splitInlineMath('식: \\[ E = mc^2 \\]');
  assert.equal(parts.length, 2);
  assert.deepEqual(parts[1], { type: 'math', value: 'E = mc^2', display: true });
});

test('splitInlineMath: 수식이 없으면 텍스트 한 조각', () => {
  assert.deepEqual(splitInlineMath('수식 없는 문장'), [{ type: 'text', value: '수식 없는 문장' }]);
  assert.deepEqual(splitInlineMath(''), [{ type: 'text', value: '' }]);
  assert.deepEqual(splitInlineMath(null), [{ type: 'text', value: '' }]);
});

test('splitInlineMath: 짝이 맞지 않는 구분자는 텍스트로 남긴다 (원문 무손실)', () => {
  assert.deepEqual(splitInlineMath('열기만 \\( 하고 끝'), [{ type: 'text', value: '열기만 \\( 하고 끝' }]);
});

test('splitInlineMath: 통화 $는 수식으로 오인하지 않는다', () => {
  assert.deepEqual(splitInlineMath('가격은 $5 입니다'), [{ type: 'text', value: '가격은 $5 입니다' }]);
});

test('splitInlineMath: 여러 수식과 빈 수식', () => {
  const parts = splitInlineMath('\\( a \\)와 \\( b \\)');
  assert.deepEqual(parts.map((p) => p.type), ['math', 'text', 'math']);
  // 내용이 비면 조각을 만들지 않는다 (빈 KaTeX 스팬 방지)
  assert.deepEqual(splitInlineMath('앞 \\(  \\) 뒤'), [
    { type: 'text', value: '앞 ' },
    { type: 'text', value: ' 뒤' },
  ]);
});

/* ---------------- readerRailBandAt ---------------- */

const RAIL_BANDS = [
  { page: 1, top: 0, height: 100 },
  { page: 2, top: 100, height: 300 },
  { page: 3, top: 400, height: 150 },
];

test('readerRailBandAt: 오프셋을 지나온 마지막 밴드', () => {
  assert.equal(readerRailBandAt(RAIL_BANDS, 0).page, 1);
  assert.equal(readerRailBandAt(RAIL_BANDS, 99).page, 1);
  assert.equal(readerRailBandAt(RAIL_BANDS, 100).page, 2);
  assert.equal(readerRailBandAt(RAIL_BANDS, 9999).page, 3);
  assert.equal(readerRailBandAt(RAIL_BANDS, -50).page, 1, '위쪽 바깥은 첫 밴드');
  assert.equal(readerRailBandAt([], 10), null);
  assert.equal(readerRailBandAt(null, 10), null, '방어: 밴드 미측정');
});

/* ---------------- railAnchorFrom / railAnchorTarget ---------------- */
// 개별(sync off) 스크롤에서 위쪽 페이지가 뒤늦게 채워져도 레일이 밀리지 않게
// 하는 계약: 렌더 전 앵커 → 렌더 후 같은 밴드 오프셋으로 복원.

test('railAnchorFrom: 눈높이 밴드와 그 밴드 기준 스크롤 오프셋', () => {
  // scrollTop 150 + 뷰포트 200 * 0.25 = 200 → page 2 (top 100)
  assert.deepEqual(railAnchorFrom(RAIL_BANDS, 150, 200, 0.25), { page: 2, offset: 50 });
  assert.deepEqual(railAnchorFrom(RAIL_BANDS, 0, 200, 0.25), { page: 1, offset: 0 });
  assert.equal(railAnchorFrom([], 150, 200, 0.25), null, '밴드가 없으면 앵커도 없다');
});

test('railAnchorTarget: 위쪽 페이지가 커져도 같은 문단이 같은 눈높이에 남는다', () => {
  const anchor = railAnchorFrom(RAIL_BANDS, 150, 200, 0.25); // page 2 + 50
  // 1페이지가 100 → 400으로 늘어난 뒤의 밴드
  const grown = [
    { page: 1, top: 0, height: 400 },
    { page: 2, top: 400, height: 300 },
    { page: 3, top: 700, height: 150 },
  ];
  assert.equal(railAnchorTarget(grown, anchor), 450, '밀린 300px만큼 되잡는다');
  assert.equal(railAnchorTarget(RAIL_BANDS, anchor), 150, '레이아웃이 그대로면 그대로');
});

test('railAnchorTarget: 앵커 페이지가 사라졌거나 앵커가 없으면 복원하지 않는다', () => {
  assert.equal(railAnchorTarget(RAIL_BANDS, null), null);
  assert.equal(railAnchorTarget([], { page: 2, offset: 50 }), null);
  assert.equal(railAnchorTarget(RAIL_BANDS, { page: 9, offset: 50 }), null);
});

test('railAnchorTarget: 음수 스크롤로 내려가지 않는다', () => {
  assert.equal(railAnchorTarget(RAIL_BANDS, { page: 1, offset: -80 }), 0);
});

/* ---------------- alignmentFailureIsPermanent ---------------- */

test('alignmentFailureIsPermanent: 409 레이아웃 불일치는 재시도해도 소용없다', () => {
  assert.equal(alignmentFailureIsPermanent(409), true, '블록/페이지 대응 불일치 — 영구');
  assert.equal(alignmentFailureIsPermanent(404), true, '좌표 없는 페이지 — 영구');
  assert.equal(alignmentFailureIsPermanent(422), true, '잘못된 페이지 번호 — 영구');
  assert.equal(alignmentFailureIsPermanent(403), true);
});

test('alignmentFailureIsPermanent: 일시 실패는 재시도 대상으로 남긴다', () => {
  assert.equal(alignmentFailureIsPermanent(500), false);
  assert.equal(alignmentFailureIsPermanent(502), false);
  assert.equal(alignmentFailureIsPermanent(408), false, '요청 타임아웃');
  assert.equal(alignmentFailureIsPermanent(429), false, '스로틀');
  assert.equal(alignmentFailureIsPermanent(0), false, '네트워크 오류');
  assert.equal(alignmentFailureIsPermanent(undefined), false, '방어: 상태 미상');
});
