// 재처리(rewind) 회귀 스위트 — 서버가 이미 보낸 출력을 폐기했을 때 라이브 뷰가
// 함께 되돌아가는지 검증한다.
//
// 배경(실측): 46페이지 논문의 3번째 청크에서 한 페이지가 출력 상한(16,384자)을
// 넘어 multi 생성이 중단되고, 서버가 그 청크를 페이지별로 다시 처리했다. 그때
//   · 폐기 통보가 없어 클라이언트 원문에 폐기된 출력이 그대로 남고
//   · 재처리 출력에는 <PAGE>가 없어 페이지가 올라가지 않아
// 18페이지 하나에 박스 107개가 쌓이고 19–24페이지는 통째로 사라졌으며,
// 잘린 <table>이 미리보기의 뒤 내용을 전부 삼켰다.
// 픽스처는 수정 후 같은 문서로 다시 캡처한 실제 서버 SSE다.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  PAGE_MARKER, createGroundState, groundAnnounce, groundDrain, groundPush,
  planPreviewRender, splitPreviewPages, structurePreview, truncateRawToPage,
} from '../app.js';

const FIXTURES = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

function parseSse(text) {
  const events = [];
  for (const record of text.split(/\r?\n\r?\n/)) {
    let event = null;
    let data = '';
    for (const line of record.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).trim();
    }
    if (event && data) events.push({ event, data: JSON.parse(data) });
  }
  return events;
}

// app.js(js/live.js + js/jobs.js + js/sse.js)의 배선을 그대로 따르는 순수 드라이버.
// token → groundPush, progress → 먼저 drain 후 announce, reset → 원문을 그 페이지
// 앞까지 잘라내고 페이지 상태머신·박스를 다시 만든다 (applyStreamReset과 동일).
function replay(events, opts = {}) {
  let g = createGroundState();
  let raw = '';
  let boxes = [];
  const resets = [];

  const drain = (final) => {
    for (const ev of groundDrain(g, final)) {
      if (ev.type === 'boxes') boxes.push(ev);
    }
  };
  for (const e of events) {
    if (e.event === 'progress') {
      drain(false);
      groundAnnounce(g, e.data.phase, e.data.current_page, e.data.total_pages);
    } else if (e.event === 'token') {
      raw += e.data.text;
      groundPush(g, e.data.text);
      if (opts.drainEveryToken) drain(false);
    } else if (e.event === 'reset') {
      drain(false);
      resets.push(e.data);
      const truncated = truncateRawToPage(raw, e.data.from_page);
      if (truncated.length !== raw.length) {
        const total = g.totalPages;
        raw = truncated;
        g = createGroundState();
        g.totalPages = total;
        g.ocrSeen = true;
        boxes = [];
        groundPush(g, raw);
        drain(false);
      }
    }
  }
  drain(true);
  return { g, raw, boxes, resets };
}

function boxesByPage(boxes) {
  const m = new Map();
  for (const ev of boxes) m.set(ev.page, (m.get(ev.page) || 0) + ev.boxes.length);
  return m;
}

const unbalancedTables = (s) =>
  (s.match(/<table[\s>]/g) || []).length - (s.match(/<\/table>/g) || []).length;

/* ================= truncateRawToPage (순수) ================= */

test('truncateRawToPage: 페이지 N의 마커 지점에서 자른다', () => {
  const raw = '<PAGE>one<PAGE>two<PAGE>three';
  assert.equal(truncateRawToPage(raw, 1), '');
  assert.equal(truncateRawToPage(raw, 2), '<PAGE>one');
  assert.equal(truncateRawToPage(raw, 3), '<PAGE>one<PAGE>two');
});

test('truncateRawToPage: 받은 적 없는 페이지는 원문을 그대로 둔다', () => {
  const raw = '<PAGE>one<PAGE>two';
  assert.equal(truncateRawToPage(raw, 3), raw, '없는 마커로 잘못 자르지 않는다');
  assert.equal(truncateRawToPage('', 1), '');
  assert.equal(truncateRawToPage(null, 2), '');
});

test('truncateRawToPage: 비정상 인자는 1페이지로 강등된다', () => {
  const raw = '<PAGE>one<PAGE>two';
  assert.equal(truncateRawToPage(raw, 0), '');
  assert.equal(truncateRawToPage(raw, -5), '');
  assert.equal(truncateRawToPage(raw, 'x'), '');
});

/* ============ reset 후 확정 페이지 캐시 길이 (순수 등가성) ============ */

// applyStreamReset은 확정 페이지 캐시를 from-1개로 자른다. 잘라낸 원문에는 마커가
// from-1개 남고 splitPreviewPages의 확정 페이지도 from-1개이므로(0번은 서두),
// 캐시가 한 칸이라도 길면 페이지 from-1이 확정본과 꼬리로 두 번 렌더된다.
test('reset 후 캐시 길이(from-1)는 처음부터 렌더한 것과 동등하다', () => {
  const raw = `${PAGE_MARKER}p1 body${PAGE_MARKER}p2 body${PAGE_MARKER}p3 body${PAGE_MARKER}p4 partial`;
  const from = 3; // 페이지 3부터 재처리
  const truncated = truncateRawToPage(raw, from);
  assert.equal(truncated, `${PAGE_MARKER}p1 body${PAGE_MARKER}p2 body`);

  const { pages, tail } = splitPreviewPages(truncated);
  // 확정 페이지는 [서두, 페이지1] 두 개이고 페이지 2는 다시 꼬리로 돌아간다.
  assert.equal(pages.length, from - 1, 'applyStreamReset이 남겨야 할 캐시 길이');
  assert.equal(tail, 'p2 body', '페이지 from-1은 미확정 꼬리가 된다');

  const scratch = planPreviewRender(truncated, [], '', false);
  const cache = pages.map((seg) => structurePreview(seg, true));
  const resumed = planPreviewRender(truncated, cache, '', false);
  assert.equal(resumed.newPages.length, 0, '캐시된 확정 페이지를 다시 렌더하지 않는다');
  assert.equal(resumed.tailMd, scratch.tailMd, '꼬리 markdown이 처음부터 렌더한 것과 같다');
  assert.equal(resumed.tailMd, 'p2 body');
});

/* ================= 실캡처: 재처리가 일어난 잡 ================= */

const fixture = fs.readFileSync(path.join(FIXTURES, 'rerun-fallback.sse.txt'), 'utf8');
const events = parseSse(fixture);
const totalPages = events.find((e) => e.event === 'progress' && e.data.total_pages)
  .data.total_pages;

test('픽스처: 재처리(reset)가 실제로 일어난 캡처다', () => {
  const resets = events.filter((e) => e.event === 'reset');
  assert.ok(resets.length >= 1, '이 픽스처는 reset 이벤트를 담고 있어야 의미가 있다');
  assert.ok(resets.every((r) => r.data.from_page >= 1));
  assert.ok(totalPages >= 2, `total_pages=${totalPages}`);
});

for (const [name, opts] of [
  ['progress에서만 drain', {}],
  ['토큰마다 drain', { drainEveryToken: true }],
]) {
  test(`실캡처 — ${name}: 폐기된 출력이 남지 않는다`, () => {
    const { raw, resets } = replay(events, opts);
    assert.ok(resets.length >= 1);
    const markers = (raw.match(/<PAGE>/g) || []).length;
    assert.equal(markers, totalPages,
      `마커 ${markers}개 ≠ 페이지 ${totalPages}개 — 프레이밍이 깨졌다`);
  });

  test(`실캡처 — ${name}: 모든 페이지가 자기 박스를 갖는다`, () => {
    const { boxes } = replay(events, opts);
    const perPage = boxesByPage(boxes);
    for (let p = 1; p <= totalPages; p += 1) {
      assert.ok((perPage.get(p) || 0) > 0, `페이지 ${p}에 박스가 하나도 없다`);
    }
    assert.ok(![...perPage.keys()].some((p) => p > totalPages),
      `총 페이지를 넘는 박스 페이지: ${[...perPage.keys()].filter((p) => p > totalPages)}`);
    // 한 페이지에 다른 페이지의 몫까지 쌓이지 않는다 (실측 버그: 18페이지에 107개)
    const counts = [...perPage.values()];
    const median = counts.slice().sort((a, b) => a - b)[Math.floor(counts.length / 2)];
    const max = Math.max(...counts);
    assert.ok(max <= median * 4 + 8,
      `한 페이지에 박스가 몰렸다: max=${max}, median=${median}, ${JSON.stringify([...perPage])}`);
  });
}

test('실캡처: 확정 미리보기 페이지에 짝 없는 <table>이 남지 않는다', () => {
  const { raw } = replay(events);
  const { pages } = splitPreviewPages(raw);
  pages.forEach((seg, i) => {
    const md = structurePreview(seg, true);
    assert.equal(unbalancedTables(md), 0,
      `확정 페이지 ${i}에 닫히지 않은 <table>이 있다 (뒤 내용을 통째로 삼킨다)`);
  });
});

test('실캡처: 확정 페이지 조각이 한 페이지 분량을 넘지 않는다', () => {
  const { raw } = replay(events);
  const { pages } = splitPreviewPages(raw);
  const sizes = pages.map((seg) => structurePreview(seg, true).length);
  // 페이지 출력 상한(16,384자) + 구조화 오버헤드 여유. 재처리분이 한 세그먼트에
  // 뭉치면 이 값을 훌쩍 넘고, 그 조각이 600ms마다 통째로 재전송된다.
  assert.ok(Math.max(...sizes, 0) < 24000, `확정 페이지 최대 길이 ${Math.max(...sizes)}자`);
});

test('실캡처: 한 세그먼트가 여러 페이지의 머리말을 담지 않는다', () => {
  // 페이지 하나에 header/page_number det는 많아야 두어 개다. 재처리분이 한
  // 세그먼트에 뭉치면 페이지 수만큼 쌓인다 (실측: 18페이지 세그먼트에 header 9개).
  const { raw } = replay(events);
  const { pages } = splitPreviewPages(raw);
  pages.forEach((seg, i) => {
    for (const label of ['header', 'page_number']) {
      const n = (seg.match(new RegExp(`<\\|det\\|>\\s*${label}\\s*\\[`, 'g')) || []).length;
      assert.ok(n <= 3, `세그먼트 ${i}에 ${label} det가 ${n}개 — 여러 페이지가 뭉쳤다`);
    }
  });
});

test('실캡처: 재처리 구간에서도 진행률이 계속 올라간다', () => {
  const ocr = events
    .filter((e) => e.event === 'progress' && e.data.phase === 'ocr')
    .map((e) => e.data.current_page);
  assert.ok(ocr.length >= totalPages,
    `ocr 진행 이벤트 ${ocr.length}건 < 페이지 ${totalPages}개 — 재처리 동안 멈춰 있었다`);
  assert.equal(Math.max(...ocr), totalPages);
  for (let p = 1; p <= totalPages; p += 1) {
    assert.ok(ocr.includes(p), `페이지 ${p}가 진행률로 한 번도 알려지지 않았다`);
  }
});

test('마커를 받지 못한 페이지의 reset은 원문을 건드리지 않는다', () => {
  const partial = [
    { event: 'progress', data: { phase: 'ocr', current_page: 1, total_pages: 4, status: 'running' } },
    { event: 'token', data: { text: `${PAGE_MARKER}<|det|>text [10, 10, 900, 60]<|/det|>a` } },
    { event: 'reset', data: { from_page: 3, reason: '테스트' } },
  ];
  const { raw, boxes } = replay(partial);
  assert.ok(raw.includes('<|det|>text'), '자를 지점을 모르면 원문을 유지한다');
  assert.equal(boxes.length, 1);
});

/* ========== 충실도 게이트: 청크 한가운데 페이지 교체 ========== */
// 서버가 8쪽 청크를 흘린 뒤 열화 페이지(6쪽)를 발견하면 reset{from_page:6}을 내고
// 6·7·8쪽을 다시 채운다. 예전에는 청크 시작 페이지에만 되감기 마크가 있어 이런
// reset 자체가 나가지 못했다 — 이제 나가므로 클라이언트가 옳게 처리하는지 본다.

function ev(name, data) { return { event: name, data }; }

function chunkStream(startPage, pages, bodyFor) {
  const out = [];
  for (let i = 0; i < pages; i += 1) {
    const page = startPage + i;
    out.push(ev('progress', { phase: 'ocr', current_page: page, total_pages: startPage + pages - 1 }));
    out.push(ev('token', { text: `<PAGE>${bodyFor(page)}` }));
  }
  return out;
}

test('게이트: 청크 중간 페이지 reset 후 다시 채우면 페이지 귀속이 유지된다', () => {
  const events = [
    ...chunkStream(1, 8, (p) => (p === 6 ? '' : `body-${p} `)),
    ev('reset', { from_page: 6, reason: '충실도 게이트 — 열화 페이지 재처리' }),
    ...chunkStream(6, 3, (p) => (p === 6 ? 'repaired-6 ' : `body-${p} `)),
  ];
  const { raw, resets } = replay(events);

  assert.equal(resets.length, 1);
  assert.equal(resets[0].from_page, 6);
  const segments = raw.split('<PAGE>');
  assert.equal(segments.length, 9, `세그먼트 ${segments.length - 1}개 (기대 8): ${JSON.stringify(segments)}`);
  assert.equal(segments[0], '', '첫 마커 앞에 본문이 남았다');
  for (let p = 1; p <= 8; p += 1) {
    const want = p === 6 ? 'repaired-6' : `body-${p}`;
    assert.ok(segments[p].includes(want), `세그먼트 ${p}에 ${want}가 없다: ${segments[p]}`);
  }
  // 되감기 전의 빈 6쪽 흔적이 남아 세그먼트를 늘리지 않았는지
  assert.ok(!raw.includes('body-6 body-7'), '되감긴 조각이 남아 있다');
});

test('게이트: 재처리가 거부돼 원래 본문을 다시 넣어도 세그먼트 수는 같다', () => {
  const events = [
    ...chunkStream(1, 4, (p) => `body-${p} `),
    ev('reset', { from_page: 3, reason: '충실도 게이트' }),
    // 3쪽 재처리 시도가 흘러나왔다가
    ev('progress', { phase: 'ocr', current_page: 3, total_pages: 4 }),
    ev('token', { text: '<PAGE>retry-attempt-3 ' }),
    // 개선되지 않아 다시 물리고 원래 본문으로 채운다
    ev('reset', { from_page: 3, reason: '재처리 결과 미채택' }),
    ...chunkStream(3, 2, (p) => `body-${p} `),
  ];
  const { raw, resets } = replay(events);

  assert.equal(resets.length, 2);
  const segments = raw.split('<PAGE>');
  assert.equal(segments.length, 5, `세그먼트 ${segments.length - 1}개 (기대 4)`);
  assert.ok(!raw.includes('retry-attempt-3'), '거부된 재처리 출력이 남았다');
  for (let p = 1; p <= 4; p += 1) {
    assert.ok(segments[p].includes(`body-${p}`), `세그먼트 ${p} 내용이 어긋났다`);
  }
});
