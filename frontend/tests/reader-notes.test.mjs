// 리더 인용·하이라이트 영속 — 순수 규칙 + 저장소 + 실제 reader.js 런타임(가짜 DOM).
//
//   node --test frontend/tests/reader-notes.test.mjs
//
// 예전에는 '인용 저장'·'하이라이트'가 쓰기 전용이었다 — 토스트는 "저장했다"고 했지만
// 볼 수도 내보낼 수도 없고 잡을 바꾸면 사라졌다(frontend-6). 하이라이트는 카드·페이지
// 경계에서 extractContents로 DOM을 복제했다(frontend-10). 여기서 지키는 계약:
//  · 잡별 localStorage에 남고(상한), 다시 열면 목록이 그대로 보인다.
//  · 목록에서 페이지 이동·삭제, Markdown 복사·내보내기를 할 수 있다.
//  · 레일을 다시 그려도 저장된 하이라이트를 텍스트 노드 조각 단위로 되살린다(중복 없이).
//  · 잡을 삭제하면 저장값도 지운다.
//  · 같은 잡을 연 다른 탭과 저장소를 공유한다 — 저장·삭제는 최신 저장값 위에 적용하고(덮어쓰기
//    없음), 다른 탭의 변경은 storage 이벤트로 목록·하이라이트에 맞춘다(frontend-1).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  READER_NOTE_MAX_CHARS, READER_NOTES_MAX, READER_NOTES_MAX_JOBS, readerNotesKey,
} from '../js/constants.js';
import {
  addReaderNote, normalizeReaderNotes, readerNotesMarkdown, readerNotesPruneKeys,
  removeReaderNote,
} from '../js/core.js';
import { forgetReaderNotes, loadReaderNotes, saveReaderNotes } from '../js/notes.js';
import { el, state } from '../js/state.js';
import {
  copyReaderNotes, deleteReaderNote, onReaderNotesStorage, renderReaderDocument, resetReaderForJob,
  saveReaderCitation,
} from '../js/reader.js';
import { deleteJob } from '../js/jobs.js';
import { installFakeStorage, mount } from './helpers/fake-dom.mjs';
import { alignment, setupReader } from './helpers/reader-setup.mjs';

const note = (over = {}) => ({
  id: 'n1', kind: 'citation', page: 3, lang: 'ko', text: '핵심 주장', at: 1, ...over,
});

/* ---------------- 순수 규칙 ---------------- */

test('normalizeReaderNotes: 잘못된 항목을 버리고 공백·길이를 정리한다', () => {
  const items = normalizeReaderNotes({
    items: [
      note({ text: '  여러   줄\n문장 ' }),
      note({ id: 'n2', kind: 'bookmark' }),          // 모르는 종류
      note({ id: 'n3', page: 0 }),                   // 페이지 없음
      note({ id: 'n4', text: '   ' }),               // 빈 문장
      note({ id: 'bad"id' }),                        // 선택자를 깨뜨릴 id
      note({ id: 'n1', text: '중복 id' }),           // 같은 id
      note({ id: 'n5', lang: 'fr', text: 'x'.repeat(READER_NOTE_MAX_CHARS + 10) }),
      null, 'text',
    ],
  });
  assert.deepEqual(items.map((n) => n.id), ['n1', 'n5']);
  assert.equal(items[0].text, '여러 줄 문장');
  assert.equal(items[1].lang, 'orig');
  assert.equal(items[1].text.length, READER_NOTE_MAX_CHARS);
  assert.deepEqual(normalizeReaderNotes('garbage'), []);
  assert.deepEqual(normalizeReaderNotes(null), []);
});

test('addReaderNote: 같은 종류·페이지·언어·문장은 다시 쌓지 않고, 상한을 넘으면 오래된 것부터 버린다', () => {
  const first = addReaderNote([], note());
  assert.equal(first.added, true);
  const dup = addReaderNote(first.items, note({ id: 'n9', text: '핵심   주장' }));
  assert.equal(dup.added, false);
  assert.equal(dup.note.id, 'n1');
  assert.equal(dup.items.length, 1);
  let items = [];
  for (let i = 0; i < 5; i += 1) items = addReaderNote(items, note({ id: `m${i}`, text: `t${i}` }), 3).items;
  assert.deepEqual(items.map((n) => n.id), ['m2', 'm3', 'm4']);
  assert.equal(addReaderNote(items, { kind: 'citation' }).note, null);
});

test('removeReaderNote: id로 지운다', () => {
  assert.deepEqual(removeReaderNote([note(), note({ id: 'n2' })], 'n1').map((n) => n.id), ['n2']);
});

test('readerNotesMarkdown: 인용·하이라이트를 페이지 순으로 묶고 줄머리 마크다운을 무력화한다', () => {
  const md = readerNotesMarkdown('paper.pdf', [
    note({ id: 'b', kind: 'highlight', page: 5, lang: 'orig', text: 'Late', at: 2 }),
    note({ id: 'a', page: 2, text: '# 제목처럼 보이는 문장', at: 3 }),
    note({ id: 'c', kind: 'highlight', page: 1, lang: 'orig', text: '- 목록처럼', at: 1 }),
  ]);
  assert.equal(md, [
    '# paper.pdf — 인용·하이라이트',
    '',
    '## 인용 (1)',
    '',
    '> \\# 제목처럼 보이는 문장',
    '',
    '— 2페이지 · 한국어',
    '',
    '## 하이라이트 (2)',
    '',
    '> \\- 목록처럼',
    '',
    '— 1페이지 · 원문',
    '',
    '> Late',
    '',
    '— 5페이지 · 원문',
    '',
  ].join('\n'));
  assert.match(readerNotesMarkdown('', []), /저장한 인용·하이라이트가 없습니다/);
});

test('readerNotesPruneKeys: 지금 잡을 빼고 가장 오래 손대지 않은 잡부터 지운다', () => {
  const entries = [
    { key: 'a', updated: 1 }, { key: 'b', updated: 30 }, { key: 'c', updated: 20 },
    { key: 'cur', updated: 0 },
  ];
  assert.deepEqual(readerNotesPruneKeys(entries, 'cur', 3), ['a']);
  assert.deepEqual(readerNotesPruneKeys(entries, 'cur', 1), ['b', 'c', 'a']);
  assert.deepEqual(readerNotesPruneKeys(entries, 'cur', 10), []);
});

/* ---------------- 저장소 ---------------- */

test('saveReaderNotes/loadReaderNotes: 잡별로 왕복하고 손댄 값은 정리해서 읽는다', (t) => {
  const storage = installFakeStorage(t);
  assert.equal(saveReaderNotes('job-a', [note()]), true);
  assert.deepEqual(loadReaderNotes('job-a').map((n) => n.id), ['n1']);
  assert.deepEqual(loadReaderNotes('job-b'), []);
  storage.setItem(readerNotesKey('job-c'), '{not json');
  assert.deepEqual(loadReaderNotes('job-c'), []);
  assert.equal(saveReaderNotes('job-a', []), true);
  assert.equal(storage.getItem(readerNotesKey('job-a')), null, '빈 목록이면 키를 지운다');
  saveReaderNotes('job-a', [note()]);
  forgetReaderNotes('job-a');
  assert.equal(storage.getItem(readerNotesKey('job-a')), null);
});

test('saveReaderNotes: 저장 공간이 모자라면 false — 호출부가 "저장했다"고 말하지 않게', (t) => {
  installFakeStorage(t, { quota: 40 });
  assert.equal(saveReaderNotes('job-a', [note({ text: '아주 긴 문장 '.repeat(20) })]), false);
});

test('saveReaderNotes: 보관 잡 수 상한을 넘으면 가장 오래된 잡의 메모를 지운다', (t) => {
  const storage = installFakeStorage(t);
  for (let i = 0; i < READER_NOTES_MAX_JOBS; i += 1) {
    storage.setItem(readerNotesKey(`old-${i}`), JSON.stringify({ v: 1, updated: 1000 + i, items: [note()] }));
  }
  storage.setItem('uocr-theme', 'dark'); // 다른 키는 건드리지 않는다
  assert.equal(saveReaderNotes('job-new', [note()]), true);
  const noteKeys = storage.keys().filter((k) => k.startsWith('uocr-reader-notes-'));
  assert.equal(noteKeys.length, READER_NOTES_MAX_JOBS);
  assert.ok(!noteKeys.includes(readerNotesKey('old-0')), '가장 오래된 잡이 지워진다');
  assert.ok(noteKeys.includes(readerNotesKey('job-new')));
  assert.equal(storage.getItem('uocr-theme'), 'dark');
});

test('저장 상한: 잡 하나에 READER_NOTES_MAX개까지만 남긴다', (t) => {
  installFakeStorage(t);
  const many = Array.from({ length: READER_NOTES_MAX + 5 }, (_, i) => note({ id: `x${i}`, text: `t${i}` }));
  saveReaderNotes('job-a', many);
  assert.equal(loadReaderNotes('job-a').length, READER_NOTES_MAX);
});

/* ---------------- 런타임: 리더 ---------------- */

function setupNotes(t) {
  const doc = setupReader(t);
  const storage = installFakeStorage(t);
  el.readerNotesList = mount(doc, 'ol');
  el.readerNotesEmpty = mount(doc, 'p');
  el.readerNotesBadge = mount(doc, 'span');
  el.readerNotesCopy = mount(doc, 'button');
  el.readerNotesExport = mount(doc, 'button');
  el.viewerToggleNav = mount(doc, 'button');
  el.viewerToggleRail = mount(doc, 'button');
  el.viewerFilename = mount(doc, 'strong');
  el.viewerFilename.textContent = 'paper.pdf';
  state.readerNotes = [];
  state.toastTimer = 0;
  t.after(() => { clearTimeout(state.toastTimer); });
  return { doc, storage };
}

test('인용 저장: 잡별로 남고 목록·배지·내보내기 버튼이 살아난다', (t) => {
  const { storage } = setupNotes(t);
  state.readerSelection = '저장할 인용 문장';
  state.readerSelectionPage = 2;
  saveReaderCitation();
  const stored = JSON.parse(storage.getItem(readerNotesKey('job-a')));
  assert.deepEqual(stored.items.map((n) => [n.kind, n.page, n.text]), [['citation', 2, '저장할 인용 문장']]);
  assert.equal(el.readerNotesList.querySelectorAll('li').length, 1);
  assert.match(el.readerNotesList.textContent, /저장할 인용 문장/);
  assert.equal(el.readerNotesBadge.hidden, false);
  assert.equal(el.readerNotesBadge.textContent, '저장 1');
  assert.equal(el.readerNotesCopy.disabled, false);
  assert.equal(el.readerNotesExport.disabled, false);
  assert.equal(el.readerNotesEmpty.hidden, true);
  assert.match(el.toast.textContent, /2페이지 인용을 저장했습니다/);

  // 잡을 다시 열어도(리더 상태 초기화) 목록이 그대로다 — 예전에는 여기서 사라졌다.
  state.readerNotes = [];
  resetReaderForJob();
  assert.equal(state.readerNotes.length, 1);
  assert.equal(el.readerNotesList.querySelectorAll('li').length, 1);
});

test('목록의 페이지 링크는 그 페이지로, 삭제는 저장소와 목록에서 지운다', (t) => {
  const { storage } = setupNotes(t);
  state.readerSelection = '두 번째 쪽 문장';
  state.readerSelectionPage = 2;
  saveReaderCitation();
  el.readerNotesList.querySelector('.reader-note-page').click();
  assert.equal(state.readerPage, 2);
  el.readerNotesList.querySelector('.reader-note-del').click();
  assert.equal(storage.getItem(readerNotesKey('job-a')), null);
  assert.equal(el.readerNotesList.querySelectorAll('li').length, 0);
  assert.equal(el.readerNotesBadge.hidden, true);
  assert.equal(el.readerNotesCopy.disabled, true);
});

test('저장된 하이라이트는 레일을 다시 그릴 때 텍스트 조각 단위로 되살아난다(중복 없이)', (t) => {
  setupNotes(t);
  state.readerNotes = normalizeReaderNotes([
    note({ id: 'h1', kind: 'highlight', page: 1, lang: 'orig', text: 'Block p1-b1' }),
    note({ id: 'h2', kind: 'highlight', page: 1, lang: 'ko', text: 'Block p1-b2' }), // 다른 언어
  ]);
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1', 'p1-b2']));
  renderReaderDocument();
  const card = el.readerContent.querySelector('[data-block-id="p1-b1"]');
  const marks = card.querySelectorAll('mark.reader-highlight');
  assert.equal(marks.length, 1);
  assert.equal(marks[0].dataset.noteId, 'h1');
  assert.equal(marks[0].textContent, 'Block p1-b1');
  assert.equal(el.readerContent.querySelectorAll('mark.reader-highlight').length, 1, '다른 언어 하이라이트는 칠하지 않는다');
  assert.equal(el.readerContent.querySelectorAll('[data-block-id="p1-b1"]').length, 1, '카드가 복제되지 않는다');

  state.readerRailKey = ''; // 언어 전환처럼 레일을 새로 만든다
  renderReaderDocument();
  assert.equal(el.readerContent.querySelectorAll('mark.reader-highlight').length, 1);

  deleteReaderNote('h1');
  assert.equal(el.readerContent.querySelectorAll('mark.reader-highlight').length, 0);
  assert.equal(el.readerContent.querySelector('[data-block-id="p1-b1"] .reader-map-target').textContent,
    'Block p1-b1', '하이라이트를 지워도 본문은 그대로');
});

test('카드 경계를 넘는 하이라이트도 조각마다 감싸고 머리말은 건너뛴다', (t) => {
  setupNotes(t);
  state.readerNotes = normalizeReaderNotes([
    note({ id: 'h1', kind: 'highlight', page: 1, lang: 'orig', text: 'p1-b1 Block p1' }),
  ]);
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1', 'p1-b2']));
  renderReaderDocument();
  const marks = el.readerContent.querySelectorAll('mark.reader-highlight');
  assert.deepEqual(marks.map((m) => m.textContent), ['p1-b1', 'Block p1']);
  assert.ok(marks.every((m) => !m.closest('.reader-map-card-head')), '카드 머리말(번호·종류)은 칠하지 않는다');
  assert.equal(el.readerContent.querySelectorAll('.reader-map-card').length, 2);
});

test('Markdown 복사: 클립보드에 내보내기 문서를 넣는다', async (t) => {
  setupNotes(t);
  state.readerSelection = '복사할 문장';
  state.readerSelectionPage = 1;
  saveReaderCitation();
  const copied = [];
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: { clipboard: { writeText: async (text) => { copied.push(text); } } }, configurable: true,
  });
  t.after(() => {
    if (saved) Object.defineProperty(globalThis, 'navigator', saved);
    else delete globalThis.navigator;
  });
  await copyReaderNotes();
  assert.equal(copied.length, 1);
  assert.match(copied[0], /^# paper\.pdf — 인용·하이라이트/);
  assert.match(copied[0], /> 복사할 문장/);
  assert.match(el.toast.textContent, /Markdown으로 복사했습니다/);
});

test('잡을 삭제하면 그 잡의 인용·하이라이트 저장값도 지운다', async (t) => {
  const { doc, storage } = setupNotes(t);
  saveReaderNotes('job-z', [note()]);
  el.jobList = mount(doc, 'ul');
  el.jobListEmpty = mount(doc, 'p');
  state.jobs = [{ job_id: 'job-z', filename: 'z.pdf', status: 'done' }];
  t.mock.method(globalThis, 'fetch', async (url, init) => ({
    ok: true, status: init && init.method === 'DELETE' ? 204 : 200,
    headers: { get: () => null }, text: async () => '{"jobs":[]}',
  }));
  await deleteJob('job-z');
  assert.equal(storage.getItem(readerNotesKey('job-z')), null);
});

/* ---------------- 런타임: 같은 잡을 연 다른 탭 (frontend-1) ---------------- */
// 다른 탭은 같은 localStorage에 쓴다 — 여기서는 그 탭의 저장을 saveReaderNotes로 흉내 낸다.

const storedNotes = (storage, jobId = 'job-a') => {
  const raw = storage.getItem(readerNotesKey(jobId));
  return raw ? JSON.parse(raw).items : [];
};

test('다른 탭이 그사이 저장한 메모를 덮어쓰지 않는다', (t) => {
  const { storage } = setupNotes(t);
  resetReaderForJob(); // 이 탭이 잡을 열 때는 메모가 없었다
  saveReaderNotes('job-a', [note({ id: 'b1', page: 1, text: '다른 탭의 인용' })]);
  state.readerSelection = '이 탭의 인용';
  state.readerSelectionPage = 2;
  saveReaderCitation();
  assert.deepEqual(storedNotes(storage).map((n) => n.text).sort(), ['다른 탭의 인용', '이 탭의 인용'].sort(),
    '예전에는 이 탭의 사본으로 목록 전체를 덮어써 다른 탭의 인용이 사라졌다');
  assert.equal(el.readerNotesList.querySelectorAll('li').length, 2, '이 탭 목록에도 다른 탭의 메모가 보인다');
  assert.match(el.toast.textContent, /2페이지 인용을 저장했습니다/);
});

test('다른 탭이 지운 메모를 다음 저장이 되살리지 않는다', (t) => {
  const { storage } = setupNotes(t);
  saveReaderNotes('job-a', [note({ id: 'x1', text: '지운 메모' }), note({ id: 'x2', text: '남은 메모' })]);
  resetReaderForJob();
  assert.equal(state.readerNotes.length, 2);
  saveReaderNotes('job-a', [note({ id: 'x2', text: '남은 메모' })]); // 다른 탭이 x1을 지웠다
  state.readerSelection = '새 인용';
  state.readerSelectionPage = 1;
  saveReaderCitation();
  const texts = storedNotes(storage).map((n) => n.text);
  assert.ok(!texts.includes('지운 메모'), texts.join(' | '));
  assert.deepEqual(texts.sort(), ['남은 메모', '새 인용'].sort());
  assert.ok(!el.readerNotesList.querySelector('[data-note-id="x1"]'), '목록에서도 사라진다');
});

test('삭제도 최신 저장값에서 뺀다 — 다른 탭이 새로 저장한 메모는 남는다', (t) => {
  const { storage } = setupNotes(t);
  saveReaderNotes('job-a', [note({ id: 'd1', text: '지울 메모' })]);
  resetReaderForJob();
  saveReaderNotes('job-a', [note({ id: 'd1', text: '지울 메모' }), note({ id: 'd2', text: '다른 탭의 새 메모' })]);
  deleteReaderNote('d1');
  assert.deepEqual(storedNotes(storage).map((n) => n.id), ['d2'],
    '예전에는 이 탭의 사본([d1])에서 지운 빈 목록으로 키를 지워 d2까지 사라졌다');
  assert.deepEqual(state.readerNotes.map((n) => n.id), ['d2']);
  // 다른 탭이 이미 지운 메모를 지우면 저장 없이 화면만 맞춘다
  saveReaderNotes('job-a', []);
  deleteReaderNote('d2');
  assert.deepEqual(state.readerNotes, []);
  assert.equal(el.readerNotesList.querySelectorAll('li').length, 0);
});

test('storage 이벤트: 다른 탭의 저장·삭제를 목록과 하이라이트에 맞춘다', (t) => {
  const { storage } = setupNotes(t);
  saveReaderNotes('job-a', [note({ id: 'h1', kind: 'highlight', page: 1, lang: 'orig', text: 'Block p1-b1' })]);
  state.readerNotes = loadReaderNotes('job-a'); // 잡을 열 때 읽은 목록
  state.readerAlignments.orig.set(1, alignment(1, ['p1-b1', 'p1-b2']));
  renderReaderDocument();
  const marks = () => el.readerContent.querySelectorAll('mark.reader-highlight')
    .map((m) => [m.dataset.noteId, m.textContent]);
  assert.deepEqual(marks(), [['h1', 'Block p1-b1']]);

  // 다른 탭: h1을 지우고 h2를 칠했다
  saveReaderNotes('job-a', [note({ id: 'h2', kind: 'highlight', page: 1, lang: 'orig', text: 'Block p1-b2' })]);
  onReaderNotesStorage({ key: readerNotesKey('job-a') });
  assert.deepEqual(state.readerNotes.map((n) => n.id), ['h2']);
  assert.deepEqual(marks(), [['h2', 'Block p1-b2']], '사라진 하이라이트는 걷고 새 하이라이트를 칠한다');
  assert.equal(el.readerContent.querySelector('[data-block-id="p1-b1"] .reader-map-target').textContent,
    'Block p1-b1', '걷어도 본문은 그대로');
  assert.deepEqual(el.readerNotesList.querySelectorAll('li').map((li) => li.dataset.noteId), ['h2']);

  // 다른 잡의 키는 무시하고, 저장소 전체 비우기(key=null)는 다시 읽는다
  storage.setItem(readerNotesKey('job-b'), JSON.stringify({ v: 1, updated: 1, items: [note({ id: 'zz' })] }));
  onReaderNotesStorage({ key: readerNotesKey('job-b') });
  assert.deepEqual(state.readerNotes.map((n) => n.id), ['h2']);
  storage.clear();
  onReaderNotesStorage({ key: null });
  assert.deepEqual(state.readerNotes, []);
  assert.deepEqual(marks(), []);
  assert.equal(el.readerNotesBadge.hidden, true);
});
