import { clampReaderPage, docLayoutIsFigureOnly } from './core.js';
import { el, state } from './state.js';
import { h, setDownload } from './ui.js';
import { applyDownloadLangs, initTranslateForJob, resetTranslateUI } from './translate.js';
import { initQaForJob, showQaPlaceholder } from './qa.js';
import { applyPdfExport, restoreReaderPosition } from './reader.js';
import { applyViewerIntent } from './viewer.js';
import { activateTab } from './tabs.js';

/* ============================ Result rendering ============================ */

export function baseName(filename) {
  return String(filename || 'document').replace(/\.pdf$/i, '') || 'document';
}

export function renderResult(job) {
  const r = job.result || {};
  const base = baseName(job.filename);

  // 결과를 새로 렌더할 때마다 번역 UI를 원문 상태로 리셋한다
  // (이전 잡의 ko 선택/EventSource 구독 정리 + 언어 속성 제거).
  resetTranslateUI();

  state.currentBaseName = base;
  // 리더 총 페이지 힌트 — 결과의 원본 페이지 목록 우선, 없으면 진행 스냅샷 (0 = 미상)
  state.readerTotalHint = (Array.isArray(r.pages) && r.pages.length) ||
    Number((job.progress || {}).total_pages) || 0;
  const figOnlyLayout = docLayoutIsFigureOnly(
    state.layoutCapability, state.currentJobEngine, state.healthEngine,
  );
  const noLayoutData = r.has_layout === false;
  // 원문·한국어 대조 PDF 내보내기 가드 — figure_only 엔진은 서버도 409(no-layout)라 함께 숨긴다.
  // has_layout 미제공(구버전 응답)은 undefined 유지 = fail-open.
  state.resultHasLayout = (noLayoutData || figOnlyLayout) ? false : r.has_layout;
  state.resultUrls = {
    markdown: r.markdown_url,
    archive: r.archive_url,
    documentHtml: `/api/jobs/${job.job_id}/document.html`,
    pdf: `/api/jobs/${job.job_id}/pdf?lang=ko&view=dual`,
    viewerManifest: r.viewer_manifest_url || `/api/jobs/${job.job_id}/viewer-manifest`,
  };
  applyDownloadLangs(); // currentLang='orig' → 원문 URL로 세팅
  applyPdfExport();     // 번역 상태가 확인되기 전까지는 숨김 (initTranslateForJob이 갱신)

  renderThumbGrid(el.layoutsGrid, r.layouts, '레이아웃 이미지가 없습니다.');
  renderThumbGrid(el.pagesGrid, r.pages, '원본 페이지 이미지가 없습니다.');

  // reset lazy caches for the newly opened result
  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
  el.mdCode.textContent = '';

  // 이어 읽기 — 이 잡에서 마지막으로 보던 페이지에서 시작한다 (주소창 ?page= 가 있으면
  // applyViewerIntent가 그 값으로 덮어쓴다). 논문은 한 번에 다 읽지 않는다.
  const resumePage = restoreReaderPosition();
  if (resumePage > 1) state.readerPage = clampReaderPage(resumePage, state.readerTotalHint || 0);

  // 완료된 잡은 읽기(리더) 뷰가 기본 — 취소·부분 결과는 기존 동작 유지.
  activateTab(job.status === 'done' ? 'reader' : 'preview');
  el.viewerOpen.hidden = job.status !== 'done';

  // 번역 컨트롤·질문 폼은 완료(done) 잡에만 붙는다 (취소본은 플레이스홀더 유지).
  if (job.status === 'done') {
    initTranslateForJob();
    initQaForJob(job);
    applyViewerIntent();
  } else {
    showQaPlaceholder();
  }
}

// Canceled job: no result object, but partial markdown endpoints still work.
export function renderPartialResult(job) {
  const id = job.job_id;
  const base = baseName(job.filename);
  resetTranslateUI(); // 취소본은 번역 대상이 아니다 (컨트롤 숨김 + PDF 내보내기 숨김)
  el.viewerOpen.hidden = true;
  showQaPlaceholder(); // 질문도 완료(done) 잡 전용 — 플레이스홀더 안내
  // 리더 탭을 열어 보는 경우를 위한 총 페이지 힌트 (부분 결과에는 pages 목록이 없다)
  state.readerTotalHint = Number((job.progress || {}).total_pages) || 0;
  setDownload(el.dlMd, `/api/jobs/${id}/markdown`, `${base}.partial.md`);
  setDownload(el.dlZip, null); // archive returns 409 for unfinished jobs
  setDownload(el.dlDoc, `/api/jobs/${id}/document.html`, `${base}.partial.html`); // 부분 문서도 유효

  renderThumbGrid(el.layoutsGrid, [], '취소된 작업에는 레이아웃 이미지가 제공되지 않습니다.');
  renderThumbGrid(el.pagesGrid, [], '취소된 작업에는 원본 페이지 목록이 제공되지 않습니다.');

  state.previewLoaded = false;
  state.markdownLoaded = false;
  state.docLayoutLoaded = false;
  el.previewBody.innerHTML = '';
  el.doclayoutBody.innerHTML = '';
  el.mdCode.textContent = '';

  activateTab('markdown');
}

export function renderThumbGrid(grid, arr, emptyMsg) {
  grid.textContent = '';
  if (!Array.isArray(arr) || !arr.length) {
    grid.appendChild(h('p', { class: 'grid-empty muted', text: emptyMsg }));
    return;
  }
  arr.forEach((url, i) => {
    const img = h('img', { class: 'thumb-img', loading: 'lazy', decoding: 'async', alt: `${i + 1}번 이미지`, src: url });
    img.addEventListener('error', () => { img.replaceWith(h('span', { class: 'grid-empty muted', text: '로드 실패' })); });
    const a = h('a', { class: 'thumb', href: url, target: '_blank', rel: 'noopener', title: `${i + 1} — 새 탭에서 원본 열기` },
      img, h('span', { class: 'thumb-no', text: String(i + 1) }));
    grid.appendChild(a);
  });
}

/* ============================ Error rendering ============================ */

export function renderError(message, canceled) {
  el.errorSection.classList.toggle('canceled', !!canceled);
  el.errorTitle.textContent = canceled ? '취소됨' : '오류';
  el.errorMessage.textContent = message || (canceled ? '작업이 취소되었습니다.' : '변환 중 오류가 발생했습니다.');
  el.errorHint.textContent = canceled
    ? '중단 시점까지의 부분 결과를 아래 Markdown 탭에서 확인할 수 있습니다.'
    : '다시 시도하려면 왼쪽에서 PDF를 다시 업로드해 주세요.';
}
