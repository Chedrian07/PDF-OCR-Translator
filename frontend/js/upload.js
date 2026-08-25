import { ICON } from './constants.js';
import { classifyFiles, fileSizeError, selectionSummary, summarizeIssues } from './core.js';
import { el, state } from './state.js';
import { safeParse, showToast } from './ui.js';
import { uploadWithProgress } from './api.js';
import { openJob, refreshJobs, upsertJob } from './jobs.js';

/* ============================ Upload ============================ */


export function validateFile(file) {
  const name = file && file.name ? file.name : '';
  if (!/\.pdf$/i.test(name)) return 'PDF 파일만 업로드할 수 있습니다. (.pdf)';
  const type = file.type || '';
  if (type && !/pdf/i.test(type)) return '올바른 PDF 파일이 아닌 것 같습니다. 파일을 확인해 주세요.';
  return null;
}

// 형식 검증에 이어 서버 상한 크기 사전 검증 — 선택·업로드 직전 공통.
export function fileValidationError(file) {
  return validateFile(file) || fileSizeError(file && file.size, state.maxUploadMb);
}

// 현재 선택(state.selectedFiles)을 file-info와 업로드 버튼에 반영.
export function renderFileInfo() {
  const s = selectionSummary(state.selectedFiles);
  if (!s) {
    el.fileInfo.hidden = true;
    el.uploadBtn.disabled = true;
    return;
  }
  el.fileName.textContent = s.name;
  el.fileName.title = s.title;
  el.fileSize.textContent = s.size;
  el.fileInfo.hidden = false;
  // 업로드 진행 중의 새 선택은 버튼을 되살리지 않는다 — 루프 이중 진입 방지.
  // 진행 중 선택은 setUploading(false)가 끝나며 재활성화된다.
  el.uploadBtn.disabled = state.uploading;
}

// 픽커·드래그드롭 공통 진입점 — 파일별 검증으로 유효분만 선택에 담고, 무효분은
// '건너뜀' 요약을 기존 업로드 에러 영역에 안내한다(전부 무효면 선택 없음).
export function setSelectedFiles(files) {
  const { valid, skipped } = classifyFiles(files, fileValidationError);
  const skipMsg = summarizeIssues('건너뜀', skipped);
  if (skipMsg) showUploadError(skipMsg);
  else hideUploadError();
  state.selectedFiles = valid;
  renderFileInfo();
}

export function clearSelectedFiles() {
  state.selectedFiles = [];
  el.fileInput.value = '';
  el.fileInfo.hidden = true;
  el.uploadBtn.disabled = true;
  hideUploadError();
}

export function showUploadError(msg) {
  el.uploadError.textContent = msg;
  el.uploadError.hidden = false;
}
export function hideUploadError() {
  el.uploadError.hidden = true;
  el.uploadError.textContent = '';
}

export function setUploading(on) {
  state.uploading = on;
  el.uploadBtn.disabled = on || !state.selectedFiles.length;
  el.uploadBtn.textContent = on ? '업로드 중…' : '변환 시작';
  el.dropzone.classList.toggle('disabled', on);
}

// 업로드 진행바 (progress-track/fill 재사용). frac=null이면 총량을 알 수 없는
// 전송(lengthComputable=false) — indeterminate 애니메이션으로 표시.
export function showUploadProgress(frac) {
  el.uploadProgress.hidden = false;
  const indet = frac == null;
  el.uploadProgress.classList.toggle('indeterminate', indet);
  el.uploadProgressFill.style.width = indet ? '' : `${Math.min(100, Math.max(0, frac * 100))}%`;
}

export function hideUploadProgress() {
  el.uploadProgress.hidden = true;
  el.uploadProgress.classList.remove('indeterminate');
  el.uploadProgressFill.style.width = '';
}

export function readMode() {
  const checked = el.modeRadios.find((r) => r.checked);
  return checked ? checked.value : 'multi';
}

export function readDpi() {
  let dpi = parseInt(el.dpiInput.value, 10);
  if (!Number.isFinite(dpi)) dpi = 200;
  dpi = Math.min(400, Math.max(72, dpi));
  el.dpiInput.value = String(dpi);
  return dpi;
}

// HTTP 실패 상태 → 실패 사유 문구. 서버 detail(실시간 상한 포함)을 우선하고,
// 413은 health로 받은 상한, 그마저 없으면 중립 문구.
export function uploadFailureMessage(status, data) {
  const detail = data && typeof data.detail === 'string' ? data.detail : null;
  if (detail) return detail;
  if (status === 413) {
    return state.maxUploadMb
      ? `파일이 너무 큽니다. 더 작은 PDF를 업로드해 주세요. (최대 ${state.maxUploadMb}MB)`
      : '파일이 너무 커서 서버 업로드 상한을 초과했습니다. 더 작은 PDF를 업로드해 주세요.';
  }
  if (status === 400) return '유효하지 않은 PDF 파일입니다.';
  return `업로드에 실패했습니다. (${status})`;
}

// 순차 다중 업로드 — 파일별로 기존 XHR 경로(uploadWithProgress)를 재사용하고
// 진행바는 파일 단위로 리셋한다. 첫 성공 잡은 즉시 openJob(대기하지 않음 —
// 업로드 도중 잡 전환이 일어나도 루프는 계속), 이후 성공은 목록 upsert만.
// 개별 실패는 수집해 끝에 요약하고, 실패분만 선택에 남겨 재시도할 수 있게 한다.
export async function handleUpload() {
  if (state.uploading || !state.selectedFiles.length) return; // 재진입 가드
  hideUploadError();

  // 선택 시점에는 health 미수신이었어도 이후 수신됐으면 여기서 한 번 더 차단
  const { valid, skipped } = classifyFiles(state.selectedFiles, fileValidationError);
  if (!valid.length) {
    showUploadError(summarizeIssues('건너뜀', skipped) || '업로드할 수 있는 파일이 없습니다.');
    return;
  }

  const selectionAtStart = state.selectedFiles;
  const mode = readMode();
  const dpi = readDpi();
  const failures = skipped.slice(); // {file, name, reason} — 뒤늦게 걸러진 파일도 요약에 포함
  let successCount = 0;
  let firstJobId = null;

  setUploading(true);
  for (let i = 0; i < valid.length; i += 1) {
    const file = valid[i];
    if (valid.length > 1) el.uploadBtn.textContent = `업로드 중… (${i + 1}/${valid.length})`;
    showUploadProgress(0); // 파일 단위 리셋

    const form = new FormData();
    form.append('file', file);
    form.append('mode', mode);
    form.append('dpi', String(dpi));

    let res;
    try {
      res = await uploadWithProgress('/api/jobs', form, showUploadProgress);
    } catch (_) {
      failures.push({ file, name: file.name, reason: '네트워크 오류가 발생했습니다. 다시 시도해 주세요.' });
      continue;
    }
    const data = res.text ? safeParse(res.text) : null;
    if (!(res.status >= 200 && res.status < 300)) {
      failures.push({ file, name: file.name, reason: uploadFailureMessage(res.status, data) });
      continue;
    }
    const jobId = data && data.job_id;
    if (!jobId) {
      failures.push({ file, name: file.name, reason: '서버 응답이 올바르지 않습니다.' });
      continue;
    }

    successCount += 1;
    // Optimistically add to history; the first success opens + streams.
    upsertJob({
      job_id: jobId,
      filename: file.name,
      status: data.status || 'queued',
      mode,
      created_at: new Date().toISOString(),
      progress: {},
      result: null,
      error: null,
    });
    if (firstJobId === null) {
      firstJobId = jobId;
      openJob(jobId); // 나머지 업로드와 병행 — 루프는 currentJobId에 의존하지 않는다
    }
  }
  hideUploadProgress();

  // 업로드 도중 사용자가 선택을 바꿨다면(드롭 등) 그 새 선택은 건드리지 않는다.
  if (state.selectedFiles === selectionAtStart) {
    state.selectedFiles = failures.map((f) => f.file); // 실패분만 남겨 재시도 가능
    if (!failures.length) el.fileInput.value = '';
    renderFileInfo();
  }
  setUploading(false);

  if (failures.length) {
    showUploadError(summarizeIssues('업로드 실패', failures));
    showToast(`업로드 요약 — 성공 ${successCount} · 실패 ${failures.length}`,
      successCount ? 'warn' : 'error');
  } else if (valid.length > 1) {
    showToast(`${successCount}개 파일이 업로드되었습니다.`);
  }
  refreshJobs();
}

/* ============================ Dropzone wiring ============================ */

export function setupDropzone() {
  const dz = el.dropzone;
  dz.addEventListener('click', () => el.fileInput.click());
  dz.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); el.fileInput.click(); }
  });
  dz.addEventListener('dragover', (ev) => { ev.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', (ev) => {
    if (ev.target === dz) dz.classList.remove('dragover');
  });
  dz.addEventListener('drop', (ev) => {
    ev.preventDefault();
    dz.classList.remove('dragover');
    const files = ev.dataTransfer && ev.dataTransfer.files;
    if (files && files.length) setSelectedFiles(files);
  });
  el.fileInput.addEventListener('change', () => {
    if (el.fileInput.files && el.fileInput.files.length) setSelectedFiles(el.fileInput.files);
  });
  el.fileClear.innerHTML = ICON.x;
  el.fileClear.addEventListener('click', clearSelectedFiles);
}
