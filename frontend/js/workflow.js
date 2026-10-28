import { el, state } from './state.js';

export function readWorkflow() {
  return el.workflowRadios?.find((radio) => radio.checked)?.value === 'markdown_zip'
    ? 'markdown_zip' : 'reader';
}

export function renderWorkflowChoice(workflow = readWorkflow()) {
  const compact = workflow === 'markdown_zip';
  if (el.workflowRadios) el.workflowRadios.forEach((radio) => { radio.checked = radio.value === workflow; });
  if (!state.uploading && el.uploadBtn) el.uploadBtn.textContent = compact ? 'Markdown ZIP 만들기' : '변환 시작';
  if (el.workflowHelp) el.workflowHelp.hidden = !compact;
  if (el.readerFeatures) el.readerFeatures.hidden = compact;
  if (el.serviceHeading) el.serviceHeading.textContent = compact ? 'PDF를 Markdown과 이미지로 받으세요' : 'PDF 구조를 그대로 읽고 번역하세요';
  if (el.serviceDescription) el.serviceDescription.textContent = compact
    ? 'PDF를 업로드하면 원문 Markdown, 추출 이미지, 모든 원본 페이지 이미지를 하나의 ZIP으로 만듭니다. 번역 설정 없이 사용할 수 있습니다.'
    : '로컬 OCR 모델이 PDF를 실시간으로 분석하고, 원본 레이아웃을 보존한 HTML과 선택적 한국어 PDF까지 만듭니다.';
}

export function setupWorkflowChoice() {
  el.workflowRadios.forEach((radio) => radio.addEventListener('change', () => renderWorkflowChoice()));
  renderWorkflowChoice();
}
