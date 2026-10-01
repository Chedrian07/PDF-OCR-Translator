/* 테마 부트스트랩 — 첫 페인트 전에 data-theme을 정해 깜빡임을 막는다.
 * 사용자가 고른 테마(localStorage 'uocr-theme')가 없으면 prefers-color-scheme을 따른다.
 *
 * index.html의 CSP(script-src 'self')는 인라인 스크립트를 막으므로 별도 파일로 두고
 * <head>에서 동기(블로킹)로 읽는다 — defer/async를 붙이면 첫 페인트 뒤에 실행된다.
 * 모듈 문법 금지(클래식 스크립트).
 */
(function () {
  try {
    var t = localStorage.getItem('uocr-theme');
    if (t !== 'light' && t !== 'dark') {
      t = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', t);
  } catch (e) {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();
