/* Unlimited-OCR — 내려받는 standalone HTML의 KaTeX 조판(크기 묶기 + 원문 폴백).
 *
 * 앱 화면은 js/ui.js renderMath(core.clampTexSizes·katexStyleOversized, constants.katexOptions)로
 * 조판한다. 다운로드용 standalone HTML(layout.py가 KaTeX 번들 뒤에 이 파일을 그대로 <script>로
 * 인라인)은 ES 모듈을 쓸 수 없어 같은 규칙을 이 클래식 스크립트로 옮겼다.
 * ⚠ SYNC: tests/katex-guard.test.mjs가 두 구현(크기 묶기·style 상한·옵션)을 대조한다.
 *
 * 수식 TeX는 업로드 PDF(OCR·텍스트 레이어)에서 온 신뢰할 수 없는 입력이다. KaTeX maxSize는
 * 양수 크기만 묶어 \raisebox{-4000em}{x} 하나로 내려받은 파일이 수만 px로 늘어났다(감사
 * frontend-4). 크기 인자를 먼저 ±10em으로 묶고, 매크로로 만든 크기처럼 그래도 거대한 결과는
 * 원문 TeX 글자로 둔다.
 * → 반드시 순수 클래식 스크립트(모듈 문법 금지)로 유지할 것.
 */
(function (root) {
  'use strict';

  var MAX_SIZE_EM = 10;   // constants.KATEX_MAX_SIZE_EM
  var MAX_EXPAND = 1000;  // constants.KATEX_MAX_EXPAND
  var MAX_BOX_EM = 100;   // constants.KATEX_MAX_BOX_EM
  var MAX_TEX_CHARS = 10000; // constants.KATEX_MAX_TEX_CHARS — 넘으면 조판하지 않고 원문 TeX

  // 단위 → em (KaTeX 기준: 1em = 10pt, ex = x-height, mu = 1/18 em) — core.js와 같은 표.
  var EM_PER_UNIT = {
    em: 1, ex: 0.431, mu: 1 / 18, pt: 0.1, mm: 7227 / 25400, cm: 7227 / 2540, in: 7.227,
    bp: 0.1 * 803 / 800, pc: 1.2, dd: 0.1 * 1238 / 1157, cc: 1.2 * 1238 / 1157, nd: 0.1 * 685 / 642,
    nc: 1.2 * 685 / 642, sp: 0.1 / 65536, px: 0.1 * 803 / 800,
  };
  // 크기 인자를 받는 명령과 그 인자 머리 — core.js TEX_SIZE_HEADER와 같은 식. 공백·괄호 인자
  // 반복의 상한(64자)이 입력 길이에 선형인 시간을 지킨다(상한 없는 \s*·[^\]]*는 닫는 괄호 없는
  // '\\[' 반복에서 제곱 시간 — 내려받은 HTML을 열 때 23초 멈췄다, delta-api-frontend-infra-1).
  var SIZE_HEADER = new RegExp([
    String.raw`\\(?:kern|mkern|hskip|mskip)(?![a-zA-Z])\s{0,64}(?:\{[^{}]{0,64}\}|(?:[-+]\s{0,64})?(?:\d+(?:\.\d*)?|\.\d+)\s{0,64}[a-z]{2})`,
    String.raw`\\(?:hspace\*?|raisebox)(?![a-zA-Z])\s{0,64}\{[^{}]{0,64}\}`,
    String.raw`\\rule(?![a-zA-Z])\s{0,64}(?:\[[^\]]{0,64}\]\s{0,64})?(?:\{[^{}]{0,64}\}\s{0,64}){1,2}`,
    String.raw`\\\\\s{0,64}\[[^\]]{0,64}\]`,
  ].join('|'), 'g');
  var SIZE_LITERAL = /([-+]?)\s*(\d+(?:\.\d*)?|\.\d+)\s*([a-z]{2})/g;
  var STYLE_EM_LENGTH = /(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)em\b/gi;

  function clampTexSizes(tex) {
    var s = String(tex == null ? '' : tex);
    if (s.indexOf('\\') < 0) return s;
    return s.replace(SIZE_HEADER, function (header) {
      return header.replace(SIZE_LITERAL, function (lit, sign, num, unit) {
        var perEm = EM_PER_UNIT[unit];
        if (!perEm || !(Number(num) * perEm > MAX_SIZE_EM)) return lit; // 모르는 단위는 KaTeX가 오류로 보인다
        return (sign === '-' ? '-' : '') + Number((MAX_SIZE_EM / perEm).toFixed(3)) + unit;
      });
    });
  }

  function styleOversized(style) {
    var text = String(style == null ? '' : style);
    var m;
    STYLE_EM_LENGTH.lastIndex = 0;
    while ((m = STYLE_EM_LENGTH.exec(text)) !== null) {
      if (Math.abs(Number(m[1])) > MAX_BOX_EM) return true;
    }
    return false;
  }

  // constants.katexOptions와 같은 옵션 — 호출마다 새 객체(KaTeX가 옵션을 바꿔도 공유 상태 오염 없음).
  function options(displayMode) {
    return {
      displayMode: !!displayMode,
      throwOnError: false,
      maxSize: MAX_SIZE_EM,
      maxExpand: MAX_EXPAND,
      strict: 'ignore',
      trust: false,
    };
  }

  function oversized(target) {
    var nodes = target.querySelectorAll('[style]');
    for (var i = 0; i < nodes.length; i += 1) {
      if (styleOversized(nodes[i].getAttribute('style'))) return true;
    }
    return false;
  }

  // 수식 하나를 조판한다. true = 끝(조판했거나 원문으로 되돌림), false = 예외(원문 유지).
  function renderMath(katex, target, tex, display) {
    if (String(tex == null ? '' : tex).length > MAX_TEX_CHARS) {
      // KaTeX 조판은 긴 입력에 초선형이다('x+' 20만 자 11초) — 조판하지 않고 원문 TeX로 둔다
      target.textContent = tex;
      target.setAttribute('data-math-fallback', 'too-long');
      target.setAttribute('title', '수식이 너무 길어 원문 TeX로 보여 줍니다');
      return true;
    }
    try {
      katex.render(clampTexSizes(tex), target, options(display));
    } catch (_) {
      target.textContent = tex; // 렌더 불가 tex는 원문 유지
      return false;
    }
    if (oversized(target)) {
      target.textContent = tex;
      target.setAttribute('data-math-fallback', 'oversized');
      target.setAttribute('title', '수식이 표시 크기 한도를 넘어 원문 TeX로 보여 줍니다');
    }
    return true;
  }

  // 서버 렌더러(render.py)가 내보낸 .math-inline/.math-display를 모두 조판한다. KaTeX가 없으면
  // (자산 누락) 원문 LaTeX가 그대로 보이는 그레이스풀 폴백.
  function typesetMath(doc) {
    var katex = root.katex;
    if (!katex || !doc || !doc.querySelectorAll) return;
    var nodes = doc.querySelectorAll('.math-inline,.math-display');
    for (var i = 0; i < nodes.length; i += 1) {
      var e = nodes[i];
      renderMath(katex, e, e.textContent, e.classList.contains('math-display'));
    }
  }

  root.uocrKatexGuard = {
    clampTexSizes: clampTexSizes,
    styleOversized: styleOversized,
    options: options,
    renderMath: renderMath,
    typesetMath: typesetMath,
  };
})(typeof window !== 'undefined' ? window : this);
