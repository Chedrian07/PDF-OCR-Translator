"""번역 PDF에서 업로드 원본의 능동 콘텐츠를 걷어낸다 (감사 security-3).

단일 보기 번역 PDF(`GET /pdf?lang=…`, view 기본값)는 업로드 원본을 열어 텍스트 블록만 바꿔
저장한다. 그래서 원본의 문서 열기 동작(/OpenAction JavaScript)·문서·페이지·주석·필드의 추가
동작(/AA)·문서 JavaScript(/Names/JavaScript)·첨부 파일이 그대로 실려, 같은 서버를 쓰는 다른
사용자가 번역 PDF를 열면 원본 작성자의 스크립트·폼 전송(외부 비컨)·실행 유도가 동작했다.
업로드 PDF는 적대적 입력이다(SECURITY.md). 나란히 보기 PDF는 새 문서에 페이지를 그려 넣어
처음부터 이런 것이 없다.

조판과 무관한 것만 지운다 — 내부 이동(GoTo)·URI 링크·목차·주석 자체는 그대로 둔다.

- 모든 사전의 /AA(추가 동작 — 문서·페이지 열기/닫기, 필드 포커스·키 입력에 자동으로 도는 동작)
- /A·/OpenAction이 위험한 동작이면(/Next 사슬 포함): JavaScript·Launch·SubmitForm·ImportData·
  GoToR·GoToE·Rendition·RichMediaExecute, 또는 /JS 키를 단 동작
- 카탈로그 /Names의 /JavaScript·/EmbeddedFiles, /AcroForm의 /XFA(폼 스크립트), 첨부 창을
  여는 /PageMode /UseAttachments
- 첨부 파일 주석(FileAttachment)의 파일(/FS) — 첨부는 원본에만 남는다

저장(`doc.save(garbage=…)`)이 더 이상 참조되지 않는 스크립트·첨부 객체를 버린다.
"""

from __future__ import annotations

# 자동 실행되거나 문서 밖으로 나가는(프로그램 실행·폼 전송·다른 파일 열기·멀티미디어) 동작
_UNSAFE_ACTIONS = frozenset({
    "JavaScript", "Launch", "SubmitForm", "ImportData", "GoToR", "GoToE",
    "Rendition", "RichMediaExecute",
})
# 직접(인라인) 객체를 따라 내려가는 깊이·동작 /Next 사슬 길이 상한 — 넘으면 위험한 것으로 본다
_MAX_DEPTH = 32


def strip_active_content(fitz, doc) -> int:
    """doc(편집 중인 원본)에서 능동 콘텐츠를 지운다 — 지운 항목 수(로그·테스트용)."""
    mu = fitz.mupdf
    pdoc = mu.pdf_document_from_fz_document(doc.this)
    removed = 0
    catalog = _get(mu, mu.pdf_trailer(pdoc), "Root")
    names = _get(mu, catalog, "Names")
    for key in ("JavaScript", "EmbeddedFiles"):
        if _get(mu, names, key) is not None:
            mu.pdf_dict_dels(names, key)
            removed += 1
    acroform = _get(mu, catalog, "AcroForm")
    if _get(mu, acroform, "XFA") is not None:
        mu.pdf_dict_dels(acroform, "XFA")
        removed += 1
    page_mode = _get(mu, catalog, "PageMode")
    if page_mode is not None and mu.pdf_to_name(page_mode) == "UseAttachments":
        mu.pdf_dict_dels(catalog, "PageMode")
    for num in range(1, mu.pdf_xref_len(pdoc)):
        try:
            obj = mu.pdf_load_object(pdoc, num)
        except Exception:  # noqa: BLE001 — 빈·깨진 xref 항목은 저장에서도 버려진다
            continue
        removed += _scrub(mu, obj, 0)
    return removed


def _get(mu, obj, key: str):
    """사전의 값(간접 참조는 따라간 것) — 사전이 아니거나 키가 없으면 None."""
    if obj is None:
        return None
    obj = mu.pdf_resolve_indirect(obj)
    if not mu.pdf_is_dict(obj):
        return None
    value = mu.pdf_dict_gets(obj, key)
    if mu.pdf_is_null(value):
        return None
    return mu.pdf_resolve_indirect(value)


def _scrub(mu, obj, depth: int) -> int:
    """객체 하나와 그 안의 직접(인라인) 사전·배열을 훑어 능동 콘텐츠 키를 지운다.

    간접 참조는 따라가지 않는다 — 그 객체는 xref를 순회하는 바깥 루프가 따로 본다."""
    if depth > _MAX_DEPTH:
        return 0
    removed = 0
    if mu.pdf_is_dict(obj):
        if _get(mu, obj, "AA") is not None:
            mu.pdf_dict_dels(obj, "AA")
            removed += 1
        for key in ("A", "OpenAction"):
            action = mu.pdf_dict_gets(obj, key)
            if not mu.pdf_is_null(action) and _unsafe(mu, action, 0, set()):
                mu.pdf_dict_dels(obj, key)
                removed += 1
        subtype = _get(mu, obj, "Subtype")
        if (
            subtype is not None and mu.pdf_is_name(subtype)
            and mu.pdf_to_name(subtype) == "FileAttachment" and _get(mu, obj, "FS") is not None
        ):
            mu.pdf_dict_dels(obj, "FS")
            removed += 1
        children = [mu.pdf_dict_get_val(obj, i) for i in range(mu.pdf_dict_len(obj))]
    elif mu.pdf_is_array(obj):
        children = [mu.pdf_array_get(obj, i) for i in range(mu.pdf_array_len(obj))]
    else:
        return 0
    for child in children:
        if not mu.pdf_is_indirect(child) and (mu.pdf_is_dict(child) or mu.pdf_is_array(child)):
            removed += _scrub(mu, child, depth + 1)
    return removed


def _unsafe(mu, action, depth: int, seen: set[int]) -> bool:
    """동작(사전·배열·간접 참조)이나 그 /Next 사슬에 위험한 동작이 있는가.

    목적지 배열(/OpenAction [페이지 /Fit])처럼 동작이 아닌 값은 안전하다."""
    if depth > _MAX_DEPTH:
        return True
    if mu.pdf_is_indirect(action):
        num = mu.pdf_to_num(action)
        if num in seen:
            return False
        seen.add(num)
        action = mu.pdf_resolve_indirect(action)
    if mu.pdf_is_array(action):
        return any(
            _unsafe(mu, mu.pdf_array_get(action, i), depth + 1, seen)
            for i in range(mu.pdf_array_len(action))
        )
    if not mu.pdf_is_dict(action):
        return False
    kind = mu.pdf_dict_gets(action, "S")
    if mu.pdf_is_name(kind) and mu.pdf_to_name(kind) in _UNSAFE_ACTIONS:
        return True
    if not mu.pdf_is_null(mu.pdf_dict_gets(action, "JS")):
        return True
    following = mu.pdf_dict_gets(action, "Next")
    return not mu.pdf_is_null(following) and _unsafe(mu, following, depth + 1, seen)
