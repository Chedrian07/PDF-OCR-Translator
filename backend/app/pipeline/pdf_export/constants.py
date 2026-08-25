"""조판 튜닝 상수 — 실측 근거는 각 상수의 주석에 있다.

이 값들은 단일 모듈 시절 `pdf_export` 전역이었고 테스트가 이름으로 참조한다.
패키지 `__init__`이 그대로 재노출하므로 기존 경로는 계속 동작한다.
"""
from __future__ import annotations

# 번역 텍스트로 교체할 수 있는 블록 타입. 이 밖의 타입(image·equation·table·
# algorithm 등)은 내용이 달라도 원본을 유지한다 — 표 HTML·수식 LaTeX를 평문으로
# 밀어 넣으면 오히려 품질이 나빠진다.
_REPLACEABLE_TYPES = frozenset({
    "text", "title", "list", "caption", "image_caption", "table_caption",
    "page_footnote", "footnote", "aside_text",
})
_SPECIALIST_TYPES = frozenset({"table", "equation", "algorithm"})
# 참고문헌은 제목을 억지로 번역하면 저자명·학술지명·URL 사이에 서로 다른 문자 폭이
# 섞여 원문보다 훨씬 불안정하게 줄바꿈된다. 학술 번역 관례대로 서지 항목은 원문
# 조판을 그대로 보존한다(본문의 인용 번호와 참고문헌 제목은 계속 검색 가능).
_PRESERVE_TYPES = frozenset({"ref_text", "header", "footer"})

# 세로쓰기 블록은 회전 조합이 페이지 회전과 얽혀 배치가 어긋나기 쉽다 — 원본 유지.
_VERTICAL_SKIP = ("up", "down")

# 과도한 축소는 한 블록만 각주처럼 작아지는 계층 붕괴를 만든다. 76%에서도
# 들어가지 않으면 원문을 보존하고 리포트에 남기는 편이 읽을 수 없는 번역보다 낫다.
# 65%까지 열어 실측한 결과 회수는 0건이고 본문 최소 한글 크기만 7.02pt→6.00pt로
# 줄었다(16p 논문 재현본). 미번역의 주 원인은 축소 부족이 아니라 OCR의 가로
# 평탄화와 flow 그룹의 전부-아니면-전무 실패다(보존 26건 → 4건: 개별 배치 13건,
# `_reflow_flattened_text` 9건 회수). 축소 하한은 76%로 유지한다.
_SHRINK_STEPS = (1.0, 0.94, 0.88, 0.82, 0.76)
_SINGLE_LINE_SCALES = (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76)
_MIN_FONT_PT = 4.0
# 본문 흐름 조판의 가독성 절대 하한. 논문 인쇄에서 6pt는 표 각주·판권 표기의
# 최소 크기이고 그 아래는 한글 받침이 뭉친다. 알고리즘 리스팅처럼 base_pt가
# 7.9pt 미만인 블록은 76% 축소만으로도 6pt 밑으로 내려가므로 비율이 아니라 실제
# pt로 막는다(실측 회수 손실 0건). 하한에 걸린 블록은 원문을 보존한다(사유: no_fit).
_MIN_BODY_FONT_PT = 6.0
_MAX_FONT_PT = 72.0
_MAX_TABLE_CELLS = 500  # search_for 셀별 탐색의 CPU 상한 + 비정상 HTML 표 방어
_MIN_TABLE_FONT_PT = 6.0
# booktabs rule의 안티앨리어싱 끝과 CJK 실제 잉크 사이에 남길 최소 간격.
# 0.5pt는 1200dpi에서 약 8px라 선과 첫 행 글자가 하나의 component로 붙지 않는다.
_TABLE_RULE_TEXT_GAP_PT = 0.5
# AppleMyungjo 실측에서 1.36까지 실제 span bbox가 겹쳤다. 1.44는 10/12pt에서
# 0.65/0.78pt의 여유가 있어 합성 bold stroke와 렌더 반올림도 견딘다.
_BODY_LINEHEIGHTS = (1.52, 1.48, 1.44)
_CAPTION_LINEHEIGHTS = (1.48, 1.44)
_TITLE_LINEHEIGHTS = (1.48, 1.44)
_BLOCK_GAP_PT = 5.0
# 서로 인접한 본문을 개별적으로 탐욕 배치하면 앞 블록이 여백을 먼저 차지해 뒤
# 제목/문단이 원문으로 되돌아간다. 이 거리 안의 같은 단 블록은 실패 시 하나의
# flow로 다시 계획한다. 글자 크기와 1.44 행간 하한은 그대로 유지한다.
_FLOW_JOIN_GAP_PT = 22.0
# 줄 단위 조판에서 번역문이 같은 줄 다음 열(또는 다음 세그먼트)의 원문 글리프에
# 닿지 않게 남기는 최소 간격. 1pt면 7pt 리스팅에서 눈에 띄는 붙음이 없다.
_LISTING_COLUMN_GAP_PT = 1.0
_FLOW_UPWARD_SLACK_PT = 48.0
_FLOW_OBSTACLE_GAP_PT = 0.9
