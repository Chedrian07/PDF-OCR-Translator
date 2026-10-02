# Ported from baidu/Unlimited-OCR modeling_unlimitedocr.py (MIT, Copyright (c) 2026 Baidu;
# 라이선스 전문: ../unlimited_ocr/LICENSE) — torch 벤더의 순수 PIL/regex 후처리와
# infer()/infer_multi()의 save_results 흐름. [local patch M4] torch-free 분리. torch 벤더
# 패치 P9(ast.literal_eval)·P13(boxes.json)·P14(raw_pages.json)·P22(좌표 clamp·크롭 번호
# 소비)·P23(페이지 분할) 포함 — 같은 입력에 torch와 같은 산출물. 출처·패치 내역: PROVENANCE.md
"""det/ref 파싱 → figure 크롭·오버레이 → 마크다운 치환 (torch·mlx 무관, PIL/numpy만).

파일 계약은 torch 경로와 같다(파일 이름·result.md 내용·크롭 픽셀·boxes.json·
raw_pages.json). 오버레이(result_with_boxes*.jpg)의 상자 색은 업스트림처럼 난수라
바이트 비교 대상이 아니다.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import STOP_STR
from .processing import text_encode

logger = logging.getLogger(__name__)


def _dump_raw_pages(output_path, pages) -> None:
    """[torch vendor patch P14] 치환 전 페이지 원문(그라운딩 태그 포함) → raw_pages.json."""
    try:
        with open(f"{output_path}/raw_pages.json", "w", encoding="utf-8") as f:
            json.dump({"pages": list(pages)}, f, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 - 레이아웃은 부가 기능 (업스트림 동작 유지)
        logger.warning("raw_pages.json 기록 실패: %s", e)


def _split_multi_pages(outputs: str) -> list[str]:
    """[torch vendor patch P23] infer_multi 출력 → 페이지 원문 목록 (torch와 같은 규칙).

    업스트림 ``outputs.split("<PAGE>")[1:]``는 첫 마커 앞을 늘 버린다 — 모델이 선행 마커를
    생략하면 1쪽이 result·raw_pages에서 사라지고 뒤 페이지 크롭이 한 칸 앞 래스터
    (images[page_idx])에서 잘렸다. 앱 merge.split_pages와 같은 규칙: 첫 마커 앞이 공백뿐일
    때만 버린다(마커가 0개면 출력 전체가 1쪽, 빈 출력이면 0쪽)."""
    parts = outputs.split("<PAGE>")
    return parts[1:] if not parts[0].strip() else parts


def re_match(text):
    ref_pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(ref_pattern, text, re.DOTALL)

    det_pattern = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"
    for full_match, label, box in re.findall(det_pattern, text, re.DOTALL):
        matches.append((full_match, label, box))

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if _is_image_ref(a_match):  # [torch vendor patch P22] 그림 번호 소비와 같은 판정
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def _is_image_ref(a_match):
    """[torch vendor patch P22] re_match가 마크다운 그림으로 치환하는 매치인가 — 크롭·번호 소비도 같은 판정."""
    return a_match[1].strip() == "image" or "<|ref|>image<|/ref|>" in a_match[0]


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        # [torch vendor patch P9] 모델 출력(=문서 내용이 조종 가능)을 eval()하지 않는다
        cor_list = ast.literal_eval(ref_text[2])
        # [torch vendor patch P22] 비어 있지 않은 list/tuple만 상자 목록 — 아니면 해석 불가(번호 1개)
        if not isinstance(cor_list, (list, tuple)) or not cor_list:
            return None
        if not isinstance(cor_list[0], (list, tuple)):  # 중첩되지 않은 목록은 상자 하나
            cor_list = [cor_list]
    except Exception as e:  # noqa: BLE001 - 이상 좌표는 건너뛴다 (업스트림 동작)
        logger.debug("det 좌표 파싱 실패: %s", e)
        return None
    return (label_type, cor_list)


def _clamp_box(points, image_width, image_height):
    """[torch vendor patch P22] 모델 좌표(0~999 정규화) 상자 → 이미지 안으로 clamp한 픽셀 상자.

    torch 벤더 ``_clamp_box``와 같은 규칙(tests/test_mlx_postprocess.py가 무작위 입력으로
    대조): x/y = int(v/999*size), x는 [0, W], y는 [0, H]로 clamp, clamp 뒤 x2<=x1 또는
    y2<=y1이면 퇴화 상자로 None. 숫자가 아니거나(bool 포함) 유한하지 않거나 4개가 아닌
    좌표도 None — 호출자는 crop/draw를 건너뛴다. 모델 출력은 PDF 내용으로 유도할 수 있어
    거대·음수·뒤집힌 좌표가 그대로 crop에 가면 거대한 검정 크롭이나 Pillow 좌표 산술
    오버플로(GHSA-6r8x-57c9-28j4)에 닿는다."""
    try:
        x1, y1, x2, y2 = points
        coords = []
        for value, size in ((x1, image_width), (y1, image_height), (x2, image_width), (y2, image_height)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            scaled = value / 999 * size
            if not math.isfinite(scaled):
                return None
            coords.append(min(max(int(scaled), 0), size))
    except (TypeError, ValueError, OverflowError):
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_bounding_boxes(image, refs, ouput_path, image_prefix=""):
    image_width, image_height = image.size

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    img_idx = 0
    crop_boxes = {}  # [torch vendor patch P13] figure bbox export용

    for ref in refs:
        try:
            is_image = _is_image_ref(ref)  # [torch vendor patch P22] 마크다운 치환과 같은 기준
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if not result and is_image:
                # [torch vendor patch P22] 좌표를 못 읽은 image ref도 마크다운에는
                # ![](images/{prefix}{idx}.jpg) 한 자리를 차지한다 — 번호를 소비해 뒤 그림이
                # 앞 그림 파일을 가리키는 어긋남을 막는다.
                img_idx += 1
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20,)
                for points in points_list:
                    # [torch vendor patch P22] 좌표를 이미지 안으로 clamp하고 쓸 수 없는 상자
                    # (퇴화·숫자 아님·비유한·개수 오류)는 크롭·boxes.json·오버레이를 건너뛴다.
                    # image 상자는 건너뛰어도 번호를 소비(업스트림의 crop 실패와 같은 규칙)해
                    # 마크다운 참조·boxes.json과 정렬을 유지한다 — 예외로 ref 전체를 버리지 않는다.
                    box = _clamp_box(points, image_width, image_height)
                    if box is None:
                        if is_image:
                            img_idx += 1
                        continue
                    x1, y1, x2, y2 = box

                    if is_image:
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{image_prefix}{img_idx}.jpg")
                            crop_boxes[f"{image_prefix}{img_idx}.jpg"] = {  # [P13]
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "image_width": image_width, "image_height": image_height,
                            }
                        except Exception as e:  # noqa: BLE001 - 업스트림 동작 유지
                            logger.debug("figure 크롭 저장 실패: %s", e)
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                       fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception:  # noqa: BLE001, S110 - 업스트림 동작 유지
                        pass
        except Exception:  # noqa: BLE001, S112 - 업스트림 동작 유지(잘못된 ref 건너뜀)
            continue
    # [torch vendor patch P13] 크롭 시점에만 알 수 있는 bbox(픽셀)·페이지 크기를 남긴다 —
    # infer_multi가 페이지별로 재호출하므로 read-modify-write로 병합 (단일 워커 전제).
    if crop_boxes:
        boxes_path = f"{ouput_path}/boxes.json"
        try:
            with open(boxes_path, encoding="utf-8") as bf:
                existing = json.load(bf)
        except Exception:  # noqa: BLE001
            existing = {}
        existing.update(crop_boxes)
        with open(boxes_path, "w", encoding="utf-8") as bf:
            json.dump(existing, bf, ensure_ascii=False, indent=1)

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path, image_prefix=""):
    return draw_bounding_boxes(image, ref_texts, output_path, image_prefix=image_prefix)


# ── infer()/infer_multi()의 생성 이후 흐름 ──


def decode_outputs(tokenizer, generated_ids) -> str:
    """torch: ``tokenizer.decode(output_ids[0, P:])`` → EOS 문자열 제거 → strip."""
    outputs = tokenizer.decode(list(generated_ids))
    if outputs.endswith(STOP_STR):
        outputs = outputs[: -len(STOP_STR)]
    return outputs.strip()


def count_output_tokens(tokenizer, outputs: str) -> int:
    """torch infer_multi의 두 번째 반환값: ``len(text_encode(tokenizer, outputs))``."""
    return len(text_encode(tokenizer, outputs, bos=False, eos=False))


def _ensure_dirs(output_path) -> None:
    # torch infer/infer_multi는 생성 전에 두 디렉터리를 만든다 (images/는 비어 있어도 존재)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/images", exist_ok=True)


def save_results_single(outputs: str, image, output_path) -> str:
    """torch ``infer(save_results=True)`` 후반부 — 처리된 마크다운을 돌려준다 (P7).

    outputs: decode_outputs() 결과, image: 전처리 전 원본(EXIF 보정 RGB)."""
    _ensure_dirs(output_path)
    _dump_raw_pages(output_path, [outputs])  # [torch vendor patch P14]
    matches_ref, matches_images, mathes_other = re_match(outputs)
    result = process_image_with_refs(image, matches_ref, output_path)

    for idx, a_match_image in enumerate(matches_images):
        outputs = outputs.replace(a_match_image, "![](images/" + str(idx) + ".jpg)\n")
    for a_match_other in mathes_other:
        outputs = outputs.replace(a_match_other, "").replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

    with open(f"{output_path}/result.md", "w", encoding="utf-8") as afile:
        afile.write(outputs)
    result.save(f"{output_path}/result_with_boxes.jpg")
    return outputs


def save_results_multi(outputs: str, images, output_path) -> str:
    """torch ``infer_multi(save_results=True)`` 후반부 — ``<PAGE>`` 마커로 묶은 마크다운."""
    _ensure_dirs(output_path)
    pages = _split_multi_pages(outputs)  # [torch vendor patch P23]
    _dump_raw_pages(output_path, [p.strip() for p in pages])  # [torch vendor patch P14]
    processed_pages = []
    for page_idx, page_output in enumerate(pages):
        page_output = page_output.strip()
        if page_idx >= len(images):
            processed_pages.append(page_output)
            continue

        matches_ref, matches_images, mathes_other = re_match(page_output)
        image_prefix = f"page_{page_idx}_"
        result = process_image_with_refs(
            images[page_idx].copy(), matches_ref, output_path, image_prefix=image_prefix
        )
        result.save(f"{output_path}/result_with_boxes_{page_idx}.jpg")

        for idx, a_match_image in enumerate(matches_images):
            page_output = page_output.replace(a_match_image, f"![](images/{image_prefix}{idx}.jpg)\n")
        for a_match_other in mathes_other:
            page_output = (
                page_output.replace(a_match_other, "").replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
            )
        processed_pages.append(page_output)

    outputs = "<PAGE>\n" + "\n<PAGE>\n".join(processed_pages)
    with open(f"{output_path}/result.md", "w", encoding="utf-8") as afile:
        afile.write(outputs)
    return outputs
