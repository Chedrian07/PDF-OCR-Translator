# Ported from baidu/Unlimited-OCR modeling_unlimitedocr.py (MIT, Copyright (c) 2026 Baidu;
# 라이선스 전문: ../unlimited_ocr/LICENSE) — infer()/infer_multi()의 전처리 부분.
# mlx-vlm processing_unlimitedocr.py 대신 torch 레퍼런스를 직접 옮겼다(앱이 검증한 경로).
# 출처·패치 내역: PROVENANCE.md
"""프롬프트 토큰·이미지 전처리 (torch-free, numpy/PIL만).

torch 레퍼런스와 같은 PIL 호출(``ImageOps.exif_transpose``·``ImageOps.pad``·
``Image.resize``·``dynamic_preprocess``)을 같은 순서로 하고, torchvision
``ToTensor()+Normalize(0.5, 0.5)``와 같은 float32 산술로 픽셀을 만든다 — 그래서
입력 토큰 ids·이미지 토큰 마스크·픽셀 값이 torch 경로와 비트 단위로 같다
(tests/test_mlx_model_parity.py가 고정). 픽셀 배열은 MLX conv 규약인 NHWC다.

두 모드:
- 멀티페이지(``prepare_multi``, torch infer_multi): 단일 ``<image>`` 자리에 페이지별
  ``([img]*q + [img])*q + [img]`` 토큰(1024px: q=16 → 273개)을 잇는다. 크롭 없음,
  ``ImageOps.pad`` 채움색 127.
- 단일 gundam(``prepare_single``, torch infer crop_mode=True): 전역 뷰(base_size) +
  ``dynamic_preprocess`` 타일(640px) — 둘 중 한 변이라도 640을 넘을 때만 타일링.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import BOS_ID, IMAGE_TOKEN, IMAGE_TOKEN_ID

PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)
# torch: color=tuple(int(x * 255) for x in image_transform.mean) → (127, 127, 127)
PAD_COLOR = tuple(int(x * 255) for x in _MEAN)


@dataclass
class OCRInputs:
    """generate()·후처리에 넘길 전처리 결과 (torch gen_kwargs와 같은 정보)."""

    input_ids: np.ndarray  # int64 [L] — BOS + 텍스트 + 이미지 토큰
    images_seq_mask: np.ndarray  # bool [L] — 이미지 특징이 들어갈 자리
    global_views: np.ndarray  # float32 [N, S, S, 3] (torch images_ori, NHWC)
    crops: np.ndarray | None  # float32 [M, 640, 640, 3] (gundam 타일) | None
    images_spatial_crop: list[list[int]]  # 이미지별 [width_crop_num, height_crop_num]
    images: list[Image.Image] = field(repr=False)  # EXIF 보정 RGB 원본 — 후처리 크롭/오버레이용
    mode: str = "multi"  # "multi" | "single"

    @property
    def prompt_length(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def spatial_crop(self) -> tuple[int, int]:
        """torch forward가 실제로 쓰는 첫 행 (zip(images, images_spatial_crop))."""
        w, h = self.images_spatial_crop[0]
        return int(w), int(h)


# ── torch modeling_unlimitedocr.py 함수 이식 (동작 동일) ──


def load_image(image_path):
    """EXIF 회전 보정 — 실패하면 원본, 그것도 실패하면 None (torch와 동일)."""
    try:
        image = Image.open(image_path)
        return ImageOps.exif_transpose(image)
    except Exception as e:  # noqa: BLE001 - 업스트림 동작 유지
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except Exception:  # noqa: BLE001
            return None


def load_pil_images(image_files) -> list[Image.Image]:
    images = []
    for image_path in image_files:
        pil_img = load_image(image_path)
        if pil_img is None:  # torch는 여기서 AttributeError — 원인이 보이게 명시
            raise ValueError(f"이미지를 열 수 없습니다: {image_path}")
        images.append(pil_img.convert("RGB"))
    return images


def format_plain_prompt(prompt: str) -> str:
    """torch format_messages(sft_format='plain')의 결과.

    plain 템플릿은 sep·sep2·system이 모두 빈 문자열이라
    [(User, prompt.strip()), (Assistant, "")] → ``prompt.strip()`` 로 축약된다."""
    return prompt.strip()


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False) -> list[int]:
    t = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        t = [BOS_ID] + t
    if eos:
        t = t + [1]
    return list(t)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=32, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio


def image_to_pixels(image: Image.Image) -> np.ndarray:
    """torchvision ``ToTensor()`` + ``Normalize(0.5, 0.5)``와 같은 float32 값 (HWC).

    ToTensor: uint8 → float32 ÷ 255, Normalize: (x - mean) ÷ std — 같은 IEEE 연산 순서."""
    a = np.asarray(image, dtype=np.uint8).astype(np.float32)
    a = a / np.float32(255.0)
    mean = np.asarray(_MEAN, dtype=np.float32)
    std = np.asarray(_STD, dtype=np.float32)
    return ((a - mean) / std).astype(np.float32)


def _num_queries(size: int) -> int:
    return math.ceil((size // PATCH_SIZE) / DOWNSAMPLE_RATIO)


def _split_single_image_prompt(formatted: str) -> list[str]:
    splits = formatted.split(IMAGE_TOKEN)
    if len(splits) != 2:
        raise ValueError(f"프롬프트에는 {IMAGE_TOKEN} 토큰이 정확히 1개 있어야 합니다: {formatted!r}")
    return splits


def prepare_multi(tokenizer, prompt: str, image_files, image_size: int = 640) -> OCRInputs:
    """torch ``infer_multi``의 전처리 — 모든 페이지를 단일 ``<image>`` 자리에 잇는다."""
    if image_files is None or len(image_files) == 0:
        raise ValueError("image_files must be a non-empty list for multi-image inference!")
    formatted = format_plain_prompt(prompt)
    images = load_pil_images([str(Path(p)) for p in image_files])
    text_splits = _split_single_image_prompt(formatted)

    num_queries = _num_queries(image_size)
    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    pixels: list[np.ndarray] = []
    spatial: list[list[int]] = []

    tokenized_sep = text_encode(tokenizer, text_splits[0], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    for image in images:
        if image_size <= 640:
            image = image.resize((image_size, image_size))
        global_view = ImageOps.pad(image, (image_size, image_size), color=PAD_COLOR)
        pixels.append(image_to_pixels(global_view))
        spatial.append([1, 1])
        tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
        tokenized_image += [IMAGE_TOKEN_ID]  # 페이지 사이 구분 토큰 (view_separator 자리)
        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = text_encode(tokenizer, text_splits[1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_str = [BOS_ID] + tokenized_str
    images_seq_mask = [False] + images_seq_mask
    return OCRInputs(
        input_ids=np.asarray(tokenized_str, dtype=np.int64),
        images_seq_mask=np.asarray(images_seq_mask, dtype=bool),
        global_views=np.stack(pixels, axis=0),
        crops=None,
        images_spatial_crop=spatial,
        images=images,
        mode="multi",
    )


def prepare_single(
    tokenizer,
    prompt: str,
    image_file,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
) -> OCRInputs:
    """torch ``infer``(prompt + image_file)의 전처리 — gundam: crop_mode=True."""
    formatted = format_plain_prompt(prompt)
    images = load_pil_images([str(Path(image_file))])
    text_splits = _split_single_image_prompt(formatted)

    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    pixels: list[np.ndarray] = []
    crop_pixels: list[np.ndarray] = []
    spatial: list[list[int]] = []

    for text_sep, image in zip(text_splits, images):
        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:
            images_crop_raw: list[Image.Image] = []
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
            else:
                # torch는 image_size 인자 없이 호출한다(기본 640) — 동작 동일하게 유지
                images_crop_raw, crop_ratio = dynamic_preprocess(image)
            global_view = ImageOps.pad(image, (base_size, base_size), color=PAD_COLOR)
            pixels.append(image_to_pixels(global_view))
            width_crop_num, height_crop_num = crop_ratio
            spatial.append([width_crop_num, height_crop_num])
            if width_crop_num > 1 or height_crop_num > 1:
                for crop in images_crop_raw:
                    crop_pixels.append(image_to_pixels(crop))
            num_queries = _num_queries(image_size)
            num_queries_base = _num_queries(base_size)
            tokenized_image = (
                [IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]
            ) * num_queries_base
            tokenized_image += [IMAGE_TOKEN_ID]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]
                ) * (num_queries * height_crop_num)
        else:
            if image_size <= 640:
                image = image.resize((image_size, image_size))
            global_view = ImageOps.pad(image, (image_size, image_size), color=PAD_COLOR)
            pixels.append(image_to_pixels(global_view))
            spatial.append([1, 1])
            num_queries = _num_queries(image_size)
            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
            tokenized_image += [IMAGE_TOKEN_ID]
        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_str = [BOS_ID] + tokenized_str
    images_seq_mask = [False] + images_seq_mask
    return OCRInputs(
        input_ids=np.asarray(tokenized_str, dtype=np.int64),
        images_seq_mask=np.asarray(images_seq_mask, dtype=bool),
        global_views=np.stack(pixels, axis=0),
        crops=np.stack(crop_pixels, axis=0) if crop_pixels else None,
        images_spatial_crop=spatial,
        images=images,
        mode="single",
    )
