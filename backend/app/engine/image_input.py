"""Conservative contrast recovery for model views of washed-out paper scans.

The caller keeps its original image for figure crops and overlays. This policy
changes only model-input intensities, without resizing or thresholding pixels.
"""

from PIL import Image, ImageOps


def _washed_paper(image: Image.Image) -> bool:
    with image.convert("L") as gray:
        histogram = gray.histogram()
    total = sum(histogram)
    if not total or sum(histogram[245:]) * 100 < total * 85:
        return False

    # Match autocontrast's lower 1% cutoff: use the first level that still
    # contains a sample after discarding the lowest total // 100 pixels.
    discarded = total // 100
    cumulative = 0
    for level, count in enumerate(histogram):
        cumulative += count
        if cumulative > discarded:
            return 48 < level <= 128
    return False


def normalize_model_image(image: Image.Image) -> Image.Image:
    """Return a new contrast-restored view only for predominantly white paper.

    Strong black ink, blank/faint-only pages, and most photographic content stay
    unchanged. The upper tail is never discarded, preserving faint details.
    Unsupported modes retain their original representation and ownership.
    """
    if image.mode not in ("RGB", "L") or not _washed_paper(image):
        return image
    normalized = ImageOps.autocontrast(image, cutoff=(1, 0), preserve_tone=True)
    # A shared grayscale LUT can leave saturated colors above the black-point
    # threshold. Abstain on those ambiguous images, rather than applying another
    # contrast adjustment the next time the same image is prepared.
    if _washed_paper(normalized):
        normalized.close()
        return image
    return normalized
