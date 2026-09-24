"""Worker-side copy of image_search/image_io.py (the workers run in other
conda envs and deliberately don't import the image_search package). Keep the
two in step; tests/test_image_io.py runs both."""

from __future__ import annotations

from PIL import Image, ImageOps


def open_rgb(path, max_side: int) -> Image.Image:
    """Decode an image upright (EXIF orientation applied) and no larger than
    `max_side` on its long side, using JPEG reduced-scale decoding."""
    im = Image.open(path)
    w, h = im.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        im.draft("RGB", (max(1, round(w * ratio)), max(1, round(h * ratio))))
    im = ImageOps.exif_transpose(im.convert("RGB"))
    im.thumbnail((max_side, max_side))
    return im
