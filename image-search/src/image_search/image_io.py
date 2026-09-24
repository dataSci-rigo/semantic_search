from __future__ import annotations

from PIL import Image, ImageOps


def open_rgb(path, max_side: int) -> Image.Image:
    """Decode an image upright (EXIF orientation applied) and no larger than
    `max_side` on its long side. JPEGs decode at a reduced DCT scale when that
    still covers the target (Image.draft) — several times cheaper than fully
    decoding a 12-megapixel photo only to shrink it for a 224px model.

    scripts/image_io.py is a copy for the worker processes, which run in
    other conda envs and deliberately don't import this package."""
    im = Image.open(path)
    w, h = im.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        im.draft("RGB", (max(1, round(w * ratio)), max(1, round(h * ratio))))
    im = ImageOps.exif_transpose(im.convert("RGB"))
    im.thumbnail((max_side, max_side))
    return im
