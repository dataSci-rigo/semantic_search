"""open_rgb: bounded, upright decodes — run against both the package helper
and the worker-side copy in scripts/, which must stay in step."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from image_search import image_io as package_io

_WORKER_COPY = Path(__file__).resolve().parents[1] / "scripts" / "image_io.py"
_spec = importlib.util.spec_from_file_location("worker_image_io", _WORKER_COPY)
worker_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker_io)


@pytest.fixture(params=[package_io.open_rgb, worker_io.open_rgb], ids=["package", "worker"])
def open_rgb(request):
    return request.param


def test_large_jpeg_is_decoded_within_max_side(tmp_path, open_rgb):
    path = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), (200, 100, 50)).save(path, quality=90)
    im = open_rgb(path, max_side=1024)
    assert im.mode == "RGB"
    assert im.size == (1024, 768)  # long side capped, aspect kept


def test_small_image_keeps_its_size(tmp_path, open_rgb):
    path = tmp_path / "small.png"
    Image.new("RGBA", (300, 200)).save(path)
    im = open_rgb(path, max_side=1024)
    assert im.size == (300, 200)
    assert im.mode == "RGB"


def test_exif_orientation_is_applied(tmp_path, open_rgb):
    # Phone photos are often stored sideways with an EXIF rotate-90 tag.
    path = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (400, 300)).save(path, exif=exif)
    assert open_rgb(path, max_side=1024).size == (300, 400)


def test_exif_orientation_survives_a_reduced_scale_decode(tmp_path, open_rgb):
    path = tmp_path / "phone_big.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (4000, 3000)).save(path, exif=exif)
    assert open_rgb(path, max_side=1000).size == (750, 1000)


def test_truncated_jpeg_still_decodes(tmp_path, open_rgb):
    # A copy cut short by bad sectors keeps its readable part.
    full = tmp_path / "full.jpg"
    Image.new("RGB", (800, 600), (10, 200, 30)).save(full, quality=95)
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(full.read_bytes()[: full.stat().st_size // 2])
    assert open_rgb(cut, max_side=1024).size == (800, 600)
