from image_search.config import Routing, load_config
from image_search.processors.base import LoadedImage
from image_search.processors.tagger import LABEL_PROMPTS, SOURCE, ClipZeroShotTagger


class FakeEmbedder:
    """Stands in for the shared image_embed processor: label vectors are unit
    axes, so an image vector picks its label by which axis it leans on."""

    model_id = "fake-clip"

    def __init__(self):
        self.image_calls = 0
        self.text_calls = 0

    def load(self):
        pass

    def embed_text(self, texts):
        self.text_calls += 1
        return [[1.0 if i == j else 0.0 for j in range(len(texts))] for i in range(len(texts))]

    def embed(self, path):
        self.image_calls += 1
        return [1.0] + [0.0] * (len(LABEL_PROMPTS) - 1)


def test_tagger_emits_one_record_per_label_with_source():
    tagger = ClipZeroShotTagger("fake-clip", embedder=FakeEmbedder())
    # Leans on the 2nd axis -> "chart" (2nd key in LABEL_PROMPTS).
    vec = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0]
    records = tagger.process(LoadedImage("id", "x.png", 4, 4, image_vector=vec))

    assert len(records) == len(LABEL_PROMPTS)
    assert {r.source for r in records} == {SOURCE}
    best = max(records, key=lambda r: r.score)
    assert best.tag == list(LABEL_PROMPTS)[1] == "chart"


def test_tagger_reuses_image_vector_instead_of_re_embedding():
    embedder = FakeEmbedder()
    tagger = ClipZeroShotTagger("fake-clip", embedder=embedder)

    tagger.process(LoadedImage("id", "x.png", 4, 4, image_vector=[1.0, 0, 0, 0, 0, 0]))
    assert embedder.image_calls == 0  # no second vision pass

    tagger.process(LoadedImage("id", "x.png", 4, 4, image_vector=None))
    assert embedder.image_calls == 1  # falls back when image_embed is off


def test_label_prompts_embedded_once():
    embedder = FakeEmbedder()
    tagger = ClipZeroShotTagger("fake-clip", embedder=embedder)
    for _ in range(3):
        tagger.process(LoadedImage("id", "x.png", 4, 4, image_vector=[1.0, 0, 0, 0, 0, 0]))
    assert embedder.text_calls == 1


# ---- routing rules ----------------------------------------------------------

def _ranked(*tags):
    """Build [(tag, score, rank)] in the given order."""
    return [(t, 1.0 - i * 0.01, i + 1) for i, t in enumerate(tags)]


def test_routing_off_runs_everything():
    routing = Routing(auto=False)
    assert routing.wants("ocr", _ranked("photo", "art"))
    assert routing.wants("caption", _ranked("photo", "art"))


def test_ocr_runs_for_text_bearing_images_within_top_two():
    routing = Routing(auto=True)
    assert routing.wants("ocr", _ranked("meme", "art"))
    assert routing.wants("ocr", _ranked("art", "meme"))  # rank 2 still counts
    assert routing.wants("ocr", _ranked("chart", "photo"))
    assert routing.wants("ocr", _ranked("screenshot", "document"))


def test_ocr_skipped_for_photos():
    routing = Routing(auto=True)
    assert not routing.wants("ocr", _ranked("photo", "art", "meme"))  # meme only rank 3


def test_caption_runs_on_text_bearing_images_too():
    """Images with writing get captioning AND ocr — a meme is findable by what
    it says and by what it depicts."""
    routing = Routing(auto=True)
    for top in ("meme", "chart", "screenshot", "document", "photo", "art"):
        assert routing.wants("caption", _ranked(top)), top


def test_routing_defaults_from_yaml_and_overrides(tmp_path):
    path = tmp_path / "folders.yaml"
    path.write_text(
        'folders:\n'
        '  "~/Mixed":\n'
        '    route: auto\n'
        '    ocr: rapidocr\n'
        '  "~/Plain":\n'
        '    ocr: rapidocr\n'
        '  "~/Narrow":\n'
        '    route: auto\n'
        '    ocr: rapidocr\n'
        '    caption_when: [photo]\n'
    )
    config = load_config(path)
    assert config.folders["~/Mixed"].routing.auto is True
    assert config.folders["~/Plain"].routing.auto is False
    narrow = config.folders["~/Narrow"].routing
    assert narrow.caption_when == ("photo",)
    assert not narrow.wants("caption", _ranked("meme"))


def test_bad_route_value_raises(tmp_path):
    path = tmp_path / "folders.yaml"
    path.write_text('folders:\n  "~/X":\n    route: sometimes\n    ocr: rapidocr\n')
    import pytest

    with pytest.raises(ValueError, match="route"):
        load_config(path)
