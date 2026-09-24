"""Persistent captioning worker, run under a separate conda env.

Same rationale and protocol as scripts/ocr_worker.py — this process is
deliberately dependency-free w.r.t. the image_search package. The model is
chosen by argv[1] (the config's caption model id, passed by
processors/subprocess_bridge.py):

  moondream2  vikhyatk/moondream2 (trust_remote_code) — the GPU default; its
              model class doesn't load under the `transformers` version
              `sem_search_gpu` needs for native SigLIP2/sentence-transformers
              support, hence the separate env. See docs/gpu-setup.md.
  blip-base   Salesforce/blip-image-captioning-base — plain transformers, no
              remote code, ~1GB: runs on CPU-only machines (set
              IMAGE_SEARCH_CAPTION_ENV to any env with torch+transformers).

  parent -> worker: one absolute image path per line
  worker -> parent: one JSON object per line: {"text": "..."} or {"error": "..."}

First line written is "READY" once the model is loaded.
"""

from __future__ import annotations

import json
import sys

BLIP_REPO = "Salesforce/blip-image-captioning-base"
BLIP_MAX_NEW_TOKENS = 40


def _load_moondream(device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()

    def caption(img) -> str:
        return model.caption(img, length="normal")["caption"]

    return caption


def _load_blip(device: str):
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_REPO)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_REPO).to(device).eval()

    def caption(img) -> str:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=BLIP_MAX_NEW_TOKENS)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    return caption


LOADERS = {
    "moondream2": _load_moondream,
    "blip-base": _load_blip,
}


def main() -> None:
    import os

    import torch

    # Same Pascal cuDNN9 conv2d issue as image_embed.py (docs/gpu-setup.md).
    torch.backends.cudnn.enabled = False

    # The parent pipeline often runs with OMP_NUM_THREADS=1 to keep ITS
    # memory down; inheriting that leaves CPU captioning generating on a
    # single core. Claim most of the machine for this worker unless told
    # otherwise.
    threads = int(
        os.environ.get("IMAGE_SEARCH_CAPTION_THREADS", max(1, (os.cpu_count() or 2) - 2))
    )
    torch.set_num_threads(threads)

    from image_io import open_rgb  # scripts/image_io.py, beside this file

    model_id = sys.argv[1] if len(sys.argv) > 1 else "moondream2"
    loader = LOADERS.get(model_id)
    if loader is None:
        print(
            json.dumps({"error": f"unknown caption model {model_id!r} "
                        f"(known: {sorted(LOADERS)})"}),
            flush=True,
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    caption = loader(device)
    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            # 1024px: BLIP resizes to 384, moondream2 tiles 378px crops.
            img = open_rgb(path, max_side=1024)
            print(json.dumps({"text": caption(img)}), flush=True)
        except Exception as exc:  # noqa: BLE001 - report to parent, keep serving
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
