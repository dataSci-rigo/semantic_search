"""Persistent Moondream2 captioning worker, run under `sem_search_caption`.

Same rationale and protocol as scripts/ocr_worker.py — this process is
deliberately dependency-free w.r.t. the image_search package, because
moondream2's trust_remote_code model class doesn't load under the
`transformers` version `sem_search_gpu` needs for native SigLIP2/
sentence-transformers support. See docs/gpu-setup.md.

  parent -> worker: one absolute image path per line
  worker -> parent: one JSON object per line: {"text": "..."} or {"error": "..."}

First line written is "READY" once the model is loaded.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    import torch

    # Same Pascal cuDNN9 conv2d issue as image_embed.py (docs/gpu-setup.md).
    torch.backends.cudnn.enabled = False

    from transformers import AutoModelForCausalLM
    from PIL import Image

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            img = Image.open(path).convert("RGB")
            result = model.caption(img, length="normal")
            print(json.dumps({"text": result["caption"]}), flush=True)
        except Exception as exc:  # noqa: BLE001 - report to parent, keep serving
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
