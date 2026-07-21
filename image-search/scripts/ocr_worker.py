"""Persistent RapidOCR worker, run under the `sem_search_ocr` conda env.

This process is deliberately dependency-free w.r.t. the image_search package
(it must run in a different env than the main pipeline — see docs/gpu-setup.md
for why OCR and text_embed can't share cuDNN versions on this GPU). It's a
line-based protocol over stdin/stdout so the parent process (running in
`sem_search_gpu`) can keep the model loaded across many images instead of
paying model-load cost per image:

  parent -> worker: one absolute image path per line
  worker -> parent: one JSON object per line: {"text": "..."} or {"error": "..."}

The first line the worker writes is "READY" once the model is loaded, so the
parent knows when it's safe to start sending paths.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    from rapidocr import RapidOCR

    engine = RapidOCR(params={"EngineConfig.onnxruntime.use_cuda": True})
    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            result = engine(path)
            text = "\n".join(result.txts) if result.txts else ""
            print(json.dumps({"text": text}), flush=True)
        except Exception as exc:  # noqa: BLE001 - report to parent, keep serving
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
