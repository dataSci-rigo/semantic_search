# GPU setup: why two conda envs

This project's GPU is a **Quadro P4000** — a Pascal-generation card (compute
capability `sm_61`, ~2017). That one fact drives everything below.

## The constraint

- **torch** (for `sentence-transformers`) drops Pascal kernels in its CUDA 13
  builds. `torch.cuda.is_available()` still returns `True` on a CUDA-13 wheel
  (it only checks driver/device presence), but any real op fails:
  `CUDA error: no kernel image is available for execution on the device`.
  torch **must** be a CUDA 11.8 build (`--index-url
  https://download.pytorch.org/whl/cu118`) on this GPU — this is a wheel
  build-time limitation, not fixable by juggling library versions.
- **onnxruntime-gpu** doesn't embed its own GPU kernels — it calls cuBLAS/cuDNN
  at runtime, so it isn't tied to a specific compute capability the way torch
  is. But the mainline PyPI `onnxruntime-gpu` (1.21+) only ships CUDA 12+
  builds. The CUDA-11 line tops out at **1.20.1**, published on a separate
  Microsoft feed, and it requires **cuDNN 8** specifically (not 9).
- torch's CUDA-11 wheels pull **cuDNN 9** (`nvidia-cudnn-cu11==9.1.0.70`).
  cuDNN 8 and 9 have different sonames (`libcudnn.so.8` vs `.so.9`) and pip's
  `nvidia-*` packages all install to the same `site-packages/nvidia/<lib>/lib`
  path — so whichever installs last wins, silently breaking the other
  framework. torch and onnxruntime-gpu **cannot share one env** here.

## The fix: two conda envs

- **`sem_search_gpu`** — torch (cu118) for `sentence-transformers` /
  `text_embed`. This is the main env `image-search` itself runs in.
- **`sem_search_ocr`** — `rapidocr` + `onnxruntime-gpu==1.20.1` (from the
  CUDA-11 feed) + `nvidia-cudnn-cu11==8.9.6.50` pinned exactly, with no torch
  at all. Nothing to conflict with.

Both envs need `LD_LIBRARY_PATH` pointed at their own `site-packages/nvidia/*/lib`
dirs — `onnxruntime`/`torch`'s pip-bundled CUDA libraries aren't on the loader
path by default. This is persisted per-env via:

```bash
conda env config vars set -n <env> LD_LIBRARY_PATH="<colon-joined nvidia/*/lib dirs>"
```

(already done for both envs on this machine; re-run if either env's packages
are reinstalled, since paths embed exact site-packages locations).

## How the pipeline bridges the two envs

`RapidOcrProcessor` (`src/image_search/processors/ocr.py`) doesn't import
`rapidocr` directly — it launches `scripts/ocr_worker.py` as a subprocess via
`conda run -n sem_search_ocr`, keeping the OCR model loaded across the whole
ingest run and communicating over stdin/stdout (one image path in, one JSON
line out). `sentence-transformers` embeds the OCR'd text in-process in
`sem_search_gpu` as normal — only OCR needs the cross-env hop.

## Installing the OCR env

```bash
conda create -n sem_search_ocr python=3.12 -y
conda run -n sem_search_ocr pip install rapidocr "onnxruntime-gpu==1.20.1" \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
conda run -n sem_search_ocr pip install \
  "nvidia-cudnn-cu11==8.9.6.50" "nvidia-cublas-cu11==11.11.3.6" \
  "nvidia-cuda-runtime-cu11==11.8.89" "nvidia-cufft-cu11==10.9.0.58" \
  "nvidia-curand-cu11==10.3.0.86" "nvidia-cusolver-cu11==11.4.1.48" \
  "nvidia-cusparse-cu11==11.7.5.86" "nvidia-nvtx-cu11==11.8.86"
```

## Phase 2 addendum: cuDNN conv2d is broken on this GPU too

`SiglipImageEmbedProcessor` (image embeddings) hits a *different* Pascal
issue: cuDNN 9's conv2d algorithm search fails outright on `sm_61` —
`RuntimeError: GET was unable to find an engine to execute this computation`.
This isn't a missing-kernel error like the CUDA-13 case above; cuDNN 9 itself
can't find a working convolution algorithm for this architecture. Fix:
`torch.backends.cudnn.enabled = False` before loading the model, which forces
torch's native (non-cuDNN) conv kernels — slower, but correct. Harmless to set
process-wide even though `sentence-transformers` shares the process: BERT-style
text models are attention/matmul-only, no conv2d, so they're unaffected.

## If the GPU ever changes

If this moves to a Volta/Turing-or-newer card (compute capability >= 7.0),
none of this applies — a single modern env (torch + onnxruntime-gpu, both on
a current CUDA line) should just work, and this doc + the two-env split can
be deleted.
