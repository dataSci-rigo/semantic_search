# GPU setup: why three conda envs

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

## The fix: three conda envs

- **`sem_search_gpu`** — torch (cu118) for `sentence-transformers` /
  `text_embed` / `image_embed`. This is the main env `image-search` itself runs in.
- **`sem_search_ocr`** — `rapidocr` + `onnxruntime-gpu==1.20.1` (from the
  CUDA-11 feed) + `nvidia-cudnn-cu11==8.9.6.50` pinned exactly, with no torch
  at all. Nothing to conflict with.
- **`sem_search_caption`** — torch (cu118) + `transformers==4.51.3` (older,
  pinned) for Moondream2 captioning. See Phase 3 addendum below for why this
  needs its own env even though it's torch-based like `sem_search_gpu`.

All three envs need `LD_LIBRARY_PATH` pointed at their own `site-packages/nvidia/*/lib`
dirs — `onnxruntime`/`torch`'s pip-bundled CUDA libraries aren't on the loader
path by default. This is persisted per-env via:

```bash
conda env config vars set -n <env> LD_LIBRARY_PATH="<colon-joined nvidia/*/lib dirs>"
```

(already done for all three envs on this machine; re-run if any env's packages
are reinstalled, since paths embed exact site-packages locations).

## How the pipeline bridges the envs

`RapidOcrProcessor` and `MoondreamCaptionProcessor`
(`src/image_search/processors/{ocr,caption}.py`) share a common base,
`SubprocessBridgeProcessor` (`processors/subprocess_bridge.py`): instead of
importing their model in-process, they launch a worker script
(`scripts/{ocr,caption}_worker.py`) via `conda run -n <env>` as a persistent
subprocess, keeping the model loaded across the whole ingest run and
communicating over stdin/stdout (one image path in, one JSON line out).
`sentence-transformers`/`image_embed` run in-process in `sem_search_gpu` as
normal — only the processors that need a conflicting dependency version take
the cross-env hop. Any future processor with the same problem (e.g.
`insightface` for faces, Phase 5 — also onnxruntime-based like OCR) should
follow the same pattern rather than inventing a new one.

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

## Installing the caption env

```bash
conda create -n sem_search_caption python=3.12 -y
conda run -n sem_search_caption pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
conda run -n sem_search_caption pip install "transformers==4.51.3" einops pillow accelerate
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

## Phase 3 addendum: transformers version skew, not a GPU issue

Captioning (Moondream2, and Microsoft's Florence-2 has the same problem) hits
a *different kind* of conflict than everything above — not Pascal, not CUDA.
Both are community `trust_remote_code` models whose custom modeling/config
code was written against older `transformers` internals. `sem_search_gpu`
runs `transformers==5.14.1` (needed for *native*, non-remote-code SigLIP2
support), and against that version:

- Moondream2 fails to load: `AttributeError: 'HfMoondream' object has no
  attribute 'all_tied_weights_keys'` — a `transformers` 5.x internal renamed
  out from under the repo's custom model class.
- Florence-2 fails the same way on its custom config class.
- Loading Florence-2 via `transformers`' own *native* `Florence2ForConditionalGeneration`
  (bypassing the repo's remote code entirely) doesn't work either: the native
  port uses completely different state-dict key names, so `microsoft/Florence-2-base`'s
  checkpoint doesn't map onto it — nearly every weight comes back `MISSING`
  (i.e. randomly initialized, not the real model).

Fix: `sem_search_caption` pins `transformers==4.51.3` (old enough to predate
the breaking internal refactor) in its own env, isolated from `sem_search_gpu`'s
newer version. Moondream2 loads and captions correctly there. Also needs
`torch.backends.cudnn.enabled = False` in the worker, same Pascal conv2d issue
as Phase 2's image_embed.

## If the GPU ever changes

If this moves to a Volta/Turing-or-newer card (compute capability >= 7.0),
none of this applies — a single modern env (torch + onnxruntime-gpu, both on
a current CUDA line) should just work, and this doc + the two-env split can
be deleted.
