# Build Spec: Local Semantic Search over Screenshots & Photos

A self-hosted indexing + search service for a local image library (screenshots and
photos). Supports: semantic text search, reverse-image search, face grouping
("people"), open-vocabulary tagging, structural attribute filtering ("are there
graphs?"), and combined topic + structure queries (e.g. *"unemployment graphs"*).

Design priority: **lightweight, self-hosted, per-folder configurable.** Storage fits
the existing SQLite/DuckDB stack. Scheduling reuses APScheduler.

---

## 0. Core mental model (read this first — it drives every decision)

There are **two retrieval primitives**:

- **Ranking** — cosine similarity over embeddings. Continuous. "Most similar to X."
- **Filtering** — discrete predicates over tags/attributes. Boolean/categorical. "Has a chart, yes/no."

There are **three vector spaces**, and they are mutually incompatible:

| Space | Produced by | Used for |
|-------|-------------|----------|
| Text  | sentence transformer | semantic search over OCR text + captions |
| Image | CLIP/SigLIP image encoder | reverse-image search (select → find similar) |
| Face  | ArcFace face embedder | face clustering / people grouping |

**The load-bearing invariant — vectors from different models are NOT comparable.**
A `bge-small` vector and a `MiniLM` vector live in different spaces; cosine between them
is meaningless. Therefore:

- **A model is LOCKED** if you consume the *vector* it emits (sentence transformer,
  image embedder, face embedder). Locked models must be identical across any set of
  items ranked against a single query.
- **A model is FREE** if you consume a *symbol* it emits — text or a tag (OCR,
  captioner, tag generators). Symbols are model-agnostic, so multiple free models may
  run and have their outputs `UNION`ed. `chart` from a layout detector == `chart` from
  CLIP-zeroshot.
- Free/locked is a property of **the output you consume, not the model.** CLIP is locked
  when you use its embedding (reverse search) and free when you threshold that embedding
  into a `chart` tag.

**Enforce this structurally:** partition vector tables by `(space, model)`. One vector
table per model per space, so a query physically cannot mix spaces. Free outputs all
land in shared `tags` / text tables with a `source`/`model` column.

The face **detector** is technically free (emits crops, not vectors), but keep it
consistent within a face-model partition anyway — ArcFace embeddings are alignment-
sensitive, and swapping detectors injects noise into the vectors you cluster.

---

## 1. Tech stack

- **Language:** Python 3.11+
- **Store:** SQLite + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for vectors,
  SQLite FTS5 for keyword fallback. (Alternative: DuckDB + VSS extension — see Open
  Decisions.)
- **Scheduling:** APScheduler (incremental ingest + nightly re-cluster).
- **Config:** YAML.
- **Models** (all swappable via config; defaults below):
  - OCR: PaddleOCR PP-OCRv5 (lightweight); PaddleOCR-VL for layout-heavy docs.
  - Captioning / tiny VLM: Moondream (0.5B/2B) or Florence-2 (0.23B/0.77B).
  - Text embeddings: `bge-small-en-v1.5` (default) or `all-MiniLM-L6-v2` (faster).
  - Image embeddings: SigLIP 2 (base) or OpenCLIP ViT-B/16.
  - Faces: InsightFace `FaceAnalysis` — SCRFD detect + ArcFace embed
    (`buffalo_l` accurate, `buffalo_s` light).
  - Tagging: RAM++ (`recognize-anything-plus-model`) for visual tags;
    KeyBERT (over OCR text) for named topics.
  - Layout / chart detection: PP-DocLayout (has a dedicated `chart` class) or
    DocLayout-YOLO (ONNX, fast, lumps charts into `figure`).
- **Clustering:** scikit-learn DBSCAN (cosine) default; HDBSCAN optional.

---

## 2. Deployment / resourcing (important constraint)

Indexing runs heavy vision models (RAM++, SigLIP, InsightFace, VLMs, layout detector).
**This will not fit on an e2-micro (≈1 GB RAM).** Split the system:

- **Indexer** (heavy, batch): runs where there's RAM/compute — laptop or a larger
  ephemeral VM (GPU helpful for VLM/face throughput, not required). Produces the SQLite
  file.
- **Search service** (light): query-time only needs to embed the query string + run
  `sqlite-vec` lookups + filters. The SQLite file is portable — build it on a capable
  machine, copy it to the small VM to serve. Even here, loading one sentence-transformer
  is the main memory cost; verify it fits before committing to the micro.

Treat the SQLite file as the artifact that moves between indexer and server.

---

## 3. Project layout

```
image-search/
  config/
    folders.yaml            # per-folder pipeline config
  src/image_search/
    config.py               # load + validate folder config
    registry.py             # processor registry + LAZY model loading
    processors/
      base.py               # Processor protocol + Record types
      ocr.py                # PaddleOCR
      caption.py            # Moondream / Florence-2
      text_embed.py         # sentence-transformer
      image_embed.py        # SigLIP / CLIP
      faces.py              # InsightFace detect + embed
      tagger.py             # RAM++
      layout.py             # PP-DocLayout / DocLayout-YOLO  (chart detection)
      topic_kw.py           # KeyBERT over OCR text  (named topics)
    store/
      db.py                 # connection, sqlite-vec load, schema migration
      images.py             # image rows + content hashing
      vectors.py            # per-(space,model) vec table create/insert/query
      tags.py               # tag insert/filter
      faces.py              # face rows + cluster ops
    ingest.py               # discover files, dispatch by folder config
    cluster.py              # DBSCAN job + incremental assignment
    search.py               # query parse + filter-then-rank
    cli.py                  # entry points (index / search / cluster / serve)
  tests/
  pyproject.toml
```

---

## 4. Data model (SQLite)

```sql
-- Stable identity: id = content hash, so moves/renames dedupe.
CREATE TABLE images (
  id          TEXT PRIMARY KEY,      -- sha256 of file bytes
  path        TEXT NOT NULL,
  folder      TEXT NOT NULL,         -- config key this image was matched to
  content_hash TEXT NOT NULL,
  mtime       REAL,
  width       INTEGER,
  height      INTEGER,
  indexed_at  REAL
);

-- FREE outputs: text. Multiple models may coexist (model column).
CREATE TABLE ocr_text  (image_id TEXT, model TEXT, text TEXT);
CREATE TABLE captions  (image_id TEXT, model TEXT, text TEXT);
CREATE VIRTUAL TABLE text_fts USING fts5(image_id, text);  -- keyword fallback

-- FREE outputs: tags. Store CONTINUOUS score; threshold at query time.
CREATE TABLE tags (
  image_id TEXT, tag TEXT, source TEXT, score REAL   -- source: ram++ | ocr-kw | pp-doclayout | clip-zs
);
CREATE INDEX idx_tags_tag ON tags(tag);

-- Faces. cluster_id assigned by the clustering job.
CREATE TABLE faces (
  face_id    TEXT PRIMARY KEY,
  image_id   TEXT,
  model      TEXT,                   -- embedding model (partition key for clustering)
  det_model  TEXT,
  bbox       TEXT,                   -- json [x,y,w,h]
  det_score  REAL,
  embedding  BLOB,                   -- raw ArcFace vector
  cluster_id INTEGER                 -- NULL until clustered
);
CREATE TABLE face_clusters (
  cluster_id INTEGER PRIMARY KEY, model TEXT, label TEXT, centroid BLOB, size INTEGER, updated_at REAL
);

-- LOCKED outputs: one vec0 table PER (space, model). Names encode the partition.
-- e.g. vec_text__bge_small_en_v1_5, vec_image__siglip2_base
-- Each maps rowid -> image_id via a sidecar table.
CREATE TABLE vec_map (vec_table TEXT, rowid INTEGER, image_id TEXT);
-- vec0 tables created dynamically by store/vectors.py per model encountered.
```

Rationale for per-`(space,model)` vec tables: it makes the locked-model invariant a
schema-level guarantee. A search picks one table = one model = one coherent space.

---

## 5. Config schema (`config/folders.yaml`)

```yaml
defaults:
  text_embed:  bge-small-en-v1.5
  image_embed: siglip2-base
  face_model:  buffalo_l
  face_detect: scrfd_10g          # part of the buffalo pack

folders:
  "~/Screenshots":
    ocr:         paddle-ppocrv5
    text_embed:  bge-small-en-v1.5
    tagger:      ram++
    layout:      pp-doclayout      # enables the "chart" attribute
    topic_kw:    on
    faces:       off
  "~/Photos/Family":
    image_embed: siglip2-base
    caption:     moondream-2b
    tagger:      ram++
    faces:       buffalo_l
    ocr:         off
  "~/Receipts":
    ocr:         paddle-ocr-vl
    text_embed:  bge-small-en-v1.5
    faces:       off
```

Validation rules (`config.py` must enforce):
- Any folders intended for **cross-folder unified search** must share the same
  `text_embed` / `image_embed`. Warn loudly if they diverge.
- Any folders intended for **unified people groups** must share the same `face_model`.
- A processor key set to `off`/absent means skip it for that folder.

---

## 6. Processor contract

```python
# processors/base.py
from typing import Protocol, Literal

Kind = Literal["ocr","caption","text_embed","image_embed","faces","tagger","layout","topic_kw"]

class Processor(Protocol):
    kind: Kind
    model_id: str
    def load(self) -> None: ...                  # heavy init; called lazily on first use
    def process(self, img: "LoadedImage") -> list["Record"]: ...
```

- `registry.py` maps `(kind, model_id)` → Processor instance, **lazy-loaded**. On startup,
  compute the *union* of `(kind, model_id)` referenced by active folders and load only
  those. Never load a model no active folder uses.
- `text_embed` is special: it consumes the text emitted by `ocr` / `caption` / `topic_kw`
  within the same image, not the raw pixels. Order the dispatch so text producers run
  before `text_embed`.

Record types (one per kind): `OcrRecord(text)`, `CaptionRecord(text)`,
`TextEmbedRecord(model, vector)`, `ImageEmbedRecord(model, vector)`,
`FaceRecord(bbox, det_score, embedding, model, det_model)`,
`TagRecord(tag, score, source)`.

---

## 7. Pipelines

### 7.1 Ingest (`ingest.py`)
1. Walk configured folders; for each file compute content hash → `image_id`.
2. Skip if `image_id` already in `images` and `mtime` unchanged.
3. Resolve folder → enabled processors (text producers first, then `text_embed`).
4. Run each processor; write records to the right tables. Vectors go to the
   `(space, model)` vec table (create on first encounter, register in `vec_map`).
5. Mirror OCR+caption text into `text_fts`.

### 7.2 Face clustering (`cluster.py`)
- Operate **per face-model partition**: `SELECT ... FROM faces WHERE model = ?`.
- Default: DBSCAN, `metric="cosine"`, tuned `eps` (start ~0.4–0.5), `min_samples` small.
  Noise points (`-1`) → singleton/"unknown" faces.
- **Incremental path** (run on ingest): for a new face, compare to existing
  `face_clusters.centroid`; if max cosine ≥ threshold, assign that `cluster_id`, else
  leave `NULL`.
- **Full re-cluster** (nightly via APScheduler): recompute clusters + centroids over the
  whole partition to heal drift; keep cluster IDs stable where possible.

### 7.3 Search (`search.py`) — filter-then-rank
1. **Parse** the query into `(structural_facets, semantic_text)`:
   - Lookup table of structural keywords → facet predicates:
     `graph|chart → tag=chart`, `table → tag=table`, `map → tag=map`,
     `code → tag=code`, `screenshot → tag=screenshot`, etc.
   - Remaining words → `semantic_text`.
   - Also accept explicit facet filters from the UI (don't rely solely on parsing).
2. **Filter** to a candidate set via predicates over `tags`
   (`WHERE tag='chart' AND score > τ`), optionally scoped to a folder.
3. **Rank** the candidates: embed `semantic_text` with the relevant `text_embed` model,
   cosine via the matching `vec_text__*` table. (If both text and image ranking apply,
   score in each space separately and fuse with normalized scores — never one cross-space
   cosine.)
4. Return ranked candidates with thumbnails + matched tags.

Worked example — *"unemployment graphs"*:
`chart` keyword → `WHERE tag='chart' AND score>τ`; `unemployment` → semantic rank over
OCR-text embeddings (the word lives in axis labels/titles, not in pixels). Filter the
structure, rank the topic.

### 7.4 Reverse-image search
Given a selected `image_id`: fetch its vector from `vec_image__<model>`, cosine-search the
same table, exclude self, return top-k. Locked to that one image model.

---

## 8. Build phases (implement in order; each is independently testable)

- [ ] **Phase 0 — Scaffold.** Project layout, `pyproject.toml`, config loader + validation,
      DB schema + migrations, image discovery + content hashing. No models yet.
      *Test:* ingest a folder → `images` populated, dedupe on re-run.
- [ ] **Phase 1 — Text path.** OCR processor + text_embed + FTS. Semantic text search +
      keyword fallback (hybrid). *Test:* query returns expected screenshot by content.
- [ ] **Phase 2 — Image path.** Image embed processor + reverse-image search.
      *Test:* select image → visually similar neighbors.
- [ ] **Phase 3 — Captions.** VLM caption processor feeding the text space.
      *Test:* image with no OCR text still retrievable by described content.
- [ ] **Phase 4 — Tags & attributes.** RAM++ tagger + layout/chart detector + KeyBERT
      topics → `tags` table (continuous scores). Facet filtering.
      *Test:* `tag=chart` filter; topic filter.
- [ ] **Phase 5 — Faces.** InsightFace detect+embed + DBSCAN clustering + incremental
      assignment. Browse-by-person. *Test:* same person across images lands in one cluster.
- [ ] **Phase 6 — Query fusion.** Query parser (structural split) + filter-then-rank +
      multi-space score fusion. *Test:* "unemployment graphs" returns unemployment charts.
- [ ] **Phase 7 — Ops.** APScheduler (incremental ingest + nightly re-cluster), CLI
      (`index` / `search` / `cluster` / `serve`), optional thin web UI / control panel.

---

## 9. Invariants Claude Code must not violate

1. Never cosine across two different models / two vector spaces. Enforced by
   per-`(space,model)` vec tables.
2. Face clustering runs within one face-model partition only.
3. Store continuous attribute scores; apply thresholds at query time (protects recall).
4. `image_id` is a content hash; identical bytes at different paths are one image.
5. Lazy-load only models referenced by active folder configs.
6. Text producers (OCR/caption/topic_kw) run before `text_embed` in dispatch order.

---

## 10. Non-goals (keep scope tight)
- No chart *data* extraction / chart QA — detection only ("is there a graph").
- No real-time streaming index; batch + scheduled incremental is fine.
- No multi-user/auth.
- No cloud/hosted vector DB; local SQLite file is the store and the portable artifact.

---

## 11. Open decisions (pick before Phase 0)
- **Vector store:** `sqlite-vec` (matches existing SQLite habit, single file) vs DuckDB
  VSS (already used for the campaign-finance project). Default: `sqlite-vec`.
- **Image embedder:** SigLIP 2 (stronger retrieval) vs OpenCLIP (ubiquitous). Default: SigLIP 2.
- **Chart detector:** PP-DocLayout (dedicated `chart` class, better for this exact use)
  vs DocLayout-YOLO (faster ONNX, `figure` only). Default: PP-DocLayout.
- **Compute for indexing:** laptop vs larger ephemeral VM vs GPU box. Determines VLM/face
  batch sizes.
