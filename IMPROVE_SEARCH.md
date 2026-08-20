Search fusion fix + caption-length investigation
(Supersedes the previous plan in this file, "Field-scoped keyword search (OCR vs. caption)" — that work shipped and is live in production.)

Context
Live testing surfaced two real, connected problems in the search/indexing pipeline:

Search quality is inconsistent. search_text() concatenates vector hits before FTS hits unconditionally — real exact-keyword matches get buried behind mediocre semantic matches whenever the vector search alone fills the result quota. Reproduced live, twice, against production data this session: querying "scuba" and "espiritu santo" both have genuine FTS matches (2 OCR + 1 caption for "scuba"; 1 OCR match for "espiritu santo") that appear nowhere in the top 15 results — every visible result is a weak vector match (cosine distance 0.78–0.94, i.e. not close).

An unplanned production change is actively causing failures. While researching a proposed caption-length upgrade (short → normal, for richer, more findable captions), discovered the switch already happened silently yesterday (commit 277a7a9) and is live now — with 1,185 CUDA OOM errors in today's index log alone on this 8GB Pascal card. Caption length only grew modestly on average (140→155 chars, +11%) despite the max jumping (244→3,719), consistent with many "normal" caption attempts failing outright and getting silently skipped/retried (per the per-file error isolation built earlier this session) rather than actually succeeding at meaningfully richer output. Confirmed live and directly, not just inferred from the log: a plain read-only diagnostic script (loading bge-small-en-v1.5, a small sentence-transformer, to check a vector rank) itself failed with CUDA error: out of memory while the indexer was running — the GPU currently has no free memory margin at all under normal indexing load.

The "scuba" reproduction has a second, distinct cause layered on top of the fusion bug. The user correctly pointed out real scuba-adjacent photos exist (divers, underwater scenes) that the initial "scuba" search didn't surface. Investigated: broadening to "diver/diving/underwater/ snorkel/wetsuit" finds 21 caption + 10 OCR matches — real relevant photos (a family exploring an underwater exhibit, a wetsuit at the beach, fish swimming) — they're captioned with different words than the literal "scuba". This is a vocabulary-mismatch problem semantic/vector search is supposed to solve (matching "scuba" to "underwater diving scene" by meaning, not exact words) — but whether bge-small-en-v1.5 actually ranks these images well for a "scuba" query is unverified, since the GPU OOM blocked the check (see finding 2). This needs resolving after Step 0, before concluding the fusion fix alone is sufficient — if vector search also fails to rank these appropriately, that points at embedding model quality (the "better sentence transformer" option from the original 4-option list), not just fusion order.

The user asked to pause the indexer and investigate the OOM issue first, before finishing the search-fusion and caption-length design work.

Step 0 (do first, before anything else): pause and diagnose the OOM issue
systemctl --user stop image-search-index.service — halt the source of ongoing OOM churn immediately. Safe by the existing atomic-commit design (verified multiple times this session: no data loss from pausing).
Diagnose, don't guess:
grep -c "out of memory" data/index.log to confirm current total (was 713 per the background agent's check, 1,185 moments later when I verified — climbing fast, confirm the current count).
Identify which processor is OOMing — grep the log for the traceback context around out of memory lines (worker env name — sem_search_caption, sem_search_ocr, or the main sem_search_gpu process — and which model/operation). The caption-length switch is the prime suspect (length="normal" plausibly needs a larger KV-cache / more activation memory than "short" on Moondream2), but confirm rather than assume — this codebase has a track record of GPU assumptions turning out wrong empirically (Pascal cuDNN quirks, transformers-version conflicts, etc.).
Check nvidia-smi for current GPU memory state and whether multiple GPU-using processes were contending simultaneously (the background agent's check found several process IDs in the log at once — confirm this wasn't just concurrent OCR+caption+embed workers, which is normal and already accounted for, vs. something new piling on).
Decide the immediate fix based on what's found — likely one of:
Revert length="normal" → "short" in scripts/caption_worker.py until a proper benchmark (Part 2 below) justifies the switch with real numbers and a plan for the memory cost.
Or, if the OOM is a separate/unrelated contention issue (e.g. multiple caption workers spawned concurrently), fix that specifically and keep "normal".
Once stable, restart image-search-index.service and confirm the OOM rate actually drops (watch data/index.log for a few minutes) before moving on to Parts 1/2 below.
With the GPU free (indexer stopped), resolve finding 3: check whether bge-small-en-v1.5 actually ranks the known-relevant underwater/diving captions well for a "scuba" query (e.g. embed "scuba", pull the vector rank of a diving-caption image_id like 37787c7c4177112cc4929b6c88fdcc6f95659b1eca332853a33a5c0418845a93 — a real "family... underwater habitat exhibit" caption). If it ranks reasonably (e.g. top 50-100), the fusion fix in Part 1 alone should surface it once fetch_k is raised. If it ranks poorly even there, note this as evidence for revisiting the embedding model (Option 3 from the original list) as a real follow-up, not just a hypothetical — but do not act on that without new user sign-off, it's a larger change (new model, likely a full re-embed) than either Part 1 or Part 2 here.
Part 1 — Fix vector/FTS score fusion in search.py
Algorithm: Reciprocal Rank Fusion (RRF)
Chosen over min-max-normalize-then-linear-combine because RRF only uses rank position, never raw score magnitude — sidesteps the fact that vector scores (-cosine_distance) and FTS scores (-bm25_rank) live on genuinely incomparable, query-dependent scales (bm25 has no fixed range; a single rare-term query like "darknet" produces wildly different bm25 magnitudes than a common multi-word query, which would badly distort any min-max normalization). RRF is also the standard, well-validated technique for exactly this two-retriever (sparse+dense) hybrid search problem — no tunable weight to guess at without relevance-labeled data.

Formula: for each candidate list (vector, FTS), score += 1 / (60 + rank) (1-indexed rank; 60 is the standard RRF constant — not worth tuning without real labeled data). Sum across lists an item appears in.

Concrete changes to src/image_search/search.py
Remove the seen-based FTS dedup (if iid not in seen) — RRF needs both full candidate lists intact so items appearing in both can be detected and their scores combined (see source="both" below).
Filter (allowed set, from tags/field) before computing RRF ranks, not after fusing — an item's fused rank must reflect its position among candidates that actually survived filtering, not its position in the original unfiltered fetch. This matches the codebase's existing filter-the-raw-lists pattern, just moved earlier in the pipeline.
Raise fetch_k, unconditionally — currently k * 5 if (active_tags or field) else k, meaning an unfiltered search only fetches exactly k per source. Under fusion, fetch_k also controls how deep FTS gets a chance to contribute at all (an FTS hit outside the fetched window can never be fused in, full stop). Change to always over-fetch, e.g. fetch_k = max(k * 5, 100) regardless of filtering — cheap for both a bm25 LIMIT scan and a sqlite-vec KNN query at this corpus size (tens of thousands of vectors). This is the direct fix for the "darknet"/"scuba"/"espiritu santo" bug: the real FTS hit needs to actually be in the fetched pool with a rank shallow enough for 1/(60+rank) to be competitive.
Add a fuse step:
RRF_K = 60

def _rrf_fuse(
    vector_hits: list[tuple[str, float]], fts_hits: list[tuple[str, float]]
) -> list[tuple[str, str, float]]:
    """Combine two ranked id lists via Reciprocal Rank Fusion. Returns
    [(image_id, source, rrf_score)] sorted best-first, source in
    "vector" | "fts" | "both". Only rank position is used — vector's
    -cosine_distance and FTS's -bm25 live on incomparable scales, so raw
    scores are never compared across sources."""
    rrf_scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for hits, tag in ((vector_hits, "vector"), (fts_hits, "fts")):
        for rank, (image_id, _score) in enumerate(hits, start=1):
            rrf_scores[image_id] = rrf_scores.get(image_id, 0.0) + 1 / (RRF_K + rank)
            sources.setdefault(image_id, set()).add(tag)
    ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        (image_id, "both" if sources[image_id] == {"vector", "fts"} else next(iter(sources[image_id])), score)
        for image_id, score in ordered
    ]
rows_for becomes a single unified resolver over the fused list (image_id, source, score) instead of being called twice with a hardcoded source string — looks up each id's DB row (image or item, as it does today), stops once k results are resolved (some fused ids won't resolve — wrong folder, dead item status — same defensive slop pattern already present in the current code).
Update SearchHit's source type/docstring: "vector" | "fts" | "both". Replace the now-false docstring ("scores are NOT comparable across sources — results are concatenated, vector hits first") with something like: "RRF-fused score (reciprocal-rank sum across the sources that matched); higher is better, comparable across all hits in one result list, but not across different queries or k values."
Webapp: _hit_to_dict and the template already do a straight passthrough of hit.source — "both" should render with no code change, but do a quick visual check of index.html's CSS in case a .source-vector/.source-fts-style selector needs a .source-both counterpart (unverified — check while implementing, not assumed clean).
Tests to add to tests/test_search.py
Reuse the vector-insertion pattern from tests/test_reverse_image_search.py (vectors_store.insert_vector(conn, "text", model_id, image_id, vector)) combined with a config using text_embed: fake-embed, since test_search.py today is FTS-only.

test_rrf_promotes_strong_fts_hit_over_many_mediocre_vector_hits — the direct regression test for the "darknet"/"scuba" bug: ~30-40 images with mediocre-but-present vectors (would fully occupy a naive top-k), plus one image with a rare/exact FTS-only match and no vector. Assert it appears in the top k — it would not, under the old concatenation code.
test_hit_found_by_both_sources_ranks_above_single_source_hits — an image matching decently in both lists should out-rank an image strong in only one, and its source should resolve to "both".
test_source_both_reported_correctly — narrow unit check of the source-labeling logic alone.
test_filter_applies_before_fusion_rank — an item that's rank-1 in the unfiltered vector list gets filtered out by allowed; assert a lower-unfiltered-rank surviving item's final RRF score reflects its rank among filtered survivors, not the original unfiltered rank (catches a regression to "filter after fuse").
Confirm existing tests still pass unmodified: test_fts_fallback_finds_keyword_match (no vector configured → source still "fts"), and the field-scoping tests (test_field_ocr_only_excludes_caption_matches etc. — FTS-only, no vector configured, so fusion shouldn't touch them; good regression coverage that this doesn't break the field-filter feature shipped earlier this session).
Deploy
Pure Python code change — no schema/DDL, so no need to pause the indexer the way the earlier field-filter migration required (that was specifically about avoiding a DB-lock race during an ALTER/table-rebuild; this change doesn't touch the DB structurally at all). systemctl --user restart image-search-web.service is sufficient. Acceptance test: re-run "darknet", "scuba", and "espiritu santo" against the live production DB (via the CLI or web UI) and confirm a real FTS/"both"-sourced hit now appears in the top 10-15 — these are the exact reproductions already captured this session.

Part 2 — Caption length: benchmark before deciding on a backfill
Reframed given the Step 0 finding: the short→normal switch already happened without the benchmark-first process originally planned. Treat this as: understand what's now inconsistent, decide whether to keep "normal" (pending Step 0's OOM diagnosis) or revert, and — only if keeping it — design the backfill mechanism and get real numbers before running it.

1. Benchmark properly (after Step 0 is resolved)
Pick a real photo folder under /home/ai1/Pictures (Camera, dw, dw2, fotos, etc. all exist), sample ~20-30 already-captioned images (stratified across photos/screenshots/memes if possible), and manually run captioning at length="normal" with the indexer stopped, to avoid the current GPU contention distorting the numbers. Compare: char-length distribution vs. existing captions, qualitative content (does "normal" actually surface details "short" misses — the real question, not just length), and seconds/image vs. the previously-benchmarked 6.4s/image at "short". Do not assume linear scaling with output length — measure it.

2. Reprocessing mechanism (new code, only build if benchmark justifies a backfill)
is_indexed() (store/images.py) gates on the images row existing at all — no per-processor granularity — so already-indexed images never get automatically reprocessed just because a config/worker changed. purge_image() deletes everything derived, too broad (would force re-running OCR/image_embed/tagger unnecessarily, each in its own conda env).

Build a narrow, purpose-built path instead of reusing purge_image:

New store/images.py helper, purge_caption(conn, image_id): deletes just the captions row, the caption-sourced text_fts row (WHERE image_id = ? AND source = 'caption' — note this only works now that the source column exists, from this session's earlier field-filter work), and the stale text_embed vector (the combined OCR+caption embedding — must be dropped too, since it's built from the old caption text).
New CLI subcommand, image-search recaption <folder> [--model MODEL_ID] [--limit N] [--force], following cmd_dupes's existing dry-run-by-default pattern (--delete there → --force here): by default, print a count and estimated GPU-hours (from the benchmark above), require --force to actually execute. Implementation: select image_ids from captions where model matches the target for re-processing, run only caption + text_embed via registry.for_processors(...) (reuses the existing dispatch machinery), recompute accumulated_text from existing OCR text + new caption (factor the existing inline logic in _ingest_image into a small shared helper rather than duplicating it), replace the caption row and its text_fts/vector rows. Does not touch images, ocr_text, tags, or the image vector.
Do not attempt the "delete the images row and let normal ingest reprocess it" shortcut — the processors' INSERT statements are unconditional appends, not upserts, so this would create duplicate ocr_text/tags/text_fts rows for anything besides the caption itself.
3. Cost structure (fill in with real numbers from the benchmark)
total_backfill_gpu_hours ≈
    (count of images still on the old caption length)
    × (measured seconds/image at length="normal", from step 1)
    / 3600
Count needs to be measured close to the actual backfill decision (it's changing continuously while the indexer runs). Compare one-time bulk backfill vs. "accept the split, only caption new images at the new length going forward" — given this is a single contended 8GB GPU, leaning toward the latter unless the benchmark shows a compellingly large recall improvement, but this is the user's call once real numbers exist.

4. model_id labeling
model_id is a pure label today — caption_worker.py doesn't receive it at all, so the caption length is fully decoupled from the moondream2 string in captions.model. Renaming to e.g. moondream2-normal is safe (nothing else keys off the literal string) but purely cosmetic unless paired with making the worker actually receive and act on it (e.g. pass it as an argv element from subprocess_bridge.py's Popen call) — otherwise the label and behavior can silently drift apart again exactly as they just did. Separately: already-written captions since yesterday's silent switch cannot be retroactively distinguished by label alone — a one-time correction pass keyed on images.indexed_at vs. the 277a7a9 deploy timestamp would be needed to backfill correct labels for the already-mixed data, if pursued.

Verification
Step 0: confirm OOM rate drops after whatever fix is applied — watch data/index.log for a clean stretch after restarting the indexer.
Part 1: conda run -n sem_search_gpu python -m pytest green, including new fusion tests. Then live: re-run "darknet"/"scuba"/"espiritu santo" against production and confirm real matches now surface in the top 10-15, where they were previously completely absent.
Part 2: benchmark numbers recorded and shared before any bulk recaption --force run; recaption dry-run output sanity-checked against known captions counts before ever passing --force.


Search fusion fix + caption-length investigation
(Supersedes the previous plan in this file, "Field-scoped keyword search (OCR vs. caption)" — that work shipped and is live in production.)

Context
Live testing surfaced two real, connected problems in the search/indexing pipeline:

Search quality is inconsistent. search_text() concatenates vector hits before FTS hits unconditionally — real exact-keyword matches get buried behind mediocre semantic matches whenever the vector search alone fills the result quota. Reproduced live, twice, against production data this session: querying "scuba" and "espiritu santo" both have genuine FTS matches (2 OCR + 1 caption for "scuba"; 1 OCR match for "espiritu santo") that appear nowhere in the top 15 results — every visible result is a weak vector match (cosine distance 0.78–0.94, i.e. not close).

An unplanned production change is actively causing failures. While researching a proposed caption-length upgrade (short → normal, for richer, more findable captions), discovered the switch already happened silently yesterday (commit 277a7a9) and is live now — with 1,185 CUDA OOM errors in today's index log alone on this 8GB Pascal card. Caption length only grew modestly on average (140→155 chars, +11%) despite the max jumping (244→3,719), consistent with many "normal" caption attempts failing outright and getting silently skipped/retried (per the per-file error isolation built earlier this session) rather than actually succeeding at meaningfully richer output. Confirmed live and directly, not just inferred from the log: a plain read-only diagnostic script (loading bge-small-en-v1.5, a small sentence-transformer, to check a vector rank) itself failed with CUDA error: out of memory while the indexer was running — the GPU currently has no free memory margin at all under normal indexing load.

The "scuba" reproduction has a second, distinct cause layered on top of the fusion bug. The user correctly pointed out real scuba-adjacent photos exist (divers, underwater scenes) that the initial "scuba" search didn't surface. Investigated: broadening to "diver/diving/underwater/ snorkel/wetsuit" finds 21 caption + 10 OCR matches — real relevant photos (a family exploring an underwater exhibit, a wetsuit at the beach, fish swimming) — they're captioned with different words than the literal "scuba". This is a vocabulary-mismatch problem semantic/vector search is supposed to solve (matching "scuba" to "underwater diving scene" by meaning, not exact words) — but whether bge-small-en-v1.5 actually ranks these images well for a "scuba" query is unverified, since the GPU OOM blocked the check (see finding 2). This needs resolving after Step 0, before concluding the fusion fix alone is sufficient — if vector search also fails to rank these appropriately, that points at embedding model quality (the "better sentence transformer" option from the original 4-option list), not just fusion order.

The user asked to pause the indexer and investigate the OOM issue first, before finishing the search-fusion and caption-length design work.

Step 0 (do first, before anything else): pause and diagnose the OOM issue
systemctl --user stop image-search-index.service — halt the source of ongoing OOM churn immediately. Safe by the existing atomic-commit design (verified multiple times this session: no data loss from pausing).
Diagnose, don't guess:
grep -c "out of memory" data/index.log to confirm current total (was 713 per the background agent's check, 1,185 moments later when I verified — climbing fast, confirm the current count).
Identify which processor is OOMing — grep the log for the traceback context around out of memory lines (worker env name — sem_search_caption, sem_search_ocr, or the main sem_search_gpu process — and which model/operation). The caption-length switch is the prime suspect (length="normal" plausibly needs a larger KV-cache / more activation memory than "short" on Moondream2), but confirm rather than assume — this codebase has a track record of GPU assumptions turning out wrong empirically (Pascal cuDNN quirks, transformers-version conflicts, etc.).
Check nvidia-smi for current GPU memory state and whether multiple GPU-using processes were contending simultaneously (the background agent's check found several process IDs in the log at once — confirm this wasn't just concurrent OCR+caption+embed workers, which is normal and already accounted for, vs. something new piling on).
Decide the immediate fix based on what's found — likely one of:
Revert length="normal" → "short" in scripts/caption_worker.py until a proper benchmark (Part 2 below) justifies the switch with real numbers and a plan for the memory cost.
Or, if the OOM is a separate/unrelated contention issue (e.g. multiple caption workers spawned concurrently), fix that specifically and keep "normal".
Once stable, restart image-search-index.service and confirm the OOM rate actually drops (watch data/index.log for a few minutes) before moving on to Parts 1/2 below.
With the GPU free (indexer stopped), resolve finding 3: check whether bge-small-en-v1.5 actually ranks the known-relevant underwater/diving captions well for a "scuba" query (e.g. embed "scuba", pull the vector rank of a diving-caption image_id like 37787c7c4177112cc4929b6c88fdcc6f95659b1eca332853a33a5c0418845a93 — a real "family... underwater habitat exhibit" caption). If it ranks reasonably (e.g. top 50-100), the fusion fix in Part 1 alone should surface it once fetch_k is raised. If it ranks poorly even there, note this as evidence for revisiting the embedding model (Option 3 from the original list) as a real follow-up, not just a hypothetical — but do not act on that without new user sign-off, it's a larger change (new model, likely a full re-embed) than either Part 1 or Part 2 here.
Part 1 — Fix vector/FTS score fusion in search.py
Algorithm: Reciprocal Rank Fusion (RRF)
Chosen over min-max-normalize-then-linear-combine because RRF only uses rank position, never raw score magnitude — sidesteps the fact that vector scores (-cosine_distance) and FTS scores (-bm25_rank) live on genuinely incomparable, query-dependent scales (bm25 has no fixed range; a single rare-term query like "darknet" produces wildly different bm25 magnitudes than a common multi-word query, which would badly distort any min-max normalization). RRF is also the standard, well-validated technique for exactly this two-retriever (sparse+dense) hybrid search problem — no tunable weight to guess at without relevance-labeled data.

Formula: for each candidate list (vector, FTS), score += 1 / (60 + rank) (1-indexed rank; 60 is the standard RRF constant — not worth tuning without real labeled data). Sum across lists an item appears in.

Concrete changes to src/image_search/search.py
Remove the seen-based FTS dedup (if iid not in seen) — RRF needs both full candidate lists intact so items appearing in both can be detected and their scores combined (see source="both" below).
Filter (allowed set, from tags/field) before computing RRF ranks, not after fusing — an item's fused rank must reflect its position among candidates that actually survived filtering, not its position in the original unfiltered fetch. This matches the codebase's existing filter-the-raw-lists pattern, just moved earlier in the pipeline.
Raise fetch_k, unconditionally — currently k * 5 if (active_tags or field) else k, meaning an unfiltered search only fetches exactly k per source. Under fusion, fetch_k also controls how deep FTS gets a chance to contribute at all (an FTS hit outside the fetched window can never be fused in, full stop). Change to always over-fetch, e.g. fetch_k = max(k * 5, 100) regardless of filtering — cheap for both a bm25 LIMIT scan and a sqlite-vec KNN query at this corpus size (tens of thousands of vectors). This is the direct fix for the "darknet"/"scuba"/"espiritu santo" bug: the real FTS hit needs to actually be in the fetched pool with a rank shallow enough for 1/(60+rank) to be competitive.
Add a fuse step:
RRF_K = 60

def _rrf_fuse(
    vector_hits: list[tuple[str, float]], fts_hits: list[tuple[str, float]]
) -> list[tuple[str, str, float]]:
    """Combine two ranked id lists via Reciprocal Rank Fusion. Returns
    [(image_id, source, rrf_score)] sorted best-first, source in
    "vector" | "fts" | "both". Only rank position is used — vector's
    -cosine_distance and FTS's -bm25 live on incomparable scales, so raw
    scores are never compared across sources."""
    rrf_scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for hits, tag in ((vector_hits, "vector"), (fts_hits, "fts")):
        for rank, (image_id, _score) in enumerate(hits, start=1):
            rrf_scores[image_id] = rrf_scores.get(image_id, 0.0) + 1 / (RRF_K + rank)
            sources.setdefault(image_id, set()).add(tag)
    ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        (image_id, "both" if sources[image_id] == {"vector", "fts"} else next(iter(sources[image_id])), score)
        for image_id, score in ordered
    ]
rows_for becomes a single unified resolver over the fused list (image_id, source, score) instead of being called twice with a hardcoded source string — looks up each id's DB row (image or item, as it does today), stops once k results are resolved (some fused ids won't resolve — wrong folder, dead item status — same defensive slop pattern already present in the current code).
Update SearchHit's source type/docstring: "vector" | "fts" | "both". Replace the now-false docstring ("scores are NOT comparable across sources — results are concatenated, vector hits first") with something like: "RRF-fused score (reciprocal-rank sum across the sources that matched); higher is better, comparable across all hits in one result list, but not across different queries or k values."
Webapp: _hit_to_dict and the template already do a straight passthrough of hit.source — "both" should render with no code change, but do a quick visual check of index.html's CSS in case a .source-vector/.source-fts-style selector needs a .source-both counterpart (unverified — check while implementing, not assumed clean).
Tests to add to tests/test_search.py
Reuse the vector-insertion pattern from tests/test_reverse_image_search.py (vectors_store.insert_vector(conn, "text", model_id, image_id, vector)) combined with a config using text_embed: fake-embed, since test_search.py today is FTS-only.

test_rrf_promotes_strong_fts_hit_over_many_mediocre_vector_hits — the direct regression test for the "darknet"/"scuba" bug: ~30-40 images with mediocre-but-present vectors (would fully occupy a naive top-k), plus one image with a rare/exact FTS-only match and no vector. Assert it appears in the top k — it would not, under the old concatenation code.
test_hit_found_by_both_sources_ranks_above_single_source_hits — an image matching decently in both lists should out-rank an image strong in only one, and its source should resolve to "both".
test_source_both_reported_correctly — narrow unit check of the source-labeling logic alone.
test_filter_applies_before_fusion_rank — an item that's rank-1 in the unfiltered vector list gets filtered out by allowed; assert a lower-unfiltered-rank surviving item's final RRF score reflects its rank among filtered survivors, not the original unfiltered rank (catches a regression to "filter after fuse").
Confirm existing tests still pass unmodified: test_fts_fallback_finds_keyword_match (no vector configured → source still "fts"), and the field-scoping tests (test_field_ocr_only_excludes_caption_matches etc. — FTS-only, no vector configured, so fusion shouldn't touch them; good regression coverage that this doesn't break the field-filter feature shipped earlier this session).
Deploy
Pure Python code change — no schema/DDL, so no need to pause the indexer the way the earlier field-filter migration required (that was specifically about avoiding a DB-lock race during an ALTER/table-rebuild; this change doesn't touch the DB structurally at all). systemctl --user restart image-search-web.service is sufficient. Acceptance test: re-run "darknet", "scuba", and "espiritu santo" against the live production DB (via the CLI or web UI) and confirm a real FTS/"both"-sourced hit now appears in the top 10-15 — these are the exact reproductions already captured this session.

Part 2 — Caption length: benchmark before deciding on a backfill
Reframed given the Step 0 finding: the short→normal switch already happened without the benchmark-first process originally planned. Treat this as: understand what's now inconsistent, decide whether to keep "normal" (pending Step 0's OOM diagnosis) or revert, and — only if keeping it — design the backfill mechanism and get real numbers before running it.

1. Benchmark properly (after Step 0 is resolved)
Pick a real photo folder under /home/ai1/Pictures (Camera, dw, dw2, fotos, etc. all exist), sample ~20-30 already-captioned images (stratified across photos/screenshots/memes if possible), and manually run captioning at length="normal" with the indexer stopped, to avoid the current GPU contention distorting the numbers. Compare: char-length distribution vs. existing captions, qualitative content (does "normal" actually surface details "short" misses — the real question, not just length), and seconds/image vs. the previously-benchmarked 6.4s/image at "short". Do not assume linear scaling with output length — measure it.

2. Reprocessing mechanism (new code, only build if benchmark justifies a backfill)
is_indexed() (store/images.py) gates on the images row existing at all — no per-processor granularity — so already-indexed images never get automatically reprocessed just because a config/worker changed. purge_image() deletes everything derived, too broad (would force re-running OCR/image_embed/tagger unnecessarily, each in its own conda env).

Build a narrow, purpose-built path instead of reusing purge_image:

New store/images.py helper, purge_caption(conn, image_id): deletes just the captions row, the caption-sourced text_fts row (WHERE image_id = ? AND source = 'caption' — note this only works now that the source column exists, from this session's earlier field-filter work), and the stale text_embed vector (the combined OCR+caption embedding — must be dropped too, since it's built from the old caption text).
New CLI subcommand, image-search recaption <folder> [--model MODEL_ID] [--limit N] [--force], following cmd_dupes's existing dry-run-by-default pattern (--delete there → --force here): by default, print a count and estimated GPU-hours (from the benchmark above), require --force to actually execute. Implementation: select image_ids from captions where model matches the target for re-processing, run only caption + text_embed via registry.for_processors(...) (reuses the existing dispatch machinery), recompute accumulated_text from existing OCR text + new caption (factor the existing inline logic in _ingest_image into a small shared helper rather than duplicating it), replace the caption row and its text_fts/vector rows. Does not touch images, ocr_text, tags, or the image vector.
Do not attempt the "delete the images row and let normal ingest reprocess it" shortcut — the processors' INSERT statements are unconditional appends, not upserts, so this would create duplicate ocr_text/tags/text_fts rows for anything besides the caption itself.
3. Cost structure (fill in with real numbers from the benchmark)
total_backfill_gpu_hours ≈
    (count of images still on the old caption length)
    × (measured seconds/image at length="normal", from step 1)
    / 3600
Count needs to be measured close to the actual backfill decision (it's changing continuously while the indexer runs). Compare one-time bulk backfill vs. "accept the split, only caption new images at the new length going forward" — given this is a single contended 8GB GPU, leaning toward the latter unless the benchmark shows a compellingly large recall improvement, but this is the user's call once real numbers exist.

4. model_id labeling
model_id is a pure label today — caption_worker.py doesn't receive it at all, so the caption length is fully decoupled from the moondream2 string in captions.model. Renaming to e.g. moondream2-normal is safe (nothing else keys off the literal string) but purely cosmetic unless paired with making the worker actually receive and act on it (e.g. pass it as an argv element from subprocess_bridge.py's Popen call) — otherwise the label and behavior can silently drift apart again exactly as they just did. Separately: already-written captions since yesterday's silent switch cannot be retroactively distinguished by label alone — a one-time correction pass keyed on images.indexed_at vs. the 277a7a9 deploy timestamp would be needed to backfill correct labels for the already-mixed data, if pursued.

Verification
Step 0: confirm OOM rate drops after whatever fix is applied — watch data/index.log for a clean stretch after restarting the indexer.
Part 1: conda run -n sem_search_gpu python -m pytest green, including new fusion tests. Then live: re-run "darknet"/"scuba"/"espiritu santo" against production and confirm real matches now surface in the top 10-15, where they were previously completely absent.
Part 2: benchmark numbers recorded and shared before any bulk recaption --force run; recaption dry-run output sanity-checked against known captions counts before ever passing --force.
