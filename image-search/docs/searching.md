# How to search your pictures

## Where

- **On this laptop:** http://localhost:9100
- **From your phone** (Tailscale on): http://100.107.4.50:9100
- **Discord:** `/fotos <query>` (once the VM's ai-prep service is fixed)
- **Terminal:**
  `conda run -n sem_search_gpu image-search --db data/pictures_index.db search "~/Pictures" "beach sunset"`

## Writing queries

Search is hybrid: your words are matched semantically (meaning, via the
combined OCR+caption text of each image) *and* by exact keyword (FTS5).
Plain language works best — describe what's IN the picture:

- `scuba diving certification`
- `birthday party piñata`
- `receipt from home depot`  ← OCR text is searchable, so words visible in
  screenshots/documents are findable verbatim

### Filters

- **Image-type words act as filters, not search terms.** "unemployment
  graphs" filters to charts and ranks on "unemployment". Recognized:
  graph/chart/plot/diagram, meme/comic, screenshot, document/scan,
  photo/picture, art/drawing. The same filters exist as chips in the web UI.
- **Field filter (web UI dropdown / `&field=` in the API):** restrict keyword
  matching to `ocr` (text visible in the image) or `caption` (what the AI
  says the image depicts). Default searches both.

### Reverse image search

Click any result → "similar" shows visually similar images (SigLIP
embedding neighbors), regardless of any text.

## API (what Discord uses)

```
GET /api/search?q=<query>&k=10&field=ocr|caption&tag=<chip>&folder=~/Pictures
```
Returns JSON hits with `image_id`, `path`, `score`, `source` ("vector" =
semantic, "fts" = keyword). Thumbnails at `/image/<id>`, originals at
`/full/<id>`.

## Guest mode (hide private/NSFW results)

```
systemctl --user start image-search-web-guest.service   # guests around
systemctl --user start image-search-web.service         # back to normal
```
Guest mode shows a GUEST badge and hides anything under the `private:`
folders in `config/folders.yaml` plus AI-flagged nsfw images. Fill in
`private:` for it to be meaningful — it ships empty.

## What's indexed, and when

Everything in `~/Pictures`: photos get AI captions + visual embeddings;
Screenshots folders get OCR + text search. New files are picked up by the
daily 9 AM indexer run (`image-search-index.timer`); force a run anytime:
`systemctl --user start image-search-index.service`.

## Quirks worth knowing

- Results always pad to k — the tail of a result list can be junk even when
  the top hits are great. Judge the top few, not the bottom.
- Exact-keyword matches currently rank below semantic matches (RRF fusion
  fix is designed but not yet implemented — see IMPROVE_SEARCH.md).
- 5 corrupt files fail indexing every run; they're damaged originals, not a
  pipeline problem.
