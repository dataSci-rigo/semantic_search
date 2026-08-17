# Ingesting "interesting stuff": saved images, links, and notes

Beyond the photo library, the index can hold everything you save on purpose:
memes you download, links worth keeping, and quick text notes. Everything
lands in one **drop folder** on the laptop; `image-search index` picks it up.

## 1. The drop folder

Create a folder (e.g. `~/Saved`) and add it to `config/folders.yaml`:

```yaml
folders:
  "~/Saved":
    ocr: rapidocr          # memes are text-heavy — OCR makes them findable
    caption: moondream2
    image_embed: siglip2-base
    # text_embed comes from defaults; it also embeds notes and links
```

What goes in it:

- **Images** — any saved meme/screenshot. Full pipeline applies.
- **Notes** — `.md` / `.txt` files. Title = first heading (or first line);
  the whole text is FTS-indexed and embedded.
- **Links** — `.links` files: one URL per line, optionally followed by a
  comment. Blank lines and `#` comments are skipped:

  ```
  https://example.com/great-thread   why this matters
  # to read later
  https://arxiv.org/abs/2401.00001
  ```

  Each URL's page title/text is fetched best-effort at ingest time (offline
  or dead links degrade to just the URL + your comment). Removing a line
  removes the item from the index on the next run.

## 2. Saving from anywhere: `POST /api/save`

The webapp exposes a capture endpoint (set `IMAGE_SEARCH_SAVE_DIR=~/Saved`
in its environment). Same Tailscale-only trust model as the rest of the app.

```bash
# a URL (appended to saved.links)
curl -X POST http://laptop:9100/api/save -H 'Content-Type: application/json' \
     -d '{"url": "https://example.com", "comment": "neat"}'

# a text note (written as note-<hash>.md)
curl -X POST http://laptop:9100/api/save -H 'Content-Type: application/json' \
     -d '{"text": "the actual note", "title": "optional title"}'

# an image
curl -X POST http://laptop:9100/api/save -F file=@meme.png
```

Run `image-search index` (cron it, or run it after saving) to ingest.

## 3. Discord: a #saved channel

Add this to the VM bot (the one already calling `/api/search`) so anything
you post or forward into a `#saved` channel gets captured:

```python
SAVE_URL = "http://<laptop-tailscale-ip>:9100/api/save"
SAVED_CHANNEL_ID = 123456789  # your #saved channel

@bot.event
async def on_message(message):
    if message.author.bot or message.channel.id != SAVED_CHANNEL_ID:
        return
    async with aiohttp.ClientSession() as session:
        for attachment in message.attachments:
            data = aiohttp.FormData()
            data.add_field("file", await attachment.read(), filename=attachment.filename)
            await session.post(SAVE_URL, data=data)
        content = message.content.strip()
        if content.startswith("http://") or content.startswith("https://"):
            url, _, comment = content.partition(" ")
            await session.post(SAVE_URL, json={"url": url, "comment": comment.strip()})
        elif content:
            await session.post(SAVE_URL, json={"text": content})
```

## 4. Browser bookmarks

`scripts/import_bookmarks.py` reads Firefox, Chrome, Edge, and Brave and writes
one de-duplicated `.links` file. Nothing is fetched at this stage — it is fast
and offline.

```bash
python scripts/import_bookmarks.py ~/Saved/bookmarks.links --dry-run  # counts only
python scripts/import_bookmarks.py ~/Saved/bookmarks.links
python scripts/import_bookmarks.py out.links --browser firefox        # one browser
```

Firefox's `places.sqlite` is opened read-only through SQLite's `immutable=1`
URI, so it works while Firefox is running and can't touch the profile.

**De-duplication** compares a normalized form of each URL — lowercased host,
no `www.`, no default port, no fragment, no tracking parameters (`utm_*`,
`fbclid`, `gclid`, …), sorted query, no trailing slash. The *original* URL is
what gets fetched; normalization is only for comparison. On this machine that
collapsed **3,992 bookmarks to 2,134 unique** — 46% were duplicates across
browsers.

### What happens at index time

Each link is embedded from its title and URL first, so it stays findable even
if the page never loads. Then the fetch classifies it:

| Outcome | `status` | Result |
|---|---|---|
| HTML with real text | `ok` | title, URL, and page text indexed |
| 404 / timeout / DNS failure | `dead` | row kept, hidden from search |
| Login, paywall, or captcha wall | `blocked` | title + URL only |
| Under 200 characters of text | `thin` | title + URL only |
| Local/private address, auth page, search results | `skipped` | never requested |

Nothing you saved is deleted — non-`ok` rows stay in the `items` table and are
simply excluded from results, so a wrong call here is visible and reversible:

```sql
SELECT status, COUNT(*) FROM items WHERE kind='link' GROUP BY status;
```

Fetching is polite by default: one request per host per second, a 10-second
timeout, a 1 MB cap, and one retry.

## 5. PDFs

Any `.pdf` in a watched folder is indexed from a **5-page sample** — the first
page plus an even spread to the last, since the opening pages of a long
document are all front matter. Pages with no text layer (scans) are OCR'd with
the folder's configured OCR model.

**Financial and tax documents are excluded by default** — filenames matching
`1099`, `W-2`, `1040`, `tax`, `statement`, `invoice`, `receipt`, `payroll`,
`K-1`. Override per folder:

```yaml
folders:
  "~/Papers":
    exclude_patterns: ["^draft-"]   # replaces the financial defaults
  "~/Everything":
    exclude_patterns: []            # index all PDFs
```

Filename matching is imperfect in both directions — check the run log, which
names every PDF it excluded.

## 6. Bookshelf photos → book records

`scripts/import_books.py` turns photos of your shelves into searchable books.

```bash
python scripts/import_books.py shelf1.jpg shelf2.jpg -o ~/Saved/books.links
```

Two stages, deliberately split:

1. **Claude vision reads the spines** (`claude-opus-5`, structured outputs) and
   returns a title, author, and confidence per book.
2. **Open Library supplies the description** and subjects. The summary is
   *looked up*, never recalled by a model — so an obscure title gets a real
   description or none at all, instead of a confident invention.

Anything below `--min-confidence` (default 0.7), or with no Open Library match,
goes to a `.review.txt` file rather than into the index — a misread spine
shouldn't become a phantom book. Reads `ANTHROPIC_API_KEY` from the
environment or a `.env` (`--env-file`). Open Library needs no credentials.

## 7. Reddit saved posts (optional, UNTESTED scaffold)

`scripts/fetch_reddit_saved.py` pulls your saved posts into the drop folder
(image posts are downloaded; text/link posts append to `saved.links`). It
needs a Reddit **script-type app** (reddit.com → prefs → apps) and env vars:

```bash
export REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=...
export REDDIT_USERNAME=... REDDIT_PASSWORD=...
python scripts/fetch_reddit_saved.py ~/Saved
```

It has not been run against the live API — treat the first run as a test
(it only writes into the drop folder, so worst case is junk files there).
