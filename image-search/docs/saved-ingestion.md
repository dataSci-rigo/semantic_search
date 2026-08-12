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

## 4. Reddit saved posts (optional, UNTESTED scaffold)

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
