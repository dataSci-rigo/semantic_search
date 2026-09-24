# Portable index (external drives)

Since DB format v2, the index is **location-independent**: paths in the DB are
stored relative to each folder's root, and the `folders:` keys are stable
logical names (`drive`, `pictures`), not paths. A drive can therefore carry
its own index and be searched wherever it mounts, on any machine or OS.

## Layout on the drive

```
<drive>/.semantic_search/folders.yaml    # copy of config/drive.yaml
<drive>/.semantic_search/index.db        # created by the first ingest
```

Everything in that config is relative to the file itself: `path: ..` is the
drive root, `db: index.db` sits beside it. Nothing machine-specific is stored
on the drive.

## Using it

```bash
# this machine (WSL): drive mounts at /mnt/d
image-search --config /mnt/d/.semantic_search/folders.yaml index
image-search --config /mnt/d/.semantic_search/folders.yaml search drive "thesis slides"

# any other Linux box: clone the repo, install deps, then
image-search --config /media/user/Seagate/.semantic_search/folders.yaml search drive "..."

# Windows (native — faster disk than the WSL 9p bridge)
image-search --config D:\.semantic_search\folders.yaml index
```

DB path precedence everywhere (CLI and webapp): `--db` flag >
`IMAGE_SEARCH_DB` > the config's `db:` key > `<project>/data/pictures_index.db`.

On Windows, `index` holds the system and display awake until it finishes:
on Modern Standby laptops the screen timing out *is* standby, which suspends
the indexer for hours even on AC. The request ends with the process (no power
setting changes); `--allow-sleep` opts out. Closing the lid still sleeps.

The first run per machine downloads the models to the HuggingFace cache
(network needed once). The receiving machine must have the same model ids
available — vector tables are named after them.

## Safety on removable media

- An **unmounted root never prunes**: a missing root skips the folder with an
  error, and a walk that finds nothing where files were known holds the prune
  back (`index --prune-empty` forces it after a genuine mass deletion).
- **Journal mode** is picked per filesystem: WAL locally, TRUNCATE on mounts
  that can't support WAL (9p/drvfs under WSL, ntfs-3g, exFAT, network shares).
  Override with `IMAGE_SEARCH_JOURNAL_MODE` or `connect(journal_mode=...)`.
- OS junk dirs (`$RECYCLE.BIN`, `System Volume Information`, `found.*`,
  `.Trash-*`, …) are excluded from the walk by default; a folder's
  `exclude_dirs:` list replaces the default set.

## Old (pre-v2) DBs

A v0 DB with data is refused with a "re-index" error — paths in it are
absolute and its `files` table is keyed by path alone. Re-index into a fresh
file (recommended), or migrate by hand: rewrite `files.path`, `images.path`,
`items.src_path` to be root-relative, rename the folder keys to logical
names, rebuild `files` with `PRIMARY KEY (folder, path)`, then set
`PRAGMA user_version = 2`.
