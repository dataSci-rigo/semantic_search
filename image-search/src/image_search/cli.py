from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

from image_search.config import load_config
from image_search.ingest import ingest_all
from image_search.registry import Registry
from image_search.search import search_similar_images, search_text
from image_search.store import images as images_store
from image_search.store.db import connect, migrate

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "folders.yaml"
DEFAULT_DB = Path(__file__).resolve().parents[2] / "index.db"


def cmd_index(args: argparse.Namespace) -> None:
    # Stream per-file failures to stderr as they happen: a full-library index
    # runs for days, so deferring them to the end (as the warning capture
    # below does) would hide them for the whole run.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config)
    conn = connect(args.db)
    migrate(conn)
    registry = Registry(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats = ingest_all(conn, config, registry)
        for w in caught:
            print(f"warning: {w.message}", file=sys.stderr)
    for folder_key, folder_stats in stats.items():
        print(f"{folder_key}: {folder_stats}")

    # Exit 0 even when individual files failed: a non-zero exit would trip
    # systemd's Restart=on-failure and recreate the restart loop that per-file
    # isolation exists to prevent. Failures are surfaced in the log + summary.
    total_failed = sum(s.get("failed", 0) for s in stats.values())
    if total_failed:
        print(
            f"{total_failed} file(s) failed and were skipped; they will be retried "
            "on the next run (see the log above for paths).",
            file=sys.stderr,
        )


def cmd_search(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    conn = connect(args.db)
    migrate(conn)
    registry = Registry(config)
    hits = search_text(conn, config, registry, args.folder, args.query, k=args.k)
    for hit in hits:
        print(f"{hit.score:.4f}\t{hit.source}\t{hit.path}")


def cmd_similar(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    conn = connect(args.db)
    migrate(conn)
    image_id = images_store.content_hash(Path(args.image))
    hits = search_similar_images(conn, config, args.folder, image_id, k=args.k)
    for hit in hits:
        print(f"{hit.score:.4f}\t{hit.path}")


def cmd_dupes(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    migrate(conn)
    groups = images_store.duplicate_groups(conn)
    if not groups:
        print("No duplicate files in the index.")
        return
    total = sum(len(paths) - 1 for _, paths in groups)
    for image_id, paths in groups:
        print(f"{image_id[:12]}  keep  {paths[0]}")
        for extra in paths[1:]:
            print(f"{'':12}  dup   {extra}")
    if not args.delete:
        print(
            f"\n{total} duplicate file(s) in {len(groups)} group(s). "
            "Re-run with --delete to remove them from disk "
            "(keeps the first path in each group)."
        )
        return
    deleted = 0
    for _, paths in groups:
        for extra in paths[1:]:
            Path(extra).unlink(missing_ok=True)
            conn.execute("DELETE FROM files WHERE path = ?", (extra,))
            deleted += 1
    conn.commit()
    print(f"\nDeleted {deleted} duplicate file(s); kept one copy per group.")


def cmd_tag_backfill(args: argparse.Namespace) -> None:
    """Tag already-indexed images from their stored image vectors — no vision
    forward passes, just dot products against the label prompts. Images with
    no stored vector (e.g. OCR-only screenshots) are skipped; guest mode
    covers those via `private:` path rules instead."""
    import struct

    from image_search.processors.image_embed import SiglipImageEmbedProcessor
    from image_search.processors.tagger import LABEL_PROMPTS, SOURCE
    from image_search.store.db import load_vec_extension
    from image_search.store.vectors import vec_table_name

    config = load_config(args.config)
    conn = connect(args.db)
    migrate(conn)
    load_vec_extension(conn)

    models = {
        folder.enabled("tagger") or folder.enabled("image_embed")
        for folder in config.folders.values()
    } - {None}
    if not models:
        print("No folder has a tagger or image_embed model configured; nothing to do.")
        return

    labels = list(LABEL_PROMPTS)
    total = 0
    nsfw_scored: list[tuple[float, str]] = []  # (nsfw score, path) for spot-checking
    for model in sorted(models):
        embedder = SiglipImageEmbedProcessor(model)
        embedder.load()
        label_vectors = embedder.embed_text(list(LABEL_PROMPTS.values()))

        table = vec_table_name("image", model)
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            print(f"{model}: no {table} table in this index; skipping.")
            continue
        rows = conn.execute(
            f"""
            SELECT vm.image_id AS image_id, v.embedding AS embedding, img.path AS path
            FROM vec_map vm
            JOIN {table} v ON v.rowid = vm.rowid
            JOIN images img ON img.id = vm.image_id
            WHERE vm.vec_table = ?
              AND vm.image_id NOT IN (SELECT DISTINCT image_id FROM tags)
            """,
            (table,),
        ).fetchall()
        print(f"{model}: {len(rows)} image(s) with a stored vector and no tags yet")

        for row in rows:
            raw = row["embedding"]
            vector = struct.unpack(f"<{len(raw) // 4}f", raw)
            scores = [
                (tag, sum(a * b for a, b in zip(vector, label_vec)))
                for tag, label_vec in zip(labels, label_vectors)
            ]
            ordered = sorted(scores, key=lambda s: s[1], reverse=True)
            conn.executemany(
                "INSERT INTO tags (image_id, tag, source, score, rank) VALUES (?, ?, ?, ?, ?)",
                [
                    (row["image_id"], tag, SOURCE, score, i + 1)
                    for i, (tag, score) in enumerate(ordered)
                ],
            )
            nsfw_rank = next(i + 1 for i, (tag, _) in enumerate(ordered) if tag == "nsfw")
            if nsfw_rank <= 3:
                nsfw_score = dict(scores)["nsfw"]
                nsfw_scored.append((nsfw_score, row["path"]))
            total += 1
            if total % 1000 == 0:
                conn.commit()
                print(f"  ...{total} tagged")
    conn.commit()
    print(f"\nTagged {total} image(s).")

    if nsfw_scored:
        nsfw_scored.sort(reverse=True)
        print(
            f"\n{len(nsfw_scored)} image(s) have nsfw in their top-3 labels. "
            "Top 30 by score — spot-check these to calibrate NSFW_EXCLUDE_RANK "
            "(guest mode hides rank <= 2):"
        )
        for score, path in nsfw_scored[:30]:
            print(f"  {score:+.4f}  {path}")
    else:
        print("\nNo image ranked nsfw in its top-3 labels.")


def cmd_cluster(args: argparse.Namespace) -> None:
    raise NotImplementedError("Face clustering lands in Phase 5 of the build spec.")


def cmd_serve(args: argparse.Namespace) -> None:
    raise NotImplementedError("The serve command lands in Phase 7 of the build spec.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-search")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to folders.yaml")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to the SQLite index file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Discover files and run enabled processors")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Hybrid semantic + keyword text search")
    p_search.add_argument("folder", help="Folder key from folders.yaml to search within")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=20)
    p_search.set_defaults(func=cmd_search)

    p_similar = sub.add_parser("similar", help="Reverse-image search: find visually similar images")
    p_similar.add_argument("folder", help="Folder key from folders.yaml to search within")
    p_similar.add_argument("image", help="Path to an already-indexed image")
    p_similar.add_argument("-k", type=int, default=20)
    p_similar.set_defaults(func=cmd_similar)

    p_dupes = sub.add_parser(
        "dupes", help="List duplicate files (same content under multiple paths)"
    )
    p_dupes.add_argument(
        "--delete",
        action="store_true",
        help="Delete all but the first path in each group — removes files from disk!",
    )
    p_dupes.set_defaults(func=cmd_dupes)

    p_backfill = sub.add_parser(
        "tag-backfill",
        help="Tag already-indexed images from stored vectors (for guest-mode nsfw filtering)",
    )
    p_backfill.set_defaults(func=cmd_tag_backfill)

    p_cluster = sub.add_parser("cluster", help="Re-cluster faces (not implemented yet)")
    p_cluster.set_defaults(func=cmd_cluster)

    p_serve = sub.add_parser("serve", help="Run the search service (not implemented yet)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
