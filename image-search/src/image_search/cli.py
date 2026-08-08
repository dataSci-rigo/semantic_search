from __future__ import annotations

import argparse
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
