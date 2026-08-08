from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from image_search.config import SearchConfig
from image_search.registry import Registry
from image_search.store import vectors as vectors_store


@dataclass(frozen=True)
class SearchHit:
    image_id: str
    path: str
    # Higher is better within a source ("vector": -distance, "fts": -bm25);
    # scores are NOT comparable across sources — results are concatenated,
    # vector hits first.
    score: float
    source: str  # "vector" | "fts"


def _fts_match_expr(query: str) -> str:
    # Quote each token so punctuation in the raw query can't break FTS5's
    # MATCH grammar (e.g. "unemployment: graphs?" would otherwise error);
    # embedded double quotes are escaped by doubling, per FTS5 string syntax.
    tokens = [tok.replace('"', '""') for tok in query.split()]
    return " ".join(f'"{tok}"' for tok in tokens) if tokens else '""'


def _fts_hits(conn: sqlite3.Connection, query: str, k: int) -> list[tuple[str, float]]:
    # FTS5 bm25(): more negative is more relevant; flip sign so higher = better.
    rows = conn.execute(
        """
        SELECT image_id, bm25(text_fts) AS rank
        FROM text_fts WHERE text_fts MATCH ?
        ORDER BY rank LIMIT ?
        """,
        (_fts_match_expr(query), k),
    ).fetchall()
    return [(r["image_id"], -r["rank"]) for r in rows]


def search_text(
    conn: sqlite3.Connection,
    config: SearchConfig,
    registry: Registry,
    folder_key: str,
    query: str,
    k: int = 20,
) -> list[SearchHit]:
    """Hybrid text search within one folder (spec section 7.3, minus facet
    parsing which lands in Phase 6): semantic rank via the folder's
    text_embed model, FTS5 keyword fallback for anything the embedding misses."""
    folder = config.folders[folder_key]
    text_embed_model = folder.enabled("text_embed")

    vector_hits: list[tuple[str, float]] = []
    if text_embed_model:
        embedder = registry.get("text_embed", text_embed_model)
        query_vector = embedder.embed(query)  # type: ignore[attr-defined]
        # sqlite-vec distance is ascending-better; flip sign for "higher is better".
        raw = vectors_store.query_nearest(conn, "text", text_embed_model, query_vector, k=k)
        vector_hits = [(image_id, -dist) for image_id, dist in raw]

    seen = {image_id for image_id, _ in vector_hits}
    fts_hits = [(iid, score) for iid, score in _fts_hits(conn, query, k) if iid not in seen]

    def rows_for(hits: list[tuple[str, float]], source: str) -> list[SearchHit]:
        out = []
        for image_id, score in hits:
            # Post-filter to the requested folder: the FTS/vector queries scan
            # every folder's rows, so fewer than k hits may survive.
            row = conn.execute(
                "SELECT path FROM images WHERE id = ? AND folder = ?", (image_id, folder_key)
            ).fetchone()
            if row is None:
                continue
            out.append(SearchHit(image_id=image_id, path=row["path"], score=score, source=source))
        return out

    return rows_for(vector_hits, "vector") + rows_for(fts_hits, "fts")


def search_similar_images(
    conn: sqlite3.Connection,
    config: SearchConfig,
    folder_key: str,
    image_id: str,
    k: int = 20,
) -> list[SearchHit]:
    """Reverse-image search (spec section 7.4): given an already-indexed
    image_id, find its nearest neighbors in the same (space, model)
    partition. Locked to that one image model — never cross-model cosine."""
    folder = config.folders[folder_key]
    image_embed_model = folder.enabled("image_embed")
    if not image_embed_model:
        raise ValueError(f"Folder {folder_key!r} has no image_embed configured")

    query_vector = vectors_store.get_vector(conn, "image", image_embed_model, image_id)
    if query_vector is None:
        raise ValueError(
            f"No {image_embed_model!r} image vector stored for image_id={image_id!r} "
            "(index it first)"
        )

    # Over-fetch by one since the query image itself is its own nearest neighbor.
    raw = vectors_store.query_nearest(conn, "image", image_embed_model, query_vector, k=k + 1)

    out = []
    for iid, dist in raw:
        if iid == image_id:
            continue
        # Same folder post-filter as search_text's rows_for.
        row = conn.execute(
            "SELECT path FROM images WHERE id = ? AND folder = ?", (iid, folder_key)
        ).fetchone()
        if row is None:
            continue
        out.append(SearchHit(image_id=iid, path=row["path"], score=-dist, source="vector"))
    return out[:k]
