from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from image_search.config import SearchConfig
from image_search.registry import Registry
from image_search.store import vectors as vectors_store


@dataclass(frozen=True)
class SearchHit:
    image_id: str
    path: str  # image path, or the note/.links file the item came from
    # Higher is better within a source ("vector": -distance, "fts": -bm25);
    # scores are NOT comparable across sources — results are concatenated,
    # vector hits first.
    score: float
    source: str  # "vector" | "fts"
    kind: str = "image"  # "image" | "note" | "link"
    title: str | None = None
    url: str | None = None
    snippet: str | None = None


# Structural words -> image-type facet (spec section 7.3). "unemployment
# graphs" filters to charts and semantically ranks on "unemployment".
FACET_KEYWORDS = {
    "graph": "chart", "graphs": "chart", "chart": "chart", "charts": "chart",
    "plot": "chart", "plots": "chart", "diagram": "chart",
    "meme": "meme", "memes": "meme", "comic": "meme", "comics": "meme",
    "screenshot": "screenshot", "screenshots": "screenshot",
    "document": "document", "documents": "document", "scan": "document",
    "photo": "photo", "photos": "photo", "photograph": "photo", "picture": "photo",
    "art": "art", "drawing": "art", "artwork": "art",
}
# A tag counts as applying to an image when it is among its top-N labels;
# raw cosines are not comparable across images, so rank is the filter.
FACET_RANK_CUTOFF = 2


def parse_facets(query: str) -> tuple[str, set[str]]:
    """Split a raw query into (semantic_text, facet tags). Words that name an
    image type are consumed as filters, not as search terms."""
    kept, tags = [], set()
    for token in query.split():
        facet = FACET_KEYWORDS.get(token.strip(".,!?").lower())
        if facet:
            tags.add(facet)
        else:
            kept.append(token)
    # An all-facet query ("memes") keeps its words so there is still something
    # to rank by; the filter does the real work.
    return (" ".join(kept) if kept else query), tags


def _tagged_ids(conn: sqlite3.Connection, tags: set[str]) -> set[str]:
    """image_ids carrying every requested tag within the rank cutoff."""
    ids: set[str] | None = None
    for tag in tags:
        rows = conn.execute(
            "SELECT image_id FROM tags WHERE tag = ? AND rank IS NOT NULL AND rank <= ?",
            (tag, FACET_RANK_CUTOFF),
        ).fetchall()
        found = {r["image_id"] for r in rows}
        ids = found if ids is None else (ids & found)
    return ids or set()


def _item_ids(conn: sqlite3.Connection, folder_key: str) -> set[str]:
    return {
        r["id"] for r in conn.execute("SELECT id FROM items WHERE folder = ?", (folder_key,))
    }


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
    tags: set[str] | None = None,
) -> list[SearchHit]:
    """Hybrid text search within one folder (spec section 7.3): filter by
    image-type facets, then semantic rank via the folder's text_embed model,
    with an FTS5 keyword fallback for anything the embedding misses.

    Facets come from the query itself ("unemployment graphs" -> charts about
    unemployment) or explicitly via `tags` from the UI."""
    folder = config.folders[folder_key]
    text_embed_model = folder.enabled("text_embed")

    explicit_tags = set(tags or set())
    semantic_text, parsed_tags = parse_facets(query)
    active_tags = explicit_tags | parsed_tags
    # Over-fetch when filtering: the vector/FTS scans don't know about facets,
    # so a plain k would often come back empty after filtering.
    fetch_k = k * 5 if active_tags else k

    allowed: set[str] | None = None
    if active_tags:
        allowed = _tagged_ids(conn, active_tags)
        if not explicit_tags:
            # Facets parsed out of the query describe *image* structure, which
            # says nothing about a saved note or link — so a typed "memes"
            # must not hide the notes that mention memes. An explicit chip is
            # a deliberate "show me this kind of image", and stays strict.
            allowed |= _item_ids(conn, folder_key)

    vector_hits: list[tuple[str, float]] = []
    if text_embed_model:
        embedder = registry.get("text_embed", text_embed_model)
        query_vector = embedder.embed(semantic_text)  # type: ignore[attr-defined]
        # sqlite-vec distance is ascending-better; flip sign for "higher is better".
        raw = vectors_store.query_nearest(
            conn, "text", text_embed_model, query_vector, k=fetch_k
        )
        vector_hits = [(image_id, -dist) for image_id, dist in raw]

    seen = {image_id for image_id, _ in vector_hits}
    fts_hits = [
        (iid, score) for iid, score in _fts_hits(conn, semantic_text, fetch_k)
        if iid not in seen
    ]

    if allowed is not None:
        vector_hits = [h for h in vector_hits if h[0] in allowed]
        fts_hits = [h for h in fts_hits if h[0] in allowed]

    def rows_for(hits: list[tuple[str, float]], source: str) -> list[SearchHit]:
        out = []
        for image_id, score in hits:
            # Post-filter to the requested folder: the FTS/vector queries scan
            # every folder's rows, so fewer than k hits may survive. An id
            # resolves either to an image or to a note/link item.
            row = conn.execute(
                "SELECT path FROM images WHERE id = ? AND folder = ?", (image_id, folder_key)
            ).fetchone()
            if row is not None:
                out.append(
                    SearchHit(image_id=image_id, path=row["path"], score=score, source=source)
                )
                continue
            # status filter: dead/blocked/thin links stay in the table (so a
            # bad filter is auditable) but never surface as results.
            item = conn.execute(
                "SELECT kind, src_path, title, url, substr(body, 1, 300) AS snippet "
                "FROM items WHERE id = ? AND folder = ? "
                "AND (status IS NULL OR status = 'ok')",
                (image_id, folder_key),
            ).fetchone()
            if item is not None:
                out.append(
                    SearchHit(
                        image_id=image_id,
                        path=item["src_path"],
                        score=score,
                        source=source,
                        kind=item["kind"],
                        title=item["title"],
                        url=item["url"],
                        snippet=item["snippet"],
                    )
                )
        return out

    return (rows_for(vector_hits, "vector") + rows_for(fts_hits, "fts"))[:k]


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
