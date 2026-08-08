from __future__ import annotations

import json
import re
import sqlite3
import struct

from image_search.store.db import load_vec_extension

Space = str  # "text" | "image"


def _sanitize(model_id: str) -> str:
    """Model ids contain characters (., -) not safe in SQL identifiers."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", model_id)


def vec_table_name(space: Space, model: str) -> str:
    return f"vec_{space}__{_sanitize(model)}"


def ensure_vec_table(conn: sqlite3.Connection, space: Space, model: str, dim: int) -> str:
    """Create the vec0 table for this (space, model) partition if it doesn't
    exist yet. Returns the table name. One table per model per space enforces
    the locked-model invariant at the schema level (spec section 0)."""
    load_vec_extension(conn)
    table = vec_table_name(space, model)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding float[{dim}])"
    )
    return table


def insert_vector(
    conn: sqlite3.Connection, space: Space, model: str, image_id: str, vector: list[float]
) -> None:
    """Insert one vector. Does not commit — the caller owns the transaction
    (ingest groups all of an image's records into one commit)."""
    table = ensure_vec_table(conn, space, model, dim=len(vector))
    cur = conn.execute(
        f"INSERT INTO {table} (embedding) VALUES (?)", (json.dumps(vector),)
    )
    conn.execute(
        "INSERT INTO vec_map (vec_table, rowid, image_id) VALUES (?, ?, ?)",
        (table, cur.lastrowid, image_id),
    )


def delete_vectors(conn: sqlite3.Connection, image_id: str) -> int:
    """Remove every stored vector for an image_id across all (space, model)
    partitions, via the vec_map sidecar. Returns the number removed."""
    rows = conn.execute(
        "SELECT vec_table, rowid AS r FROM vec_map WHERE image_id = ?", (image_id,)
    ).fetchall()
    if not rows:
        return 0
    load_vec_extension(conn)
    for row in rows:
        conn.execute(f'DELETE FROM {row["vec_table"]} WHERE rowid = ?', (row["r"],))
    conn.execute("DELETE FROM vec_map WHERE image_id = ?", (image_id,))
    return len(rows)


def get_vector(
    conn: sqlite3.Connection, space: Space, model: str, image_id: str
) -> list[float] | None:
    """Fetch the stored vector for one image_id within a (space, model)
    partition, e.g. to drive reverse-image search from an already-indexed
    image rather than re-embedding it."""
    table = vec_table_name(space, model)
    load_vec_extension(conn)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return None
    row = conn.execute(
        f"""
        SELECT v.embedding AS embedding
        FROM vec_map vm
        JOIN {table} v ON v.rowid = vm.rowid
        WHERE vm.vec_table = ? AND vm.image_id = ?
        """,
        (table, image_id),
    ).fetchone()
    if row is None:
        return None
    # vec0 returns the embedding column as packed float32 bytes, not JSON
    # (only INSERT accepts the JSON-text form).
    raw = row["embedding"]
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw))


def query_nearest(
    conn: sqlite3.Connection, space: Space, model: str, vector: list[float], k: int = 20
) -> list[tuple[str, float]]:
    """Cosine-nearest image_ids within one (space, model) partition.
    Returns [(image_id, distance)], ascending distance (most similar first)."""
    table = vec_table_name(space, model)
    load_vec_extension(conn)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return []
    rows = conn.execute(
        f"""
        SELECT vm.image_id AS image_id, v.distance AS distance
        FROM (
            SELECT rowid, distance FROM {table}
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
        ) v
        JOIN vec_map vm ON vm.vec_table = ? AND vm.rowid = v.rowid
        ORDER BY v.distance
        """,
        (json.dumps(vector), k, table),
    ).fetchall()
    return [(r["image_id"], r["distance"]) for r in rows]
