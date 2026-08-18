"""Persistent, content-addressed source corpus for dataset factory jobs.

Session workspaces (``dataset_tools.dataset_workspace``) remain the unit that
curation operates on; the corpus store is the durable asset behind them:

- Source text is stored once as a content-addressed blob keyed by SHA-256
  (``blobs/<sha[:2]>/<sha>``), so the same document scraped by many jobs is
  never stored twice — global de-duplication by construction.
- Metadata lives in a SQLite index (title, url, license, kind, size,
  retrieval time) for cheap listing and lookup without touching blobs.
- A persisted bloom filter pre-screens membership; SQLite remains the source
  of truth when the bloom filter reports a possible hit.
- ``export_to_workspace`` bridges the corpus into the session-workspace
  layout (``sources.jsonl``) that ``curate_dataset`` consumes, so all
  existing quality gates apply unchanged to corpus-backed datasets.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .dedup import BloomFilter
from .storage import app_home

MIN_SOURCE_CHARS = 180  # same floor as dataset_tools.save_source_bundle

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    sha256 TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    license TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    chars INTEGER NOT NULL DEFAULT 0,
    retrieved_at TEXT NOT NULL DEFAULT ''
);
"""


def corpus_root() -> Path:
    root = app_home() / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_source(source: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and hash one candidate source; None when it is too short."""
    text = str(source.get("text") or "").strip()
    if len(text) < MIN_SOURCE_CHARS:
        return None
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "title": str(source.get("title") or source.get("url") or "Source"),
        "url": str(source.get("url") or ""),
        "text": text,
        "license": str(source.get("license") or "documentation"),
        "kind": str(source.get("kind") or "web-documentation"),
        "sha256": digest,
        "retrieved_at": str(source.get("retrieved_at") or datetime.now(UTC).isoformat()),
    }


class CorpusStore:
    """Content-addressed corpus with SQLite index and bloom membership."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else corpus_root()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "index.db"
        self._bloom_path = self.root / "membership.bloom"
        # check_same_thread=False: job runners execute the store from worker
        # threads (server endpoints use asyncio.to_thread). Access is
        # serialized per store instance, and WAL mode keeps writes safe.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._bloom = self._load_bloom()

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def _load_bloom(self) -> BloomFilter:
        if self._bloom_path.exists():
            try:
                return BloomFilter.load(self._bloom_path)
            except Exception:  # noqa: BLE001 - rebuild below if unreadable
                pass
        bloom = BloomFilter(capacity=1_000_000, error_rate=1e-6)
        for (sha,) in self._conn.execute("SELECT sha256 FROM sources"):
            bloom.add(sha)
        bloom.save(self._bloom_path)
        return bloom

    # ------------------------------------------------------------ membership

    def has_source(self, sha256: str) -> bool:
        """Global de-duplication check: is this content already in the corpus?"""
        if sha256 not in self._bloom:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM sources WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return row is not None

    def source_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0])

    # ------------------------------------------------------------ mutations

    def add_sources(self, sources: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Add many sources in one call; content-addressed de-dup applies.

        Returns per-call accounting plus the hashes of all valid sources
        selected by this call, allowing jobs to preserve their provenance
        boundary even when a source was already in the shared corpus.
        """
        added_hashes: list[str] = []
        selected_hashes: list[str] = []
        duplicates = 0
        rejected_short = 0
        for candidate in sources:
            normalized = _normalize_source(candidate)
            if normalized is None:
                rejected_short += 1
                continue
            sha = normalized["sha256"]
            selected_hashes.append(sha)
            if self.has_source(sha):
                duplicates += 1
                continue
            blob_path = self._blob_path(sha)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.write_text(normalized["text"], encoding="utf-8")
            self._conn.execute(
                "INSERT INTO sources (sha256, title, url, license, kind, chars, retrieved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    sha,
                    normalized["title"],
                    normalized["url"],
                    normalized["license"],
                    normalized["kind"],
                    len(normalized["text"]),
                    normalized["retrieved_at"],
                ),
            )
            self._bloom.add(sha)
            added_hashes.append(sha)
        self._conn.commit()
        self._bloom.save(self._bloom_path)
        return {
            "added": len(added_hashes),
            "duplicates": duplicates,
            "rejected_short": rejected_short,
            "total": self.source_count(),
            "added_hashes": added_hashes,
            # All valid candidates selected by this call, including content
            # already present in the shared corpus. Jobs use this set to keep
            # their workspace scoped to their own acquisition request.
            "source_hashes": sorted(set(selected_hashes)),
        }

    # -------------------------------------------------------------- queries

    def _blob_path(self, sha256: str) -> Path:
        return self.root / "blobs" / sha256[:2] / sha256

    def get_source(self, sha256: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT sha256, title, url, license, kind, chars, retrieved_at "
            "FROM sources WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        blob_path = self._blob_path(sha256)
        text = blob_path.read_text(encoding="utf-8") if blob_path.exists() else ""
        return {
            "sha256": row[0],
            "title": row[1],
            "url": row[2],
            "license": row[3],
            "kind": row[4],
            "chars": row[5],
            "retrieved_at": row[6],
            "text": text,
        }

    def iter_source_meta(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        query = "SELECT sha256, title, url, license, kind, chars, retrieved_at FROM sources"
        args: tuple = ()
        if kind:
            query += " WHERE kind = ?"
            args = (kind,)
        query += " ORDER BY sha256"
        cursor = self._conn.execute(query, args)
        for row in cursor:
            yield {
                "sha256": row[0],
                "title": row[1],
                "url": row[2],
                "license": row[3],
                "kind": row[4],
                "chars": row[5],
                "retrieved_at": row[6],
            }

    # -------------------------------------------------------------- exports

    def export_to_workspace(
        self,
        workspace_id: str,
        *,
        kind: str | None = None,
        shas: list[str] | None = None,
    ) -> dict[str, Any]:
        """Materialize corpus sources as the workspace's ``sources.jsonl``.

        Uses the same append-and-dedupe semantics as
        ``dataset_tools.save_source_bundle``, so repeated exports are
        idempotent and existing workspace layouts stay valid.
        """
        from .dataset_tools import load_source_bundle, save_source_bundle

        if shas is not None:
            wanted = set(shas)
            candidates = [
                self.get_source(sha)
                for sha in sorted(wanted)
                if self.has_source(sha)
            ]
            candidates = [c for c in candidates if c is not None]
        else:
            candidates = []
            for meta in self.iter_source_meta(kind=kind):
                full = self.get_source(meta["sha256"])
                if full is not None:
                    candidates.append(full)
        result = save_source_bundle(workspace_id, candidates)
        return {
            "exported": len(candidates),
            "workspace_sources": len(load_source_bundle(workspace_id)),
            "manifest": result,
        }
