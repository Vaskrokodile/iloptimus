"""Audited dataset-workspace tools for long-horizon local-model research."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .storage import app_home, atomic_write_json

WORKSPACE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class DatasetAudit:
    input_rows: int
    quality_rows: int
    accepted_rows: int
    exact_duplicates: int
    near_duplicates: int
    contaminated_rows: int
    short_rows: int
    repetitive_rows: int
    source_dominated_rows: int
    capped_rows: int
    source_count: int
    origin_count: int
    dataset_sha256: str

    def public(self) -> dict[str, Any]:
        return asdict(self)


def dataset_workspace(workspace_id: str) -> Path:
    if not WORKSPACE_ID.fullmatch(workspace_id):
        raise ValueError("Dataset workspace ids may contain only lowercase letters, digits, '_' and '-'")
    root = app_home() / "dataset-workspaces" / workspace_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_dataset_workspace(workspace_id: str | None = None) -> dict[str, str]:
    workspace_id = workspace_id or uuid.uuid4().hex[:12]
    root = dataset_workspace(workspace_id)
    return {"workspace_id": workspace_id, "root": str(root)}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    # `str.splitlines()` also splits valid JSON strings at Unicode U+2028 and
    # U+2029. JSONL records are delimited only by ASCII LF.
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
        if line.strip():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                rejected.append({"line": line_number, "error": str(error)})
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    if rejected:
        atomic_write_json(path.with_suffix(path.suffix + ".rejected.json"), {"rejected": rejected})
    return rows


def save_source_bundle(workspace_id: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist immutable source text plus hashes inside a bounded workspace."""
    root = dataset_workspace(workspace_id)
    sources = _read_jsonl(root / "sources.jsonl") + sources
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        text = str(source.get("text") or "").strip()
        if len(text) < 180:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        normalized.append(
            {
                "title": str(source.get("title") or source.get("url") or "Source"),
                "url": str(source.get("url") or ""),
                "text": text,
                "license": str(source.get("license") or "documentation"),
                "kind": str(source.get("kind") or "web-documentation"),
                "sha256": digest,
                "retrieved_at": str(source.get("retrieved_at") or datetime.now(UTC).isoformat()),
            }
        )
    path = root / "sources.jsonl"
    _write_jsonl(path, normalized)
    manifest = {
        "version": 1,
        "workspace_id": workspace_id,
        "source_count": len(normalized),
        "source_urls": sorted({row["url"] for row in normalized if row["url"]}),
        "source_hashes": [row["sha256"] for row in normalized],
    }
    atomic_write_json(root / "sources-manifest.json", manifest)
    return {**manifest, "path": str(path)}


def _windows(text: str, *, target_chars: int, overlap_chars: int) -> list[str]:
    """Split on syntactic boundaries while retaining enough local context."""
    raw_blocks = [
        block.strip()
        for block in re.split(r"(?<=[;{}])\s*\n|(?=\b(?:class|function|const|let|def)\b)", text)
        if block.strip()
    ]
    blocks: list[str] = []
    for block in raw_blocks:
        if len(block) <= target_chars:
            blocks.append(block)
            continue
        # Bound pathological minified/generated blocks. The source hash and
        # URL remain attached, and the overlap preserves local continuity.
        stride = max(1, target_chars - overlap_chars)
        blocks.extend(
            block[start : start + target_chars].strip()
            for start in range(0, len(block), stride)
            if len(block[start : start + target_chars].strip()) >= 120
        )
    windows: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) > target_chars:
            windows.append(current.strip())
            current = current[-overlap_chars:] + "\n" + block
        else:
            current = f"{current}\n{block}" if current else block
    if len(current.strip()) >= 240:
        windows.append(current.strip())
    if not windows:
        for start in range(0, len(text), max(1, target_chars - overlap_chars)):
            chunk = text[start : start + target_chars].strip()
            if len(chunk) >= 240:
                windows.append(chunk)
    return windows


def assemble_dataset(
    workspace_id: str,
    *,
    task: str,
    artifact_kind: str,
    requested_features: list[str],
    target_examples: int = 128,
    chunk_chars: int = 2_400,
) -> dict[str, Any]:
    """Assemble source-balanced, provenance-carrying raw demonstrations."""
    root = dataset_workspace(workspace_id)
    all_sources = _read_jsonl(root / "sources.jsonl")
    # Documentation is evidence for API/research audits, but source-completion
    # training must learn from runnable, permissively licensed code rather than
    # prose that happens to mention an API.
    sources = [
        source
        for source in all_sources
        if source.get("kind") == "repository-code" and _looks_like_code(str(source.get("text") or ""))
    ]
    if not sources:
        raise ValueError("The dataset workspace has no usable permissively licensed repository code")
    task_key = _normalize(task)
    rows: list[dict[str, Any]] = []
    source_windows = [_windows(str(source["text"]), target_chars=chunk_chars, overlap_chars=64) for source in sources]
    max_windows = max((len(chunks) for chunks in source_windows), default=0)
    for window_index in range(max_windows):
        for source, chunks in zip(sources, source_windows):
            if window_index >= len(chunks):
                continue
            excerpt = chunks[window_index]
            if task_key and task_key in _normalize(excerpt):
                continue
            features = [feature for feature in requested_features if _feature_in_text(feature, excerpt)]
            focus = ", ".join(features) or artifact_kind
            rows.append(
                {
                    "prompt": (
                        f"Implement a reusable {artifact_kind} pattern focused on {focus}. "
                        "Return complete, syntactically valid source and preserve the verified APIs."
                    ),
                    "ideal_response": excerpt,
                    "expected_answer": excerpt[:1_200],
                    "source_url": source.get("url", ""),
                    "source_hash": source.get("sha256", ""),
                    "source_origin": _source_origin(str(source.get("url") or "")),
                    "license": source.get("license", "documentation"),
                    "features": features,
                    "view": "implementation",
                }
            )
    rows = _select_diverse_rows(rows, target_examples)
    path = root / "dataset-raw.jsonl"
    _write_jsonl(path, rows)
    assembly = {
        "workspace_id": workspace_id,
        "rows": len(rows),
        "eligible_code_sources": len(sources),
        "excluded_non_code_sources": len(all_sources) - len(sources),
        "chunk_chars": chunk_chars,
        "target_examples": target_examples,
        "requested_features": requested_features,
        "path": str(path),
    }
    atomic_write_json(root / "dataset-assembly.json", assembly)
    return assembly


def expand_dataset(workspace_id: str, *, target_examples: int = 192) -> dict[str, Any]:
    """Expand into distinct source-grounded units without model hallucination."""
    root = dataset_workspace(workspace_id)
    rows = _read_jsonl(root / "dataset-raw.jsonl")
    if not rows:
        raise ValueError("Assemble the dataset before expanding it")
    expanded = list(rows)
    for row in rows:
        source = str(row["ideal_response"])
        # Non-overlapping, syntax-aware spans are genuinely different examples.
        # No local model is asked to fabricate an answer during expansion.
        spans = _windows(source, target_chars=max(480, len(source) // 2), overlap_chars=0)
        for span_index, response in enumerate(spans):
            if len(expanded) >= target_examples:
                break
            if len(response) < 240 or _normalize(response) == _normalize(source):
                continue
            features = ", ".join(str(item) for item in row.get("features", [])) or "the verified API"
            expanded.append(
                {
                    **row,
                    "prompt": (
                        f"Implement focused source unit {span_index + 1} using {features}. "
                        "Return only the complete source unit with the verified API usage."
                    ),
                    "ideal_response": response,
                    "expected_answer": response[:1_200],
                    "view": "source-unit",
                }
            )
        if len(expanded) >= target_examples:
            break
    path = root / "dataset-expanded.jsonl"
    _write_jsonl(path, expanded)
    return {"workspace_id": workspace_id, "rows": len(expanded), "path": str(path)}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_$]+", " ", text.casefold())).strip()


def _shingles(text: str, width: int = 5) -> set[int]:
    tokens = _normalize(text).split()
    if not tokens:
        return set()
    if len(tokens) < width:
        return {int.from_bytes(hashlib.blake2b(" ".join(tokens).encode(), digest_size=8).digest(), "big")}
    return {
        int.from_bytes(hashlib.blake2b(" ".join(tokens[index : index + width]).encode(), digest_size=8).digest(), "big")
        for index in range(len(tokens) - width + 1)
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def filter_dataset(
    workspace_id: str,
    *,
    holdout_task: str,
    near_duplicate_threshold: float = 0.84,
    minimum_response_chars: int = 220,
    maximum_rows: int = 512,
) -> dict[str, Any]:
    """Remove exact/near duplicates, contamination, tiny rows, and source domination."""
    root = dataset_workspace(workspace_id)
    input_path = root / "dataset-expanded.jsonl"
    if not input_path.exists():
        input_path = root / "dataset-raw.jsonl"
    rows = _read_jsonl(input_path)
    accepted: list[dict[str, Any]] = []
    fingerprints: list[set[int]] = []
    exact: set[str] = set()
    source_counts: Counter[str] = Counter()
    holdout = _normalize(holdout_task)
    exact_duplicates = near_duplicates = contaminated = short = repetitive = source_dominated = 0
    max_per_source = 3
    max_per_origin = max(6, math.ceil(maximum_rows * 0.2))
    origin_counts: Counter[str] = Counter()
    for row in rows:
        response = str(row.get("ideal_response") or "").strip()
        if len(response) < minimum_response_chars:
            short += 1
            continue
        if _line_diversity(response) < 0.45:
            repetitive += 1
            continue
        normalized = _normalize(response)
        if holdout and holdout in normalized:
            contaminated += 1
            continue
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in exact:
            exact_duplicates += 1
            continue
        source = str(row.get("source_url") or row.get("source_hash") or "unknown")
        origin = str(row.get("source_origin") or _source_origin(source) or source)
        if source_counts[source] >= max_per_source:
            source_dominated += 1
            continue
        if origin_counts[origin] >= max_per_origin:
            source_dominated += 1
            continue
        fingerprint = _shingles(response)
        if any(_jaccard(fingerprint, prior) >= near_duplicate_threshold for prior in fingerprints):
            near_duplicates += 1
            continue
        exact.add(digest)
        fingerprints.append(fingerprint)
        source_counts[source] += 1
        origin_counts[origin] += 1
        accepted.append({**row, "row_sha256": digest})
    quality_rows = len(accepted)
    maximum_rows = max(24, min(2_048, maximum_rows))
    if len(accepted) > maximum_rows:
        accepted = _select_diverse_rows(accepted, maximum_rows)
    capped_rows = quality_rows - len(accepted)
    output_path = root / "dataset-filtered.jsonl"
    _write_jsonl(output_path, accepted)
    dataset_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    audit = DatasetAudit(
        input_rows=len(rows),
        quality_rows=quality_rows,
        accepted_rows=len(accepted),
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        contaminated_rows=contaminated,
        short_rows=short,
        repetitive_rows=repetitive,
        source_dominated_rows=source_dominated,
        capped_rows=capped_rows,
        source_count=len({str(row.get("source_url") or "") for row in accepted}),
        origin_count=len({str(row.get("source_origin") or _source_origin(str(row.get("source_url") or ""))) for row in accepted}),
        dataset_sha256=dataset_hash,
    )
    atomic_write_json(root / "dataset-audit.json", audit.public())
    return {**audit.public(), "workspace_id": workspace_id, "path": str(output_path)}


def load_filtered_dataset(workspace_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(dataset_workspace(workspace_id) / "dataset-filtered.jsonl")


def load_source_bundle(workspace_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(dataset_workspace(workspace_id) / "sources.jsonl")


def audit_feature_coverage(
    rows: list[dict[str, Any]],
    requested_features: Iterable[str],
    *,
    minimum_rows: int = 4,
    minimum_sources: int = 2,
    minimum_origins: int = 2,
) -> dict[str, Any]:
    """Prove each requested capability has enough independent demonstrations."""
    coverage: dict[str, dict[str, Any]] = {}
    for feature in dict.fromkeys(str(item) for item in requested_features):
        matching = [row for row in rows if feature in row.get("features", [])]
        sources = {str(row.get("source_url") or "") for row in matching if row.get("source_url")}
        origins = {
            str(
                row.get("source_origin")
                or _source_origin(str(row.get("source_url") or ""))
                or row.get("source_hash")
                or "unknown"
            )
            for row in matching
        }
        passed = (
            len(matching) >= minimum_rows
            and len(sources) >= minimum_sources
            and len(origins) >= minimum_origins
        )
        coverage[feature] = {
            "passed": passed,
            "rows": len(matching),
            "minimum_rows": minimum_rows,
            "sources": len(sources),
            "minimum_sources": minimum_sources,
            "origins": len(origins),
            "minimum_origins": minimum_origins,
        }
    missing = [feature for feature, audit in coverage.items() if not audit["passed"]]
    return {"passed": not missing, "features": coverage, "missing_features": missing}


def _feature_in_text(feature: str, text: str) -> bool:
    aliases = {
        "three.js": ("three.", "three.js", "from 'three", 'from "three'),
        "voxel": ("voxel", "instancedmesh", "boxgeometry"),
        "shader": ("shadermaterial", "vertexshader", "fragmentshader", "gl_position"),
        "animation": (
            "requestanimationframe",
            "setanimationloop",
            "animationmixer",
            "clock.getdelta",
            "tween",
            "utime",
        ),
        "interaction": ("orbitcontrols", "addeventlistener", "raycaster"),
        "responsive": ("resize", "devicepixelratio", "innerwidth", "setsize", "updateprojectionmatrix"),
        "island": ("island", "terrain", "shore", "water", "heightmap"),
        "sakura": ("sakura", "cherry", "blossom", "petal", "particle", "sprite", "points", "wind"),
    }
    lowered = text.casefold()
    return any(alias in lowered for alias in aliases.get(feature, (feature.casefold(),)))


def _looks_like_code(text: str) -> bool:
    markers = (
        r"\b(?:const|let|var|function|class|def|import|export)\b",
        r"[{};]\s*(?://|/\*|\n|$)",
        r"</?(?:html|script|style|canvas)\b",
        r"\b(?:THREE\.|requestAnimationFrame|addEventListener)\b",
    )
    return len(text) >= 240 and sum(bool(re.search(marker, text, re.I)) for marker in markers) >= 2


def _select_diverse_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Greedily retain rare capabilities and underrepresented source origins."""
    remaining = list(enumerate(rows))
    selected: list[tuple[int, dict[str, Any]]] = []
    origin_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        best_position = max(
            range(len(remaining)),
            key=lambda position: (
                sum(1.0 / (1 + feature_counts[str(feature)]) for feature in remaining[position][1].get("features", [])),
                1.0
                / (
                    1
                    + origin_counts[
                        str(
                            remaining[position][1].get("source_origin")
                            or _source_origin(str(remaining[position][1].get("source_url") or ""))
                            or remaining[position][1].get("source_hash")
                            or "unknown"
                        )
                    ]
                ),
                remaining[position][1].get("view") == "implementation",
                -remaining[position][0],
            ),
        )
        original_index, row = remaining.pop(best_position)
        selected.append((original_index, row))
        origin = str(
            row.get("source_origin")
            or _source_origin(str(row.get("source_url") or ""))
            or row.get("source_hash")
            or "unknown"
        )
        origin_counts[origin] += 1
        feature_counts.update(str(feature) for feature in row.get("features", []))
    return [row for _, row in sorted(selected)]


def _source_origin(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.hostname in {"github.com", "www.github.com", "raw.githubusercontent.com"} and len(parts) >= 2:
        return f"github:{parts[0].casefold()}/{parts[1].removesuffix('.git').casefold()}"
    return (parsed.hostname or "unknown").casefold()


def _line_diversity(text: str) -> float:
    lines = [_normalize(line) for line in text.splitlines() if _normalize(line)]
    if len(lines) < 8:
        return 1.0
    return len(set(lines)) / len(lines)
