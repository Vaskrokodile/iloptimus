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

from .dedup import ExactJaccardGuard, MinHashDuplicateGuard
from .storage import app_home, atomic_write_json

WORKSPACE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class CurationConfig:
    """Caps and thresholds for dataset curation.

    ``LEGACY_CURATION`` reproduces the original session-scale behavior
    exactly: the 2,048-row cap, 3 rows per source, exact O(n^2) Jaccard
    de-duplication. ``FACTORY_CURATION`` lifts the caps for corpus-scale
    curation while keeping every quality gate active, and switches
    near-duplicate detection to MinHash/LSH so filtering stays sub-quadratic
    at tens of thousands of rows.
    """

    maximum_rows: int = 2_048
    near_duplicate_threshold: float = 0.84
    minimum_response_chars: int = 220
    minimum_quality_score: float = 0.5
    max_per_source: int = 3
    max_per_origin_fraction: float = 0.2
    max_per_origin_floor: int = 6
    # MinHash/LSH near-duplicate detection instead of exact Jaccard scans.
    minhash_dedup: bool = False


LEGACY_CURATION = CurationConfig()
FACTORY_CURATION = CurationConfig(
    maximum_rows=250_000,
    max_per_source=64,
    max_per_origin_fraction=0.5,
    max_per_origin_floor=64,
    minhash_dedup=True,
)


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
    low_quality_rows: int
    source_dominated_rows: int
    capped_rows: int
    source_count: int
    origin_count: int
    mean_quality_score: float
    minimum_quality_score: float
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
    priority_features: list[str] | None = None,
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
    priority = set(priority_features or [])
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
            quality_score = score_source_unit(excerpt, features)
            curriculum_role = (
                "integration"
                if len(features) >= 3
                else "remediation"
                if priority.intersection(features)
                else "capability"
            )
            rows.append(
                {
                    "prompt": (
                        f"Write a complete {artifact_kind} source unit for {focus}. "
                        "Return syntactically valid code only."
                    ),
                    "ideal_response": excerpt,
                    "expected_answer": excerpt[:1_200],
                    "source_url": source.get("url", ""),
                    "source_hash": source.get("sha256", ""),
                    "source_origin": _source_origin(str(source.get("url") or "")),
                    "license": source.get("license", "documentation"),
                    "features": features,
                    "view": "implementation",
                    "quality_score": quality_score,
                    "curriculum_role": curriculum_role,
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
        "priority_features": sorted(priority),
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
                        f"Write source unit {span_index + 1} for {features}. "
                        "Return valid code only."
                    ),
                    "ideal_response": response,
                    "expected_answer": response[:1_200],
                    "view": "source-unit",
                    "quality_score": score_source_unit(response, row.get("features", [])),
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
    minimum_quality_score: float = 0.5,
    config: CurationConfig | None = None,
) -> dict[str, Any]:
    """Remove exact/near duplicates, contamination, tiny rows, and source domination.

    With ``config=None`` the original session-scale behavior runs unchanged
    (2,048-row cap, exact Jaccard scans). Passing a ``CurationConfig`` (e.g.
    ``FACTORY_CURATION``) switches to the streaming, sub-quadratic
    implementation so corpus-scale datasets can be filtered in bounded time;
    the config's fields then govern all thresholds and the scalar keyword
    arguments are ignored.
    """
    if config is not None:
        return _filter_dataset_streaming(
            workspace_id, holdout_task=holdout_task, config=config
        )
    root = dataset_workspace(workspace_id)
    input_path = root / "dataset-expanded.jsonl"
    if not input_path.exists():
        input_path = root / "dataset-raw.jsonl"
    rows = _read_jsonl(input_path)
    feature_frequency: Counter[str] = Counter(
        str(feature)
        for row in rows
        for feature in row.get("features", [])
    )
    # Enforce rare-capability reservations before per-source and per-origin
    # caps. A sequential filter can otherwise spend an origin's allowance on
    # generic rows before reaching its only island/accessibility example.
    rows = [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda item: (
                min(
                    (feature_frequency[str(feature)] for feature in item[1].get("features", [])),
                    default=10**9,
                ),
                item[1].get("curriculum_role") != "remediation",
                -float(item[1].get("quality_score") or 0.0),
                item[0],
            ),
        )
    ]
    accepted: list[dict[str, Any]] = []
    fingerprints: list[set[int]] = []
    exact: set[str] = set()
    source_counts: Counter[str] = Counter()
    holdout = _normalize(holdout_task)
    exact_duplicates = near_duplicates = contaminated = short = repetitive = low_quality = source_dominated = 0
    max_per_source = 3
    max_per_origin = max(6, math.ceil(maximum_rows * 0.2))
    origin_counts: Counter[str] = Counter()
    for row in rows:
        response = str(row.get("ideal_response") or "").strip()
        if len(response) < minimum_response_chars:
            short += 1
            continue
        normalized = _normalize(response)
        if holdout and holdout in normalized:
            contaminated += 1
            continue
        if _line_diversity(response) < 0.45:
            repetitive += 1
            continue
        quality_score = float(row.get("quality_score") or score_source_unit(response, row.get("features", [])))
        if quality_score < minimum_quality_score:
            low_quality += 1
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
        accepted.append({**row, "quality_score": quality_score, "row_sha256": digest})
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
        low_quality_rows=low_quality,
        source_dominated_rows=source_dominated,
        capped_rows=capped_rows,
        source_count=len({str(row.get("source_url") or "") for row in accepted}),
        origin_count=len({str(row.get("source_origin") or _source_origin(str(row.get("source_url") or ""))) for row in accepted}),
        mean_quality_score=round(
            sum(float(row.get("quality_score") or 0.0) for row in accepted) / max(1, len(accepted)), 4
        ),
        minimum_quality_score=round(
            min((float(row.get("quality_score") or 0.0) for row in accepted), default=0.0), 4
        ),
        dataset_sha256=dataset_hash,
    )
    atomic_write_json(root / "dataset-audit.json", audit.public())
    return {**audit.public(), "workspace_id": workspace_id, "path": str(output_path)}


def _filter_dataset_streaming(
    workspace_id: str,
    *,
    holdout_task: str,
    config: CurationConfig,
) -> dict[str, Any]:
    """Streaming, sub-quadratic implementation of ``filter_dataset``.

    Rows are never all held in memory: pass 1 scans byte offsets and
    lightweight sort fields, pass 2 seeks to each row in rare-first order and
    applies the same gates with the same ordering as the legacy path.
    Near-duplicate detection uses MinHash/LSH when ``config.minhash_dedup``
    is set, otherwise the exact Jaccard guard.
    """
    root = dataset_workspace(workspace_id)
    input_path = root / "dataset-expanded.jsonl"
    if not input_path.exists():
        input_path = root / "dataset-raw.jsonl"

    # Pass 1 — lightweight scan: byte offsets plus the fields the rare-first
    # sort needs. Rows themselves stay on disk.
    offsets: list[int] = []
    sort_fields: list[tuple[list[str], str, float]] = []
    feature_frequency: Counter[str] = Counter()
    with input_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            features = [str(feature) for feature in payload.get("features", [])]
            offsets.append(offset)
            sort_fields.append(
                (
                    features,
                    str(payload.get("curriculum_role") or ""),
                    float(payload.get("quality_score") or 0.0),
                )
            )
            feature_frequency.update(features)
    input_rows = len(offsets)

    # Same rare-first reservation order as the legacy path.
    order = sorted(
        range(input_rows),
        key=lambda index: (
            min(
                (feature_frequency[feature] for feature in sort_fields[index][0]),
                default=10**9,
            ),
            sort_fields[index][1] != "remediation",
            -sort_fields[index][2],
            index,
        ),
    )

    guard: ExactJaccardGuard | MinHashDuplicateGuard
    if config.minhash_dedup:
        guard = MinHashDuplicateGuard(threshold=config.near_duplicate_threshold)
    else:
        guard = ExactJaccardGuard(threshold=config.near_duplicate_threshold)

    holdout = _normalize(holdout_task)
    max_per_source = config.max_per_source
    max_per_origin = max(
        config.max_per_origin_floor,
        math.ceil(config.maximum_rows * config.max_per_origin_fraction),
    )

    accepted: list[dict[str, Any]] = []
    exact: set[str] = set()
    source_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    exact_duplicates = near_duplicates = contaminated = short = repetitive = low_quality = source_dominated = 0

    # Pass 2 — apply the gates in legacy order, seeking row by row.
    with input_path.open("rb") as handle:
        for index in order:
            handle.seek(offsets[index])
            row = json.loads(handle.readline().decode("utf-8", errors="replace"))
            response = str(row.get("ideal_response") or "").strip()
            if len(response) < config.minimum_response_chars:
                short += 1
                continue
            normalized = _normalize(response)
            if holdout and holdout in normalized:
                contaminated += 1
                continue
            if _line_diversity(response) < 0.45:
                repetitive += 1
                continue
            quality_score = float(
                row.get("quality_score") or score_source_unit(response, row.get("features", []))
            )
            if quality_score < config.minimum_quality_score:
                low_quality += 1
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
            if guard.is_duplicate(response):
                near_duplicates += 1
                continue
            exact.add(digest)
            source_counts[source] += 1
            origin_counts[origin] += 1
            accepted.append({**row, "quality_score": quality_score, "row_sha256": digest})

    quality_rows = len(accepted)
    maximum_rows = max(24, config.maximum_rows)
    if len(accepted) > maximum_rows:
        accepted = _select_diverse_rows(accepted, maximum_rows)
    capped_rows = quality_rows - len(accepted)
    output_path = root / "dataset-filtered.jsonl"
    _write_jsonl(output_path, accepted)
    dataset_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    audit = DatasetAudit(
        input_rows=input_rows,
        quality_rows=quality_rows,
        accepted_rows=len(accepted),
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        contaminated_rows=contaminated,
        short_rows=short,
        repetitive_rows=repetitive,
        low_quality_rows=low_quality,
        source_dominated_rows=source_dominated,
        capped_rows=capped_rows,
        source_count=len({str(row.get("source_url") or "") for row in accepted}),
        origin_count=len({str(row.get("source_origin") or _source_origin(str(row.get("source_url") or ""))) for row in accepted}),
        mean_quality_score=round(
            sum(float(row.get("quality_score") or 0.0) for row in accepted) / max(1, len(accepted)), 4
        ),
        minimum_quality_score=round(
            min((float(row.get("quality_score") or 0.0) for row in accepted), default=0.0), 4
        ),
        dataset_sha256=dataset_hash,
    )
    atomic_write_json(root / "dataset-audit.json", audit.public())
    return {**audit.public(), "workspace_id": workspace_id, "path": str(output_path)}


def load_filtered_dataset(workspace_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(dataset_workspace(workspace_id) / "dataset-filtered.jsonl")


def load_source_bundle(workspace_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(dataset_workspace(workspace_id) / "sources.jsonl")


def curate_dataset(
    workspace_id: str,
    *,
    task: str,
    artifact_kind: str,
    requested_features: list[str],
    priority_features: list[str] | None = None,
    assembled_examples: int = 144,
    expanded_examples: int = 192,
    maximum_rows: int = 80,
    chunk_chars: int = 2_400,
    minimum_response_chars: int = 1_000,
    minimum_quality_score: float = 0.5,
    config: CurationConfig | None = None,
) -> dict[str, Any]:
    """Run the deterministic assemble/expand/filter/audit pipeline in one tool call.

    ``config=None`` preserves the legacy session-scale behavior. Passing a
    ``CurationConfig`` (e.g. ``FACTORY_CURATION``) makes the filtering stage
    use the config's caps and the streaming sub-quadratic path; ``maximum_rows``
    then comes from the config as well.
    """
    started = datetime.now(UTC)
    assembly = assemble_dataset(
        workspace_id,
        task=task,
        artifact_kind=artifact_kind,
        requested_features=requested_features,
        target_examples=assembled_examples,
        chunk_chars=chunk_chars,
        priority_features=priority_features,
    )
    expansion = expand_dataset(workspace_id, target_examples=expanded_examples)
    if config is not None:
        filtering = filter_dataset(
            workspace_id,
            holdout_task=task,
            config=config,
        )
    else:
        filtering = filter_dataset(
            workspace_id,
            holdout_task=task,
            minimum_response_chars=minimum_response_chars,
            maximum_rows=maximum_rows,
            minimum_quality_score=minimum_quality_score,
        )
    # Workspace paths are useful at the filter API boundary but are not part
    # of the deterministic quality audit. Keeping them out of the curation
    # manifest makes identical inputs compare equal across workspaces.
    filtering_path = filtering.get("path")
    filtering = {
        key: value for key, value in filtering.items() if key not in {"workspace_id", "path"}
    }
    coverage = audit_feature_coverage(load_filtered_dataset(workspace_id), requested_features)
    result = {
        "version": 1,
        "workspace_id": workspace_id,
        "assembly": assembly,
        "expansion": expansion,
        "filtering": filtering,
        "filtered_dataset_path": filtering_path,
        "feature_coverage": coverage,
        "elapsed_ms": round((datetime.now(UTC) - started).total_seconds() * 1_000),
    }
    atomic_write_json(dataset_workspace(workspace_id) / "curation-manifest.json", result)
    return result


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


def score_source_unit(text: str, features: Iterable[str] = ()) -> float:
    """Score reusable source mechanically; no model tokens are spent on curation."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    score = 0.15
    score += 0.22 if _looks_like_code(stripped) else 0.0
    score += 0.14 if 600 <= len(stripped) <= 6_000 else 0.07 if len(stripped) >= 240 else 0.0
    score += 0.12 * min(1.0, _line_diversity(stripped) / 0.8)
    score += 0.15 * min(1.0, len(set(str(item) for item in features)) / 3)
    score += 0.12 if re.search(r"\b(?:function|class|const|let|def|import|export)\b", stripped) else 0.0
    score += 0.1 if re.search(r"(?:[;}\]])\s*$", stripped) else 0.0
    longest_line = max((len(line) for line in stripped.splitlines()), default=0)
    if longest_line > 1_200:
        score -= 0.18
    if re.search(r"\b(?:deprecated|legacy|polyfill|vendor|minified)\b", stripped, re.I):
        score -= 0.08
    return round(max(0.0, min(1.0, score)), 4)


def _row_origin(row: dict[str, Any]) -> str:
    return str(
        row.get("source_origin")
        or _source_origin(str(row.get("source_url") or ""))
        or row.get("source_hash")
        or "unknown"
    )


def _select_single_feature_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select one-feature rows with exact greedy ordering in near-linear time.

    The common factory path emits one capability per row. Grouping by
    capability and source origin means a count update only invalidates the
    affected groups instead of rescanning every remaining row.
    """
    import heapq

    feature_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    by_feature: dict[str, set[tuple[str, str]]] = {}
    by_origin: dict[str, set[tuple[str, str]]] = {}

    for index, row in enumerate(rows):
        feature = str((row.get("features") or [""])[0])
        origin = _row_origin(row)
        key = (feature, origin)
        group = groups.setdefault(key, {"rows": [], "next": 0, "version": 0})
        group["rows"].append((index, row))
        by_feature.setdefault(feature, set()).add(key)
        by_origin.setdefault(origin, set()).add(key)

    for group in groups.values():
        group["rows"].sort(
            key=lambda item: (
                item[1].get("view") == "implementation",
                float(item[1].get("quality_score") or 0.0),
                -item[0],
            ),
            reverse=True,
        )

    def candidate(key: tuple[str, str]):
        feature, origin = key
        group = groups[key]
        pointer = group["next"]
        if pointer >= len(group["rows"]):
            return None
        index, row = group["rows"][pointer]
        return (
            -1.0 / (1 + feature_counts[feature]),
            -1.0 / (1 + origin_counts[origin]),
            0 if row.get("view") == "implementation" else 1,
            -float(row.get("quality_score") or 0.0),
            index,
        )

    def push(key: tuple[str, str], heap: list[tuple[Any, ...]]) -> None:
        current = candidate(key)
        if current is not None:
            feature, origin = key
            heapq.heappush(
                heap,
                (*current, groups[key]["version"], feature, origin),
            )

    heap: list[tuple[Any, ...]] = []
    for key in groups:
        push(key, heap)

    selected: list[tuple[int, dict[str, Any]]] = []
    while heap and len(selected) < limit:
        *priority, version, feature, origin = heapq.heappop(heap)
        key = (feature, origin)
        group = groups[key]
        current = candidate(key)
        if version != group["version"] or current is None or tuple(priority) != current:
            continue

        index, row = group["rows"][group["next"]]
        group["next"] += 1
        selected.append((index, row))
        feature_counts[feature] += 1
        origin_counts[origin] += 1

        affected = by_feature[feature] | by_origin[origin]
        for affected_key in affected:
            groups[affected_key]["version"] += 1
            push(affected_key, heap)

    return [row for _, row in sorted(selected)]


def _select_diverse_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Greedily retain rare capabilities and underrepresented source origins."""
    import heapq

    if limit <= 0:
        return []
    if limit >= len(rows):
        return list(rows)
    if all(len(row.get("features") or []) == 1 for row in rows):
        return _select_single_feature_rows(rows, limit)

    feature_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()

    def priority(index: int) -> tuple[float, float, int, float, int]:
        row = rows[index]
        rarity = sum(
            1.0 / (1 + feature_counts[str(feature)])
            for feature in row.get("features", [])
        )
        origin_rarity = 1.0 / (1 + origin_counts[_row_origin(row)])
        return (
            -rarity,
            -origin_rarity,
            0 if row.get("view") == "implementation" else 1,
            -float(row.get("quality_score") or 0.0),
            index,
        )

    heap = [(*priority(index), index) for index in range(len(rows))]
    heapq.heapify(heap)
    selected: list[tuple[int, dict[str, Any]]] = []
    while heap and len(selected) < limit:
        *key, index = heapq.heappop(heap)
        current = priority(index)
        if tuple(key) != current:
            heapq.heappush(heap, (*current, index))
            continue
        row = rows[index]
        origin_counts[_row_origin(row)] += 1
        feature_counts.update(str(feature) for feature in row.get("features", []))
        selected.append((index, row))
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
