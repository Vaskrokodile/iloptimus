"""Dataset factory tests: scaled curation internals, persistent corpus,
bulk acquisition, job orchestration, server endpoints, and the factory-scale
curation acceptance test (>=5x the legacy 2,048-row cap with every quality
gate intact and deterministic output).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pytest

from iloptimus.core import dataset_tools
from iloptimus.core.dataset_factory import DatasetJobRunner, DatasetJobSpec, harvest_urls
from iloptimus.core.dataset_store import CorpusStore
from iloptimus.core.dataset_tools import (
    FACTORY_CURATION,
    LEGACY_CURATION,
    CurationConfig,
    _select_diverse_rows,
    _shingles,
    create_dataset_workspace,
    curate_dataset,
    dataset_workspace,
    filter_dataset,
    load_filtered_dataset,
    load_source_bundle,
    save_source_bundle,
)
from iloptimus.core.dedup import (
    BloomFilter,
    ExactJaccardGuard,
    MinHashDuplicateGuard,
    MinHashSignature,
    jaccard,
    shingle_set,
)

FACTORY_SCALE_TARGET = 5 * 2_048  # acceptance criterion: >= 5x the legacy cap


# ---------------------------------------------------------------------------
# Synthetic corpus helpers
# ---------------------------------------------------------------------------


def _code_source(name: str, feature: str, origin_index: int = 0, paragraphs: int = 6) -> dict:
    """~7KB of unique, code-like text that passes _looks_like_code and the
    mechanical quality scorer."""
    lines: list[str] = []
    for p in range(paragraphs):
        lines.append(f"const {feature}_unit{p}_{name} = function (scene{p}) {{")
        for i in range(8):
            lines.append(
                f"  const value{p}_{i} = new {feature.title()}Widget({p}, {i}, '{name}-{p}-{i}');"
            )
            lines.append(
                f"  scene{p}.add(value{p}_{i}); // {feature} line {p}.{i} token-{name}-{p}-{i}"
            )
        lines.append("};")
    return {
        "title": name,
        "url": f"https://origin{origin_index % 40}.example/{name}",
        "text": "\n".join(lines),
        "license": "MIT",
        "kind": "repository-code",
    }


FEATURE_POOL = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _synthetic_corpus(count: int) -> list[dict]:
    return [
        _code_source(f"src{index}", FEATURE_POOL[index % len(FEATURE_POOL)], origin_index=index)
        for index in range(count)
    ]


def _row(
    index: int,
    response: str,
    *,
    features: list[str] | None = None,
    source: str = "https://origin0.example/repo",
    quality: float = 0.8,
    view: str = "implementation",
    role: str = "capability",
) -> dict:
    return {
        "prompt": f"Write unit {index}",
        "ideal_response": response,
        "source_url": source,
        "features": features or ["alpha"],
        "quality_score": quality,
        "view": view,
        "curriculum_role": role,
    }


def _write_expanded_rows(workspace_id: str, rows: list[dict]) -> None:
    path = dataset_workspace(workspace_id) / "dataset-expanded.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _unique_text(index: int, words: int = 90) -> str:
    return " ".join(f"token{index}x{w} const let value{index}_{w};" for w in range(words))


# ---------------------------------------------------------------------------
# Dedup internals
# ---------------------------------------------------------------------------


def test_shingle_set_matches_legacy_shingles():
    text = "const alpha = new BetaWidget(1, 2);\nscene.add(alpha); // gamma delta epsilon"
    assert shingle_set(text) == _shingles(text)


def test_minhash_guard_detects_near_duplicate_and_passes_distinct():
    base_words = [f"word{i}" for i in range(200)]
    text_a = " ".join(base_words)
    # Change a single word -> true Jaccard ~0.95, well above the threshold.
    mutated = list(base_words)
    mutated[100] = "changed"
    text_b = " ".join(mutated)
    text_c = " ".join(f"entirely{i} different{i} content{i} here{i}" for i in range(200))

    true_sim = jaccard(shingle_set(text_a), shingle_set(text_b))
    assert true_sim >= 0.84

    guard = MinHashDuplicateGuard(threshold=0.84)
    assert guard.is_duplicate(text_a) is False  # first occurrence accepted
    assert guard.is_duplicate(text_b) is True  # near-duplicate rejected
    assert guard.is_duplicate(text_c) is False  # distinct text accepted
    assert guard.is_duplicate("") is True  # empty never passes


def test_minhash_estimates_true_jaccard():
    signer = MinHashSignature(num_perm=128)
    text_a = " ".join(f"shared{i} token{i}" for i in range(200))
    text_b = " ".join(f"shared{i} token{i}" for i in range(160)) + " " + " ".join(
        f"extra{i} filler{i}" for i in range(40)
    )
    estimated = MinHashSignature.estimated_jaccard(
        signer.signature(text_a), signer.signature(text_b)
    )
    actual = jaccard(shingle_set(text_a), shingle_set(text_b))
    assert abs(estimated - actual) < 0.15


def test_bloom_membership_and_persistence(tmp_path: Path):
    bloom = BloomFilter(capacity=10_000, error_rate=1e-6)
    for i in range(500):
        bloom.add(f"item-{i}")
    assert "item-42" in bloom
    assert "never-added" not in bloom
    path = tmp_path / "membership.bloom"
    bloom.save(path)
    reloaded = BloomFilter.load(path)
    assert "item-42" in reloaded
    assert "never-added" not in reloaded
    assert len(reloaded) == 500


def test_minhash_guard_handles_tens_of_thousands_of_rows(monkeypatch, tmp_path: Path):
    """Sub-quadratic scale proof: 20k unique rows accepted, planted
    near-duplicates rejected, in bounded wall time."""
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    guard = MinHashDuplicateGuard(threshold=0.84)
    started = time.monotonic()
    unique_rows = 20_000
    for i in range(unique_rows):
        assert guard.is_duplicate(_unique_text(i)) is False
    # Plant near-duplicates of a few earlier rows (a single word changed,
    # keeping true Jaccard far above the 0.84 threshold).
    for victim in (0, 7_000, 19_999):
        words = _unique_text(victim).split()
        words[len(words) // 2] = f"mutated{victim}"
        assert guard.is_duplicate(" ".join(words)) is True
    elapsed = time.monotonic() - started
    assert len(guard) == unique_rows
    assert elapsed < 120, f"dedup of {unique_rows} rows took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Heap-based diverse selection
# ---------------------------------------------------------------------------


def _oracle_greedy(rows: list[dict], limit: int) -> list[dict]:
    """Reference implementation of the original scan-for-max greedy loop."""
    remaining = list(enumerate(rows))
    selected: list[tuple[int, dict]] = []
    feature_counts: Counter = Counter()
    origin_counts: Counter = Counter()

    def origin(row: dict) -> str:
        return str(
            row.get("source_origin")
            or dataset_tools._source_origin(str(row.get("source_url") or ""))
            or row.get("source_hash")
            or "unknown"
        )

    while remaining and len(selected) < limit:
        best = max(
            range(len(remaining)),
            key=lambda pos: (
                sum(1.0 / (1 + feature_counts[str(f)]) for f in remaining[pos][1].get("features", [])),
                1.0 / (1 + origin_counts[origin(remaining[pos][1])]),
                remaining[pos][1].get("view") == "implementation",
                float(remaining[pos][1].get("quality_score") or 0.0),
                -remaining[pos][0],
            ),
        )
        idx, row = remaining.pop(best)
        selected.append((idx, row))
        origin_counts[origin(row)] += 1
        feature_counts.update(str(f) for f in row.get("features", []))
    return [row for _, row in sorted(selected)]


def test_heap_selection_matches_greedy_oracle():
    import random

    rng = random.Random(7)
    rows = []
    for i in range(140):
        rows.append(
            {
                "id": i,
                "features": rng.sample(FEATURE_POOL, k=rng.randint(1, 3)),
                "source_url": f"https://origin{rng.randint(0, 9)}.example/repo{rng.randint(0, 30)}",
                "quality_score": round(rng.random(), 3),
                "view": rng.choice(["implementation", "source-unit"]),
            }
        )
    for limit in (1, 17, 60, 140, 500):
        assert _select_diverse_rows(rows, limit) == _oracle_greedy(rows, limit)


def test_single_feature_selection_matches_greedy_oracle():
    rows = [
        {
            "id": i,
            "features": [FEATURE_POOL[i % len(FEATURE_POOL)]],
            "source_url": f"https://origin{i % 5}.example/repo{i % 11}",
            "quality_score": round((i % 29) / 29, 3),
            "view": "implementation" if i % 3 else "source-unit",
        }
        for i in range(180)
    ]
    for limit in (1, 17, 60, 140, 179):
        assert _select_diverse_rows(rows, limit) == _oracle_greedy(rows, limit)


def test_heap_selection_scales_to_50k_rows():
    rows = [
        {
            "id": i,
            "features": [FEATURE_POOL[i % len(FEATURE_POOL)]],
            "source_url": f"https://origin{i % 40}.example/repo{i % 500}",
            "quality_score": (i % 97) / 100,
            "view": "implementation",
        }
        for i in range(50_000)
    ]
    started = time.monotonic()
    selected = _select_diverse_rows(rows, 20_000)
    elapsed = time.monotonic() - started
    assert len(selected) == 20_000
    assert elapsed < 120, f"heap selection over 50k rows took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Persistent corpus store
# ---------------------------------------------------------------------------


def test_corpus_store_dedupes_and_persists(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    sources = _synthetic_corpus(3)
    store = CorpusStore()
    first = store.add_sources(sources)
    assert first["added"] == 3 and first["duplicates"] == 0
    # Same content again — global content-addressed dedup.
    again = store.add_sources(sources + [{"title": "tiny", "url": "", "text": "short", "kind": "repository-code"}])
    assert again["added"] == 0
    assert again["duplicates"] == 3
    assert again["rejected_short"] == 1
    assert again["total"] == 3
    # A fresh instance (fresh SQLite handle + persisted bloom) sees the corpus.
    reopened = CorpusStore()
    assert reopened.source_count() == 3
    sha = first["added_hashes"][0]
    assert reopened.has_source(sha)
    full = reopened.get_source(sha)
    assert full is not None and full["text"] == sources[0]["text"].strip()
    assert (tmp_path / "corpus" / "membership.bloom").exists()


def test_corpus_export_to_workspace_is_idempotent(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    store = CorpusStore()
    store.add_sources(_synthetic_corpus(4))
    create_dataset_workspace("export-check")
    first = store.export_to_workspace("export-check")
    assert first["workspace_sources"] == 4
    second = store.export_to_workspace("export-check")
    assert second["workspace_sources"] == 4  # no growth on re-export
    bundle = dataset_workspace("export-check") / "sources.jsonl"
    assert bundle.exists()
    lines = [line for line in bundle.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert len(lines) == 4


# ---------------------------------------------------------------------------
# Streaming filter: gates and legacy equivalence
# ---------------------------------------------------------------------------


def test_streaming_filter_enforces_every_gate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    create_dataset_workspace("gate-check")

    good = _unique_text(1)
    near = good.split()
    near[len(near) // 2] = "nearword"  # one-word change keeps Jaccard ~0.97
    rows = [
        _row(0, good, features=["alpha"]),
        _row(1, good, features=["alpha"]),  # exact duplicate of row 0
        _row(2, " ".join(near), features=["alpha"]),  # near duplicate of row 0
        _row(3, _unique_text(3)[:180], features=["beta"]),  # too short
        _row(4, _unique_text(4) + " forbidden-holdout-task content", features=["beta"]),
        _row(5, "\n".join(["const same = line();"] * 40), features=["gamma"]),  # repetitive
        _row(6, _unique_text(6), features=["gamma"], quality=0.1),  # low quality
        # source domination: four rows from one source, cap is 3
        _row(7, _unique_text(7), features=["delta"], source="https://solo.example/repo"),
        _row(8, _unique_text(8), features=["delta"], source="https://solo.example/repo"),
        _row(9, _unique_text(9), features=["delta"], source="https://solo.example/repo"),
        _row(10, _unique_text(10), features=["delta"], source="https://solo.example/repo"),
    ]
    _write_expanded_rows("gate-check", rows)
    config = CurationConfig(maximum_rows=1_000, max_per_source=3, minhash_dedup=True)
    result = filter_dataset("gate-check", holdout_task="forbidden-holdout-task", config=config)

    # Accepted: rows 0, 7, 8, 9. Every other row trips exactly one gate.
    assert result["accepted_rows"] == 4
    assert result["exact_duplicates"] == 1
    assert result["near_duplicates"] == 1
    assert result["short_rows"] == 1
    assert result["contaminated_rows"] == 1
    assert result["repetitive_rows"] == 1
    assert result["low_quality_rows"] == 1
    assert result["source_dominated_rows"] == 1
    accepted = load_filtered_dataset("gate-check")
    assert len(accepted) == 4
    # The audit file carries the same fields for API consumers.
    audit = json.loads((dataset_workspace("gate-check") / "dataset-audit.json").read_text())
    for key in (
        "input_rows",
        "exact_duplicates",
        "near_duplicates",
        "contaminated_rows",
        "source_count",
        "origin_count",
        "mean_quality_score",
        "dataset_sha256",
    ):
        assert key in audit


def test_streaming_filter_matches_legacy_exactly(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    create_dataset_workspace("equivalence")
    save_source_bundle("equivalence", _synthetic_corpus(8))
    legacy = curate_dataset(
        "equivalence",
        task="equivalence holdout",
        artifact_kind="code",
        requested_features=FEATURE_POOL,
        assembled_examples=512,
        expanded_examples=768,
        maximum_rows=2_048,
        minimum_response_chars=220,
    )
    legacy_rows = load_filtered_dataset("equivalence")

    streaming = filter_dataset(
        "equivalence",
        holdout_task="equivalence holdout",
        config=CurationConfig(
            maximum_rows=2_048,
            minimum_response_chars=220,
            minimum_quality_score=0.5,
            max_per_source=3,
            max_per_origin_fraction=0.2,
            max_per_origin_floor=6,
            minhash_dedup=False,  # exact Jaccard, same math as legacy
        ),
    )
    streaming_rows = load_filtered_dataset("equivalence")

    assert legacy_rows == streaming_rows
    assert legacy["filtering"]["accepted_rows"] == streaming["accepted_rows"]


def test_streaming_filter_processes_30k_rows(monkeypatch, tmp_path: Path):
    """The streaming path + heap cap handle tens of thousands of rows."""
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    create_dataset_workspace("scaled-filter")
    rows = [
        _row(
            i,
            _unique_text(i, words=25),
            features=[FEATURE_POOL[i % len(FEATURE_POOL)]],
            source=f"https://origin{i % 40}.example/repo{i % 900}",
        )
        for i in range(30_000)
    ]
    _write_expanded_rows("scaled-filter", rows)
    started = time.monotonic()
    result = filter_dataset(
        "scaled-filter", holdout_task="unused-holdout", config=FACTORY_CURATION
    )
    elapsed = time.monotonic() - started
    assert result["input_rows"] == 30_000
    assert result["accepted_rows"] >= FACTORY_SCALE_TARGET
    assert elapsed < 300, f"30k-row streaming filter took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Factory-scale curation acceptance test
# ---------------------------------------------------------------------------


def test_factory_curation_exceeds_legacy_cap_deterministically(monkeypatch, tmp_path: Path):
    """Acceptance criterion 1: the shipped curation path produces >=5x the
    legacy 2,048-row cap from a multi-source corpus, with all gates present,
    and does so deterministically across identical runs."""
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    corpus = _synthetic_corpus(2_000)

    results = []
    for workspace_id in ("factory-a", "factory-b"):
        create_dataset_workspace(workspace_id)
        save_source_bundle(workspace_id, corpus)
        started = time.monotonic()
        result = curate_dataset(
            workspace_id,
            task="factory scale holdout probe",
            artifact_kind="code",
            requested_features=FEATURE_POOL,
            assembled_examples=40_000,
            expanded_examples=60_000,
            chunk_chars=2_400,
            config=FACTORY_CURATION,
        )
        elapsed = time.monotonic() - started
        results.append(result)
        filtering = result["filtering"]
        assert filtering["accepted_rows"] >= FACTORY_SCALE_TARGET, (
            f"expected >= {FACTORY_SCALE_TARGET} rows, got {filtering['accepted_rows']}"
        )
        # Every legacy gate still reports in the audit.
        for counter in (
            "exact_duplicates",
            "near_duplicates",
            "contaminated_rows",
            "short_rows",
            "repetitive_rows",
            "low_quality_rows",
            "source_dominated_rows",
            "capped_rows",
        ):
            assert isinstance(filtering[counter], int)
        assert filtering["source_count"] >= 100
        assert filtering["origin_count"] >= 10
        assert filtering["mean_quality_score"] >= 0.5
        assert filtering["dataset_sha256"]
        # Capability coverage audit ran for every requested feature.
        coverage = result["feature_coverage"]
        assert set(coverage["features"]) == set(FEATURE_POOL)
        assert elapsed < 900, f"factory curation took {elapsed:.1f}s"

    # Determinism: two identical runs agree row-for-row in counts and audits.
    assert results[0]["filtering"]["accepted_rows"] == results[1]["filtering"]["accepted_rows"]
    assert results[0]["filtering"] == results[1]["filtering"]
    assert results[0]["assembly"]["rows"] == results[1]["assembly"]["rows"]
    assert results[0]["expansion"]["rows"] == results[1]["expansion"]["rows"]


# ---------------------------------------------------------------------------
# Jobs: bulk acquisition, resume, dedup across jobs
# ---------------------------------------------------------------------------


def test_dataset_job_runs_end_to_end_and_resumes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    runner = DatasetJobRunner()
    spec = DatasetJobSpec(
        task="job holdout probe",
        artifact_kind="code",
        requested_features=["alpha", "beta"],
        sources=_synthetic_corpus(60)[:60],
        maximum_rows=5_000,
        assembled_examples=5_000,
        expanded_examples=8_000,
    )
    state = runner.create(spec)
    finished = runner.run(state.job_id)
    assert finished.status == "done", finished.error
    for stage in ("acquire", "export", "curate", "finalize"):
        assert stage in finished.checkpoints
    assert finished.checkpoints["acquire"]["added"] == 60
    assert finished.result["accepted_rows"] > 0
    assert finished.result["feature_coverage"]["features"]

    # Resume: running again skips stages and returns the same outcome.
    rerun = runner.run(state.job_id)
    assert rerun.status == "done"
    assert rerun.result["accepted_rows"] == finished.result["accepted_rows"]

    listed = runner.list()
    assert any(job["job_id"] == state.job_id for job in listed)
    assert runner.get(state.job_id) is not None
    assert runner.get("nonexistent") is None
    assert runner.get("../outside-job") is None
    assert not (tmp_path / "outside-job").exists()


def test_dataset_job_acquire_dedupes_across_jobs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    runner = DatasetJobRunner()
    corpus = _synthetic_corpus(20)
    first = runner.run(
        runner.create(
            DatasetJobSpec(
                task="dedup holdout one",
                artifact_kind="code",
                requested_features=["alpha"],
                sources=corpus,
                maximum_rows=1_000,
                assembled_examples=1_000,
                expanded_examples=2_000,
            )
        ).job_id
    )
    assert first.status == "done"
    second = runner.run(
        runner.create(
            DatasetJobSpec(
                task="dedup holdout two",
                artifact_kind="code",
                requested_features=["alpha"],
                sources=corpus,
                maximum_rows=1_000,
                assembled_examples=1_000,
                expanded_examples=2_000,
            )
        ).job_id
    )
    assert second.status == "done"
    assert second.checkpoints["acquire"]["added"] == 0
    assert second.checkpoints["acquire"]["duplicates"] == len(corpus)


def test_dataset_jobs_export_only_their_acquired_sources(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    runner = DatasetJobRunner()
    corpus = _synthetic_corpus(4)

    first = runner.run(
        runner.create(
            DatasetJobSpec(
                task="scoped holdout one",
                artifact_kind="code",
                sources=corpus[:3],
                assembled_examples=1_000,
                expanded_examples=2_000,
            )
        ).job_id
    )
    second = runner.run(
        runner.create(
            DatasetJobSpec(
                task="scoped holdout two",
                artifact_kind="code",
                sources=[corpus[0], corpus[3]],
                assembled_examples=1_000,
                expanded_examples=2_000,
            )
        ).job_id
    )

    assert first.status == "done", first.error
    assert second.status == "done", second.error
    assert first.checkpoints["export"]["workspace_sources"] == 3
    assert second.checkpoints["export"]["workspace_sources"] == 2
    assert len(load_source_bundle(first.spec["workspace_id"])) == 3
    assert len(load_source_bundle(second.spec["workspace_id"])) == 2
    assert runner.store.source_count() == 4


def test_harvest_urls_bulk_with_partial_failures(monkeypatch, tmp_path: Path):
    """Bulk acquisition fetches many candidates per invocation, keeps the
    successes when some URLs fail, and can complete a full job end-to-end."""
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))

    def _code_like_text(seed: int) -> str:
        lines = [f"const harvestUnit{seed}_{i} = function (scene{i}) {{" for i in range(30)]
        lines += [
            f"  const value{i} = new HarvestWidget({seed}, {i}); scene{i}.add(value{i}); // unique harvest token {seed}-{i}"
            for i in range(30)
        ]
        lines += ["};"] * 30
        return "\n".join(lines)

    async def fake_web_fetch(url: str):
        if "fail" in url:
            raise ValueError("simulated fetch failure")
        seed = abs(hash(url)) % 100_000
        return {
            "url": url,
            "status": 200,
            "content_type": "text/plain",
            "text": _code_like_text(seed),
            "truncated": False,
        }

    import iloptimus.core.tools as tools_module

    monkeypatch.setattr(tools_module, "web_fetch", fake_web_fetch)

    import asyncio

    urls = [f"https://candidate{i}.example/doc" for i in range(6)] + [
        "https://fail-a.example/doc",
        "https://fail-b.example/doc",
    ]
    report = asyncio.run(harvest_urls(urls))
    assert report["fetched"] == 6
    assert len(report["failed"]) == 2

    runner = DatasetJobRunner()
    state = runner.create(
        DatasetJobSpec(
            task="harvest holdout",
            artifact_kind="code",
            requested_features=[],
            urls=urls,
            source_kind="repository-code",
            maximum_rows=1_000,
            assembled_examples=1_000,
            expanded_examples=2_000,
        )
    )
    finished = runner.run(state.job_id)
    assert finished.status == "done", finished.error
    assert finished.checkpoints["acquire"]["url_harvest"]["fetched"] == 6
    assert finished.checkpoints["acquire"]["url_harvest"]["failed_count"] == 2
    assert finished.checkpoints["acquire"]["added"] == 6
    assert finished.result["accepted_rows"] > 0


# ---------------------------------------------------------------------------
# Server endpoints
# ---------------------------------------------------------------------------


def test_server_dataset_endpoints(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    from starlette.testclient import TestClient

    from iloptimus.server import create_app

    client = TestClient(create_app())

    healthy = client.get("/api/health")
    assert healthy.status_code == 200

    payload = {
        "task": "endpoint holdout probe",
        "artifact_kind": "code",
        "requested_features": ["alpha", "beta"],
        "sources": _synthetic_corpus(40),
        "maximum_rows": 5_000,
        "assembled_examples": 5_000,
        "expanded_examples": 8_000,
    }
    created = client.post("/api/datasets/jobs", json=payload)
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["status"] == "done", body.get("error")
    job_id = body["job_id"]
    workspace_id = body["spec"]["workspace_id"]
    assert body["result"]["accepted_rows"] > 0

    jobs = client.get("/api/datasets/jobs").json()
    assert any(job["job_id"] == job_id for job in jobs["jobs"])

    single = client.get(f"/api/datasets/jobs/{job_id}")
    assert single.status_code == 200
    assert single.json()["status"] == "done"

    missing = client.get("/api/datasets/jobs/does-not-exist")
    assert missing.status_code == 404

    audit = client.get(f"/api/datasets/{workspace_id}/audit")
    assert audit.status_code == 200, audit.text
    audit_body = audit.json()
    assert audit_body["rows"] == body["result"]["accepted_rows"]
    assert audit_body["audit"]["accepted_rows"] == body["result"]["accepted_rows"]
    assert audit_body["audit"]["dataset_sha256"]


def test_legacy_preset_preserves_original_caps():
    """The legacy preset must keep the historical session-scale caps."""
    assert LEGACY_CURATION.maximum_rows == 2_048
    assert LEGACY_CURATION.max_per_source == 3
    assert LEGACY_CURATION.max_per_origin_floor == 6
    assert LEGACY_CURATION.near_duplicate_threshold == 0.84
    assert LEGACY_CURATION.minhash_dedup is False
    # Factory preset lifts the row cap far beyond legacy while keeping gates.
    assert FACTORY_CURATION.maximum_rows >= 50 * LEGACY_CURATION.maximum_rows
    assert FACTORY_CURATION.near_duplicate_threshold == 0.84
