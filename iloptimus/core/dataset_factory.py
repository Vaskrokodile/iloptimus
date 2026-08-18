"""Dataset factory: bulk acquisition + declarative, resumable dataset jobs.

A ``DatasetJobSpec`` declares *what* to build — capability targets, inline
and/or URL candidate sources, curation caps, budget — and the runner executes
the same staged pipeline for every job:

    acquire  → bulk-add candidate sources to the persistent corpus
               (many sources per invocation, content-addressed dedup)
    export   → materialize corpus sources into the job's dataset workspace
    curate   → shipped curate_dataset with FACTORY_CURATION (streaming,
               MinHash/LSH dedup, all legacy quality gates active)
    finalize → record the audit + capability coverage in the job state

Jobs are checkpointed to ``app_home()/dataset-jobs/<id>/job.json`` after each
stage, so ``run`` is resumable: re-running a job skips completed stages and
re-runs are idempotent because acquisition is content-addressed and curation
is deterministic.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset_store import CorpusStore
from .dataset_tools import (
    FACTORY_CURATION,
    CurationConfig,
    audit_feature_coverage,
    create_dataset_workspace,
    curate_dataset,
    dataset_workspace,
    load_filtered_dataset,
)
from .storage import app_home, atomic_write_json

HARVEST_CONCURRENCY = 8
JOB_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def jobs_root() -> Path:
    root = app_home() / "dataset-jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Spec + state
# ---------------------------------------------------------------------------


@dataclass
class DatasetJobSpec:
    """Declarative description of one factory dataset build."""

    task: str
    artifact_kind: str
    requested_features: list[str] = field(default_factory=list)
    # Inline sources: {"title", "url", "text", "license", "kind"}. This is the
    # bulk path — dozens to thousands of sources in a single call.
    sources: list[dict[str, Any]] = field(default_factory=list)
    # URL candidates harvested via the shipped web_fetch path.
    urls: list[str] = field(default_factory=list)
    # Corpus kind assigned to URL-harvested sources.
    source_kind: str = "web-documentation"
    workspace_id: str | None = None
    maximum_rows: int = 20_000
    assembled_examples: int = 40_000
    expanded_examples: int = 60_000
    chunk_chars: int = 2_400
    minimum_response_chars: int = 220
    # Factory curation by default; callers may pin LEGACY_CURATION.
    curation: CurationConfig = FACTORY_CURATION
    budget_seconds: float = 3_600.0

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["curation"] = asdict(self.curation)
        return payload


@dataclass
class DatasetJobState:
    job_id: str
    spec: dict[str, Any]
    status: str = "pending"  # pending | acquiring | exporting | curating | done | failed
    checkpoints: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _job_dir(job_id: str, *, create: bool = True) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise ValueError("Invalid dataset job id")
    directory = jobs_root() / job_id
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_state(state: DatasetJobState) -> None:
    state.updated_at = datetime.now(UTC).isoformat()
    atomic_write_json(_job_dir(state.job_id) / "job.json", state.public())


def _load_state(job_id: str) -> DatasetJobState | None:
    try:
        path = _job_dir(job_id, create=False) / "job.json"
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    curation_payload = payload.get("spec", {}).get("curation") or {}
    spec_payload = dict(payload.get("spec") or {})
    spec_payload["curation"] = CurationConfig(**curation_payload) if curation_payload else FACTORY_CURATION
    return DatasetJobState(
        job_id=payload.get("job_id", job_id),
        spec=spec_payload,
        status=payload.get("status", "pending"),
        checkpoints=payload.get("checkpoints", {}),
        result=payload.get("result", {}),
        error=payload.get("error"),
        created_at=payload.get("created_at", ""),
        updated_at=payload.get("updated_at", ""),
    )


# ---------------------------------------------------------------------------
# Bulk acquisition
# ---------------------------------------------------------------------------


async def harvest_urls(urls: list[str], *, kind: str = "web-documentation") -> dict[str, Any]:
    """Fetch many URL candidates in one bounded-concurrency pass.

    Failures on individual URLs are recorded, not raised — a bulk harvest
    keeps whatever succeeded. Returns source dicts ready for the corpus.
    """
    from .tools import web_fetch

    semaphore = asyncio.Semaphore(HARVEST_CONCURRENCY)

    async def fetch_one(url: str) -> dict[str, Any]:
        async with semaphore:
            try:
                fetched = await web_fetch(url)
            except Exception as error:  # noqa: BLE001 - bulk harvest skips failures
                return {"url": url, "error": str(error)}
            return {"url": url, "fetched": fetched}

    outcomes = await asyncio.gather(*(fetch_one(url) for url in urls))
    sources: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for outcome in outcomes:
        if "error" in outcome:
            failed.append({"url": outcome["url"], "error": outcome["error"]})
            continue
        fetched = outcome["fetched"]
        sources.append(
            {
                "title": fetched.get("url", outcome["url"]),
                "url": fetched.get("url", outcome["url"]),
                "text": fetched.get("text", ""),
                "kind": kind,
                "license": "web-content",
            }
        )
    return {"sources": sources, "failed": failed, "fetched": len(sources)}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_STAGES = ("acquire", "export", "curate", "finalize")


class DatasetJobRunner:
    """Executes dataset jobs with per-stage checkpoints and resume support."""

    def __init__(self, store: CorpusStore | None = None) -> None:
        self.store = store or CorpusStore()

    # ------------------------------------------------------------- job CRUD

    def create(self, spec: DatasetJobSpec) -> DatasetJobState:
        job_id = uuid.uuid4().hex[:12]
        workspace_id = spec.workspace_id or f"factory-{job_id}"
        spec.workspace_id = workspace_id
        state = DatasetJobState(
            job_id=job_id,
            spec=spec.public(),
            status="pending",
            created_at=datetime.now(UTC).isoformat(),
        )
        _save_state(state)
        return state

    def get(self, job_id: str) -> DatasetJobState | None:
        return _load_state(job_id)

    def list(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in sorted(jobs_root().glob("*/job.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            jobs.append(
                {
                    "job_id": payload.get("job_id"),
                    "status": payload.get("status"),
                    "task": (payload.get("spec") or {}).get("task"),
                    "workspace_id": (payload.get("spec") or {}).get("workspace_id"),
                    "accepted_rows": ((payload.get("result") or {}).get("filtering") or {}).get("accepted_rows"),
                    "updated_at": payload.get("updated_at"),
                }
            )
        return jobs

    # ------------------------------------------------------------ execution

    def run(self, job_id: str) -> DatasetJobState:
        """Run (or resume) a job. Completed stages are skipped; re-runs are
        idempotent because acquisition is content-addressed and curation is
        deterministic."""
        state = _load_state(job_id)
        if state is None:
            raise ValueError(f"Unknown dataset job: {job_id}")
        if state.status == "done":
            return state

        spec_payload = dict(state.spec)
        curation_payload = spec_payload.pop("curation", None)
        if isinstance(curation_payload, CurationConfig):
            curation = curation_payload
        elif curation_payload:
            curation = CurationConfig(**curation_payload)
        else:
            curation = FACTORY_CURATION
        spec = DatasetJobSpec(curation=curation, **spec_payload)
        started = time.monotonic()

        try:
            acquire_checkpoint = state.checkpoints.get("acquire")
            if not isinstance(acquire_checkpoint, dict) or "source_hashes" not in acquire_checkpoint:
                # Migrate checkpoints written before acquisition manifests were
                # introduced. Re-acquisition is idempotent and avoids the old
                # unsafe behavior of exporting the entire shared corpus.
                state.status = "acquiring"
                _save_state(state)
                state.checkpoints["acquire"] = self._stage_acquire(spec, started)
                _save_state(state)
            if "export" not in state.checkpoints:
                state.status = "exporting"
                _save_state(state)
                state.checkpoints["export"] = self._stage_export(
                    spec,
                    state.checkpoints["acquire"].get("source_hashes", []),
                )
                _save_state(state)
            if "curate" not in state.checkpoints:
                state.status = "curating"
                _save_state(state)
                state.checkpoints["curate"] = self._stage_curate(spec, started)
                _save_state(state)
            if "finalize" not in state.checkpoints:
                state.result = self._stage_finalize(spec)
                state.checkpoints["finalize"] = {"completed": True}
                state.status = "done"
                state.error = None
                _save_state(state)
        except Exception as error:  # noqa: BLE001 - job state must record failure
            state.status = "failed"
            state.error = str(error)
            _save_state(state)
        return state

    # --------------------------------------------------------------- stages

    def _budget_ok(self, started: float, budget_seconds: float) -> None:
        if time.monotonic() - started > budget_seconds:
            raise TimeoutError("Dataset job budget exhausted")

    def _stage_acquire(self, spec: DatasetJobSpec, started: float) -> dict[str, Any]:
        candidates = list(spec.sources)
        harvest_report: dict[str, Any] = {"fetched": 0, "failed": []}
        if spec.urls:
            self._budget_ok(started, spec.budget_seconds)
            harvest_report = asyncio.run(harvest_urls(spec.urls, kind=spec.source_kind))
            candidates.extend(harvest_report["sources"])
        accounting = self.store.add_sources(candidates)
        return {
            "candidates": len(candidates),
            **accounting,
            "url_harvest": {
                "fetched": harvest_report.get("fetched", 0),
                "failed_count": len(harvest_report.get("failed", [])),
            },
        }

    def _stage_export(
        self,
        spec: DatasetJobSpec,
        source_hashes: list[str],
    ) -> dict[str, Any]:
        create_dataset_workspace(spec.workspace_id)
        return self.store.export_to_workspace(
            spec.workspace_id,
            shas=source_hashes,
        )

    def _stage_curate(self, spec: DatasetJobSpec, started: float) -> dict[str, Any]:
        from dataclasses import replace

        self._budget_ok(started, spec.budget_seconds)
        # The job's row cap overrides the preset's cap; all other gates stay.
        curation = replace(spec.curation, maximum_rows=spec.maximum_rows)
        return curate_dataset(
            spec.workspace_id,
            task=spec.task,
            artifact_kind=spec.artifact_kind,
            requested_features=spec.requested_features,
            assembled_examples=spec.assembled_examples,
            expanded_examples=spec.expanded_examples,
            maximum_rows=spec.maximum_rows,
            chunk_chars=spec.chunk_chars,
            minimum_response_chars=spec.minimum_response_chars,
            config=curation,
        )

    def _stage_finalize(self, spec: DatasetJobSpec) -> dict[str, Any]:
        rows = load_filtered_dataset(spec.workspace_id)
        coverage = audit_feature_coverage(rows, spec.requested_features)
        root = dataset_workspace(spec.workspace_id)
        audit_path = root / "dataset-audit.json"
        audit = (
            json.loads(audit_path.read_text(encoding="utf-8"))
            if audit_path.exists()
            else {}
        )
        return {
            "workspace_id": spec.workspace_id,
            "accepted_rows": len(rows),
            "filtering": audit,
            "feature_coverage": coverage,
            "dataset_path": str(root / "dataset-filtered.jsonl"),
        }
