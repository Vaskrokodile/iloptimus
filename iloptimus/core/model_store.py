"""Model download registry backed by Hugging Face snapshots."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, try_to_load_from_cache

from .hardware import HardwareInfo
from .models import ModelInfo, get_model
from .storage import atomic_write_json, directory_size, models_dir


@dataclass
class DownloadState:
    model_id: str
    precision: str
    repository: str
    status: str = "not-downloaded"
    path: str = ""
    bytes_downloaded: int = 0
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["size_gb"] = round(self.bytes_downloaded / 1024**3, 2)
        return result


_downloads: dict[str, DownloadState] = {}
_download_lock = threading.Lock()


def repository_for(model: ModelInfo, precision: str, backend: str) -> str:
    if backend == "mlx" and precision in {"int4", "int8"}:
        bits = "4" if precision == "int4" else "8"
        return f"mlx-community/{model.huggingface_id.split('/')[-1]}-{bits}bit"
    # vLLM / CUDA backend: download the base HuggingFace checkpoint. The
    # VLLMBackend applies bitsandbytes NF4 4-bit quantization (or 8-bit) at load
    # time via transformers, so we do not fetch a separate AWQ/GPTQ repo — those
    # quant schemes are incompatible with PEFT QLoRA training. fp16/int8/int4
    # all resolve to the same base repo; the precision is realized at load.
    return model.huggingface_id


def _local_path(model_id: str, precision: str) -> Path:
    return models_dir() / model_id / precision


def _cached_snapshot(repository: str) -> Optional[Path]:
    cached_config = try_to_load_from_cache(repository, "config.json")
    if isinstance(cached_config, str):
        path = Path(cached_config).parent
        if path.exists():
            return path
    return None


def resolve_model_source(model_id: str, precision: str, backend: str) -> Optional[str]:
    model = get_model(model_id)
    if not model:
        return None
    local = _local_path(model_id, precision)
    if (local / ".iloptimus-model.json").exists() and (local / "config.json").exists():
        return str(local)
    cached = _cached_snapshot(repository_for(model, precision, backend))
    return str(cached) if cached else None


def model_status(model_id: str, precision: str, backend: str) -> dict:
    with _download_lock:
        active = _downloads.get(model_id)
        if active and active.status in {"queued", "downloading", "failed"}:
            active.bytes_downloaded = directory_size(Path(active.path)) if active.path else 0
            return active.to_dict()
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Unknown model: {model_id}")
    source = resolve_model_source(model_id, precision, backend)
    repository = repository_for(model, precision, backend)
    if source:
        size = directory_size(Path(source))
        return DownloadState(
            model_id=model_id,
            precision=precision,
            repository=repository,
            status="downloaded",
            path=source,
            bytes_downloaded=size,
        ).to_dict()
    return DownloadState(model_id=model_id, precision=precision, repository=repository).to_dict()


def download_model(model_id: str, precision: str, backend: str) -> DownloadState:
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Unknown model: {model_id}")
    source = resolve_model_source(model_id, precision, backend)
    repository = repository_for(model, precision, backend)
    if source:
        return DownloadState(
            model_id=model_id,
            precision=precision,
            repository=repository,
            status="downloaded",
            path=source,
            bytes_downloaded=directory_size(Path(source)),
        )

    target = _local_path(model_id, precision)
    state = DownloadState(
        model_id=model_id,
        precision=precision,
        repository=repository,
        status="downloading",
        path=str(target),
        started_at=time.time(),
    )
    with _download_lock:
        existing = _downloads.get(model_id)
        if existing and existing.status == "downloading":
            return existing
        _downloads[model_id] = state

    try:
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repository, local_dir=target)
        if not (target / "config.json").exists():
            raise RuntimeError("The downloaded repository does not contain a model config")
        state.status = "downloaded"
        state.completed_at = time.time()
        state.bytes_downloaded = directory_size(target)
        atomic_write_json(
            target / ".iloptimus-model.json",
            {
                "model_id": model_id,
                "repository": repository,
                "precision": precision,
                "backend": backend,
                "completed_at": state.completed_at,
            },
        )
    except Exception as error:
        state.status = "failed"
        state.error = str(error)
        state.bytes_downloaded = directory_size(target)
    return state


def compatible_precision(model: ModelInfo, hardware: HardwareInfo) -> str:
    return "int4"
