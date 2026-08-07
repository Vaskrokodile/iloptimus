"""FastAPI server — serves the API and the built frontend."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
    HTMLResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core import (
    detect_hardware,
    get_all_models,
    get_model,
    check_compatibility,
    get_all_tasksets,
    get_taskset,
    create_run,
    get_run,
    get_all_runs,
    run_pipeline,
    _stream_events,
    RunConfig,
)

class CreateRunRequest(BaseModel):
    model_id: str
    taskset_id: str
    backend: Optional[str] = None
    precision: Optional[str] = None
    sft_iters: int = 100
    sft_lr: float = 1e-4
    grpo_iters: int = 50
    grpo_group_size: int = 4
    grpo_lr: float = 1e-5
    grpo_temperature: float = 0.6
    max_seq_length: int = 768
    benchmark_tasks: int = 12
    rollouts_per_example: int = 4


# Cache hardware detection
_hw_cache = None


def _get_hardware():
    global _hw_cache
    if _hw_cache is None:
        _hw_cache = detect_hardware()
    return _hw_cache


def create_app() -> FastAPI:
    app = FastAPI(title="IL Optimus", version="0.1.0")

    # ---- API routes ----

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/hardware")
    async def hardware():
        hw = _get_hardware()
        return {
            "cpu_name": hw.cpu_name,
            "cpu_cores": hw.cpu_cores,
            "ram_gb": hw.ram_gb,
            "os": hw.os,
            "arch": hw.arch,
            "gpu": {
                "name": hw.gpu.name,
                "vram_gb": hw.gpu.vram_gb,
                "type": hw.gpu.type,
            },
            "python_version": hw.python_version,
            "mlx_available": hw.mlx_available,
            "vllm_available": hw.vllm_available,
            "torch_available": hw.torch_available,
            "recommended_backend": hw.recommended_backend,
            "total_memory_gb": hw.total_memory_gb,
            "labels": hw.labels,
        }

    @app.get("/api/models")
    async def models():
        hw = _get_hardware()
        result = []
        for m in get_all_models():
            compat = check_compatibility(m, hw)
            result.append({
                "id": m.id,
                "name": m.name,
                "huggingface_id": m.huggingface_id,
                "params_b": m.params_b,
                "fp16_gb": m.fp16_gb,
                "int8_gb": m.int8_gb,
                "int4_gb": m.int4_gb,
                "family": m.family,
                "context_length": m.context_length,
                "backends": m.backends,
                "description": m.description,
                "tags": m.tags,
                "compatibility": {
                    "status": compat.status,
                    "best_precision": compat.best_precision,
                    "best_precision_gb": compat.best_precision_gb,
                    "reason": compat.reason,
                    "score": compat.score,
                },
            })
        return result

    @app.get("/api/models/{model_id}")
    async def model_detail(model_id: str):
        hw = _get_hardware()
        m = get_model(model_id)
        if not m:
            raise HTTPException(404, "Model not found")
        compat = check_compatibility(m, hw)
        return {
            "id": m.id,
            "name": m.name,
            "huggingface_id": m.huggingface_id,
            "params_b": m.params_b,
            "fp16_gb": m.fp16_gb,
            "fp32_gb": m.fp32_gb,
            "int8_gb": m.int8_gb,
            "int4_gb": m.int4_gb,
            "family": m.family,
            "context_length": m.context_length,
            "backends": m.backends,
            "description": m.description,
            "tags": m.tags,
            "compatibility": {
                "status": compat.status,
                "best_precision": compat.best_precision,
                "best_precision_gb": compat.best_precision_gb,
                "reason": compat.reason,
                "score": compat.score,
            },
        }

    @app.get("/api/tasksets")
    async def tasksets():
        return [
            {
                "id": t.id,
                "name": t.name,
                "package_name": t.package_name,
                "domain": t.domain,
                "description": t.description,
                "num_tasks": t.num_tasks,
                "needs_sandbox": t.needs_sandbox,
                "tags": t.tags,
                "eval_config": t.eval_config,
            }
            for t in get_all_tasksets()
        ]

    @app.get("/api/tasksets/{taskset_id}")
    async def taskset_detail(taskset_id: str):
        t = get_taskset(taskset_id)
        if not t:
            raise HTTPException(404, "Taskset not found")
        return {
            "id": t.id,
            "name": t.name,
            "package_name": t.package_name,
            "domain": t.domain,
            "description": t.description,
            "num_tasks": t.num_tasks,
            "needs_sandbox": t.needs_sandbox,
            "tags": t.tags,
            "eval_config": t.eval_config,
        }

    # ---- Run management ----

    @app.post("/api/runs")
    async def create_run_endpoint(req: CreateRunRequest):
        hw = _get_hardware()
        model = get_model(req.model_id)
        if not model:
            raise HTTPException(400, f"Unknown model: {req.model_id}")
        taskset = get_taskset(req.taskset_id)
        if not taskset:
            raise HTTPException(400, f"Unknown taskset: {req.taskset_id}")

        backend = req.backend or hw.recommended_backend
        compat = check_compatibility(model, hw)
        precision = req.precision or compat.best_precision

        config = RunConfig(
            model_id=req.model_id,
            taskset_id=req.taskset_id,
            backend=backend,
            precision=precision,
            sft_iters=req.sft_iters,
            sft_lr=req.sft_lr,
            grpo_iters=req.grpo_iters,
            grpo_group_size=req.grpo_group_size,
            grpo_lr=req.grpo_lr,
            grpo_temperature=req.grpo_temperature,
            max_seq_length=req.max_seq_length,
            benchmark_tasks=req.benchmark_tasks,
            rollouts_per_example=req.rollouts_per_example,
        )
        state = create_run(config)

        # Launch pipeline in background
        asyncio.create_task(run_pipeline(state.id, config, hw))

        return {"id": state.id, "status": state.status, "config": config.__dict__}

    @app.get("/api/runs")
    async def list_runs():
        return [r.to_dict() for r in get_all_runs()]

    @app.get("/api/runs/{run_id}")
    async def get_run_endpoint(run_id: str):
        state = get_run(run_id)
        if not state:
            raise HTTPException(404, "Run not found")
        return state.to_dict()

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(run_id: str):
        state = get_run(run_id)
        if not state:
            raise HTTPException(404, "Run not found")

        async def event_generator():
            # First send all past events
            for event in state.events:
                yield f"data: {json.dumps(event)}\n\n"
            # Then stream new events
            async for event in _stream_events(run_id):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---- Static frontend serving ----

    web_dist = Path(__file__).parent / "web" / "dist"

    if web_dist.exists():
        # Serve static assets
        assets_dir = web_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # SPA fallback: serve index.html for all non-API routes
        @app.get("/", response_class=HTMLResponse)
        async def index():
            return FileResponse(str(web_dist / "index.html"))

        @app.get("/{path:path}", response_class=HTMLResponse)
        async def spa_fallback(path: str):
            # Don't intercept API routes
            if path.startswith("api/"):
                raise HTTPException(404)
            # Try to serve a static file
            file_path = web_dist / path
            if file_path.is_file():
                return FileResponse(str(file_path))
            # Fallback to index.html for SPA routing
            return FileResponse(str(web_dist / "index.html"))
    else:
        @app.get("/", response_class=HTMLResponse)
        async def no_frontend():
            return HTMLResponse(
                "<html><body><h1>IL Optimus</h1>"
                "<p>Frontend not built. Run <code>npm run build</code> in the web/ directory.</p>"
                "<p>API is available at <a href='/api/health'>/api/health</a></p>"
                "</body></html>"
            )

    return app


# Module-level app instance for uvicorn
app = create_app()
