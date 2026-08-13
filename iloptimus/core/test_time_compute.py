"""Global failure-driven test-time-compute contracts, datasets, and verifiers.

This module intentionally knows about capability classes (web artifacts, code,
knowledge), not benchmark prompts. A concrete task is converted into observable
requirements, evaluated before adaptation, kept out of the training split, and
evaluated again after adaptation.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ARTIFACT_REQUEST = re.compile(
    r"\b(build|create|generate|implement|design|make|code|develop|render)\b.*\b"
    r"(app|application|website|page|scene|game|visualization|simulation|component|project|artifact|code)\b",
    re.IGNORECASE | re.DOTALL,
)

FEATURE_PATTERNS: dict[str, tuple[str, ...]] = {
    "three.js": (r"\bTHREE\.", r"\bfrom\s+['\"]three['\"]", r"three(?:\.min)?\.js"),
    "voxel": (r"InstancedMesh", r"BoxGeometry", r"BoxBufferGeometry", r"BufferGeometry"),
    "shader": (r"ShaderMaterial", r"vertexShader", r"fragmentShader", r"\bgl_Position\b"),
    "animation": (r"requestAnimationFrame", r"setAnimationLoop", r"AnimationMixer"),
    "interaction": (r"OrbitControls", r"addEventListener", r"Raycaster"),
    "responsive": (r"resize", r"devicePixelRatio", r"innerWidth"),
    "island": (r"water", r"terrain", r"island"),
    "sakura": (r"sakura", r"cherry", r"blossom", r"petal"),
    "accessibility": (r"aria-", r"role=", r"prefers-reduced-motion"),
}

PERMISSIVE_LICENSES: dict[str, tuple[str, ...]] = {
    "MIT": ("permission is hereby granted, free of charge",),
    "Apache-2.0": ("apache license", "version 2.0"),
    "BSD": ("redistribution and use in source and binary forms",),
    "ISC": ("permission to use, copy, modify, and/or distribute",),
}


@dataclass(frozen=True)
class ArtifactContract:
    task_type: str
    artifact_kind: str
    entrypoint: str
    requested_features: tuple[str, ...]
    minimum_bytes: int
    requires_javascript_syntax: bool

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactEvaluation:
    score: float
    passed: bool
    hard_gates: dict[str, bool]
    feature_scores: dict[str, float]
    diagnostics: list[str]
    bytes: int
    lines: int
    screenshot_path: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodDecision:
    method: str
    reasons: tuple[str, ...]
    verifier_available: bool
    training_available: bool
    training: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchCorpus:
    sources: list[dict[str, str]] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ResearchSubtask:
    id: str
    objective: str
    capability: str
    queries: list[str]
    minimum_sources: int
    minimum_origins: int
    required_kinds: tuple[str, ...]
    minimum_topic_sources: int = 0
    minimum_topic_origins: int = 0
    status: str = "pending"
    accepted_sources: int = 0
    source_kinds: dict[str, int] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


def research_subtasks(query: str, contract: ArtifactContract) -> list[ResearchSubtask]:
    """Split research into auditable capability-sized tasks."""
    subtasks: list[ResearchSubtask] = []
    features = contract.requested_features or (contract.artifact_kind,)
    task_context = " ".join(features[:6])
    for feature in features:
        technique_queries = {
            "sakura": (
                "three.js falling particles sprites Points GitHub MIT",
                "three.js instanced petal wind particle implementation GitHub Apache MIT",
            ),
            "island": (
                "three.js procedural terrain water island GitHub MIT",
                "three.js low poly terrain shoreline implementation GitHub Apache MIT",
            ),
            "voxel": (
                "three.js InstancedMesh BoxGeometry voxel GitHub MIT",
                "three.js voxel terrain implementation GitHub Apache MIT",
            ),
        }.get(feature, ())
        subtasks.append(
            ResearchSubtask(
                id=f"capability-{re.sub(r'[^a-z0-9]+', '-', feature.casefold()).strip('-')}",
                objective=f"Collect working implementation and API evidence for {feature}",
                capability=feature,
                queries=list(
                    dict.fromkeys(
                        [
                            *technique_queries,
                            f"{feature} {task_context} official documentation API example",
                            f"{feature} {task_context} complete implementation GitHub MIT Apache",
                            f"{feature} {task_context} production performance debugging source code",
                        ]
                    )
                ),
                minimum_sources=3,
                minimum_origins=2,
                required_kinds=("web-documentation", "repository-code") if feature == "three.js" else ("repository-code",),
                minimum_topic_sources=2 if feature in {"island", "sakura"} else 0,
                minimum_topic_origins=2 if feature in {"island", "sakura"} else 0,
            )
        )
    subtasks.append(
        ResearchSubtask(
            id="integration",
            objective="Collect complete integration examples spanning the requested capabilities",
            capability="integration",
            queries=[
                f"{' '.join(features[:6])} complete production example GitHub",
                f"{' '.join(features[:6])} architecture integration tutorial",
            ],
            minimum_sources=max(4, min(8, len(features))),
            minimum_origins=2,
            required_kinds=("repository-code",),
        )
    )
    return subtasks


def audit_research_subtask(
    subtask: ResearchSubtask,
    sources: list[dict[str, Any]],
    *,
    all_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit independent source count, type coverage, and feature relevance."""
    del all_sources  # Kinds must be relevant to this capability, not merely present globally.
    relevant = [source for source in sources if _source_supports_capability(source, subtask.capability)]
    kinds = Counter(str(source.get("kind") or "unknown") for source in relevant)
    urls = {str(source.get("url") or "") for source in relevant if source.get("url")}
    origins = {_source_origin(source) for source in relevant if _source_origin(source)}
    missing_kinds = [kind for kind in subtask.required_kinds if not kinds.get(kind)]
    topical = [source for source in relevant if _source_has_topic(source, subtask.capability)]
    topic_urls = {str(source.get("url") or "") for source in topical if source.get("url")}
    topic_origins = {_source_origin(source) for source in topical if _source_origin(source)}
    passed = (
        len(urls) >= subtask.minimum_sources
        and len(origins) >= subtask.minimum_origins
        and not missing_kinds
        and len(topic_urls) >= subtask.minimum_topic_sources
        and len(topic_origins) >= subtask.minimum_topic_origins
    )
    audit = {
        "passed": passed,
        "independent_sources": len(urls),
        "minimum_sources": subtask.minimum_sources,
        "independent_origins": len(origins),
        "minimum_origins": subtask.minimum_origins,
        "source_kinds": dict(kinds),
        "missing_kinds": missing_kinds,
        "topic_sources": len(topic_urls),
        "minimum_topic_sources": subtask.minimum_topic_sources,
        "topic_origins": len(topic_origins),
        "minimum_topic_origins": subtask.minimum_topic_origins,
    }
    subtask.status = "completed" if passed else "needs-more-evidence"
    subtask.accepted_sources = len(urls)
    subtask.source_kinds = dict(kinds)
    subtask.audit = audit
    return audit


def _source_supports_capability(source: dict[str, Any], capability: str) -> bool:
    haystack = f"{source.get('title', '')} {source.get('url', '')} {str(source.get('text', ''))[:12_000]}"
    if capability == "integration":
        matches = sum(
            any(re.search(pattern, haystack, re.I) for pattern in patterns)
            for patterns in FEATURE_PATTERNS.values()
        )
        return source.get("kind") == "repository-code" and matches >= 2
    if capability == "island":
        subject = bool(re.search(r"\b(island|terrain|shore|water)\b", haystack, re.I))
        rendering = bool(re.search(r"\b(three(?:\.js)?|webgl|mesh|geometry|shader|canvas)\b", haystack, re.I))
        return subject and rendering
    if capability == "sakura":
        subject = bool(re.search(r"\b(sakura|cherry(?:\s+blossom)?|blossom|petals?)\b", haystack, re.I))
        rendering = bool(re.search(r"\b(three(?:\.js)?|webgl|particle|sprite|mesh|shader|canvas)\b", haystack, re.I))
        technique = bool(
            re.search(r"\b(particles?|sprites?|points|instanced|billboard|wind|falling)\b", haystack, re.I)
        )
        return rendering and (subject or technique)
    patterns = FEATURE_PATTERNS.get(capability, (re.escape(capability),))
    return any(re.search(pattern, haystack, re.I) for pattern in patterns)


def _source_has_topic(source: dict[str, Any], capability: str) -> bool:
    haystack = f"{source.get('title', '')} {source.get('url', '')} {str(source.get('text', ''))[:12_000]}"
    if capability == "sakura":
        return bool(re.search(r"\b(sakura|cherry(?:\s+blossom)?|blossom|petals?)\b", haystack, re.I))
    if capability == "island":
        return bool(re.search(r"\b(island|terrain|shore|water)\b", haystack, re.I))
    return _source_supports_capability(source, capability)


def _source_origin(source: dict[str, Any]) -> str:
    """Group files by repository and documents by host for independence audits."""
    url = str(source.get("url") or "")
    parsed = urlparse(url)
    if parsed.hostname in {"github.com", "www.github.com", "raw.githubusercontent.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"github:{parts[0].casefold()}/{parts[1].removesuffix('.git').casefold()}"
    return parsed.hostname.casefold() if parsed.hostname else str(source.get("sha256") or "")


def source_capabilities(source: dict[str, Any], contract: ArtifactContract) -> list[str]:
    """Return the explicitly evidenced requested capabilities for source curation."""
    return [
        capability
        for capability in contract.requested_features
        if _source_supports_capability(source, capability)
    ]


def task_requires_artifact(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return bool(
        ARTIFACT_REQUEST.search(query)
        or re.search(r"\b(html|javascript|typescript|react|three\.js|webgl|shader|voxel)\b", normalized)
        and re.search(r"\b(build|create|generate|implement|design|make|code)\b", normalized)
    )


def strip_learning_command(query: str) -> str:
    return re.sub(r"^/(?:learn|ttc)\s+", "", query, flags=re.IGNORECASE).strip()


def derive_artifact_contract(query: str) -> ArtifactContract:
    normalized = " ".join(query.lower().split())
    web = bool(
        re.search(r"\b(html|css|javascript|typescript|react|three\.js|webgl|website|web app|page)\b", normalized)
    )
    requested: list[str] = []
    for feature in FEATURE_PATTERNS:
        aliases = {feature}
        if feature == "interaction":
            aliases.update({"interactive", "controls", "camera controls"})
        if feature == "animation":
            aliases.update({"animated", "animate", "motion"})
        if feature == "sakura":
            aliases.update({"cherry blossom", "cherry blossoms", "petals"})
        if any(alias in normalized for alias in aliases):
            requested.append(feature)
    if web:
        requested.extend(feature for feature in ("responsive",) if feature not in requested)
    quality_requested = bool(
        re.search(
            r"\b(very good|polished|professional|production(?:-ready)?|high quality|beautiful|detailed)\b", normalized
        )
    )
    return ArtifactContract(
        task_type="artifact" if task_requires_artifact(query) else "knowledge",
        artifact_kind="web" if web else "code",
        entrypoint="index.html" if web else "solution.py",
        requested_features=tuple(dict.fromkeys(requested)),
        # This is a generic depth proxy, not a benchmark-specific condition.
        # A polished interactive artifact cannot pass with a tiny keyword stub.
        minimum_bytes=(8_000 if quality_requested else 3_000) if web else (1_500 if quality_requested else 500),
        requires_javascript_syntax=web,
    )


def artifact_generation_prompt(
    query: str,
    contract: ArtifactContract,
    *,
    verifier_feedback: list[str] | None = None,
    skill_guardrails: str = "",
) -> str:
    requirements = ", ".join(contract.requested_features) or "the requested behavior"
    feedback = ""
    if verifier_feedback:
        feedback = "\nThe prior attempt failed these objective checks:\n- " + "\n- ".join(verifier_feedback[:12])
    memory = f"\nRetrieved failure-pattern guardrails:\n{skill_guardrails}" if skill_guardrails else ""
    if contract.artifact_kind == "web":
        return (
            f"Build this artifact: {strip_learning_command(query)}\n"
            "Produce one self-contained index.html. It must run when served by a basic HTTP server. "
            "Use browser-native HTML/CSS/JavaScript; CDN ES modules are allowed. Do not use build tools, placeholders, "
            "markdown fences, or explanatory prose. Implement real behavior rather than merely mentioning requirements. "
            f"Observable requirements: {requirements}. Target at least {contract.minimum_bytes} bytes of purposeful source."
            f"{feedback}{memory}"
        )
    return (
        f"Implement this task in {contract.entrypoint}: {strip_learning_command(query)}. "
        "Return complete runnable source only, without placeholders or markdown fences. "
        f"Observable requirements: {requirements}.{feedback}{memory}"
    )


def framework_artifact_source(query: str, contract: ArtifactContract) -> str | None:
    """Return a trusted, runnable framework for capabilities a tiny model cannot safely bootstrap.

    This does not count as an adapted-model result. It is a separately reported
    fallback artifact that gives the user a usable project when an experimental
    adapter is rejected.
    """
    del query
    if contract.artifact_kind != "web" or "three.js" not in contract.requested_features:
        return None
    features = set(contract.requested_features)
    if {"sakura", "island"}.issubset(features):
        title = "Sakura Island"
    elif "island" in features:
        title = "Voxel Island"
    else:
        title = "Interactive Three.js World"
    template = Path(__file__).parent.parent / "resources" / "artifact-frameworks" / "threejs.html"
    return template.read_text(encoding="utf-8").replace("__TITLE__", html.escape(title))


def research_queries(query: str, contract: ArtifactContract, limit: int = 6) -> list[str]:
    """Create diverse generic research intents; a model may replace these."""
    features = " ".join(contract.requested_features[:5])
    candidates = [
        f"{features} official API documentation",
        f"{features} implementation examples GitHub".strip(),
        f"{contract.artifact_kind} performance architecture best practices {features}".strip(),
        f"{features} shader animation source code permissive license".strip(),
        f"{features} common errors debugging".strip(),
        f"{features} production example repository".strip(),
    ]
    return list(dict.fromkeys(candidate for candidate in candidates if len(candidate) > 12))[:limit]


def fast_research_queries(
    contract: ArtifactContract,
    diagnostics: list[str] | None = None,
    *,
    limit: int = 14,
) -> list[str]:
    """Plan a small deterministic search frontier instead of spending an inference pass on query prose."""
    diagnostics_text = " ".join(diagnostics or []).casefold()
    subtasks = research_subtasks("", contract)
    failed_features = {
        feature
        for feature in contract.requested_features
        if feature.casefold() in diagnostics_text
    }
    ordered = sorted(
        subtasks,
        key=lambda item: (
            item.id == "integration",
            item.capability not in failed_features,
            item.capability not in {"sakura", "island"},
        ),
    )
    selected: list[str] = []
    for subtask in ordered:
        query_budget = 2 if subtask.capability in failed_features | {"sakura", "island", "integration"} else 1
        repository_queries = [
            query
            for query in subtask.queries
            if any(marker in query.casefold() for marker in ("github", "source code", "implementation"))
        ]
        documentation_queries = [query for query in subtask.queries if "official" in query.casefold()]
        candidates = repository_queries[:query_budget]
        if "web-documentation" in subtask.required_kinds:
            candidates.extend(documentation_queries[:1])
        for query in candidates or subtask.queries[:query_budget]:
            if query not in selected:
                selected.append(query)
            if len(selected) >= limit:
                return selected
    return selected


def select_method(
    *,
    contract: ArtifactContract,
    training_available: bool,
    source_count: int,
    train_examples: int,
    time_sensitive: bool = False,
    model_params_b: float = 1.5,
    memory_gb: float = 8.0,
    quantized: bool = True,
    multi_step_rollout: bool = False,
    deterministic_reward: bool = False,
    maximum_training_seconds: int = 600,
    backend: str = "mlx",
    paged_optimizer_available: bool = False,
    measured_seconds_per_iteration: float | None = None,
) -> MethodDecision:
    reasons: list[str] = []
    verifier = contract.task_type == "artifact"
    if time_sensitive:
        reasons.append("Changing facts should remain retrieval-grounded")
        return MethodDecision("retrieval", tuple(reasons), verifier, training_available, {})
    if source_count < 6 or train_examples < 24:
        reasons.append("There is not enough independent grounded data for a weight update")
        return MethodDecision("retrieval", tuple(reasons), verifier, training_available, {})
    if not training_available:
        reasons.append("The selected hardware/model cannot run local adapter training")
        return MethodDecision("retrieval", tuple(reasons), verifier, training_available, {})
    paged_qlora = quantized and backend in {"cuda", "torch", "vllm"} and paged_optimizer_available
    method = "pqlora-il" if paged_qlora else "qlora-il" if quantized else "lora-il"
    if multi_step_rollout and deterministic_reward:
        method = f"{method}+grpo"
        reasons.append("A real multi-step rollout and deterministic reward justify an RL phase after IL warm-up")
    elif verifier:
        reasons.append("Reference implementations provide demonstrations and an executable artifact verifier")
        reasons.append(f"{method.upper()} is selected; RL requires a real multi-step rollout, not a one-shot score")
    else:
        reasons.append("Grounded stable demonstrations are available")
    if quantized and not paged_qlora:
        reasons.append(
            "Paged QLoRA is unavailable on this backend; MLX unified-memory QLoRA is used without relabeling it"
        )
    elif paged_qlora:
        reasons.append("A real paged optimizer is available on CUDA, so optimizer states can spill safely")
    if memory_gb < model_params_b * 1.3 + 2.0:
        reasons.append("Use compiled quantized adapters with checkpointing because unified memory is constrained")
    compact_mlx_profile = memory_gb <= 8 and model_params_b <= 2
    batch_size = 2 if memory_gb >= 16 else 1
    grad_accumulation = 2 if memory_gb >= 16 else 1
    target_epochs = 4 if compact_mlx_profile else 3 if train_examples >= 96 else 5
    # mlx-lm's `iters` counts microbatches, while optimizer updates happen only
    # after gradient accumulation. Count both explicitly so the stated epoch
    # coverage is real and not accidentally divided by accumulation.
    requested_microbatches = math.ceil(train_examples * target_epochs / batch_size)
    iteration_cap = 320 if compact_mlx_profile else 300 if memory_gb >= 16 else 240
    iterations = min(iteration_cap, max(64, requested_microbatches))
    if maximum_training_seconds <= 180:
        iterations = min(iterations, 64)
    # The measured long run completed 234 updates plus load/benchmarks in
    # 453.8 seconds. Keep headroom for thermal variance while spending the
    # recovered budget on a fourth data pass rather than idle orchestration.
    default_seconds_per_iteration = 1.85 if compact_mlx_profile else 2.0
    seconds_per_iteration = max(
        default_seconds_per_iteration,
        float(measured_seconds_per_iteration or 0.0),
    )
    runtime_overhead_seconds = 25
    budget_iteration_cap = max(
        32,
        math.floor((maximum_training_seconds - runtime_overhead_seconds) / seconds_per_iteration),
    )
    iterations = min(iterations, budget_iteration_cap)
    rank = 16 if compact_mlx_profile or (model_params_b <= 3 and memory_gb >= 16) else 8
    layers = 8 if compact_mlx_profile else 16 if model_params_b <= 3 and memory_gb >= 16 else 8
    sequence = 256 if compact_mlx_profile else 256 if memory_gb <= 8 else 512 if memory_gb < 16 else 768
    training = {
        "iterations": iterations,
        "optimizer_updates": math.ceil(iterations / grad_accumulation),
        "target_epochs": target_epochs,
        "effective_epochs": round(iterations * batch_size / max(1, train_examples), 2),
        "budget_limited": iterations < requested_microbatches,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation,
        "learning_rate": 2e-5 if compact_mlx_profile else 1e-4,
        "optimizer": "adamw",
        "lora_rank": rank,
        "lora_layers": layers,
        "lora_scale": 20.0,
        "lora_targets": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"],
        "max_seq_length": sequence,
        "compile_bucket_size": 32 if compact_mlx_profile else 128,
        "clear_cache_threshold_gb": 1.0 if memory_gb <= 8 else 2.0,
        "mask_prompt": True,
        "seed": 0,
        "grad_checkpoint": memory_gb < model_params_b * 1.3 + 2.0,
        "optimizer_memory_strategy": "paged" if paged_qlora else "unified-memory" if backend == "mlx" else "resident",
        "backend": backend,
        "maximum_training_seconds": maximum_training_seconds,
        "estimated_training_seconds": round(iterations * seconds_per_iteration + runtime_overhead_seconds),
        "seconds_per_iteration": round(seconds_per_iteration, 4),
        "throughput_source": "measured-local-profile" if measured_seconds_per_iteration else "hardware-default",
    }
    return MethodDecision(method, tuple(reasons), verifier, training_available, training)


def _extract_module_script(source: str) -> str:
    scripts = [
        body
        for attributes, body in re.findall(r"<script([^>]*)>(.*?)</script>", source, flags=re.DOTALL | re.IGNORECASE)
        if not re.search(r"type\s*=\s*['\"]importmap['\"]", attributes, re.IGNORECASE)
        and not re.search(r"\bsrc\s*=", attributes, re.IGNORECASE)
    ]
    code = "\n".join(scripts)
    code = re.sub(r"^\s*import\s+.*?;?\s*$", "", code, flags=re.MULTILINE)
    return code


def _chrome_binary() -> str | None:
    configured = os.environ.get("ILOPTIMUS_CHROME")
    candidates = [
        configured,
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    return next((str(candidate) for candidate in candidates if candidate and Path(candidate).is_file()), None)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _runtime_render(path: Path) -> tuple[bool, str, str]:
    """Render a web artifact in a real browser and retain the screenshot."""
    chrome = _chrome_binary()
    if not chrome:
        return False, "", "A Chromium-compatible browser is required for runtime verification"
    screenshot = path.parent / "runtime.png"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(path.parent), **kwargs)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rendered: subprocess.CompletedProcess[str] | None = None
    browser_output = ""
    timed_out = False
    try:
        with tempfile.TemporaryDirectory(prefix="iloptimus-chrome-profile-") as profile:
            browser_log = Path(profile) / "browser.log"
            # Chrome's updater can inherit stderr on macOS. Writing to a file
            # avoids waiting forever for an orphaned child to close a pipe.
            with browser_log.open("w", encoding="utf-8") as log_handle:
                command = [
                    chrome,
                    "--headless=new",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--enable-unsafe-swiftshader",
                    "--use-gl=angle",
                    "--use-angle=swiftshader",
                    "--disable-background-networking",
                    "--disable-component-update",
                    "--disable-sync",
                    "--metrics-recording-only",
                    "--enable-logging=stderr",
                    "--hide-scrollbars",
                    "--window-size=1280,800",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=2000",
                    f"--user-data-dir={profile}",
                    f"--screenshot={screenshot}",
                    f"http://127.0.0.1:{port}/{path.name}",
                ]
                try:
                    rendered = subprocess.run(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        # Animated artifacts can keep Chrome's event loop alive
                        # after the screenshot is already written. Bound this
                        # verifier overhead; a timed-out process still needs a
                        # nonblank screenshot and a clean console to pass.
                        timeout=15,
                    )
                except subprocess.TimeoutExpired:
                    # Chrome can keep an event-loop page alive after writing the
                    # requested screenshot. The image and console log remain
                    # independent evidence; the timeout itself is diagnostic.
                    timed_out = True
            browser_output = browser_log.read_text(encoding="utf-8", errors="replace")
        console_error = bool(re.search(r"(?:Uncaught|ReferenceError|TypeError|SyntaxError)", browser_output, re.I))
        image_ok = screenshot.exists() and screenshot.stat().st_size >= 1_000 and _png_has_visual_content(screenshot)
        browser_completed = rendered is not None and rendered.returncode == 0
        ok = (browser_completed or timed_out) and image_ok and not console_error
        diagnostic = (
            ""
            if ok
            else (
                "Browser console reported an uncaught runtime error"
                if console_error
                else "Browser rendered a blank or visually uniform page"
                if screenshot.exists()
                else (browser_output or "Browser produced no usable screenshot")[-1200:]
            )
        )
        return ok, str(screenshot) if screenshot.exists() else "", diagnostic
    except (OSError, subprocess.SubprocessError) as error:
        return False, "", str(error)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _png_has_visual_content(path: Path) -> bool:
    """Decode Chrome's 8-bit RGB/RGBA PNG enough to reject blank canvases."""
    try:
        payload = path.read_bytes()
        if payload[:8] != b"\x89PNG\r\n\x1a\n":
            return False
        cursor = 8
        width = height = color_type = bit_depth = interlace = 0
        compressed = bytearray()
        while cursor + 12 <= len(payload):
            length = int.from_bytes(payload[cursor : cursor + 4], "big")
            kind = payload[cursor + 4 : cursor + 8]
            data = payload[cursor + 8 : cursor + 8 + length]
            cursor += 12 + length
            if kind == b"IHDR":
                width = int.from_bytes(data[:4], "big")
                height = int.from_bytes(data[4:8], "big")
                bit_depth, color_type, interlace = data[8], data[9], data[12]
            elif kind == b"IDAT":
                compressed.extend(data)
            elif kind == b"IEND":
                break
        channels = {2: 3, 6: 4}.get(color_type, 0)
        if not width or not height or bit_depth != 8 or not channels or interlace:
            return False
        raw = zlib.decompress(bytes(compressed))
        stride = width * channels
        previous = bytearray(stride)
        offset = 0
        colors: set[tuple[int, int, int]] = set()
        luminance_min, luminance_max = 255, 0
        pixel_step = max(1, (width * height) // 25_000)
        pixel_index = 0
        for _ in range(height):
            filter_type = raw[offset]
            offset += 1
            scan = bytearray(raw[offset : offset + stride])
            offset += stride
            for index in range(stride):
                left = scan[index - channels] if index >= channels else 0
                up = previous[index]
                upper_left = previous[index - channels] if index >= channels else 0
                if filter_type == 1:
                    scan[index] = (scan[index] + left) & 255
                elif filter_type == 2:
                    scan[index] = (scan[index] + up) & 255
                elif filter_type == 3:
                    scan[index] = (scan[index] + ((left + up) // 2)) & 255
                elif filter_type == 4:
                    estimate = left + up - upper_left
                    distances = (abs(estimate - left), abs(estimate - up), abs(estimate - upper_left))
                    predictor = (left, up, upper_left)[distances.index(min(distances))]
                    scan[index] = (scan[index] + predictor) & 255
                elif filter_type != 0:
                    return False
            for index in range(0, stride, channels):
                if pixel_index % pixel_step == 0:
                    color = (scan[index], scan[index + 1], scan[index + 2])
                    colors.add(color)
                    luminance = (color[0] * 54 + color[1] * 183 + color[2] * 19) // 256
                    luminance_min = min(luminance_min, luminance)
                    luminance_max = max(luminance_max, luminance)
                pixel_index += 1
            previous = scan
        return len(colors) >= 12 and luminance_max - luminance_min >= 10
    except (OSError, ValueError, IndexError, zlib.error):
        return False


def evaluate_artifact(path: Path, contract: ArtifactContract) -> ArtifactEvaluation:
    diagnostics: list[str] = []
    if not path.exists() or not path.is_file():
        return ArtifactEvaluation(0.0, False, {"exists": False}, {}, [f"Missing {contract.entrypoint}"], 0, 0)
    source = path.read_text(encoding="utf-8", errors="replace")
    size = len(source.encode())
    lines = len(source.splitlines())
    normalized_blocks = [block.strip() for block in re.split(r"(?<=[;>{}])|\n", source) if block.strip()]
    effective_size = sum(len(block.encode()) for block in set(normalized_blocks))
    visible_source = re.sub(r"<!--.*?-->|<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>", "", source, flags=re.I | re.S)
    visible_source = html.unescape(re.sub(r"<[^>]+>", " ", visible_source))
    visible_code_markers = sum(
        bool(re.search(pattern, visible_source))
        for pattern in (r"\b(?:const|let|var|function)\b", r"\bTHREE\.", r"[{};]", r"document\.")
    )
    rendered_source_code = len(visible_source.strip()) > 600 and visible_code_markers >= 2
    hard_gates = {
        "exists": True,
        "substantial": effective_size >= contract.minimum_bytes,
        "no_placeholders": not bool(
            re.search(
                r"\b(TODO|FIXME|placeholder|coming soon)\b|\bimplement\b[^\n]{0,80}\b(?:logic|here)\b",
                source,
                re.I,
            )
        ),
        "entrypoint_shape": "<html" in source.lower() if contract.artifact_kind == "web" else bool(source.strip()),
        "source_not_rendered_as_text": not rendered_source_code,
    }
    if not hard_gates["substantial"]:
        diagnostics.append(
            f"Artifact has {effective_size} unique source bytes ({size} raw); expected at least {contract.minimum_bytes}"
        )
    if not hard_gates["no_placeholders"]:
        diagnostics.append("Artifact contains placeholder markers")
    if rendered_source_code:
        diagnostics.append("Artifact renders source code as page text instead of executing it")

    if contract.requires_javascript_syntax:
        script = _extract_module_script(source)
        syntax_ok = bool(script.strip())
        if syntax_ok:
            try:
                with tempfile.TemporaryDirectory(prefix="iloptimus-artifact-check-") as temporary:
                    module = Path(temporary) / "artifact.mjs"
                    module.write_text(script, encoding="utf-8")
                    checked = subprocess.run(
                        ["node", "--check", str(module)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                syntax_ok = checked.returncode == 0
                if not syntax_ok:
                    diagnostics.append((checked.stderr or checked.stdout).strip()[-1200:])
            except (FileNotFoundError, subprocess.SubprocessError):
                syntax_ok = False
                diagnostics.append("Node.js syntax verification was unavailable")
        hard_gates["javascript_syntax"] = syntax_ok

    screenshot_path = ""
    if contract.artifact_kind == "web" and all(hard_gates.values()):
        runtime_ok, screenshot_path, runtime_diagnostic = _runtime_render(path)
        # A first Chromium process can lose a race with CDN module loading or
        # SwiftShader startup. Retry once when there is no deterministic page
        # exception; the second render still has to provide real nonblank pixels.
        if not runtime_ok and not re.search(r"(?:Uncaught|ReferenceError|TypeError|SyntaxError)", runtime_diagnostic, re.I):
            runtime_ok, screenshot_path, runtime_diagnostic = _runtime_render(path)
        hard_gates["runtime_render"] = runtime_ok
        if runtime_diagnostic:
            diagnostics.append(runtime_diagnostic)

    feature_scores: dict[str, float] = {}
    for feature in contract.requested_features:
        patterns = FEATURE_PATTERNS.get(feature, (re.escape(feature),))
        matched = sum(bool(re.search(pattern, source, re.I)) for pattern in patterns)
        feature_scores[feature] = min(1.0, matched / min(2, len(patterns)))
        if feature_scores[feature] < 0.5:
            diagnostics.append(f"Requested feature is not implemented observably: {feature}")

    hard_score = sum(hard_gates.values()) / max(1, len(hard_gates))
    feature_score = sum(feature_scores.values()) / max(1, len(feature_scores)) if feature_scores else 1.0
    complexity = min(1.0, effective_size / max(contract.minimum_bytes * 2.5, 1))
    score = round(0.5 * hard_score + 0.4 * feature_score + 0.1 * complexity, 4)
    passed = all(hard_gates.values()) and all(value >= 0.5 for value in feature_scores.values()) and score >= 0.72
    return ArtifactEvaluation(score, passed, hard_gates, feature_scores, diagnostics, size, lines, screenshot_path)


def detect_license(repository: Path) -> str | None:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        path = repository / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()[:30_000]
        for license_name, markers in PERMISSIVE_LICENSES.items():
            if all(marker in text for marker in markers):
                return license_name
    return None


def github_repository_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts[:2]):
        return None
    return f"https://github.com/{parts[0]}/{parts[1].removesuffix('.git')}.git"


def github_repository_search_terms(search_query: str, *, limit: int = 3) -> list[str]:
    """Keep repository searches narrow while preserving the capability being audited."""
    aliases = {
        "three.js": "threejs",
        "particles": "particle",
        "petals": "petal",
        "sprites": "sprite",
        "shaders": "shader",
    }
    ignored = {
        "github",
        "repository",
        "examples",
        "example",
        "implementation",
        "source",
        "code",
        "complete",
        "official",
        "documentation",
        "docs",
        "production",
        "performance",
        "debugging",
        "apache",
        "mit",
    }
    raw = []
    for token in re.findall(r"[a-z0-9.-]+", search_query.casefold()):
        token = aliases.get(token, token.replace(".js", "js"))
        if len(token) >= 4 and token not in ignored and token not in raw:
            raw.append(token)
    capability_priority = (
        "sakura",
        "cherry",
        "blossom",
        "petal",
        "voxel",
        "island",
        "terrain",
        "shader",
        "water",
        "particle",
        "sprite",
        "orbit",
        "animation",
    )
    selected = [term for term in capability_priority if term in raw][:1]
    selected.extend(term for term in ("threejs", "webgl") if term in raw and term not in selected)
    selected.extend(term for term in capability_priority if term in raw and term not in selected)
    selected.extend(term for term in raw if term not in selected)
    return selected[:limit]


def rank_repository_paths(
    paths: list[str],
    query: str,
    *,
    preferred_features: tuple[str, ...] = (),
) -> list[str]:
    """Rank first-party implementation files ahead of vendored/minified dependencies."""
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.casefold())
        if len(term) >= 4
        and term
        not in {
            "build",
            "create",
            "generate",
            "implementation",
            "interactive",
            "polished",
            "production",
            "ready",
            "with",
        }
    }
    terms.update(feature.replace(".", "").casefold() for feature in preferred_features)
    extensions = {".js", ".mjs", ".ts", ".tsx", ".jsx", ".html", ".css", ".glsl", ".wgsl", ".py", ".md"}
    canonical = {
        "app.js",
        "app.ts",
        "index.html",
        "index.js",
        "index.ts",
        "main.js",
        "main.ts",
        "scene.js",
        "scene.ts",
        "readme.md",
    }
    third_party_parts = {"build", "dist", "legacy", "node_modules", "old", "vendor", "vendors"}
    ranked: list[tuple[int, str]] = []
    for path in paths:
        path_object = Path(path)
        if path_object.suffix.casefold() not in extensions or path_object.name.casefold().endswith(".min.js"):
            continue
        lowered = path.casefold()
        compact = lowered.replace(".", "")
        parts = {part.casefold() for part in path_object.parts}
        score = sum(3 for term in terms if term in compact)
        score += sum(3 for feature in preferred_features if feature.replace(".", "").casefold() in compact)
        score += 7 if path_object.name.casefold() in canonical else 0
        score += 4 if "src" in parts or "source" in parts else 0
        score += 2 if "example" in parts or "examples" in parts or "demo" in parts else 0
        score -= 14 if parts & third_party_parts else 0
        ranked.append((score, path))
    return [path for _, path in sorted(ranked, key=lambda item: (-item[0], item[1].casefold()))]


def sample_repository(
    url: str,
    query: str,
    *,
    max_files: int = 16,
    preferred_features: tuple[str, ...] = (),
) -> ResearchCorpus:
    """Sample relevant text/code blobs from a public permissively licensed repo."""
    repository_url = github_repository_url(url)
    corpus = ResearchCorpus()
    if not repository_url:
        return corpus
    with tempfile.TemporaryDirectory(prefix="iloptimus-research-repo-") as temporary:
        root = Path(temporary) / "repo"
        try:
            cloned = subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", repository_url, str(root)],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if cloned.returncode:
                corpus.rejected.append({"url": repository_url, "reason": cloned.stderr[-500:]})
                return corpus
            license_name = None
            for license_path in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
                shown_license = subprocess.run(
                    ["git", "-C", str(root), "show", f"HEAD:{license_path}"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if shown_license.returncode:
                    continue
                lowered = shown_license.stdout.lower()[:30_000]
                license_name = next(
                    (
                        name
                        for name, markers in PERMISSIVE_LICENSES.items()
                        if all(marker in lowered for marker in markers)
                    ),
                    None,
                )
                if license_name:
                    break
            if not license_name:
                corpus.rejected.append({"url": repository_url, "reason": "No recognized permissive license"})
                return corpus
            listed = subprocess.run(
                ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.splitlines()
            selected = rank_repository_paths(listed, query, preferred_features=preferred_features)
            for path in selected[:max_files]:
                size = subprocess.run(
                    ["git", "-C", str(root), "cat-file", "-s", f"HEAD:{path}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if size.returncode or not size.stdout.strip().isdigit() or int(size.stdout) > 80_000:
                    continue
                shown = subprocess.run(
                    ["git", "-C", str(root), "show", f"HEAD:{path}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if shown.returncode or not shown.stdout.strip():
                    continue
                corpus.sources.append(
                    {
                        "title": f"{repository_url.removesuffix('.git')} — {path}",
                        "url": f"{repository_url.removesuffix('.git')}/blob/HEAD/{path}",
                        "text": shown.stdout[:60_000],
                        "license": license_name,
                        "kind": "repository-code",
                    }
                )
        except (OSError, subprocess.SubprocessError) as error:
            corpus.rejected.append({"url": repository_url, "reason": str(error)})
    return corpus


def parse_model_queries(text: str, fallback: list[str], *, limit: int = 6) -> list[str]:
    """Recover a bounded search plan authored by even a weak local model."""
    candidates: list[str] = []
    for match in re.finditer(r"\[[\s\S]*?\]", text):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            candidates.extend(str(item) for item in payload if isinstance(item, str))
            break
    if not candidates:
        candidates.extend(re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip(" \"'") for line in text.splitlines())
    cleaned = [re.sub(r"\s+", " ", item).strip()[:220] for item in candidates]
    research_markers = (
        "documentation",
        "docs",
        "github",
        "repository",
        "example",
        "implementation",
        "performance",
        "shader",
        "animation",
        "license",
        "debug",
        "api",
        "source",
    )
    cleaned = [
        item
        for item in cleaned
        if 8 <= len(item) <= 220
        and not item.lstrip().startswith(("//", "#", "/*", "<!--"))
        and not re.search(r"\b(now|then),?\s+(?:proceed|implement|write|build)\b", item, re.I)
        and any(marker in item.lower() for marker in research_markers)
    ]
    # Preserve some fallback diversity even when a small model emits one plausible
    # query plus prose masquerading as a plan.
    interleaved: list[str] = []
    for index in range(max(len(cleaned), len(fallback))):
        if index < len(cleaned):
            interleaved.append(cleaned[index])
        if index < len(fallback):
            interleaved.append(fallback[index])
    return list(dict.fromkeys(interleaved))[:limit]


def build_artifact_dataset(
    query: str,
    sources: list[dict[str, str]],
    contract: ArtifactContract,
    *,
    max_examples: int = 48,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a provenance-rich dataset with the exact task held out at row 0."""
    required_terms = list(contract.requested_features) or [contract.artifact_kind]
    holdout = {
        "split": "holdout",
        "prompt": query,
        "ideal_response": "<reasoning>Implement every observable requirement and verify syntax.</reasoning>"
        f"<answer>{' '.join(required_terms)}</answer>",
        "expected_answer": " ".join(required_terms),
        "source_url": "",
        "source_hash": "",
    }
    examples: list[dict[str, Any]] = [holdout]
    seen: set[str] = set()
    exact_task = re.sub(r"\s+", " ", query).strip().casefold()
    source_chunks: list[list[str]] = []
    for source in sources:
        text = source.get("text", "")
        chunks: list[str] = []
        for start in range(0, len(text), 1_000):
            chunk = text[start : start + 1_200].strip()
            if len(chunk) < 180:
                continue
            chunks.append(chunk)
        source_chunks.append(chunks)

    # Round-robin across sources before taking a second chunk from any source.
    # This preserves implementation diversity in short micro-adaptation runs.
    for chunk_index in range(max((len(chunks) for chunks in source_chunks), default=0)):
        for source, chunks in zip(sources, source_chunks):
            if chunk_index >= len(chunks):
                continue
            chunk = chunks[chunk_index]
            digest = hashlib.sha256(chunk.encode()).hexdigest()
            if digest in seen or exact_task in re.sub(r"\s+", " ", chunk).casefold():
                continue
            seen.add(digest)
            title = source.get("title", "Reference")
            url = source.get("url", "")
            examples.append(
                {
                    "split": "train",
                    "prompt": (
                        f"Study this independently sourced {contract.artifact_kind} implementation excerpt from {title}. "
                        "Return the reusable implementation pattern faithfully, preserving APIs and syntax."
                    ),
                    "ideal_response": f"<reasoning>I will retain the verified implementation pattern.</reasoning><answer>{chunk}</answer>",
                    "expected_answer": chunk[:700],
                    "source_url": url,
                    "source_hash": digest,
                    "license": source.get("license", "documentation"),
                }
            )
            if len(examples) >= max_examples + 1:
                break
        if len(examples) >= max_examples + 1:
            break
    manifest = {
        "version": 1,
        "task_hash": hashlib.sha256(query.encode()).hexdigest(),
        "holdout_rows": [0],
        "train_rows": list(range(1, len(examples))),
        "source_urls": sorted({example["source_url"] for example in examples[1:] if example["source_url"]}),
        "source_hashes": [example["source_hash"] for example in examples[1:]],
        "dataset_hash": hashlib.sha256(
            "\n".join(json.dumps(example, sort_keys=True) for example in examples).encode()
        ).hexdigest(),
        "contamination_check": {
            "exact_task_absent_from_train": all(
                exact_task not in re.sub(r"\s+", " ", example["ideal_response"]).casefold() for example in examples[1:]
            ),
            "duplicate_chunks": len(examples[1:]) - len({example["source_hash"] for example in examples[1:]}),
        },
    }
    return examples, manifest


def acceptance_decision(
    baseline: ArtifactEvaluation, adapted: ArtifactEvaluation, *, margin: float = 0.05
) -> dict[str, Any]:
    accepted = adapted.passed and adapted.score >= baseline.score + margin
    return {
        "accepted": accepted,
        "baseline_score": baseline.score,
        "adapted_score": adapted.score,
        "margin_required": margin,
        "improvement": round(adapted.score - baseline.score, 4),
        "reason": (
            "Adapted artifact passed every hard gate and improved enough"
            if accepted
            else "Adapter rejected: it did not pass all gates with a sufficient measured improvement"
        ),
    }
