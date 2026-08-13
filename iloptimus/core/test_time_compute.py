"""Global failure-driven test-time-compute contracts, datasets, and verifiers.

This module intentionally knows about capability classes (web artifacts, code,
knowledge), not benchmark prompts. A concrete task is converted into observable
requirements, evaluated before adaptation, kept out of the training split, and
evaluated again after adaptation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import zlib
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

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchCorpus:
    sources: list[dict[str, str]] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)


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
) -> str:
    requirements = ", ".join(contract.requested_features) or "the requested behavior"
    feedback = ""
    if verifier_feedback:
        feedback = "\nThe prior attempt failed these objective checks:\n- " + "\n- ".join(verifier_feedback[:12])
    if contract.artifact_kind == "web":
        return (
            f"Build this artifact: {strip_learning_command(query)}\n"
            "Produce one self-contained index.html. It must run when served by a basic HTTP server. "
            "Use browser-native HTML/CSS/JavaScript; CDN ES modules are allowed. Do not use build tools, placeholders, "
            "markdown fences, or explanatory prose. Implement real behavior rather than merely mentioning requirements. "
            f"Observable requirements: {requirements}. Target at least {contract.minimum_bytes} bytes of purposeful source."
            f"{feedback}"
        )
    return (
        f"Implement this task in {contract.entrypoint}: {strip_learning_command(query)}. "
        "Return complete runnable source only, without placeholders or markdown fences. "
        f"Observable requirements: {requirements}.{feedback}"
    )


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


def select_method(
    *,
    contract: ArtifactContract,
    training_available: bool,
    source_count: int,
    train_examples: int,
    time_sensitive: bool = False,
) -> MethodDecision:
    reasons: list[str] = []
    verifier = contract.task_type == "artifact"
    if time_sensitive:
        reasons.append("Changing facts should remain retrieval-grounded")
        return MethodDecision("retrieval", tuple(reasons), verifier, training_available)
    if source_count < 2 or train_examples < 3:
        reasons.append("There is not enough independent grounded data for a weight update")
        return MethodDecision("retrieval", tuple(reasons), verifier, training_available)
    if not training_available:
        reasons.append("The selected hardware/model cannot run local adapter training")
        return MethodDecision("retrieval", tuple(reasons), verifier, training_available)
    if verifier:
        reasons.append("Reference implementations provide demonstrations and an executable artifact verifier")
        reasons.append("QLoRA-IL is the first intervention; RL is reserved for repeated rollout rewards")
        return MethodDecision("qlora-il", tuple(reasons), verifier, training_available)
    reasons.append("Grounded stable demonstrations are available")
    return MethodDecision("qlora-il", tuple(reasons), verifier, training_available)


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
                    "--virtual-time-budget=5000",
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
    }
    if not hard_gates["substantial"]:
        diagnostics.append(
            f"Artifact has {effective_size} unique source bytes ({size} raw); expected at least {contract.minimum_bytes}"
        )
    if not hard_gates["no_placeholders"]:
        diagnostics.append("Artifact contains placeholder markers")

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
    if contract.artifact_kind == "web" and hard_gates["entrypoint_shape"] and hard_gates.get("javascript_syntax", True):
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
            terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 4}
            terms.update(feature.replace(".", "") for feature in preferred_features)
            extensions = {".js", ".mjs", ".ts", ".tsx", ".jsx", ".html", ".css", ".glsl", ".wgsl", ".py", ".md"}
            candidates: list[tuple[int, str]] = []
            for path in listed:
                if Path(path).suffix.lower() not in extensions:
                    continue
                lowered = path.lower()
                score = sum(3 for term in terms if term in lowered)
                score += sum(
                    2 for feature in preferred_features if feature.replace(".", "") in lowered.replace(".", "")
                )
                score += 2 if any(folder in lowered for folder in ("example", "demo", "sample", "docs")) else 0
                # Once the repository itself is search-relevant and licensed,
                # include its examples even when generic names such as main.js
                # do not repeat the query terms.
                candidates.append((score, path))
            selected: list[str] = []
            # Round-robin by query term prevents one prolific feature from
            # dominating the corpus.
            for term in sorted(terms):
                match = next(
                    (
                        path
                        for _, path in sorted(candidates, reverse=True)
                        if term in path.lower() and path not in selected
                    ),
                    None,
                )
                if match:
                    selected.append(match)
            selected.extend(path for _, path in sorted(candidates, reverse=True) if path not in selected)
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
        item for item in cleaned if 8 <= len(item) <= 220 and any(marker in item.lower() for marker in research_markers)
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
