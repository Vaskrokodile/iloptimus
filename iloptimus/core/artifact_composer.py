"""Model-authored component generation for reliable local web artifacts.

The harness owns only imports, ordering, and verification. Substantive scene
logic remains attributable to bounded model completions recorded in a manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ArtifactComponent:
    id: str
    function_name: str
    objective: str
    required_patterns: tuple[str, ...]
    maximum_tokens: int = 768
    minimum_bytes: int = 420

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentAudit:
    passed: bool
    diagnostics: tuple[str, ...]
    bytes: int
    sha256: str

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComposedGeneration:
    tokens_generated: int
    elapsed: float
    tokens_per_sec: float
    manifest: dict[str, Any]
    attempts: dict[str, int]


def threejs_component_plan() -> tuple[ArtifactComponent, ...]:
    """Return stable interfaces for model-owned Three.js scene components."""
    return (
        ArtifactComponent(
            "world",
            "initializeWorld",
            (
                "Initialize world.scene, world.camera, world.renderer, world.controls, lighting, fog, and a real "
                "resize handler. Append the renderer canvas to #app. Use THREE.PerspectiveCamera, THREE.WebGLRenderer, "
                "OrbitControls, devicePixelRatio, innerWidth, and innerHeight."
            ),
            (
                r"THREE\.Scene",
                r"THREE\.PerspectiveCamera",
                r"THREE\.WebGLRenderer",
                r"OrbitControls",
                r"addEventListener\s*\(\s*['\"]resize",
                r"devicePixelRatio",
            ),
            720,
            650,
        ),
        ArtifactComponent(
            "terrain",
            "createVoxelIsland",
            (
                "Build a substantial rounded voxel island from many boxes or an InstancedMesh. Give it visible height "
                "variation, grass/stone/sand colors, a shoreline silhouette, and store it on world."
            ),
            (r"InstancedMesh", r"BoxGeometry", r"setMatrixAt", r"(?:terrain|island)", r"\bfor\s*\("),
            820,
            850,
        ),
        ArtifactComponent(
            "water",
            "createAnimatedWater",
            (
                "Create a large water surface around the island using THREE.ShaderMaterial with explicit vertexShader "
                "and fragmentShader strings plus a time uniform. Store the mesh and uniforms on world."
            ),
            (r"ShaderMaterial", r"vertexShader", r"fragmentShader", r"uniforms", r"(?:uTime|time)"),
            820,
            750,
        ),
        ArtifactComponent(
            "sakura",
            "createSakuraSystem",
            (
                "Create multiple recognizable cherry trees with trunks and pink blossom crowns, plus a falling-petal "
                "particle system using THREE.Points or InstancedMesh. Store petals and their original positions on world."
            ),
            (r"(?:sakura|cherry|blossom)", r"(?:petal|Points)", r"BufferGeometry", r"THREE\.(?:Mesh|Points)"),
            900,
            900,
        ),
        ArtifactComponent(
            "details",
            "createWorldDetails",
            (
                "Polish the scene with several model-authored details such as rocks, lanterns, a bridge, stepping stones, "
                "clouds, or distant decorative geometry. Store any animated detail state on world."
            ),
            (r"THREE\.(?:Mesh|Group)", r"(?:rock|lantern|bridge|stone|cloud|detail)", r"scene\.add|world\.scene\.add"),
            720,
            600,
        ),
        ArtifactComponent(
            "animation",
            "startExperience",
            (
                "Start one requestAnimationFrame loop. Advance the water time uniform, animate falling petals with wraparound, "
                "add subtle camera motion without overriding OrbitControls, call controls.update, and render every frame."
            ),
            (
                r"requestAnimationFrame",
                r"renderer\.render",
                r"controls\.update",
                r"(?:uTime|uniforms)",
                r"(?:petal|position)",
            ),
            780,
            700,
        ),
    )


def component_prompt(
    user_request: str,
    component: ArtifactComponent,
    *,
    failure_guardrails: str = "",
    verifier_feedback: Iterable[str] = (),
) -> str:
    feedback = "\n".join(f"- {item}" for item in verifier_feedback)
    return (
        f"We are building this requested artifact: {user_request}\n\n"
        f"Write exactly one JavaScript function named {component.function_name}(world).\n"
        f"Component contract: {component.objective}\n"
        "Assume THREE and OrbitControls are already imported and `world` is a shared plain object. "
        "Do not import modules, write HTML, call the function, redefine another component, use markdown fences, "
        "or include prose. Return one complete function and stop after its closing brace. Use concrete executable code, "
        "bounded loops, and unique purposeful statements; never pad or repeat blocks."
        + (f"\nVerifier-derived guardrails:\n{failure_guardrails}" if failure_guardrails else "")
        + (f"\nRepair these objective failures:\n{feedback}" if feedback else "")
    )


def balanced_function_end(text: str, function_name: str) -> tuple[int, int] | None:
    """Locate the first complete named function while respecting JS strings/comments."""
    match = re.search(rf"\bfunction\s+{re.escape(function_name)}\s*\(\s*world\s*\)\s*\{{", text)
    if not match:
        return None
    start = match.start()
    brace_start = text.find("{", match.start())
    depth = 0
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    index = brace_start
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
        elif block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char == "/" and next_char == "/":
            line_comment = True
            index += 1
        elif char == "/" and next_char == "*":
            block_comment = True
            index += 1
        elif char in {'"', "'", "`"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
        index += 1
    return None


def clean_component_source(text: str, component: ArtifactComponent) -> str:
    source = text.strip()
    fenced = re.search(r"```(?:javascript|js)?\s*([\s\S]*?)```", source, re.I)
    if fenced:
        source = fenced.group(1).strip()
    source = re.sub(r"^\s*(?:Here(?:'s| is)[^\n]*|JavaScript:)\s*\n", "", source, flags=re.I)
    bounds = balanced_function_end(source, component.function_name)
    return source[bounds[0] : bounds[1]].strip() if bounds else source.strip()


def audit_component(source: str, component: ArtifactComponent) -> ComponentAudit:
    diagnostics: list[str] = []
    encoded = source.encode()
    if len(encoded) < component.minimum_bytes:
        diagnostics.append(f"Component has {len(encoded)} bytes; expected at least {component.minimum_bytes}")
    if not re.search(rf"\bfunction\s+{re.escape(component.function_name)}\s*\(\s*world\s*\)", source):
        diagnostics.append(f"Missing exact function {component.function_name}(world)")
    if re.search(r"```|<html|<script|\b(?:TODO|placeholder)\b", source, re.I):
        diagnostics.append("Component contains markup, fences, or placeholder text")
    for pattern in component.required_patterns:
        if not re.search(pattern, source, re.I):
            diagnostics.append(f"Missing observable pattern: {pattern}")
    try:
        with tempfile.TemporaryDirectory(prefix="iloptimus-component-") as temporary:
            path = Path(temporary) / "component.mjs"
            path.write_text(source, encoding="utf-8")
            checked = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=8)
        if checked.returncode:
            diagnostics.append("JavaScript syntax: " + (checked.stderr or checked.stdout).strip()[-600:])
    except (OSError, subprocess.SubprocessError) as error:
        diagnostics.append(f"JavaScript syntax check unavailable: {error}")
    return ComponentAudit(
        passed=not diagnostics,
        diagnostics=tuple(diagnostics),
        bytes=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def assemble_threejs_artifact(components: dict[str, str]) -> str:
    plan = threejs_component_plan()
    missing = [component.id for component in plan if not components.get(component.id, "").strip()]
    if missing:
        raise ValueError("Missing model-authored components: " + ", ".join(missing))
    functions = "\n\n".join(components[component.id].strip() for component in plan)
    calls = "\n".join(f"{component.function_name}(world);" for component in plan)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Model-authored Three.js experience</title>
  <style>
    html,body,#app{{width:100%;height:100%;margin:0;overflow:hidden;background:#08111d}}
    canvas{{display:block;width:100%;height:100%}}
    #credit{{position:fixed;left:16px;bottom:14px;padding:8px 11px;border:1px solid #ffffff2b;border-radius:10px;
      color:#f7dce8;background:#08111db8;font:12px/1.25 ui-sans-serif,system-ui;backdrop-filter:blur(8px);z-index:2}}
  </style>
  <script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"}}}}</script>
</head>
<body>
  <main id="app" aria-label="Interactive Three.js scene"></main>
  <div id="credit">Drag to orbit · Scroll to zoom</div>
  <script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const world = {{}};

{functions}

{calls}
  </script>
</body>
</html>
"""


def authorship_manifest(
    destination: Path,
    components: dict[str, str],
    audits: dict[str, ComponentAudit],
    *,
    model_id: str,
    adapter_path: str = "",
) -> dict[str, Any]:
    artifact = destination.read_text(encoding="utf-8")
    model_bytes = sum(len(source.encode()) for source in components.values())
    manifest = {
        "version": 1,
        "authorship": "local-model-components",
        "fallback_used": False,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "artifact_sha256": hashlib.sha256(artifact.encode()).hexdigest(),
        "artifact_bytes": len(artifact.encode()),
        "model_authored_bytes": model_bytes,
        "model_authored_ratio": round(model_bytes / max(1, len(artifact.encode())), 4),
        "components": {
            component_id: {
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "bytes": len(source.encode()),
                "audit": audits[component_id].public(),
            }
            for component_id, source in components.items()
        },
    }
    destination.with_suffix(destination.suffix + ".authorship.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def audit_model_authorship(manifest: dict[str, Any], *, minimum_ratio: float = 0.65) -> list[str]:
    errors: list[str] = []
    if manifest.get("authorship") != "local-model-components" or manifest.get("fallback_used") is not False:
        errors.append("Artifact is not exclusively attributed to model-authored components")
    if float(manifest.get("model_authored_ratio") or 0.0) < minimum_ratio:
        errors.append("Model-authored source ratio is below the required threshold")
    components = dict(manifest.get("components") or {})
    required = {component.id for component in threejs_component_plan()}
    if set(components) != required:
        errors.append("Authorship manifest does not cover every required component")
    if any(not item.get("audit", {}).get("passed") for item in components.values()):
        errors.append("At least one attributed model component failed its contract")
    return errors
