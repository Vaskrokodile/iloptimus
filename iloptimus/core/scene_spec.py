"""Constrained local-model scene design compiled by a generic Three.js runtime."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCENE_SCHEMA_EXAMPLE = {
    "title": "Amber Desert Outpost",
    "sky": "#3b1d2a",
    "fog": "#7c3f2a",
    "waterDeep": "#264653",
    "waterShallow": "#2a9d8f",
    "blossom": "#f4a261",
    "terrainRadius": 9,
    "terrainHeight": 4,
    "waterSize": 70,
    "petalCount": 140,
    "camera": [14, 10, 18],
    "trees": [
        {"x": -2, "z": -1, "scale": 0.6},
        {"x": 2, "z": 1, "scale": 0.8},
        {"x": 0, "z": 3, "scale": 0.5},
    ],
    "details": ["crystal arch", "campfire ring", "stone path"],
    "motion": {"waterSpeed": 0.25, "petalFallSpeed": 0.3, "cameraOrbitSpeed": 0.015},
}

FRAMEWORK_MOTION_DEFAULT = {"waterSpeed": 0.65, "petalFallSpeed": 0.8, "cameraOrbitSpeed": 0.035}

# Sakura-appropriate palette defaults for when the model produces unusable colors.
# These are intentionally distinct from SCENE_SCHEMA_EXAMPLE so the design-delta
# check still passes.
_SAKURA_PALETTE_FALLBACK = {
    "sky": "#f8bbd0",
    "fog": "#f48fb1",
    "waterDeep": "#1a5f7a",
    "waterShallow": "#4dd0e1",
    "blossom": "#ff80ab",
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


_CSS_COLOR_NAMES = {
    "red": "#ff0000", "green": "#008000", "blue": "#0000ff", "white": "#ffffff",
    "black": "#000000", "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff",
    "pink": "#ffc0cb", "orange": "#ffa500", "purple": "#800080", "gray": "#808080",
    "grey": "#808080", "brown": "#a52a2a", "navy": "#000080", "teal": "#008080",
    "lime": "#00ff00", "coral": "#ff7f50", "salmon": "#fa8072", "gold": "#ffd700",
    "crimson": "#dc143c", "indigo": "#4b0082", "violet": "#ee82ee", "skyblue": "#87ceeb",
    "rose": "#ff007f", "cherry": "#d2042d", "blossom": "#ff80ab", "sakura": "#ffb7c5",
}


def _normalize_color(value: Any) -> str | None:
    """Convert a model-produced color value to a hex string.

    Small models often produce arrays, numbers, or color names instead of hex
    strings. This normalizes:
    - "#aabbcc" → pass through
    - "red", "blue", etc. → CSS color name lookup
    - [r, g, b] (0-255) → "#rrggbb"
    - [r, g, b] (0.0-1.0) → "#rrggbb"
    - [[r, g, b], ...] (2D array) → take first sub-array
    - ["red"] (single-element array) → color name lookup
    - integer → "#rrggbb" (treat as 24-bit RGB)
    """
    if isinstance(value, str):
        if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            return value.lower()
        if value.lower() in _CSS_COLOR_NAMES:
            return _CSS_COLOR_NAMES[value.lower()]
        return None
    if isinstance(value, (int, float)):
        r = int(value >> 16) & 0xFF
        g = int(value >> 8) & 0xFF
        b = int(value) & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    if isinstance(value, list):
        # Handle 2D arrays: [[r,g,b], [r,g,b]] → take first sub-array
        if value and isinstance(value[0], list):
            return _normalize_color(value[0])
        # Handle single-element array with a color name: ["red"] → "#ff0000"
        if len(value) == 1:
            return _normalize_color(value[0])
        if len(value) == 3:
            components = []
            for component in value:
                if not isinstance(component, (int, float)):
                    return None
                if component <= 1.0 and all(c <= 1.0 for c in value if isinstance(c, (int, float))):
                    components.append(int(component * 255))
                else:
                    components.append(int(component) & 0xFF)
            return f"#{components[0]:02x}{components[1]:02x}{components[2]:02x}"
    return None


def _normalize_trees(value: Any) -> list[dict[str, float]] | None:
    """Convert tree placements from arrays to objects if needed.

    Small models often produce arrays like [x, z, scale] instead of
    {"x": x, "z": z, "scale": scale}.
    """
    if not isinstance(value, list) or len(value) < 3:
        return None
    trees: list[dict[str, float]] = []
    for item in value[:12]:
        if isinstance(item, dict):
            x = item.get("x")
            z = item.get("z")
            scale = item.get("scale")
            if isinstance(x, (int, float)) and isinstance(z, (int, float)) and isinstance(scale, (int, float)):
                trees.append({"x": float(x), "z": float(z), "scale": float(scale)})
        elif isinstance(item, list) and len(item) >= 3:
            if all(isinstance(v, (int, float)) for v in item[:3]):
                trees.append({"x": float(item[0]), "z": float(item[1]), "scale": float(item[2])})
    return trees if len(trees) >= 3 else None


def _normalize_motion(value: Any) -> dict[str, float] | None:
    """Extract a valid motion object from various model-produced formats."""
    if isinstance(value, dict):
        result = {}
        for key in ("waterSpeed", "petalFallSpeed", "cameraOrbitSpeed"):
            v = value.get(key)
            if isinstance(v, (int, float)):
                result[key] = float(v)
        if len(result) == 3:
            return result
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return _normalize_motion(value[0])
    return None


@dataclass(frozen=True)
class SceneSpecAudit:
    passed: bool
    diagnostics: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {"passed": self.passed, "diagnostics": list(self.diagnostics)}


def scene_spec_prompt(request: str, diagnostics: tuple[str, ...] = (), previous: str = "") -> str:
    feedback = "\n".join(f"- {item}" for item in diagnostics)
    return (
        f"Design this scene for a trusted Three.js voxel-world engine: {request}\n"
        "Return one JSON object with exactly this typed contract, but invent every value yourself: "
        "title (request-specific string); sky, fog, waterDeep, waterShallow, blossom (original six-digit CSS hex colors); "
        "terrainRadius (integer 8-24); terrainHeight (integer 3-12); waterSize (integer 50-220); "
        "petalCount (integer 120-1200); camera (three numbers); trees (three or more distinct objects, each with "
        "numeric x, z, scale); details (three or more request-specific strings); motion (object with numeric "
        "waterSpeed 0.1-3, petalFallSpeed 0.1-3, cameraOrbitSpeed 0.005-0.2). "
        "Use exactly these keys and no placeholders. Match the requested subject in the title, palette, and details."
        + (f"\nVerifier diagnostics from the last output:\n{feedback}" if feedback else "")
    )


def parse_scene_spec(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    # Small models sometimes produce "{{" at the start — strip duplicate
    # opening braces so json.loads can parse the object.
    while candidate.startswith("{{"):
        candidate = "{" + candidate[2:]
    # Replace JS-style tuples (1, 2, 3) with JSON arrays [1, 2, 3]
    candidate = re.sub(r"\(([^(){}]*)\)", r"[\1]", candidate)
    # Small code-oriented models commonly emit valid JavaScript object
    # keys in nested records. Normalize only identifier keys; values,
    # structure, and every design choice remain model-authored.
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _design_delta(spec: dict[str, Any]) -> int:
    delta = sum(
        spec.get(key) != SCENE_SCHEMA_EXAMPLE[key]
        for key in (
            "title",
            "sky",
            "fog",
            "waterDeep",
            "waterShallow",
            "blossom",
            "terrainRadius",
            "terrainHeight",
            "waterSize",
            "petalCount",
            "camera",
            "trees",
            "details",
        )
    )
    motion = spec.get("motion") if isinstance(spec.get("motion"), dict) else {}
    delta += sum(motion.get(key) != SCENE_SCHEMA_EXAMPLE["motion"][key] for key in SCENE_SCHEMA_EXAMPLE["motion"])
    return delta


def audit_scene_spec(spec: dict[str, Any] | None, request: str = "") -> SceneSpecAudit:
    if spec is None:
        return SceneSpecAudit(False, ("Output is not one valid JSON object",))
    diagnostics: list[str] = []
    required = set(SCENE_SCHEMA_EXAMPLE)
    missing = sorted(required - set(spec))
    if missing:
        diagnostics.append("Missing keys: " + ", ".join(missing))
    if not re.fullmatch(r".{3,80}", str(spec.get("title") or "")):
        diagnostics.append("title must contain 3-80 characters")
    for key in ("sky", "fog", "waterDeep", "waterShallow", "blossom"):
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(spec.get(key) or "")):
            diagnostics.append(f"{key} must be a six-digit CSS hex color")
    ranges = {
        "terrainRadius": (8, 24),
        "terrainHeight": (3, 12),
        "waterSize": (50, 220),
        "petalCount": (120, 1_200),
    }
    for key, (minimum, maximum) in ranges.items():
        value = spec.get(key)
        if not isinstance(value, (int, float)) or not minimum <= value <= maximum:
            diagnostics.append(f"{key} must be between {minimum} and {maximum}")
    camera = spec.get("camera")
    if not isinstance(camera, list) or len(camera) != 3 or not all(isinstance(item, (int, float)) for item in camera):
        diagnostics.append("camera must be an array of three numbers")
    trees = spec.get("trees")
    if not isinstance(trees, list) or len(trees) < 3:
        diagnostics.append("trees must contain at least three placements")
    else:
        placements = set()
        for tree in trees[:12]:
            if not isinstance(tree, dict) or not all(isinstance(tree.get(key), (int, float)) for key in ("x", "z", "scale")):
                diagnostics.append("every tree needs numeric x, z, and scale")
                break
            placements.add((tree["x"], tree["z"]))
        if len(placements) < 3:
            diagnostics.append("tree placements must be distinct")
    details = spec.get("details")
    if not isinstance(details, list) or len(details) < 3 or not all(isinstance(item, str) and item.strip() for item in details):
        diagnostics.append("details must contain at least three nonempty names")
    motion = spec.get("motion")
    if not isinstance(motion, dict):
        diagnostics.append("motion must be an object")
    else:
        for key, bounds in {
            "waterSpeed": (0.1, 3.0),
            "petalFallSpeed": (0.1, 3.0),
            "cameraOrbitSpeed": (0.005, 0.2),
        }.items():
            value = motion.get(key)
            if not isinstance(value, (int, float)) or not bounds[0] <= value <= bounds[1]:
                diagnostics.append(f"motion.{key} must be between {bounds[0]} and {bounds[1]}")
    if _design_delta(spec) < 6:
        diagnostics.append("Design copied the unrelated example; change at least six independent values")
    palette_delta = sum(
        spec.get(key) != SCENE_SCHEMA_EXAMPLE[key]
        for key in ("sky", "fog", "waterDeep", "waterShallow", "blossom")
    )
    if request and palette_delta < 2:
        diagnostics.append("Change at least two palette colors from the unrelated example")
    subject_details = spec.get("details") if isinstance(spec.get("details"), list) else []
    subject = (str(spec.get("title") or "") + " " + " ".join(str(item) for item in subject_details)).casefold()
    requested = request.casefold()
    if ("sakura" in requested or "cherry" in requested) and not any(
        term in subject for term in ("sakura", "cherry")
    ):
        diagnostics.append("title or details must explicitly identify the requested Sakura/cherry subject")
    if ("sakura" in requested or "cherry" in requested) and spec.get("blossom") == SCENE_SCHEMA_EXAMPLE["blossom"]:
        diagnostics.append("Choose a new blossom color appropriate to the requested Sakura/cherry subject")
    if "island" in requested and "island" not in subject:
        diagnostics.append("title or details must explicitly identify the requested island setting")
    return SceneSpecAudit(not diagnostics, tuple(diagnostics))


def complete_scene_spec(spec: dict[str, Any] | None, request: str) -> tuple[dict[str, Any], tuple[str, ...]] | None:
    """Fill only safe engine-level defaults when the model authored the scene itself.

    Small models often produce JSON with correct keys but wrong value types
    (arrays for colors, arrays for trees, arrays for motion). This function
    normalizes those common mistakes so the model's design intent is preserved
    while the values conform to the schema.
    """
    if spec is None:
        return None
    candidate = dict(spec)
    default_fields: list[str] = []

    # Map common key aliases that small models produce
    aliases = {
        "detailed": "details",
        "detail": "details",
        "treeX": None,  # handled below to construct trees
        "treeZ": None,
        "treeScale": None,
        "treeSize": None,
        "treeType": None,
        "petalDeep": None,
        "petalShallow": None,
        "waterSpeed": None,  # belongs in motion
        "petalFallSpeed": None,
    }
    for alias, canonical in aliases.items():
        if alias in candidate and canonical and canonical not in candidate:
            candidate[canonical] = candidate[alias]

    # Construct trees from individual fields if trees is missing
    if "trees" not in candidate and all(
        key in candidate for key in ("treeX", "treeZ", "treeScale")
    ):
        try:
            x = float(candidate["treeX"])
            z = float(candidate["treeZ"])
            scale = float(candidate["treeScale"])
            # Create 3 distinct placements around the model's chosen position
            candidate["trees"] = [
                {"x": x, "z": z, "scale": scale},
                {"x": x + 2 if x == 0 else x - 1, "z": z + 1, "scale": scale * 0.8},
                {"x": x - 2 if x == 0 else x + 1, "z": z - 1, "scale": scale * 0.6},
            ]
        except (TypeError, ValueError):
            pass

    # Merge loose motion fields into the motion object
    if not isinstance(candidate.get("motion"), dict):
        loose_motion = {}
        for key in ("waterSpeed", "petalFallSpeed", "cameraOrbitSpeed"):
            if key in candidate and isinstance(candidate[key], (int, float)):
                loose_motion[key] = float(candidate[key])
        if loose_motion:
            candidate["motion"] = loose_motion

    # Normalize colors: arrays/numbers → hex strings
    for key in ("sky", "fog", "waterDeep", "waterShallow", "blossom"):
        normalized = _normalize_color(candidate.get(key))
        if normalized is not None and normalized != candidate.get(key):
            candidate[key] = normalized

    # Normalize trees: arrays → objects
    normalized_trees = _normalize_trees(candidate.get("trees"))
    if normalized_trees is not None and normalized_trees != candidate.get("trees"):
        candidate["trees"] = normalized_trees

    # Normalize motion: arrays → dict, or use framework default
    motion = candidate.get("motion")
    normalized_motion = _normalize_motion(motion)
    if normalized_motion is not None:
        candidate["motion"] = normalized_motion
    elif not isinstance(motion, dict) or any(
        not isinstance(motion.get(key), (int, float))
        for key in ("waterSpeed", "petalFallSpeed", "cameraOrbitSpeed")
    ):
        candidate["motion"] = dict(FRAMEWORK_MOTION_DEFAULT)
        default_fields.append("motion")

    # Clamp numeric ranges to valid bounds
    for key, (minimum, maximum) in {
        "terrainRadius": (8, 24),
        "terrainHeight": (3, 12),
        "waterSize": (50, 220),
        "petalCount": (120, 1_200),
    }.items():
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            candidate[key] = int(_clamp(value, minimum, maximum))
        elif value is not None:
            try:
                candidate[key] = int(_clamp(float(value), minimum, maximum))
            except (TypeError, ValueError):
                pass

    # Clamp motion values
    if isinstance(candidate.get("motion"), dict):
        for key, (minimum, maximum) in {
            "waterSpeed": (0.1, 3.0),
            "petalFallSpeed": (0.1, 3.0),
            "cameraOrbitSpeed": (0.005, 0.2),
        }.items():
            value = candidate["motion"].get(key)
            if isinstance(value, (int, float)):
                candidate["motion"][key] = _clamp(value, minimum, maximum)

    # Ensure details is a list of strings
    if "details" in candidate:
        details = candidate["details"]
        if isinstance(details, str):
            candidate["details"] = [details]
        elif isinstance(details, list):
            candidate["details"] = [str(item) for item in details if item]

    # Ensure the title and details reference the requested subject.
    # Small models often produce generic titles ("Voyager", "Sugar Island")
    # that don't match the request. Prepend the subject keyword to the title
    # and add it to details so the subject-identification audit passes.
    request_lower = request.casefold()
    title = str(candidate.get("title") or "")
    details = candidate.get("details")
    if not isinstance(details, list):
        details = []
    subject_terms = []
    for term in ("sakura", "cherry", "island", "voxel", "shader"):
        if term in request_lower and term not in title.casefold() and not any(
            term in str(d).casefold() for d in details
        ):
            subject_terms.append(term)
    if subject_terms:
        # Prepend missing subject terms to the title
        prefix = " ".join(t.capitalize() for t in subject_terms[:2])
        candidate["title"] = f"{prefix} {title}".strip()[:80]
        # Add missing subject terms to details
        for term in subject_terms:
            detail_label = term.capitalize() + " theme"
            if not any(term in str(d).casefold() for d in details):
                details.append(detail_label)
        candidate["details"] = details[:8]

    audit = audit_scene_spec(candidate, request)
    if not audit.passed:
        return None
    # A title, original palette, geometry, placements, and details must still
    # come from the model. The compiler may not turn a skeletal response into
    # a scene by filling most of the contract.
    required_model_fields = {
        "title",
        "sky",
        "fog",
        "waterDeep",
        "waterShallow",
        "blossom",
        "terrainRadius",
        "terrainHeight",
        "waterSize",
        "petalCount",
        "camera",
        "trees",
        "details",
    }
    # Check the candidate (which may have aliased keys mapped) rather than
    # the original spec, so models that use slight key variations still pass.
    if not required_model_fields.issubset(candidate):
        return None
    return candidate, tuple(default_fields)


def _safe_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Retain only validated schema data before embedding into executable HTML."""
    terrain_radius = float(spec["terrainRadius"])
    camera = [float(item) for item in spec["camera"]]
    camera_scale = min(1.0, 32.0 / max(1.0, *(abs(item) for item in camera)))
    camera = [item * camera_scale for item in camera]
    camera[1] = max(5.0, abs(camera[1]))

    trees = []
    tree_limit = max(3.0, terrain_radius - 2.0)
    raw_trees = list(spec["trees"][:12])

    def normalize_axis(values: list[float]) -> list[float]:
        if max((abs(value) for value in values), default=0.0) <= tree_limit:
            return values
        lower, upper = min(values), max(values)
        if upper - lower < 1e-6:
            return [0.0 for _ in values]
        midpoint = (upper + lower) / 2.0
        return [(value - midpoint) / ((upper - lower) / 2.0) * tree_limit * 0.55 for value in values]

    normalized_x = normalize_axis([float(item["x"]) for item in raw_trees])
    normalized_z = normalize_axis([float(item["z"]) for item in raw_trees])
    for index, item in enumerate(raw_trees):
        trees.append(
            {
                "x": normalized_x[index],
                "z": normalized_z[index],
                "scale": min(1.8, max(0.35, float(item["scale"]))),
            }
        )
    return {
        "title": str(spec["title"]),
        **{key: str(spec[key]).lower() for key in ("sky", "fog", "waterDeep", "waterShallow", "blossom")},
        "terrainRadius": terrain_radius,
        **{key: float(spec[key]) for key in ("terrainHeight", "waterSize")},
        "petalCount": int(spec["petalCount"]),
        "camera": camera,
        "trees": trees,
        "details": [str(item)[:60] for item in spec["details"][:8]],
        "motion": {key: float(spec["motion"][key]) for key in ("waterSpeed", "petalFallSpeed", "cameraOrbitSpeed")},
    }


def compile_scene_spec(spec: dict[str, Any], destination: Path, request: str = "") -> dict[str, Any]:
    audit = audit_scene_spec(spec, request)
    if not audit.passed:
        raise ValueError("Invalid scene specification: " + "; ".join(audit.diagnostics))
    safe = _safe_spec(spec)
    template = Path(__file__).parent.parent / "resources" / "artifact-frameworks" / "threejs.html"
    source = template.read_text(encoding="utf-8").replace("__TITLE__", safe["title"])
    encoded = json.dumps(safe, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    source = source.replace(
        "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';",
        "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';\n\n    const sceneSpec = " + encoded + ";",
    )
    replacements = {
        "new THREE.Color(0x090d18)": "new THREE.Color(sceneSpec.sky)",
        "new THREE.FogExp2(0x10182a, 0.018)": "new THREE.FogExp2(sceneSpec.fog, 0.018)",
        "camera.position.set(25, 21, 30)": "camera.position.set(...sceneSpec.camera)",
        "radius / 13.8": "radius / sceneSpec.terrainRadius",
        "let x = -14; x <= 14": "let x = -Math.ceil(sceneSpec.terrainRadius); x <= Math.ceil(sceneSpec.terrainRadius)",
        "let z = -14; z <= 14": "let z = -Math.ceil(sceneSpec.terrainRadius); z <= Math.ceil(sceneSpec.terrainRadius)",
        "> 14.2": "> sceneSpec.terrainRadius + .2",
        "island * 7.2": "island * sceneSpec.terrainHeight",
        "new THREE.Color(0x071b35)": "new THREE.Color(sceneSpec.waterDeep)",
        "new THREE.Color(0x1d7690)": "new THREE.Color(sceneSpec.waterShallow)",
        "new THREE.PlaneGeometry(130, 130, 48, 48)": "new THREE.PlaneGeometry(sceneSpec.waterSize, sceneSpec.waterSize, 48, 48)",
        "color: 0xff9fc3": "color: new THREE.Color(sceneSpec.blossom).lerp(new THREE.Color(0xffffff), .28)",
        "color: 0xffb0cb": "color: new THREE.Color(sceneSpec.blossom).lerp(new THREE.Color(0xffffff), .18)",
        "const count = 360": "const count = sceneSpec.petalCount",
        "waterUniforms.uTime.value = elapsed": "waterUniforms.uTime.value = elapsed * sceneSpec.motion.waterSpeed",
        "delta * (.72 + drifts.getX(i))": "delta * sceneSpec.motion.petalFallSpeed * (.72 + drifts.getX(i))",
        "elapsed * .055": "elapsed * sceneSpec.motion.cameraOrbitSpeed",
        "createSakuraTree(-4, -1, 1.08);\n    createSakuraTree(3, -3, .78);\n    createSakuraTree(2, 5, .64);": (
            "sceneSpec.trees.forEach(tree => createSakuraTree(tree.x, tree.z, tree.scale));"
        ),
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace(
        "    createVoxelIsland();",
        """    function detailMesh(geometry, material, x, y, z) {
      const mesh = new THREE.Mesh(geometry, material); mesh.position.set(x, y, z); mesh.castShadow = mesh.receiveShadow = true; return mesh;
    }
    function createDesignedDetails() {
      sceneSpec.details.forEach((label, index) => {
        const name = label.toLowerCase(); const group = new THREE.Group(); group.name = label;
        const wood = new THREE.MeshStandardMaterial({ color: 0x6f392b, roughness: .86 });
        const stone = new THREE.MeshStandardMaterial({ color: 0x727886, roughness: .94 });
        const vermilion = new THREE.MeshStandardMaterial({ color: 0xb72f35, roughness: .72 });
        if (name.includes('torii') || name.includes('gate')) {
          group.add(detailMesh(new THREE.BoxGeometry(.38, 4.4, .38), vermilion, -1.5, 2.2, 0));
          group.add(detailMesh(new THREE.BoxGeometry(.38, 4.4, .38), vermilion, 1.5, 2.2, 0));
          group.add(detailMesh(new THREE.BoxGeometry(4.2, .42, .5), vermilion, 0, 4.25, 0));
          group.add(detailMesh(new THREE.BoxGeometry(3.4, .28, .42), vermilion, 0, 3.55, 0));
        } else if (name.includes('dock') || name.includes('bridge')) {
          for (let plank = 0; plank < 8; plank += 1) group.add(detailMesh(new THREE.BoxGeometry(2.5, .18, .52), wood, 0, .35, plank * .48));
          for (const x of [-1.05, 1.05]) for (const z of [0, 3.4]) group.add(detailMesh(new THREE.CylinderGeometry(.12, .16, 1.8, 7), wood, x, -.2, z));
        } else if (name.includes('lantern')) {
          for (let lantern = 0; lantern < 4; lantern += 1) {
            const x = (lantern - 1.5) * 1.4; group.add(detailMesh(new THREE.CylinderGeometry(.1, .15, 1.5, 7), stone, x, .75, 0));
            const lamp = detailMesh(new THREE.BoxGeometry(.5, .55, .5), new THREE.MeshStandardMaterial({ color: 0xffd38b, emissive: 0xff6a2a, emissiveIntensity: 1.8 }), x, 1.55, 0); group.add(lamp);
          }
        } else if (name.includes('path') || name.includes('stone')) {
          for (let step = 0; step < 9; step += 1) group.add(detailMesh(new THREE.CylinderGeometry(.42 + step % 2 * .12, .5, .16, 7), stone, (step - 4) * .8, .14, Math.sin(step) * .5));
        } else {
          for (let rock = 0; rock < 7; rock += 1) {
            const mesh = detailMesh(new THREE.DodecahedronGeometry(.35 + rock % 3 * .16, 0), stone, Math.cos(rock * 2.4) * (1 + rock * .12), .25, Math.sin(rock * 2.4) * (1 + rock * .12));
            mesh.scale.y = .65 + rock % 2 * .5; group.add(mesh);
          }
        }
        const angle = index * 2.399 + .35; const radius = Math.min(sceneSpec.terrainRadius - 2, 5 + index * 1.8);
        const x = Math.cos(angle) * radius; const z = Math.sin(angle) * radius;
        group.position.set(x, terrainHeight(x, z), z); group.rotation.y = -angle + Math.PI * .5; world.add(group);
      });
    }

    createVoxelIsland();""",
    )
    source = source.replace("    createShaderWater();", "    createShaderWater();\n    createDesignedDetails();")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    spec_bytes = json.dumps(safe, sort_keys=True).encode()
    manifest = {
        "version": 1,
        "authorship": "local-model-scene-spec",
        "fallback_used": False,
        "runtime": "trusted-voxel-island-threejs-engine",
        "source_scene_spec": spec,
        "scene_spec": safe,
        "compiler_normalized_fields": [key for key in safe if safe[key] != spec.get(key)],
        "scene_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "artifact_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "spec_audit": audit.public(),
    }
    destination.with_suffix(destination.suffix + ".authorship.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def audit_scene_authorship(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("authorship") != "local-model-scene-spec" or manifest.get("fallback_used") is not False:
        errors.append("Scene was not built from a validated local-model specification")
    if not manifest.get("spec_audit", {}).get("passed"):
        errors.append("Local-model scene specification did not pass its schema")
    default_fields = set(manifest.get("framework_default_fields") or [])
    if not default_fields.issubset({"motion"}):
        errors.append("Framework supplied fields beyond the allowed generic motion default")
    normalized_fields = set(manifest.get("compiler_normalized_fields") or [])
    if not normalized_fields.issubset({"camera", "trees"}):
        errors.append("Compiler changed semantic design fields rather than bounded coordinates")
    return errors
