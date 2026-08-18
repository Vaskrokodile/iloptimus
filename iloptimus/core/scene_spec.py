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
    "sceneType": "island",
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

# Supported scene types and their descriptions for the model prompt.
SCENE_TYPES = {
    "island": "a voxel island with water, trees, and falling petals (sakura/cherry blossom theme)",
    "sakura": "a cherry blossom voxel island with sakura trees, water, and falling petals",
    "desert": "a deserted island with a large mountain, sand colors, and sparse vegetation",
    "city": "a city grid with buildings of varying heights, streets, and window lights",
    "paris": "Paris with an Eiffel Tower, surrounding Haussmann buildings, and the Seine river",
    "sky_island": "a floating sky island with a Chinese pagoda building, clouds, and floating rocks",
}

# Scene-type-specific optional fields.
SCENE_TYPE_FIELDS = {
    "island": set(),
    "sakura": set(),
    "desert": set(),
    "city": {"buildings"},
    "paris": {"buildings"},
    "sky_island": {"floatHeight", "buildings"},
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


# Keywords that map a natural-language request to a scene type.
_SCENE_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sakura": ("sakura", "cherry blossom", "cherry tree", "blossom island"),
    "desert": ("desert", "deserted island", "mountain island", "sand island", "canyon"),
    "city": ("new york", "nyc", "city", "skyscraper", "manhattan", "urban", "downtown"),
    "paris": ("paris", "eiffel", "eiffel tower", "seine", "arc de triomphe"),
    "sky_island": ("sky island", "floating island", "cloud island", "chinese building", "pagoda", "temple in the sky"),
    "island": ("island", "voxel island", "tropical"),
}


def detect_scene_type(request: str) -> str:
    """Infer the scene type from the natural-language request text."""
    lowered = request.casefold()
    # Check specific types before the generic "island" fallback.
    for scene_type in ("sakura", "desert", "city", "paris", "sky_island", "island"):
        for keyword in _SCENE_TYPE_KEYWORDS.get(scene_type, ()):
            if keyword in lowered:
                return scene_type
    return "island"


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
    detected_type = detect_scene_type(request)
    type_description = SCENE_TYPES.get(detected_type, SCENE_TYPES["island"])
    type_hint = (
        f" The scene type is '{detected_type}' ({type_description})."
        if detected_type != "island"
        else ""
    )
    # Add scene-type-specific field instructions.
    extra_fields = ""
    if detected_type in ("city", "paris"):
        extra_fields = (
            " For buildings, include a 'buildings' array of 4-12 objects, each with numeric "
            "x, z (position), w or width, d or depth, h or height (all 2-30), and optional "
            "hue (0-1), saturation (0-1), lightness (0-1) for color."
        )
    elif detected_type == "sky_island":
        extra_fields = (
            " Include floatHeight (integer 8-30, the height the island floats at)."
            " You may also include a 'buildings' array but the pagoda is auto-generated."
        )
    return (
        f"Design this scene for a trusted Three.js voxel-world engine: {request}\n"
        f"Return one JSON object with exactly this typed contract, but invent every value yourself:"
        f" sceneType (string, one of: {', '.join(SCENE_TYPES)});"
        + type_hint
        + " title (request-specific string); sky, fog, waterDeep, waterShallow, blossom (original six-digit CSS hex colors); "
        "terrainRadius (integer 8-24); terrainHeight (integer 3-12); waterSize (integer 50-220); "
        "petalCount (integer 120-1200); camera (three numbers); trees (three or more distinct objects, each with "
        "numeric x, z, scale); details (three or more request-specific strings); motion (object with numeric "
        "waterSpeed 0.1-3, petalFallSpeed 0.1-3, cameraOrbitSpeed 0.005-0.2)."
        + extra_fields
        + " Use exactly these keys and no placeholders. Match the requested subject in the title, palette, and details."
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
            "sceneType",
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
    # sceneType is required and must be a known type.
    scene_type = str(spec.get("sceneType") or "")
    if scene_type not in SCENE_TYPES:
        diagnostics.append(f"sceneType must be one of: {', '.join(SCENE_TYPES)}")
    # Required fields are the base schema fields (sceneType is checked above).
    required = {key for key in SCENE_SCHEMA_EXAMPLE if key != "sceneType"}
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
    # Validate scene-type-specific optional fields.
    if scene_type in SCENE_TYPES:
        type_fields = SCENE_TYPE_FIELDS.get(scene_type, set())
        if "buildings" in type_fields:
            buildings = spec.get("buildings")
            if buildings is not None:
                if not isinstance(buildings, list) or len(buildings) < 3:
                    diagnostics.append("buildings must contain at least three entries")
                else:
                    for b in buildings[:20]:
                        if not isinstance(b, dict):
                            diagnostics.append("each building must be an object")
                            break
        if "floatHeight" in type_fields:
            fh = spec.get("floatHeight")
            if fh is not None and (not isinstance(fh, (int, float)) or not 8 <= fh <= 30):
                diagnostics.append("floatHeight must be between 8 and 30")
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
    # Relax the "island" subject check for non-island scene types.
    if scene_type in ("island", "sakura") and "island" in requested and "island" not in subject:
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

    # Inject sceneType if the model didn't provide it — infer from the request.
    if not candidate.get("sceneType") or candidate.get("sceneType") not in SCENE_TYPES:
        detected = detect_scene_type(request)
        candidate["sceneType"] = detected
        default_fields.append("sceneType")

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
    for term in ("sakura", "cherry", "island", "voxel", "shader", "paris", "eiffel", "city", "desert", "mountain", "sky", "pagoda", "chinese", "new york", "nyc"):
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
    safe: dict[str, Any] = {
        "title": str(spec["title"]),
        "sceneType": str(spec.get("sceneType") or "island"),
        **{key: str(spec[key]).lower() for key in ("sky", "fog", "waterDeep", "waterShallow", "blossom")},
        "terrainRadius": terrain_radius,
        **{key: float(spec[key]) for key in ("terrainHeight", "waterSize")},
        "petalCount": int(spec["petalCount"]),
        "camera": camera,
        "trees": trees,
        "details": [str(item)[:60] for item in spec["details"][:8]],
        "motion": {key: float(spec["motion"][key]) for key in ("waterSpeed", "petalFallSpeed", "cameraOrbitSpeed")},
    }
    # Pass through scene-type-specific optional fields.
    scene_type = safe["sceneType"]
    if scene_type in ("city", "paris") and isinstance(spec.get("buildings"), list):
        safe["buildings"] = [
            {
                "x": float(b.get("x", 0)),
                "z": float(b.get("z", 0)),
                "width": float(b.get("width", b.get("w", 4))),
                "depth": float(b.get("depth", b.get("d", 4))),
                "height": float(b.get("height", b.get("h", 8))),
                "hue": float(b.get("hue", 0.55)),
                "saturation": float(b.get("saturation", 0.08)),
                "lightness": float(b.get("lightness", 0.18)),
            }
            for b in spec["buildings"][:20]
            if isinstance(b, dict)
        ]
    if scene_type == "sky_island" and spec.get("floatHeight") is not None:
        safe["floatHeight"] = float(spec["floatHeight"])
    return safe


def compile_scene_spec(spec: dict[str, Any], destination: Path, request: str = "") -> dict[str, Any]:
    audit = audit_scene_spec(spec, request)
    if not audit.passed:
        raise ValueError("Invalid scene specification: " + "; ".join(audit.diagnostics))
    safe = _safe_spec(spec)
    scene_type = safe["sceneType"]
    # Use the multi-scene template for all scene types. It has a built-in
    # dispatcher that reads sceneSpec.sceneType and builds the right terrain.
    template = Path(__file__).parent.parent / "resources" / "artifact-frameworks" / "threejs-multi.html"
    source = template.read_text(encoding="utf-8").replace("__TITLE__", safe["title"])
    encoded = json.dumps(safe, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    source = source.replace(
        "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';",
        "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';\n\n    const sceneSpec = " + encoded + ";",
    )
    # The multi-scene template reads sceneSpec at runtime, so we only need
    # to replace the static background/fog/camera colors that are set before
    # the sceneSpec is available.
    replacements = {
        "new THREE.Color(0x090d18)": "new THREE.Color(sceneSpec.sky)",
        "new THREE.FogExp2(0x10182a, 0.018)": "new THREE.FogExp2(sceneSpec.fog, 0.018)",
        "camera.position.set(25, 21, 30)": "camera.position.set(...sceneSpec.camera)",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    spec_bytes = json.dumps(safe, sort_keys=True).encode()
    manifest = {
        "version": 2,
        "authorship": "local-model-scene-spec",
        "fallback_used": False,
        "runtime": "trusted-multi-scene-threejs-engine",
        "scene_type": scene_type,
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
    if not default_fields.issubset({"motion", "sceneType"}):
        errors.append("Framework supplied fields beyond the allowed generic motion/sceneType default")
    # Allow camera/trees (bounded coordinate normalization), sceneType (inferred
    # from request), and color fields (lowercased hex normalization — not semantic).
    normalized_fields = set(manifest.get("compiler_normalized_fields") or [])
    allowed_normalizations = {"camera", "trees", "sceneType", "sky", "fog", "waterDeep", "waterShallow", "blossom"}
    if not normalized_fields.issubset(allowed_normalizations):
        errors.append("Compiler changed semantic design fields rather than bounded coordinates")
    return errors
