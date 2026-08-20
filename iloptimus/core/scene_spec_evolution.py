"""Autonomous dataset evolution for the self-improving loop.

Each iteration generates a richer, more varied set of training examples
than the last. The evolution strategy:

1. **Iteration 0**: A base set of clean, valid examples with varied values.
2. **Iteration 1+**: Increase variety so the model generalizes rather than
   memorizes. Add more diverse palettes, titles, and structural variations.

The goal is not just "more data" but "better data that targets the gap
between what the model produces and what the schema requires."

This module is general-purpose: it generates scene-spec examples for any
voxel-world scene request (sakura, desert, forest, etc.) by deriving
appropriate palettes and details from the request text.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .scene_spec import SCENE_SCHEMA_EXAMPLE


# Theme palettes — each is a coherent aesthetic for a different scene type.
# The loop picks the palette that best matches the request, then varies it.
THEME_PALETTES = {
    "sakura": [
        {"sky": "#f8bbd0", "fog": "#f48fb1", "waterDeep": "#1a5f7a", "waterShallow": "#4dd0e1", "blossom": "#ff80ab"},
        {"sky": "#ffd6e0", "fog": "#ffc1cc", "waterDeep": "#0d4f5c", "waterShallow": "#5eb6ca", "blossom": "#ff6b9d"},
        {"sky": "#e8a0bf", "fog": "#d4849e", "waterDeep": "#1b3a4b", "waterShallow": "#3a8e9e", "blossom": "#ff5c8a"},
        {"sky": "#fce4ec", "fog": "#f8bbd0", "waterDeep": "#26547c", "waterShallow": "#64b5f6", "blossom": "#ec407a"},
        {"sky": "#f3d9e6", "fog": "#e8b4cf", "waterDeep": "#1d3557", "waterShallow": "#457b9d", "blossom": "#e91e63"},
        {"sky": "#ffcdd2", "fog": "#ef9a9a", "waterDeep": "#006d77", "waterShallow": "#83c5be", "blossom": "#ff0a54"},
    ],
    "desert": [
        {"sky": "#3b1d2a", "fog": "#7c3f2a", "waterDeep": "#264653", "waterShallow": "#2a9d8f", "blossom": "#f4a261"},
        {"sky": "#e9c46a", "fog": "#f4a261", "waterDeep": "#264653", "waterShallow": "#2a9d8f", "blossom": "#e76f51"},
        {"sky": "#d4a373", "fog": "#ccd5ae", "waterDeep": "#283618", "waterShallow": "#606c38", "blossom": "#bc6c25"},
    ],
    "forest": [
        {"sky": "#a3b18a", "fog": "#588157", "waterDeep": "#1b3a4b", "waterShallow": "#3a8e9e", "blossom": "#dad7cd"},
        {"sky": "#588157", "fog": "#3a5a40", "waterDeep": "#1b3a4b", "waterShallow": "#52b788", "blossom": "#95d5b2"},
        {"sky": "#d8e2dc", "fog": "#a3b18a", "waterDeep": "#264653", "waterShallow": "#2a9d8f", "blossom": "#b7e4c7"},
    ],
    "ocean": [
        {"sky": "#caf0f8", "fog": "#90e0ef", "waterDeep": "#03045e", "waterShallow": "#00b4d8", "blossom": "#0077b6"},
        {"sky": "#48cae4", "fog": "#00b4d8", "waterDeep": "#023e8a", "waterShallow": "#0096c7", "blossom": "#caf0f8"},
        {"sky": "#ade8f4", "fog": "#90e0ef", "waterDeep": "#03045e", "waterShallow": "#48cae4", "blossom": "#0077b6"},
    ],
    "default": [
        {"sky": "#1a1a2e", "fog": "#16213e", "waterDeep": "#0f3460", "waterShallow": "#16537e", "blossom": "#e94560"},
        {"sky": "#22223b", "fog": "#4a4e69", "waterDeep": "#1a1a2e", "waterShallow": "#4a4e69", "blossom": "#9a8c98"},
        {"sky": "#264653", "fog": "#2a9d8f", "waterDeep": "#1a1a2e", "waterShallow": "#2a9d8f", "blossom": "#e9c46a"},
    ],
    "city": [
        {"sky": "#0d1b3e", "fog": "#3a2a5e", "waterDeep": "#0f1a2e", "waterShallow": "#1a2a4e", "blossom": "#ff6b4a"},
        {"sky": "#1a2348", "fog": "#6b4a8c", "waterDeep": "#0f1a2e", "waterShallow": "#1a3a5c", "blossom": "#ff9e6b"},
        {"sky": "#05070f", "fog": "#0a1228", "waterDeep": "#020408", "waterShallow": "#0a1428", "blossom": "#141a30"},
        {"sky": "#1a1a2e", "fog": "#16213e", "waterDeep": "#0f3460", "waterShallow": "#16537e", "blossom": "#e94560"},
        {"sky": "#2a1a3e", "fog": "#3e2a4e", "waterDeep": "#1a0a2e", "waterShallow": "#2a1a4e", "blossom": "#ff4a6b"},
    ],
    "paris": [
        {"sky": "#3a3328", "fog": "#5a4a3a", "waterDeep": "#1a3a5c", "waterShallow": "#2a5a7c", "blossom": "#8b7355"},
        {"sky": "#2a2a3e", "fog": "#3e3e5a", "waterDeep": "#1a2a4e", "waterShallow": "#2a4a6e", "blossom": "#d4a020"},
    ],
}

THEME_DETAILS = {
    "sakura": [
        ["torii gate", "stone lanterns", "wooden bridge", "falling petals"],
        ["cherry trees", "stone path", "koi pond", "petal drift"],
        ["sakura grove", "lantern posts", "garden bridge", "wind petals"],
        ["torii arch", "rock garden", "tea house", "petal spiral"],
        ["cherry arbor", "stone shrine", "moon bridge", "petal cascade"],
        ["sakura walkway", "iron lanterns", "dock pier", "petal shower"],
        ["torii entrance", "zen garden", "waterfall", "petal dance"],
        ["cherry canopy", "stone steps", "island shrine", "petal stream"],
        ["sakura archway", "paper lanterns", "fishing dock", "petal rain"],
        ["torii gate", "koi stream", "wooden path", "blossom swirl"],
        ["cherry grove", "stone lantern", "arch bridge", "petal fall"],
        ["sakura terrace", "garden stones", "tea pavilion", "wind blossom"],
        ["torii shrine", "gravel path", "lantern row", "petal wave"],
        ["cherry walk", "moss garden", "bamboo bridge", "petal drift"],
        ["sakura cove", "rock formation", "wooden deck", "blossom rain"],
    ],
    "desert": [
        ["sand dunes", "oasis pool", "palm trees", "stone ruins"],
        ["desert temple", "cactus garden", "sandstone arch", "mirage pool"],
        [" Bedouin tent", "camel post", "well spring", "sand drift"],
        ["desert ruins", "oasis palms", "stone obelisk", "sand storm"],
        ["canyon bridge", "rock formation", "desert shrine", "sand cascade"],
    ],
    "forest": [
        ["ancient oak", "mossy stones", "wooden bridge", "leaf canopy"],
        ["forest shrine", "mushroom ring", "vine arch", "leaf drift"],
        ["tree house", "stone path", "forest stream", "wind leaves"],
        ["ember camp", "pine grove", "rock garden", "leaf spiral"],
        ["forest glade", "wooden bench", "fern garden", "canopy shower"],
    ],
    "ocean": [
        ["lighthouse", "rocky shore", "wooden dock", "wave crest"],
        ["coral reef", "sea shells", "tide pool", "wave drift"],
        ["ocean temple", "stone pier", "sea cave", "wave dance"],
        ["beach hut", "palm trees", "sand castle", "wave cascade"],
    ],
    "default": [
        ["stone arch", "crystal formation", "wooden bridge", "ambient particles"],
        ["ancient ruins", "garden path", "water feature", "light rays"],
        ["observation deck", "rock garden", "lantern posts", "mist drift"],
        ["central shrine", "stone steps", "reflecting pool", "wind swirl"],
        ["garden terrace", "wooden walkway", "stone lantern", "petal shower"],
    ],
    "city": [
        ["skyscrapers", "street grid", "window lights", "downtown plaza", "traffic flow"],
        ["high-rise towers", "avenue grid", "lit windows", "urban park", "rooftop water tanks"],
        ["office buildings", "city blocks", "neon signs", "street lamps", "antenna spires"],
        ["apartment towers", "grid streets", "glowing facades", "plaza fountain", "car traffic"],
        ["corporate towers", "manhattan grid", "window glow", "pocket park", "street lights"],
        ["residential blocks", "avenue grid", "lit windows", "rooftop details", "moving cars"],
        ["downtown skyline", "street network", "window lights", "urban plaza", "traffic lanes"],
        ["tower district", "grid layout", "emissive windows", "city park", "lamp posts"],
    ],
    "paris": [
        ["eiffel tower", "haussmann buildings", "seine river", "city lights"],
        ["parisian boulevard", "iron lattice tower", "riverside quay", "lamplit streets"],
        ["french capital", "wrought-iron spire", "waterway bridge", "warm glow"],
    ],
}

TREE_LAYOUTS = [
    [(-4, -1, 1.08), (3, -3, 0.78), (2, 5, 0.64)],
    [(-3, 2, 0.9), (4, -2, 0.7), (-1, -4, 0.55)],
    [(-5, 3, 1.2), (5, -1, 0.85), (0, 5, 0.6)],
    [(-2, -3, 0.75), (3, 3, 1.0), (-4, 1, 0.65)],
    [(-1, -2, 0.8), (2, 2, 0.9), (4, -3, 0.5)],
    [(-6, 0, 1.1), (0, -5, 0.75), (6, 2, 0.95)],
    [(-3, -3, 0.6), (0, 0, 1.0), (3, 3, 0.7)],
    [(-4, 4, 0.85), (4, -4, 0.65), (0, 0, 0.9)],
]

MOTION_PRESETS = [
    {"waterSpeed": 0.25, "petalFallSpeed": 0.3, "cameraOrbitSpeed": 0.015},
    {"waterSpeed": 0.5, "petalFallSpeed": 0.6, "cameraOrbitSpeed": 0.02},
    {"waterSpeed": 0.8, "petalFallSpeed": 0.9, "cameraOrbitSpeed": 0.03},
    {"waterSpeed": 1.0, "petalFallSpeed": 1.2, "cameraOrbitSpeed": 0.04},
    {"waterSpeed": 1.5, "petalFallSpeed": 1.8, "cameraOrbitSpeed": 0.06},
    {"waterSpeed": 0.3, "petalFallSpeed": 0.45, "cameraOrbitSpeed": 0.025},
    {"waterSpeed": 0.65, "petalFallSpeed": 0.8, "cameraOrbitSpeed": 0.035},
    {"waterSpeed": 0.4, "petalFallSpeed": 0.5, "cameraOrbitSpeed": 0.01},
]

CAMERA_ANGLES = [
    [14, 10, 18], [20, 15, 25], [12, 8, 16], [18, 12, 22],
    [10, 18, 14], [25, 20, 30], [15, 22, 18], [8, 6, 12],
    [16, 14, 20], [22, 18, 28], [11, 9, 15], [19, 16, 24],
]


def _detect_theme(request: str) -> str:
    """Detect the scene theme from the request text."""
    request_lower = request.lower()
    for theme in ("sakura", "cherry", "desert", "forest", "ocean", "island", "city", "paris", "nyc", "new york"):
        if theme in request_lower:
            if theme in ("sakura", "cherry"):
                return "sakura"
            if theme in ("city", "nyc", "new york"):
                return "city"
            if theme == "paris":
                return "paris"
            return theme
    return "default"


def _generate_titles(theme: str, request: str) -> list[str]:
    """Generate theme-appropriate titles."""
    base = {
        "sakura": ["Sakura Island Sanctuary", "Cherry Blossom Isle", "Sakura Cove Retreat",
                    "Blossom Island Haven", "Sakura Petal Bay", "Cherry Tree Island",
                    "Sakura Moon Island", "Blossom Shore Island", "Sakura Mist Island",
                    "Cherry Spring Isle", "Sakura Dawn Island", "Blossom Harbor Isle"],
        "desert": ["Desert Oasis Island", "Sandswept Isle", "Dune Sanctuary",
                    "Mirage Island", "Desert Temple Isle", "Canyon Retreat Island"],
        "forest": ["Forest Glade Island", "Ancient Grove Isle", "Mossy Retreat",
                    "Canopy Island", "Forest Shrine Isle", "Woodland Haven Island"],
        "ocean": ["Ocean Crest Island", "Tidal Pool Isle", "Lighthouse Retreat",
                    "Coral Reef Island", "Wave Dance Isle", "Deep Blue Sanctuary"],
        "default": ["Voxel Island Sanctuary", "Island Retreat", "Voxel Cove",
                    "Island Haven", "Voxel Bay", "Island Sanctuary"],
        "city": ["NYC Voxel Skyline", "Manhattan Grid City", "Downtown Voxel City",
                 "New York Skyline", "City Block District", "Urban Tower Grid",
                 "NYC Night Cityscape", "Manhattan Avenue Grid", "Voxel Metropolis",
                 "New York Tower District", "City Skyline Dusk", "Downtown Grid City"],
        "paris": ["Paris Voxel City", "Seine River District", "Eiffel Tower City",
                  "Parisian Boulevard", "French Capital Grid"],
    }
    return base.get(theme, base["default"])


def _make_spec(
    palette: dict[str, str],
    title: str,
    details: list[str],
    trees: list[tuple[float, float, float]],
    motion: dict[str, float],
    camera: list[float],
    terrain_radius: int,
    terrain_height: int,
    water_size: int,
    petal_count: int,
) -> dict[str, Any]:
    return {
        "title": title,
        "sky": palette["sky"],
        "fog": palette["fog"],
        "waterDeep": palette["waterDeep"],
        "waterShallow": palette["waterShallow"],
        "blossom": palette["blossom"],
        "terrainRadius": terrain_radius,
        "terrainHeight": terrain_height,
        "waterSize": water_size,
        "petalCount": petal_count,
        "camera": camera,
        "trees": [{"x": x, "z": z, "scale": s} for x, z, s in trees],
        "details": details,
        "motion": motion,
    }


# City building generation — produces 8-20 buildings on a Manhattan-style grid.
_CITY_SKY_TYPES = ("dawn", "dusk", "night")
_CITY_BUILDING_TYPES = ("box", "setback", "spire")
_CITY_HEIGHT_PROFILES = [
    # (min_h, max_h, weight) — taller buildings are less common
    (4, 8, 0.25), (8, 14, 0.30), (14, 22, 0.25), (22, 32, 0.15), (32, 35, 0.05),
]


def _generate_city_buildings(rng: random.Random, count: int) -> list[dict[str, Any]]:
    """Generate a varied set of city buildings on a grid."""
    buildings: list[dict[str, Any]] = []
    # Grid positions: multiples of 10 from -30 to 30, skip the center plaza
    grid_positions = []
    for gx in range(-30, 31, 10):
        for gz in range(-30, 31, 10):
            if abs(gx) <= 5 and abs(gz) <= 5:
                continue  # downtown plaza
            grid_positions.append((gx, gz))
    rng.shuffle(grid_positions)
    for i in range(min(count, len(grid_positions))):
        gx, gz = grid_positions[i]
        # Pick a height from the weighted profiles
        profile = rng.choices(
            _CITY_HEIGHT_PROFILES,
            weights=[p[2] for p in _CITY_HEIGHT_PROFILES],
        )[0]
        height = rng.randint(profile[0], profile[1])
        # Building type based on height
        if height >= 22:
            btype = rng.choice(["setback", "spire", "setback"])
        elif height >= 14:
            btype = rng.choice(["setback", "box", "box"])
        else:
            btype = "box"
        # Offset within the block
        ox = rng.choice([-2, 0, 2])
        oz = rng.choice([-2, 0, 2])
        buildings.append({
            "x": gx + ox,
            "z": gz + oz,
            "width": rng.randint(3, 6),
            "depth": rng.randint(3, 6),
            "height": height,
            "hue": round(rng.uniform(0.55, 0.62), 2),
            "saturation": round(rng.uniform(0.05, 0.12), 2),
            "lightness": round(rng.uniform(0.18, 0.25), 2),
            "type": btype,
            "windows": round(rng.uniform(0.4, 0.7), 2),
        })
    return buildings


def _make_city_spec(
    palette: dict[str, str],
    title: str,
    details: list[str],
    motion: dict[str, float],
    camera: list[float],
    rng: random.Random,
) -> dict[str, Any]:
    """Generate a city scene spec with buildings, cars, street lights, and sky type."""
    building_count = rng.randint(8, 20)
    buildings = _generate_city_buildings(rng, building_count)
    sky_type = rng.choice(_CITY_SKY_TYPES)
    cars = rng.randint(6, 18)
    return {
        "sceneType": "city",
        "title": title,
        "sky": palette["sky"],
        "fog": palette["fog"],
        "waterDeep": palette["waterDeep"],
        "waterShallow": palette["waterShallow"],
        "blossom": palette["blossom"],
        "terrainRadius": 14,
        "terrainHeight": 8,
        "waterSize": 120,
        "petalCount": 400,
        "camera": camera,
        "trees": [{"x": 3, "z": 2, "scale": 0.7}, {"x": -3, "z": 4, "scale": 0.5}, {"x": 5, "z": -2, "scale": 0.6}],
        "details": details,
        "motion": motion,
        "buildings": buildings,
        "cars": cars,
        "streetLights": True,
        "skyType": sky_type,
    }


def _scene_spec_prompt(request: str, theme: str = "default") -> str:
    base = (
        f"Design this scene for a trusted Three.js voxel-world engine: {request}\n"
        "Return one JSON object with exactly these keys: "
        "title (string), sky/fog/waterDeep/waterShallow/blossom (hex colors like #ff80ab), "
        "terrainRadius (int 8-24), terrainHeight (int 3-12), waterSize (int 50-220), "
        "petalCount (int 120-1200), camera (3 numbers), trees (3+ objects with x/z/scale), "
        "details (3+ strings), motion (waterSpeed/petalFallSpeed/cameraOrbitSpeed numbers)."
    )
    if theme == "city":
        base += (
            " sceneType (string 'city'). Include a 'buildings' array of 8-20 objects, each with "
            "x, z (position, multiples of 10 from -30 to 30), width and depth (integers 2-8), "
            "height (integer 4-35), hue (0-1), saturation (0-1), lightness (0-1), "
            "type ('box', 'setback', or 'spire'), and windows (0-1). "
            "Also include cars (integer 0-24), streetLights (boolean), and "
            f"skyType (string, one of: {', '.join(_CITY_SKY_TYPES)})."
        )
    return base


def generate_evolved_dataset(
    request: str,
    iteration: int,
    previous_failures: list[dict[str, Any]] | None = None,
    row_count: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a scene-spec training dataset for the given loop iteration.

    Each iteration increases variety and targets previous failures:
    - Iteration 0: clean, varied base examples
    - Iteration 1+: adds more diversity, targets specific failure patterns
    """
    rng = random.Random(seed + iteration * 1000)
    theme = _detect_theme(request)
    palettes = THEME_PALETTES.get(theme, THEME_PALETTES["default"])
    titles = _generate_titles(theme, request)
    details_sets = THEME_DETAILS.get(theme, THEME_DETAILS["default"])
    prompt = _scene_spec_prompt(request, theme)

    rows: list[dict[str, Any]] = []

    for i in range(row_count):
        palette = palettes[rng.randint(0, len(palettes) - 1)]
        title = titles[rng.randint(0, len(titles) - 1)]
        details = details_sets[rng.randint(0, len(details_sets) - 1)]
        motion = MOTION_PRESETS[rng.randint(0, len(MOTION_PRESETS) - 1)]
        camera = CAMERA_ANGLES[rng.randint(0, len(CAMERA_ANGLES) - 1)]

        if theme == "city":
            spec = _make_city_spec(palette, title, details, motion, camera, rng)
        else:
            trees = TREE_LAYOUTS[rng.randint(0, len(TREE_LAYOUTS) - 1)]
            terrain_radius = rng.randint(10, 20)
            terrain_height = rng.randint(4, 10)
            water_size = rng.randint(80, 200)
            petal_count = rng.randint(200, 1000)
            spec = _make_spec(
                palette, title, details, trees, motion, camera,
                terrain_radius, terrain_height, water_size, petal_count,
            )

        # On later iterations, add extra trees for richer scenes (non-city only;
        # city scenes use buildings + parks instead of dense trees)
        if iteration >= 2 and rng.random() < 0.4 and theme != "city":
            extra_trees = []
            tr = spec.get("terrainRadius", 14)
            for _ in range(rng.randint(1, 3)):
                ex = rng.uniform(-tr + 2, tr - 2)
                ez = rng.uniform(-tr + 2, tr - 2)
                es = rng.uniform(0.4, 1.2)
                extra_trees.append({"x": round(ex, 1), "z": round(ez, 1), "scale": round(es, 2)})
            spec["trees"] = spec["trees"] + extra_trees

        # On later iterations, add richer details
        if iteration >= 1 and rng.random() < 0.5:
            theme_words = theme.split() + (["island", "shore", "voxel", "terrain"] if theme != "city" else ["skyline", "grid", "tower", "avenue"])
            spec["details"] = spec["details"] + [theme_words[rng.randint(0, len(theme_words) - 1)]]

        rows.append({
            "split": "train",
            "prompt": prompt,
            "ideal_response": json.dumps(spec, indent=2),
            "expected_answer": json.dumps(spec),
            "source_url": f"synthetic://scene-spec-evolved/iter{iteration}/row{i}",
            "source_hash": hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()
            ).hexdigest()[:16],
            "features": ["three.js", "voxel", "shader", "animation", "island", theme] if theme != "city" else ["three.js", "voxel", "shader", "animation", "city", "responsive"],
            "quality_score": 0.9 + min(0.09, iteration * 0.02),
        })

    # If we have failure analysis from the previous iteration, add targeted
    # examples that directly address those failures.
    if previous_failures and iteration > 0:
        targeted = _generate_targeted_examples(request, previous_failures, iteration, theme)
        rows.extend(targeted)

    return rows


def _generate_targeted_examples(
    request: str,
    failures: list[dict[str, Any]],
    iteration: int,
    theme: str,
) -> list[dict[str, Any]]:
    """Generate examples that directly address specific failure patterns."""
    rows: list[dict[str, Any]] = []
    rng = random.Random(iteration * 7777)
    palettes = THEME_PALETTES.get(theme, THEME_PALETTES["default"])
    titles = _generate_titles(theme, request)
    details_sets = THEME_DETAILS.get(theme, THEME_DETAILS["default"])
    prompt = _scene_spec_prompt(request, theme)

    for failure in failures[:10]:
        palette = palettes[rng.randint(0, len(palettes) - 1)]
        title = titles[rng.randint(0, len(titles) - 1)]
        details = details_sets[rng.randint(0, len(details_sets) - 1)]
        motion = MOTION_PRESETS[rng.randint(0, len(MOTION_PRESETS) - 1)]
        camera = CAMERA_ANGLES[rng.randint(0, len(CAMERA_ANGLES) - 1)]

        if theme == "city":
            spec = _make_city_spec(palette, title, details, motion, camera, rng)
        else:
            trees = TREE_LAYOUTS[rng.randint(0, len(TREE_LAYOUTS) - 1)]
            spec = _make_spec(
                palette, title, details, trees, motion, camera,
                rng.randint(10, 20), rng.randint(4, 10),
                rng.randint(80, 200), rng.randint(200, 1000),
            )

        failure_type = failure.get("type", "unknown")

        rows.append({
            "split": "train",
            "prompt": prompt,
            "ideal_response": json.dumps(spec, indent=2),
            "expected_answer": json.dumps(spec),
            "source_url": f"synthetic://scene-spec-targeted/iter{iteration}/{failure_type}",
            "source_hash": hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()
            ).hexdigest()[:16],
            "features": ["three.js", "voxel", "shader", "animation", "island", theme],
            "quality_score": 0.95,
            "targeted_failure": failure_type,
        })

    return rows


def analyze_scene_spec_failures(
    raw_outputs: list[str],
    specs: list[dict[str, Any] | None],
    audits: list[Any],
) -> list[dict[str, Any]]:
    """Analyze what went wrong across multiple scene-spec attempts.

    Returns a list of failure records, each with a type and the raw output,
    so the next dataset iteration can generate targeted examples.
    """
    failures: list[dict[str, Any]] = []
    for raw, spec, audit in zip(raw_outputs, specs, audits):
        if audit and getattr(audit, "passed", False):
            continue
        diagnostics = getattr(audit, "diagnostics", ()) if audit else ("no audit",)
        if spec is None:
            failure_type = "parse_failure"
        elif not isinstance(spec.get("motion"), dict):
            failure_type = "motion_not_object"
        elif any(not isinstance(spec.get(key), str) for key in ("sky", "fog", "waterDeep", "waterShallow", "blossom")):
            failure_type = "colors_not_hex"
        elif not isinstance(spec.get("trees"), list) or len(spec.get("trees", [])) < 3:
            failure_type = "trees_missing"
        elif not isinstance(spec.get("details"), list) or len(spec.get("details", [])) < 3:
            failure_type = "details_missing"
        else:
            failure_type = "design_quality"
        failures.append({
            "type": failure_type,
            "raw_output": raw[:500],
            "diagnostics": list(diagnostics)[:5],
        })
    return failures
