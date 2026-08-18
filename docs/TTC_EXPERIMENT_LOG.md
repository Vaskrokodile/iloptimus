# TTC Experiment Log — 5 Three.js Environments

**Started:** 2026-08-18
**Model:** boosted-v1-small (DeepSeek-R1-Distill-Qwen-1.5B + LoRA adapter from Akahsizrr/boosted-v1-small)
**Hardware:** NVIDIA GeForce RTX 3060, 12GB VRAM, 15.9 GB RAM, Windows
**Backend:** vllm (recommended) → HF Transformers fallback on Windows

## Goal

Run the model through RSI + test-time compute loops to generate 5 different Three.js environments:
1. Sakura Island
2. New York City representation
3. Deserted island with a mountain
4. Island in the sky with a Chinese building
5. Paris with the Eiffel Tower

The model must generate the environments. We improve the harness (loop, tools, dataset factory, scene framework, skills) but do not write the model's output code.

## Experiment Progress

### Pre-flight
- [x] Model downloaded (DeepSeek-R1-Distill-Qwen-1.5B fp16, 3.4GB safetensors)
- [x] Adapter downloaded (Akahsizrr/boosted-v1-small, 1.9MB)
- [x] Server running (port 8765, ILOPTIMUS_HOME=E:\iloptimus-home)
- [x] First TTC session triggered

### Harness Improvements Made
1. **Multi-scene Three.js framework** (`threejs-multi.html`): Built a new template supporting 5 scene types (island/sakura, desert, city, paris, sky_island) with scene-type-specific terrain generators, buildings, Eiffel Tower, pagoda, clouds, and floating rocks.
2. **Scene spec schema updated** (`scene_spec.py`): Added `sceneType` field, `detect_scene_type()` function for NL→type mapping, scene-type-specific optional fields (buildings, floatHeight), updated prompt/audit/compiler.
3. **Feature detection expanded** (`test_time_compute.py`): Added feature patterns for city, paris, desert, sky_island scene types.
4. **Composed three.js check expanded** (`server.py`): Now recognizes all 5 scene types for the framework-scene-design path.
5. **torch.compile disabled on Windows** (`vllm_backend.py`): MSVC `cl.exe` not available, causing "Compiler: cl is not found" errors.
6. **Adapter config format bug fixed** (`vllm_backend.py`): `model.save_pretrained()` writes PEFT-format `adapter_config.json`, but the code was overwriting it with MLX format, breaking `PeftModel.from_pretrained()`. Now saves MLX config to `mlx_adapter_config.json` instead.
7. **Authorship audit relaxed** (`scene_spec.py`): Allow color field normalization (lowercasing hex) in `audit_scene_authorship` — this is not a semantic change.

### Test 1: Sakura Island
_Status:_ In progress (session f647f3d03615)

**Session 1 (d77c2771a2d3)** — Failed: `RuntimeError: Compiler: cl is not found` (torch.compile on Windows)
**Session 2 (844095e88b42)** — Failed: `'peft_type'` KeyError (adapter config overwritten with MLX format)
- Baseline score: 0.93 (all feature scores 1.0, only `runtime_render` failed — no Chromium)
- Training completed: 32 iterations, final loss 1.23
**Session 3 (f647f3d03615)** — Running with both fixes applied

### Test 2: New York City
_Status:_ Not started

### Test 3: Deserted Island with Mountain
_Status:_ Not started

### Test 4: Sky Island with Chinese Building
_Status:_ Not started

### Test 5: Paris with Eiffel Tower
_Status:_ Not started

## Harness Improvements Made
(none yet)

## Results Summary
(none yet)
