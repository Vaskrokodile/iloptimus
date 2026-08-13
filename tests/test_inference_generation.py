import sys
import types

from iloptimus.core.inference import ModelHandle, run_completion, run_source_completion, run_tool_completion


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages and tokenize is False and add_generation_prompt is True
        return "chat prompt"

    def encode(self, text):
        return text.split()


def test_all_generation_paths_apply_repetition_control(monkeypatch):
    marker = lambda tokens, logits: logits
    calls = []

    def generate(*_args, **kwargs):
        calls.append(kwargs)
        return "const scene = new THREE.Scene();"

    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: object()
    sample_utils.make_logits_processors = lambda **kwargs: [marker]
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.generate = generate
    mlx_lm.sample_utils = sample_utils
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.clear_cache = lambda: None
    mlx = types.ModuleType("mlx")
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    handle = ModelHandle(object(), _Tokenizer(), "model", "example/non-reasoning", "int4", True)

    run_completion(handle, "answer")
    run_source_completion(handle, "build it", "index.html")
    run_tool_completion(handle, "search", "web_search")

    assert len(calls) == 3
    assert all(marker in call["logits_processors"] for call in calls)
