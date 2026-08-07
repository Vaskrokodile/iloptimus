"""Handcrafted agentic-coding tasks for the IL pipeline.

Each task is a handcrafted multi-file codebase scenario (mechanize.work-style):
the model reads a mini-codebase with a bug or missing feature, reasons through
the code, and produces fixes as ```python:filename``` code blocks. The grader
applies the changes in a sandbox and runs hidden tests.

IL efficiency-aware reward shaping:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

Plus anti-laziness: degenerate fixes (no-op, deleting code, constant returns)
get penalized.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field


@dataclass
class AgenticCodingTask:
    """A single handcrafted multi-file codebase task."""
    idx: int
    name: str
    spec: str
    # {filename: content} — the buggy/incomplete codebase shown to the model
    codebase: dict[str, str]
    # the test harness run against the MERGED codebase (after applying fixes)
    test_harness: str
    expected_concepts: list[str]
    token_budget: int = 800
    difficulty: str = "hard"
    # the file(s) the model is expected to edit
    target_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 10 handcrafted agentic-coding tasks
# ---------------------------------------------------------------------------

TASKS: list[AgenticCodingTask] = [
    AgenticCodingTask(
        idx=0,
        name="cascading_bug_chain",
        difficulty="hard",
        token_budget=900,
        target_files=["parser.py", "transformer.py", "formatter.py"],
        spec=(
            "A 3-module data pipeline has THREE bugs stacked in sequence. Each "
            "bug is only visible after the previous one is fixed.\n\n"
            "Bug 1 (parser.py): `parse_line` drops the last character of each "
            "field (off-by-one in slicing).\n"
            "Bug 2 (transformer.py): `to_number` converts '3.0' to int 3 instead "
            "of float 3.0 when the string has no decimal point.\n"
            "Bug 3 (formatter.py): `format_value` uses %d instead of %f, "
            "truncating float values.\n\n"
            "Fix all three bugs. Return each fixed file as a "
            "```python:filename``` block."
        ),
        codebase={
            "parser.py": textwrap.dedent('''\
                """Input parser — splits CSV-like lines into named fields."""

                FIELDS = ("name", "age", "score")


                def parse_line(line: str) -> dict:
                    parts = line.split(",")
                    result = {}
                    for i, field in enumerate(FIELDS):
                        if i < len(parts):
                            result[field] = parts[i][:-1]
                    return result


                def parse_lines(lines: list[str]) -> list[dict]:
                    return [parse_line(line) for line in lines]
            '''),
            "transformer.py": textwrap.dedent('''\
                """Transforms parsed field dicts — converts score to number."""

                from parser import parse_line


                def to_number(value: str):
                    if "." not in value:
                        return int(value)
                    return float(value)


                def transform(record: dict) -> dict:
                    result = dict(record)
                    result["score"] = to_number(record["score"])
                    return result
            '''),
            "formatter.py": textwrap.dedent('''\
                """Formats transformed records for output."""

                from transformer import transform


                def format_value(value) -> str:
                    return "%d" % value


                def format_record(record: dict) -> str:
                    t = transform(record)
                    return f"{t['name']},{t['age']},{format_value(t['score'])}"
            '''),
        },
        test_harness=textwrap.dedent('''\
            from parser import parse_line
            from transformer import transform, to_number
            from formatter import format_record, format_value

            # Bug 1: parser must not drop last char
            r = parse_line("alice,30,3.5")
            assert r == {"name": "alice", "age": "30", "score": "3.5"}, f"parser: {r}"

            # Bug 2: to_number must return float for "3" (no decimal) -> 3.0
            assert to_number("3") == 3.0, f"to_number('3')={to_number('3')}"
            assert to_number("3.5") == 3.5
            assert isinstance(to_number("3"), float)

            # Bug 3: format_value must not truncate floats
            assert format_value(3.5) == "3.5", f"format_value(3.5)={format_value(3.5)}"
            assert format_value(3.0) == "3.0"

            # End-to-end
            out = format_record(parse_line("alice,30,3.5"))
            assert out == "alice,30,3.5", f"e2e: {out}"
            print("ALL_PASS")
        '''),
        expected_concepts=["parse", "slice", "last", "char", "float", "int", "decimal", "format", "%f", "%d", "cascade", "trace", "fix", "verify"],
    ),
    AgenticCodingTask(
        idx=1,
        name="codebase_nav_bug",
        difficulty="hard",
        token_budget=800,
        target_files=["utils.py"],
        spec=(
            "A small web app has a bug: user lookups by ID always return None "
            "even when the user exists. Trace the call chain across the files "
            "and fix the bug. Return the fixed file(s) as ```python:filename``` blocks.\n\n"
            "Hint: the bug is in the data layer, not the route handler."
        ),
        codebase={
            "app.py": textwrap.dedent('''\
                """Route handlers for the user API."""

                from store import get_user


                def handle_get_user(user_id: str) -> dict:
                    user = get_user(user_id)
                    if user is None:
                        return {"error": "not found", "status": 404}
                    return {"user": user, "status": 200}
            '''),
            "store.py": textwrap.dedent('''\
                """Data store facade — delegates to the storage layer."""

                from utils import lookup


                def get_user(user_id: str) -> dict | None:
                    return lookup(user_id)
            '''),
            "utils.py": textwrap.dedent('''\
                """Low-level storage utilities."""

                _DB = {"1": {"id": "1", "name": "alice"}, "2": {"id": "2", "name": "bob"}}


                def lookup(key: str) -> dict | None:
                    # BUG: checks for key as int, but keys are strings
                    if int(key) in _DB:
                        return _DB[int(key)]
                    return None
            '''),
        },
        test_harness=textwrap.dedent('''\
            from store import get_user
            from app import handle_get_user

            assert get_user("1") == {"id": "1", "name": "alice"}, f"get_user('1')={get_user('1')}"
            assert get_user("2") == {"id": "2", "name": "bob"}
            assert get_user("3") is None

            r = handle_get_user("1")
            assert r["status"] == 200 and r["user"]["name"] == "alice", f"handle: {r}"
            r404 = handle_get_user("99")
            assert r404["status"] == 404
            print("ALL_PASS")
        '''),
        expected_concepts=["trace", "call", "chain", "store", "lookup", "key", "string", "int", "type", "bug", "fix", "verify"],
    ),
    AgenticCodingTask(
        idx=2,
        name="refactor_preserve_behavior",
        difficulty="hard",
        token_budget=800,
        target_files=["calculator.py"],
        spec=(
            "Refactor `calculator.py` to use a dispatch dict instead of the "
            "if/elif chain, while preserving EXACT behavior (including error "
            "handling for unknown operators). The tests must still pass. "
            "Return the refactored file as a ```python:calculator.py``` block."
        ),
        codebase={
            "calculator.py": textwrap.dedent('''\
                """A simple calculator with if/elif dispatch."""

                def calculate(op: str, a: float, b: float) -> float:
                    if op == "add":
                        return a + b
                    elif op == "sub":
                        return a - b
                    elif op == "mul":
                        return a * b
                    elif op == "div":
                        if b == 0:
                            raise ValueError("division by zero")
                        return a / b
                    else:
                        raise ValueError(f"unknown operator: {op}")
            '''),
        },
        test_harness=textwrap.dedent('''\
            from calculator import calculate

            assert calculate("add", 1, 2) == 3
            assert calculate("sub", 5, 3) == 2
            assert calculate("mul", 3, 4) == 12
            assert calculate("div", 10, 2) == 5.0
            try:
                calculate("div", 1, 0)
                assert False, "should raise"
            except ValueError as e:
                assert "zero" in str(e).lower()
            try:
                calculate("pow", 2, 3)
                assert False, "should raise for unknown op"
            except ValueError as e:
                assert "unknown" in str(e).lower() or "pow" in str(e)
            print("ALL_PASS")
        '''),
        expected_concepts=["refactor", "dispatch", "dict", "preserve", "behavior", "if", "elif", "chain", "error", "unknown", "zero", "verify"],
    ),
    AgenticCodingTask(
        idx=3,
        name="missing_error_handling",
        difficulty="medium",
        token_budget=700,
        target_files=["processor.py"],
        spec=(
            "The `process_items` function crashes on empty lists and on items "
            "that are None. Add error handling so that:\n"
            "- empty list returns []\n"
            "- None items are skipped (not included in output)\n"
            "- non-None items are processed normally\n\n"
            "Return the fixed file as a ```python:processor.py``` block."
        ),
        codebase={
            "processor.py": textwrap.dedent('''\
                """Processes a list of items, doubling each numeric value."""

                def process_items(items: list) -> list:
                    return [item * 2 for item in items]
            '''),
        },
        test_harness=textwrap.dedent('''\
            from processor import process_items

            assert process_items([1, 2, 3]) == [2, 4, 6]
            assert process_items([]) == []
            assert process_items([None, 1, None, 2]) == [2, 4]
            assert process_items([None]) == []
            assert process_items([0]) == [0]
            assert process_items([1, None, 3, None, 5]) == [2, 6, 10]
            print("ALL_PASS")
        '''),
        expected_concepts=["error", "handle", "none", "empty", "skip", "filter", "list", "process", "edge", "verify"],
    ),
    AgenticCodingTask(
        idx=4,
        name="api_client_impl",
        difficulty="hard",
        token_budget=800,
        target_files=["client.py"],
        spec=(
            "Implement the missing `fetch_user` method in `client.py` per this "
            "spec:\n"
            "- Takes a user_id (int) and optional fields list (default ['name']).\n"
            "- Returns a dict with 'id' always present.\n"
            "- For each field in fields: 'name' -> f'user_{id}', 'email' -> f'{id}@test.com'.\n"
            "- Unknown fields are ignored.\n"
            "- Raises TypeError if user_id is not an int.\n\n"
            "Return the completed file as a ```python:client.py``` block."
        ),
        codebase={
            "client.py": textwrap.dedent('''\
                """API client with a missing method."""

                class UserClient:
                    def __init__(self, base_url: str = "https://api.test.com"):
                        self.base_url = base_url

                    def fetch_user(self, user_id, fields=None):
                        # TODO: implement per spec
                        pass
            '''),
        },
        test_harness=textwrap.dedent('''\
            from client import UserClient

            c = UserClient()
            assert c.fetch_user(1) == {"id": 1, "name": "user_1"}
            assert c.fetch_user(1, ["name", "email"]) == {"id": 1, "name": "user_1", "email": "1@test.com"}
            assert c.fetch_user(5, ["email"]) == {"id": 5, "email": "5@test.com"}
            assert c.fetch_user(0, []) == {"id": 0}
            assert c.fetch_user(1, ["unknown"]) == {"id": 1}
            try:
                c.fetch_user("1")
                assert False, "should raise TypeError"
            except TypeError:
                pass
            print("ALL_PASS")
        '''),
        expected_concepts=["implement", "method", "id", "fields", "name", "email", "dict", "type", "int", "default", "unknown", "verify"],
    ),
    AgenticCodingTask(
        idx=5,
        name="dead_code_and_logic_error",
        difficulty="hard",
        token_budget=800,
        target_files=["validator.py"],
        spec=(
            "`validator.py` has TWO problems:\n"
            "1. Dead code: the `format_error` function is never used and the "
            "`_legacy_check` function is unreachable.\n"
            "2. A logic error: `validate_age` rejects age 0 (which should be "
            "valid) and accepts age 200 (which should be invalid — max is 150).\n\n"
            "Remove the dead code AND fix the logic error. Return the cleaned "
            "file as a ```python:validator.py``` block."
        ),
        codebase={
            "validator.py": textwrap.dedent('''\
                """User input validator."""

                def format_error(msg: str) -> str:
                    return f"[ERROR] {msg}"


                def validate_age(age: int) -> bool:
                    if age > 0:
                        return True
                    return False


                def _legacy_check(age: int) -> bool:
                    if False:
                        return True
                    return age > 18


                def validate_name(name: str) -> bool:
                    return len(name) >= 2 and name.isalpha()
            '''),
        },
        test_harness=textwrap.dedent('''\
            from validator import validate_age, validate_name

            assert validate_age(0) == True, "age 0 should be valid"
            assert validate_age(25) == True
            assert validate_age(150) == True
            assert validate_age(200) == False, "age 200 should be invalid"
            assert validate_age(-1) == False
            assert validate_name("alice") == True
            assert validate_name("a") == False
            assert validate_name("al1") == False

            # dead code removed
            import validator
            assert not hasattr(validator, "format_error"), "format_error should be removed"
            assert not hasattr(validator, "_legacy_check"), "_legacy_check should be removed"
            print("ALL_PASS")
        '''),
        expected_concepts=["dead", "code", "remove", "logic", "error", "age", "valid", "invalid", "max", "boundary", "fix", "verify"],
    ),
    AgenticCodingTask(
        idx=6,
        name="type_annotation_add",
        difficulty="medium",
        token_budget=700,
        target_files=["graph.py"],
        spec=(
            "Add complete type annotations to `graph.py`:\n"
            "- All function parameters and return types.\n"
            "- The adjacency dict should be typed as dict[str, list[str]].\n"
            "- Do NOT change the runtime behavior — only add annotations.\n\n"
            "Return the annotated file as a ```python:graph.py``` block."
        ),
        codebase={
            "graph.py": textwrap.dedent('''\
                """Simple directed graph utilities."""

                from collections import deque


                def build_graph(edges):
                    graph = {}
                    for src, dst in edges:
                        graph.setdefault(src, []).append(dst)
                        graph.setdefault(dst, [])
                    return graph


                def shortest_path(graph, start, end):
                    if start == end:
                        return [start]
                    queue = deque([[start]])
                    visited = {start}
                    while queue:
                        path = queue.popleft()
                        node = path[-1]
                        for neighbor in graph.get(node, []):
                            if neighbor == end:
                                return path + [neighbor]
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(path + [neighbor])
                    return None
            '''),
        },
        test_harness=textwrap.dedent('''\
            from graph import build_graph, shortest_path
            import inspect
            import typing

            # Behavior preserved
            g = build_graph([("a", "b"), ("b", "c")])
            assert g == {"a": ["b"], "b": ["c"], "c": []}
            assert shortest_path(g, "a", "c") == ["a", "b", "c"]
            assert shortest_path(g, "a", "a") == ["a"]
            assert shortest_path(g, "a", "z") is None

            # Annotations present
            sig_build = inspect.signature(build_graph)
            sig_path = inspect.signature(shortest_path)
            assert sig_build.return_annotation is not inspect.Parameter.empty, "build_graph needs return annotation"
            assert sig_path.return_annotation is not inspect.Parameter.empty, "shortest_path needs return annotation"
            for p in sig_build.parameters.values():
                assert p.annotation is not inspect.Parameter.empty, f"build_graph param {p.name} needs annotation"
            for p in sig_path.parameters.values():
                assert p.annotation is not inspect.Parameter.empty, f"shortest_path param {p.name} needs annotation"
            print("ALL_PASS")
        '''),
        expected_concepts=["type", "annotation", "dict", "list", "str", "return", "parameter", "graph", "preserve", "behavior", "verify"],
    ),
    AgenticCodingTask(
        idx=7,
        name="perf_optimize_list",
        difficulty="hard",
        token_budget=800,
        target_files=["search.py"],
        spec=(
            "`find_duplicates` in `search.py` is O(n^2) — it scans the list for "
            "each element. Rewrite it to be O(n) using a set, preserving the "
            "EXACT same output: a list of duplicate values in the order they "
            "are first detected as duplicates. Return the optimized file as a "
            "```python:search.py``` block."
        ),
        codebase={
            "search.py": textwrap.dedent('''\
                """Find duplicate values in a list (slow O(n^2) version)."""

                def find_duplicates(items: list) -> list:
                    duplicates = []
                    for i, item in enumerate(items):
                        for j in range(i):
                            if items[j] == item and item not in duplicates:
                                duplicates.append(item)
                    return duplicates
            '''),
        },
        test_harness=textwrap.dedent('''\
            from search import find_duplicates

            assert find_duplicates([1, 2, 3, 2, 1]) == [2, 1]
            assert find_duplicates([1, 1, 1, 1]) == [1]
            assert find_duplicates([1, 2, 3]) == []
            assert find_duplicates([]) == []
            assert find_duplicates([3, 1, 3, 2, 1, 2]) == [3, 1, 2]
            assert find_duplicates(["a", "b", "a", "c", "b"]) == ["a", "b"]
            print("ALL_PASS")
        '''),
        expected_concepts=["optimize", "o(n)", "set", "duplicate", "scan", "order", "preserve", "output", "performance", "verify"],
    ),
    AgenticCodingTask(
        idx=8,
        name="config_fix_multi",
        difficulty="medium",
        token_budget=700,
        target_files=["config.py"],
        spec=(
            "`config.py` has three configuration bugs:\n"
            "1. `DEFAULT_TIMEOUT` is a string '300' but should be int 300.\n"
            "2. `MAX_RETRIES` is 0 but should be 3.\n"
            "3. `get_config` returns the raw dict including internal keys "
            "(starting with '_'). It should filter those out.\n\n"
            "Fix all three. Return the fixed file as a ```python:config.py``` block."
        ),
        codebase={
            "config.py": textwrap.dedent('''\
                """Application configuration."""

                DEFAULT_TIMEOUT = "300"
                MAX_RETRIES = 0

                _CONFIG = {
                    "timeout": DEFAULT_TIMEOUT,
                    "retries": MAX_RETRIES,
                    "host": "localhost",
                    "port": 8080,
                    "_internal": "secret",
                    "_version": "1.0",
                }


                def get_config() -> dict:
                    return dict(_CONFIG)
            '''),
        },
        test_harness=textwrap.dedent('''\
            from config import DEFAULT_TIMEOUT, MAX_RETRIES, get_config

            assert DEFAULT_TIMEOUT == 300, f"timeout={DEFAULT_TIMEOUT}"
            assert isinstance(DEFAULT_TIMEOUT, int)
            assert MAX_RETRIES == 3, f"retries={MAX_RETRIES}"

            cfg = get_config()
            assert cfg["timeout"] == 300
            assert cfg["retries"] == 3
            assert cfg["host"] == "localhost"
            assert cfg["port"] == 8080
            assert "_internal" not in cfg, "internal keys must be filtered"
            assert "_version" not in cfg
            print("ALL_PASS")
        '''),
        expected_concepts=["config", "timeout", "int", "string", "retries", "filter", "internal", "key", "fix", "verify"],
    ),
    AgenticCodingTask(
        idx=9,
        name="test_writing_catch_mutant",
        difficulty="hard",
        token_budget=800,
        target_files=["tests.py"],
        spec=(
            "Write tests for `solution.py` (a `Stack` class) that would catch a "
            "MUTANT version where `pop()` returns the BOTTOM element instead of "
            "the top. Your tests must pass on the correct implementation AND "
            "fail on the mutant. Write them in `tests.py` as functions named "
            "`test_*` that use assert statements. Return as a "
            "```python:tests.py``` block."
        ),
        codebase={
            "solution.py": textwrap.dedent('''\
                """A correct Stack implementation (do NOT modify this file)."""

                class Stack:
                    def __init__(self):
                        self._items = []

                    def push(self, item):
                        self._items.append(item)

                    def pop(self):
                        if not self._items:
                            raise IndexError("pop from empty stack")
                        return self._items.pop()

                    def peek(self):
                        if not self._items:
                            raise IndexError("peek from empty stack")
                        return self._items[-1]

                    def size(self):
                        return len(self._items)

                    def is_empty(self):
                        return len(self._items) == 0
            '''),
            "tests.py": textwrap.dedent('''\
                """Write your tests here. They must catch a mutant where pop() returns the bottom element."""

                # TODO: write test_* functions
                pass
            '''),
        },
        test_harness=textwrap.dedent('''\
            import importlib
            import sys

            # Run the student's tests against the CORRECT solution
            sys.modules.pop("tests", None)
            import tests
            test_fns = [getattr(tests, n) for n in dir(tests) if n.startswith("test_") and callable(getattr(tests, n))]
            assert len(test_fns) >= 2, f"need at least 2 tests, got {len(test_fns)}"

            for fn in test_fns:
                fn()  # must pass on correct solution

            # Now run against the MUTANT (pop returns bottom)
            import types
            mutant_src = """
            class Stack:
                def __init__(self):
                    self._items = []
                def push(self, item):
                    self._items.append(item)
                def pop(self):
                    if not self._items:
                        raise IndexError("pop from empty stack")
                    return self._items.pop(0)  # MUTANT: bottom not top
                def peek(self):
                    if not self._items:
                        raise IndexError("peek from empty stack")
                    return self._items[-1]
                def size(self):
                    return len(self._items)
                def is_empty(self):
                    return len(self._items) == 0
            """
            mutant_mod = types.ModuleType("solution_mutant")
            exec(mutant_src, mutant_mod.__dict__)
            sys.modules["solution"] = mutant_mod
            sys.modules.pop("tests", None)
            importlib.reload(tests)

            caught = False
            for fn in test_fns:
                try:
                    fn()
                except (AssertionError, IndexError, Exception):
                    caught = True
                    break
            assert caught, "tests must catch the mutant (pop returns bottom)"
            print("ALL_PASS")
        '''),
        expected_concepts=["test", "mutant", "catch", "pop", "top", "bottom", "stack", "push", "assert", "order", "verify"],
    ),
]


__all__ = ["AgenticCodingTask", "TASKS"]
