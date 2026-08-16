# /// script
# dependencies = []
# ///
"""Run hidden tests against a candidate HumanEval solution without exposing them.

Usage: python verify.py <payload_path> <timeout>
Payload JSON: {"code": "...", "tests": ["assert ...", ...], "entry_point": "..."}
Prints: {"passed": N, "total": M, "pass_rate": float}

Execs the candidate code, then runs each test against the resulting namespace.
Tests never appear in the candidate's source — they are injected after exec.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

RUNNER_TEMPLATE = textwrap.dedent('''\
    import json, sys

    payload = json.loads(sys.argv[1])
    code = payload["code"]
    tests = payload["tests"]
    entry_point = payload.get("entry_point", "")

    namespace = {}
    try:
        exec(code, namespace)
    except BaseException as e:
        print(json.dumps({"passed": 0, "total": len(tests), "pass_rate": 0.0, "error": str(e)}))
        sys.exit(0)

    # Verify the entry_point function exists
    if entry_point and entry_point not in namespace:
        msg = "function " + repr(entry_point) + " not defined"
        print(json.dumps({"passed": 0, "total": len(tests), "pass_rate": 0.0, "error": msg}))
        sys.exit(0)

    passed = 0
    for test_src in tests:
        try:
            exec(test_src, namespace)
            passed += 1
        except BaseException:
            pass

    total = len(tests)
    pass_rate = passed / total if total > 0 else 0.0
    print(json.dumps({"passed": passed, "total": total, "pass_rate": pass_rate}))
''')


def main() -> None:
    payload_path = Path(sys.argv[1])
    timeout = float(sys.argv[2])
    payload = json.loads(payload_path.read_text())
    payload_path.unlink()

    runner = RUNNER_TEMPLATE
    result = subprocess.run(
        [sys.executable, "-c", runner, json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    lines = out.splitlines()
    if lines:
        try:
            parsed = json.loads(lines[-1])
            print(json.dumps(parsed))
            return
        except json.JSONDecodeError:
            pass
    print(json.dumps({"passed": 0, "total": len(payload.get("tests", [])), "pass_rate": 0.0, "error": err[-1000:]}))


if __name__ == "__main__":
    main()
