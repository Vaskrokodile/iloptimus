# /// script
# dependencies = []
# ///
"""Run a test harness against a multi-file codebase with applied fixes.

Usage: python verify.py <payload_path> <timeout>
Payload JSON: {
    "files": {"filename": "content", ...},   # full codebase AFTER merging fixes
    "harness": "test code that imports from the files and prints ALL_PASS"
}

Writes files to a temp dir, runs the harness, returns pass/fail.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    payload_path = Path(sys.argv[1])
    timeout = float(sys.argv[2])
    payload = json.loads(payload_path.read_text())
    payload_path.unlink()

    files: dict[str, str] = payload["files"]
    harness: str = payload["harness"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpd = Path(tmpdir)

        # Write all codebase files
        for fname, content in files.items():
            fpath = tmpd / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        # Write the harness
        harness_path = tmpd / "_harness.py"
        harness_path.write_text(harness)

        # Run the harness in the temp dir (so imports resolve)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmpd)
        try:
            result = subprocess.run(
                [sys.executable, str(harness_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(tmpd),
            )
            output = (result.stdout or "") + (result.stderr or "")
            passed = "ALL_PASS" in (result.stdout or "")
        except subprocess.TimeoutExpired:
            output = "TIMEOUT"
            passed = False
        except BaseException as e:
            output = str(e)
            passed = False

    print(json.dumps({"passed": passed, "output": output[-2000:]}))


if __name__ == "__main__":
    main()
