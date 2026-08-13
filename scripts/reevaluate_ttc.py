#!/usr/bin/env python3
"""Re-run the current artifact verifier and rewrite a TTC session's verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from iloptimus.core.storage import app_home, atomic_write_json
from iloptimus.core.test_time_compute import (
    ArtifactContract,
    acceptance_decision,
    artifact_generation_prompt,
    evaluate_artifact,
    strip_learning_command,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    arguments = parser.parse_args()

    root = app_home() / "learning" / arguments.session_id
    session_path = root / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    contract = ArtifactContract(**session["contract"])
    baseline = evaluate_artifact(Path(session["baseline_artifact_path"]), contract)
    adapted = evaluate_artifact(Path(session["adapted_artifact_path"]), contract)
    acceptance = acceptance_decision(baseline, adapted)
    session["baseline_evaluation"] = baseline.public()
    session["adapted_evaluation"] = adapted.public()
    session["acceptance"] = acceptance
    session["accepted_adapter_path"] = session.get("accepted_adapter_path", "") if acceptance["accepted"] else ""
    session["final_answer"] = (
        f"Test-time adapter {'accepted' if acceptance['accepted'] else 'rejected'}. "
        f"Baseline {baseline.score:.3f}; retry {adapted.score:.3f}; "
        f"measured change {acceptance['improvement']:+.3f}."
    )
    atomic_write_json(session_path, session)
    atomic_write_json(root / "acceptance.json", acceptance)
    experiment_path = root / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
    generation_prompt = artifact_generation_prompt(strip_learning_command(session["query"]), contract)
    baseline_path = Path(session["baseline_artifact_path"])
    adapted_path = Path(session["adapted_artifact_path"])
    generation = experiment.get("generation", {})
    generation.update(
        {
            "identical_prompt": True,
            "prompt_sha256": hashlib.sha256(generation_prompt.encode()).hexdigest(),
        }
    )
    experiment.update(
        {
            "contract": contract.public(),
            "generation": generation,
            "artifact_sha256": {
                "baseline": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                "adapted": hashlib.sha256(adapted_path.read_bytes()).hexdigest(),
            },
            "baseline_evaluation": baseline.public(),
            "adapted_evaluation": adapted.public(),
            "acceptance": acceptance,
            "reevaluated": True,
        }
    )
    atomic_write_json(experiment_path, experiment)
    print(json.dumps({"session_id": arguments.session_id, "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
