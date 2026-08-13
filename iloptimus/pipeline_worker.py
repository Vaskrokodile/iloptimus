"""Isolated entry point for one durable MLX training run."""

from __future__ import annotations

import argparse
import asyncio

from .core.hardware import detect_hardware
from .core.pipeline import _persist_state, get_run, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one IL Optimus pipeline worker")
    parser.add_argument("run_id")
    args = parser.parse_args()
    state = get_run(args.run_id)
    if state is None:
        parser.error(f"unknown run: {args.run_id}")

    # Loading persisted state intentionally marks orphaned runs as interrupted.
    # This process is the owner now, so remove that synthetic in-memory marker.
    if state.events and state.events[-1].get("message") == "Run was interrupted when IL Optimus stopped":
        state.events.pop()
    state.status = "pending"
    _persist_state(state)
    asyncio.run(run_pipeline(state.id, state.config, detect_hardware()))
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
