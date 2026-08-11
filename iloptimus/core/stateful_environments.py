"""Trusted declarative state-machine runtime for no-code environments."""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any

CONDITION_OPS = {"eq", "ne", "gt", "gte", "lt", "lte"}
EFFECT_OPS = {"set", "add", "subtract", "toggle"}
STATEFUL_HINTS = {
    "agent",
    "battery",
    "environment",
    "game",
    "grid",
    "inventory",
    "maze",
    "navigate",
    "robot",
    "simulation",
    "state",
    "steps",
    "tool",
    "world",
}


def is_stateful_request(description: str) -> bool:
    words = set(re.findall(r"[a-z]+", description.casefold()))
    return bool(words & STATEFUL_HINTS)


def _compare(left: Any, op: str, right: Any) -> bool:
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    raise ValueError(f"Unsupported condition operator: {op}")


def evaluate_condition(condition: dict[str, Any] | None, state: dict[str, Any]) -> bool:
    if not condition:
        return True
    if "all" in condition:
        return all(evaluate_condition(item, state) for item in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(item, state) for item in condition["any"])
    if "not" in condition:
        return not evaluate_condition(condition["not"], state)
    variable = str(condition.get("var") or "")
    if variable not in state:
        raise ValueError(f"Condition references unknown state variable: {variable}")
    op = str(condition.get("op") or "eq")
    right = state.get(condition["value_from"]) if "value_from" in condition else condition.get("value")
    return _compare(state[variable], op, right)


def _normalise_effect(effect: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    variable = str(effect.get("var") or "")
    operation = str(effect.get("op") or "set")
    if variable not in state:
        raise ValueError(f"Effect references unknown state variable: {variable}")
    if operation not in EFFECT_OPS:
        raise ValueError(f"Unsupported effect operator: {operation}")
    return {
        "var": variable,
        "op": operation,
        "value": effect.get("value"),
        "min": effect.get("min"),
        "max": effect.get("max"),
    }


def validate_simulator(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("A state-machine environment needs initial state variables")
    initial_state = {str(key): value for key, value in state.items()}
    if not all(isinstance(value, (str, int, float, bool)) for value in initial_state.values()):
        raise ValueError("State values must be strings, numbers, or booleans")

    actions = []
    names = set()
    for raw in payload.get("actions", []):
        name = str(raw.get("name") or "").strip()
        if not name or name in names:
            raise ValueError("Every simulator action needs a unique name")
        names.add(name)
        effects = [_normalise_effect(effect, initial_state) for effect in raw.get("effects", [])]
        if not effects:
            raise ValueError(f"Action {name} needs at least one state effect")
        actions.append(
            {
                "name": name,
                "description": str(raw.get("description") or name.replace("_", " ")),
                "requires": raw.get("requires"),
                "effects": effects,
                "reward": float(raw.get("reward", 0.0)),
            }
        )
    if not actions:
        raise ValueError("A state-machine environment needs at least one action")

    terminals = []
    for raw in payload.get("terminals", []):
        condition = raw.get("when")
        if not isinstance(condition, dict):
            raise ValueError("Every terminal outcome needs a condition")
        terminals.append(
            {
                "when": condition,
                "outcome": str(raw.get("outcome") or "terminal"),
                "success": bool(raw.get("success", False)),
                "reward": float(raw.get("reward", 1.0 if raw.get("success") else -1.0)),
            }
        )
    if not terminals:
        raise ValueError("A state-machine environment needs a terminal condition")

    scenarios = []
    for index, raw in enumerate(payload.get("scenarios", [])):
        scenario_state = {**initial_state, **raw.get("initial_state", {})}
        if set(scenario_state) != set(initial_state):
            raise ValueError("Scenario state variables must match the simulator state")
        solution = [str(action) for action in raw.get("solution", [])]
        if any(action not in names for action in solution):
            raise ValueError("Scenario solutions may only use declared actions")
        scenarios.append(
            {
                "name": str(raw.get("name") or f"Episode {index + 1}"),
                "initial_state": scenario_state,
                "solution": solution,
            }
        )
    if not scenarios:
        scenarios = [{"name": "Default episode", "initial_state": initial_state, "solution": []}]

    rewards = payload.get("rewards", {})
    return {
        "template_id": str(payload.get("template_id") or "custom-state-machine-v1"),
        "observation": str(payload.get("observation") or "State: {state}"),
        "state": initial_state,
        "actions": actions,
        "terminals": terminals,
        "rewards": {
            "step": float(rewards.get("step", -0.01)),
            "invalid": float(rewards.get("invalid", -0.2)),
            "timeout": float(rewards.get("timeout", -1.0)),
        },
        "max_steps": max(1, min(64, int(payload.get("max_steps", 12)))),
        "scenarios": scenarios,
    }


@dataclass
class StepResult:
    observation: str
    state: dict[str, Any]
    reward: float
    terminated: bool
    success: bool
    outcome: str
    step: int
    valid: bool
    actions: list[str]


class StateMachineRuntime:
    def __init__(self, simulator: dict[str, Any], scenario_index: int = 0):
        self.spec = validate_simulator(simulator)
        self.scenario_index = max(0, min(scenario_index, len(self.spec["scenarios"]) - 1))
        self.state: dict[str, Any] = {}
        self.steps = 0
        self.terminated = False
        self.success = False
        self.outcome = "running"
        self.total_reward = 0.0
        self.reset(self.scenario_index)

    @property
    def action_names(self) -> list[str]:
        return [action["name"] for action in self.spec["actions"]]

    def observation(self) -> str:
        values = {**self.state, "state": json.dumps(self.state, sort_keys=True)}
        try:
            return self.spec["observation"].format_map(values)
        except (KeyError, ValueError):
            return f"State: {json.dumps(self.state, sort_keys=True)}"

    def reset(self, scenario_index: int = 0) -> StepResult:
        self.scenario_index = max(0, min(scenario_index, len(self.spec["scenarios"]) - 1))
        self.state = copy.deepcopy(self.spec["scenarios"][self.scenario_index]["initial_state"])
        self.steps = 0
        self.terminated = False
        self.success = False
        self.outcome = "running"
        self.total_reward = 0.0
        return self._result(0.0, True)

    def _result(self, reward: float, valid: bool) -> StepResult:
        return StepResult(
            observation=self.observation(),
            state=copy.deepcopy(self.state),
            reward=reward,
            terminated=self.terminated,
            success=self.success,
            outcome=self.outcome,
            step=self.steps,
            valid=valid,
            actions=self.action_names,
        )

    def step(self, action_name: str) -> StepResult:
        if self.terminated:
            raise ValueError("This episode has already terminated")
        action = next((item for item in self.spec["actions"] if item["name"] == action_name), None)
        self.steps += 1
        if action is None or not evaluate_condition(action.get("requires"), self.state):
            reward = self.spec["rewards"]["invalid"]
            self.total_reward += reward
            if self.steps >= self.spec["max_steps"]:
                self.terminated = True
                self.outcome = "timeout"
            return self._result(reward, False)

        for effect in action["effects"]:
            variable = effect["var"]
            if effect["op"] == "set":
                value = effect["value"]
            elif effect["op"] == "add":
                value = self.state[variable] + effect["value"]
            elif effect["op"] == "subtract":
                value = self.state[variable] - effect["value"]
            else:
                value = not bool(self.state[variable])
            if effect.get("min") is not None:
                value = max(effect["min"], value)
            if effect.get("max") is not None:
                value = min(effect["max"], value)
            self.state[variable] = value

        reward = self.spec["rewards"]["step"] + action["reward"]
        for terminal in self.spec["terminals"]:
            if evaluate_condition(terminal["when"], self.state):
                self.terminated = True
                self.success = terminal["success"]
                self.outcome = terminal["outcome"]
                reward += terminal["reward"]
                break
        if not self.terminated and self.steps >= self.spec["max_steps"]:
            self.terminated = True
            self.outcome = "timeout"
            reward += self.spec["rewards"]["timeout"]
        self.total_reward += reward
        return self._result(reward, True)


def _trajectory_from_response(response: str, action_names: list[str]) -> list[str]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(response):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return [str(item) for item in value]
    tagged = re.findall(r"<action>(.*?)</action>", response, re.DOTALL | re.IGNORECASE)
    if tagged:
        return [item.strip() for item in tagged]
    tokens = re.findall(r"[a-z][a-z0-9_-]*", response.casefold())
    return [token for token in tokens if token in action_names]


def simulate_response(environment: dict[str, Any], task_index: int, response: str) -> dict[str, Any]:
    runtime = StateMachineRuntime(environment["simulator"], task_index)
    trajectory = _trajectory_from_response(response, runtime.action_names)
    trace = []
    for action in trajectory[: runtime.spec["max_steps"]]:
        result = runtime.step(action)
        trace.append({"action": action, **asdict(result)})
        if result.terminated:
            break
    score = 0.0
    if runtime.success:
        efficiency = max(0.0, 1.0 - max(0, runtime.steps - 1) / runtime.spec["max_steps"])
        score = 0.8 + 0.2 * efficiency
    else:
        score = min(0.79, max(0.0, runtime.total_reward))
    return {
        "score": min(1.0, max(0.0, score)),
        "success": runtime.success,
        "outcome": runtime.outcome,
        "steps": runtime.steps,
        "cumulative_reward": runtime.total_reward,
        "trajectory": trajectory,
        "trace": trace,
        "final_state": runtime.state,
    }


def build_stateful_prompt(environment: dict[str, Any], task_index: int) -> str:
    simulator = environment["simulator"]
    scenario = simulator["scenarios"][task_index]
    actions = "\n".join(f'- "{item["name"]}": {item["description"]}' for item in simulator["actions"])
    return f"""Environment goal: {environment["goal"]}
Episode: {scenario["name"]}
Initial observation: {StateMachineRuntime(simulator, task_index).observation()}

Available actions:
{actions}

Choose a sequence of at most {simulator["max_steps"]} actions that reaches a successful terminal state.
Think inside <reasoning> tags. Put only a JSON array of action names inside <answer> tags.
Example output format: <answer>["action_one", "action_two"]</answer>"""


def stateful_tasks(simulator: dict[str, Any]) -> list[dict[str, Any]]:
    spec = validate_simulator(simulator)
    tasks = []
    for index, scenario in enumerate(spec["scenarios"]):
        solution = scenario["solution"]
        tasks.append(
            {
                "id": f"episode-{index + 1}",
                "name": scenario["name"],
                "prompt": f"Complete state-machine episode: {scenario['name']}",
                "expected_answer": json.dumps(solution),
                "ideal_response": (
                    "<reasoning>I choose valid actions and check the state after every transition.</reasoning>"
                    f"<answer>{json.dumps(solution)}</answer>"
                ),
                "criteria": ["valid actions", "terminal state"],
                "grader": {"type": "exact", "target": json.dumps(solution)},
                "difficulty": ("easy", "medium", "hard")[min(index, 2)],
            }
        )
    return tasks


def scaffold_simulator(description: str, template_id: str | None = None) -> dict[str, Any]:
    lowered = description.casefold()
    selected = template_id or (
        "tool-workflow-v1"
        if any(word in lowered for word in ("tool", "api", "search", "submit"))
        else "grid-navigation-v1"
        if any(word in lowered for word in ("navigate", "robot", "grid", "maze", "position"))
        else "resource-control-v1"
    )
    if selected == "tool-workflow-v1":
        return validate_simulator(
            {
                "template_id": selected,
                "observation": "gathered={gathered}, verified={verified}, submitted={submitted}",
                "state": {"gathered": False, "verified": False, "submitted": False},
                "actions": [
                    {
                        "name": "gather",
                        "description": "collect the required information",
                        "effects": [{"var": "gathered", "op": "set", "value": True}],
                        "reward": 0.1,
                    },
                    {
                        "name": "verify",
                        "description": "verify gathered information",
                        "requires": {"var": "gathered", "op": "eq", "value": True},
                        "effects": [{"var": "verified", "op": "set", "value": True}],
                        "reward": 0.15,
                    },
                    {
                        "name": "submit",
                        "description": "submit the verified result",
                        "requires": {"var": "verified", "op": "eq", "value": True},
                        "effects": [{"var": "submitted", "op": "set", "value": True}],
                        "reward": 0.2,
                    },
                ],
                "terminals": [
                    {
                        "when": {"var": "submitted", "op": "eq", "value": True},
                        "outcome": "completed",
                        "success": True,
                        "reward": 1,
                    }
                ],
                "max_steps": 6,
                "scenarios": [
                    {
                        "name": name,
                        "initial_state": {"gathered": False, "verified": False, "submitted": False},
                        "solution": ["gather", "verify", "submit"],
                    }
                    for name in ("Standard workflow", "Verification-first discipline", "Reliable submission")
                ],
            }
        )
    if selected == "grid-navigation-v1":
        scenarios = []
        for goal in (2, 3, 4):
            scenarios.append(
                {
                    "name": f"Reach position {goal}",
                    "initial_state": {"position": 0, "energy": goal + 1, "goal": goal},
                    "solution": ["move_forward"] * goal,
                }
            )
        return validate_simulator(
            {
                "template_id": selected,
                "observation": "position={position}, goal={goal}, energy={energy}",
                "state": {"position": 0, "energy": 4, "goal": 3},
                "actions": [
                    {
                        "name": "move_forward",
                        "description": "advance one position and consume one energy",
                        "requires": {"var": "energy", "op": "gt", "value": 0},
                        "effects": [
                            {"var": "position", "op": "add", "value": 1},
                            {"var": "energy", "op": "subtract", "value": 1},
                        ],
                        "reward": 0.1,
                    },
                    {
                        "name": "move_back",
                        "description": "move back one position",
                        "effects": [{"var": "position", "op": "subtract", "value": 1, "min": 0}],
                        "reward": -0.1,
                    },
                ],
                "terminals": [
                    {
                        "when": {"var": "position", "op": "gte", "value_from": "goal"},
                        "outcome": "goal_reached",
                        "success": True,
                        "reward": 1,
                    },
                    {
                        "when": {
                            "all": [
                                {"var": "energy", "op": "lte", "value": 0},
                                {"var": "position", "op": "lt", "value_from": "goal"},
                            ]
                        },
                        "outcome": "out_of_energy",
                        "success": False,
                        "reward": -1,
                    },
                ],
                "max_steps": 8,
                "scenarios": scenarios,
            }
        )
    scenarios = []
    for goal in (2, 3, 4):
        scenarios.append(
            {
                "name": f"Complete {goal} milestones",
                "initial_state": {"progress": 0, "energy": goal, "goal": goal},
                "solution": ["advance"] * goal,
            }
        )
    return validate_simulator(
        {
            "template_id": "resource-control-v1",
            "observation": "progress={progress}/{goal}, energy={energy}",
            "state": {"progress": 0, "energy": 3, "goal": 3},
            "actions": [
                {
                    "name": "advance",
                    "description": "make progress and consume energy",
                    "requires": {"var": "energy", "op": "gt", "value": 0},
                    "effects": [
                        {"var": "progress", "op": "add", "value": 1},
                        {"var": "energy", "op": "subtract", "value": 1},
                    ],
                    "reward": 0.1,
                },
                {
                    "name": "recharge",
                    "description": "recover one energy",
                    "effects": [{"var": "energy", "op": "add", "value": 1, "max": 6}],
                    "reward": -0.05,
                },
            ],
            "terminals": [
                {
                    "when": {"var": "progress", "op": "gte", "value_from": "goal"},
                    "outcome": "goal_reached",
                    "success": True,
                    "reward": 1,
                }
            ],
            "max_steps": 10,
            "scenarios": scenarios,
        }
    )


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]
