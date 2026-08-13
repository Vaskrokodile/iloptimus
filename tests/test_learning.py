import json

from iloptimus.core.learning import (
    LearningManager,
    assess_uncertainty,
    build_research_dataset,
    select_learning_method,
)
from iloptimus.core.sft import generate_sft_data


def test_uncertainty_detector_is_conservative_and_observable():
    certain = assess_uncertainty("Explain recursion.", "Recursion is a function calling itself with a base case and a reducing step.")
    assert certain.needs_research is False

    unknown = assess_uncertainty("Who currently leads Example Corp?", "I'm not sure; it might be Jane Doe.")
    assert unknown.needs_research is True
    assert unknown.time_sensitive is True
    assert unknown.score >= 0.58
    assert select_learning_method(unknown, training_available=True) == "retrieval"

    explicit = assess_uncertainty("/learn Explain stable concept X", "Here is an answer.")
    assert explicit.explicit is True
    assert select_learning_method(explicit, training_available=True) == "qlora-il"


def test_grounded_dataset_retains_source_urls_and_text():
    dataset = build_research_dataset(
        "What is X?",
        [{"title": "Primary source", "url": "https://example.com/x", "text": "X is a tested mechanism with property Y."}],
    )
    assert len(dataset) == 2
    assert "https://example.com/x" in dataset[1]["ideal_response"]
    assert "property Y" in dataset[0]["ideal_response"]


def test_learning_sessions_persist_events_and_terminal_state(tmp_path):
    manager = LearningManager(tmp_path)
    session = manager.create("model", "question", "uncertain", "retrieval", "missing source")
    manager.emit(session.id, "researching", "Found a source", 0.4)
    manager.complete(session.id, "Grounded answer")

    reloaded = LearningManager(tmp_path)
    saved = reloaded.get(session.id)
    assert saved is not None
    assert saved.status == "completed"
    assert saved.final_answer == "Grounded answer"
    assert [event["stage"] for event in reloaded.events(session.id)] == [
        "uncertainty-detected",
        "researching",
        "completed",
    ]
    dataset_path = tmp_path / session.id / "session.json"
    assert json.loads(dataset_path.read_text())["status"] == "completed"


def test_sft_data_offset_reserves_the_first_custom_task_for_evaluation(monkeypatch):
    environment = {
        "mode": "IL",
        "tasks": [
            {"ideal_response": "holdout"},
            {"ideal_response": "training one"},
            {"ideal_response": "training two"},
        ],
    }
    monkeypatch.setattr("iloptimus.core.sft.get_num_tasks", lambda _domain: 3)
    monkeypatch.setattr("iloptimus.core.environments.get_environment", lambda _environment_id: environment)
    monkeypatch.setattr("iloptimus.core.sft.build_prompt", lambda _domain, index: f"prompt {index}")

    examples = generate_sft_data(object(), "custom:learned", num_tasks=2, task_offset=1)

    assert [example.prompt for example in examples] == ["prompt 1", "prompt 2"]
    assert [example.response for example in examples] == ["training one", "training two"]
