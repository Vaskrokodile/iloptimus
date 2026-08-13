from iloptimus.core.training_performance import (
    load_training_seconds_per_iteration,
    record_training_throughput,
    training_profile_key,
)


def test_sustained_training_profile_is_persisted_and_conservative(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    key = training_profile_key("small-model", sequence_length=256, rank=16, layers=8, backend="mlx")
    reports = [
        {"iterations_per_second": rate}
        for rate in (1.0, 0.8, 0.55, 0.5, 0.4, 0.35, 0.5, 0.45)
    ]
    profile = record_training_throughput(key, reports, run_id="run-one")
    assert profile is not None
    assert profile["seconds_per_iteration"] >= 2.0
    assert load_training_seconds_per_iteration(key) == profile["seconds_per_iteration"]


def test_training_profile_ignores_invalid_reports(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    key = training_profile_key("model", sequence_length=128, rank=8, layers=4, backend="mlx")
    assert record_training_throughput(key, [{"iterations_per_second": 0}], run_id="bad") is None
    assert load_training_seconds_per_iteration(key) is None
