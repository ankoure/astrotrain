import mlflow

from astrotrain import tracking


def test_evaluation_of_a_model_from_another_database_is_not_nested():
    elsewhere = "0" * 32

    with tracking.start_run("test", "evaluation", training_run_id=elsewhere) as run:
        pass

    tags = mlflow.get_run(run.info.run_id).data.tags
    assert tags["training_run_id"] == elsewhere
    assert "mlflow.parentRunId" not in tags


def test_model_params_are_flat():
    metadata = {
        "model": "lightgbm",
        "settings": {"sequence_length": 5, "params": {"num_leaves": 255}},
        "training": {"days": 21, "trees": 900},
        "features": ["stops_ahead", "typical_time"],
    }

    assert tracking.model_params(metadata) == {
        "model": "lightgbm",
        "sequence_length": 5,
        "num_leaves": 255,
        "features": 2,
        "training_days": 21,
        "training_trees": 900,
    }
