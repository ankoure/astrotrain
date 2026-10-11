"""
Experiment tracking with MLflow. The training and evaluation commands record
each run's settings and results (and training runs, the saved model) so runs
can be compared later:

    uv run mlflow ui

Runs go to mlflow.db, and their files to mlruns/, in the working directory, or
wherever MLFLOW_TRACKING_URI points. MLflow tags each run with the git commit;
uncommitted_changes says whether the checkout had changes on top of it.

CountdownClockGBM.train() and save() log to the active run if there is one, so
they work the same without tracking. To track a script, wrap it in start_run().
"""

import subprocess
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.exceptions import MlflowException


def start_run(
    experiment: str,
    kind: str,
    run_name: Optional[str] = None,
    training_run_id: Optional[str] = None,
) -> mlflow.ActiveRun:
    """
    Start a run in `experiment`, tagged with its kind ("training" or
    "evaluation"). An evaluation of a model from a tracked training run is
    nested under that run, if it's in this tracking database.
    """
    tags = {"kind": kind}
    _, dirty = git_commit()
    if dirty is not None:
        tags["uncommitted_changes"] = str(dirty)
    experiment_id = None
    if training_run_id:
        tags["training_run_id"] = training_run_id
        try:
            training_run = mlflow.get_run(training_run_id)
        except MlflowException:
            training_run = None  # trained against another tracking database
        if training_run:
            tags["mlflow.parentRunId"] = training_run_id
            experiment_id = training_run.info.experiment_id
    if experiment_id is None:
        experiment_id = mlflow.set_experiment(experiment).experiment_id
    return mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags)


def active() -> bool:
    return mlflow.active_run() is not None


def active_run_id() -> Optional[str]:
    run = mlflow.active_run()
    return run.info.run_id if run else None


def model_params(metadata: dict) -> dict:
    """
    A saved model's settings (its metadata.json) as flat run parameters, the
    same for its training run and its evaluations so they line up in the UI
    """
    settings = dict(metadata["settings"])
    lightgbm_params = settings.pop("params", {})
    return {
        "model": metadata["model"],
        **settings,
        **lightgbm_params,
        "features": len(metadata["features"]),
        **{f"training_{key}": value for key, value in metadata["training"].items()},
    }


def log_lightgbm_evaluation(env):
    """LightGBM callback: log each round's validation metrics (val_l1)"""
    for _, metric, value, *_ in env.evaluation_result_list:
        # Asynchronous: a synchronous write takes ~15 ms, thousands of rounds
        mlflow.log_metric(f"val_{metric}", value, step=env.iteration, synchronous=False)


def log_training(metadata: dict, best_score: dict):
    """A trained model's settings, and its best validation scores (best_val_l1)"""
    mlflow.log_params(model_params(metadata))
    for scores in best_score.values():
        for metric, value in scores.items():
            mlflow.log_metric(f"best_val_{metric}", value)


def log_files(folder: Path, artifact_path: str):
    mlflow.log_artifacts(str(folder), artifact_path)


def git_commit():
    """This checkout's commit and whether it has uncommitted changes, if in git"""
    here = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())
