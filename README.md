# astrotrain

[![test](https://github.com/ankoure/astrotrain/actions/workflows/test.yml/badge.svg)](https://github.com/ankoure/astrotrain/actions/workflows/test.yml)
[![lint](https://github.com/ankoure/astrotrain/actions/workflows/lint.yml/badge.svg)](https://github.com/ankoure/astrotrain/actions/workflows/lint.yml)

Learned countdown clocks for the MBTA subway. When a train leaves a station,
astrotrain predicts how many seconds it will take to reach each of the next 10
stops. It learns from months of recorded train movements and is scored against
the published schedule.

## The name

Astrotrain is a Decepticon from the original *Transformers*. He's a
triple-changer whose two vehicle modes are a train and a space shuttle. This
project is about trains, and it's built on neural networks, where "transformer"
is the architecture behind most of modern AI.

Despite the name, the current model is an LSTM, not a transformer.

## How it works

### Data

The model trains on the MBTA's [LAMP](https://performancedata.mbta.com) subway
on-time performance data. There's one parquet file per service day, with a row
for each time a train stops at a station. Each row has the recorded travel,
dwell and headway times, plus their scheduled values.

### What it predicts

Each example is a window of the last 5 stops on one trip. Its targets are the
seconds from the train leaving its current stop to arriving at each of the next
10. A train gets no prediction until it has passed 5 stops, so a shorter window
covers more of each trip. Near the end of a trip, some of the next 10 stops
don't exist. Their targets are marked unknown and masked out of the loss, so
the trip itself stays in the data.

### Features

Each stop in the window has 14 features:

- where it falls in the trip (stop number and the trip's total stop count)
- recorded travel time, with a flag for when it was missing
- dwell time and trunk and branch headways
- scheduled travel time and trunk headway
- route, station and direction
- the hour the trip started and the day of the week

### Cleaning

- A trip is identified by service date, route and trip ID together. Trip IDs
  repeat every day the timetable runs, and some Green Line trips appear under
  two branches.
- A travel time over 30 minutes between adjacent stops is treated as a
  recording error, usually a train parked at a terminal. As a feature it's
  swapped for the scheduled time and flagged as missing. Any target that passes
  through it is marked unknown.
- Other missing values are filled with medians learned from the training data.
  The medians are saved with the model, so prediction fills gaps the same way.

### Model

The network has three stacked LSTM layers (128, 64 and 32 units) followed by
two dense layers (32 and 16), with dropout in between. It ends in 10 linear
outputs, one for each stop ahead. The loss is Huber over the known targets,
which is squared error for small misses and linear for large ones, so rare
delays of 10+ minutes don't dominate training.

Files are sorted by date. The first 80% are used for training and the last 20%
for validation, so the model is always validated on days after the ones it
trained on. Files are loaded a few at a time to keep memory use bounded, and
they're shuffled on every epoch.

### Baseline

The schedule's prediction for the same moment is the sum of the scheduled
stop-to-stop times along the way, minus the dwell the train has just finished.
`evaluate.py` scores the model and the schedule on the same targets.

## Getting started

You'll need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ankoure/astrotrain.git
cd astrotrain
uv sync
```

Run every command below from the repository root, because the scripts use
paths relative to it.

### 1. Download data

```bash
uv run python src/prep_data.py                    # the latest 180 days
uv run python src/prep_data.py --days 30          # the latest 30 days
uv run python src/prep_data.py --start 2026-04-01 --end 2026-09-30
```

This fetches the LAMP index and the daily parquet files into `data/`. That's
about 1.3 MB per day, so 180 days is roughly 240 MB. Re-running skips files
that are already fully downloaded, so it's cheap to top up with new days.

### 2. Train

```bash
uv run python -m astrotrain.train_lstm
```

This trains on every parquet file in `data/` and saves the model and its
preprocessing (scalers, encoders and fill values) to `models/countdown_lstm/`.
The epoch count and batch sizes are set at the bottom of
[train_lstm.py](src/astrotrain/train_lstm.py). If you run out of memory, lower
`file_batch_size`. TensorFlow uses a GPU if it finds one.

### 3. Evaluate

```bash
uv run python -m astrotrain.evaluate data/2026-10-0*.parquet
```

Evaluate on days the model hasn't seen. Training uses every file in `data/`, so
either download newer days after training, or move some days out of `data/`
before training and point `evaluate` at them.

It prints two tables, one by stops ahead and one by route. All values are in
seconds:

| Column | Meaning |
| --- | --- |
| `targets` | Number of predictions compared |
| `actual_median` | Median actual time to arrival |
| `model_mae`, `model_median`, `model_p90` | Model's mean, median and 90th percentile absolute error |
| `schedule_mae`, `schedule_median` | Schedule's mean and median absolute error |

### 4. Compare runs

Training the LightGBM model (`uv run python -m astrotrain.gbm`) and `evaluate`
record each run with [MLflow](https://mlflow.org): the model's settings, the
training and evaluated days, validation error by round, the summary errors as
metrics, every printed table as a CSV, and the saved model. Evaluations are
nested under the run that trained the model. Pass `--run-name` to label a run.

Runs are kept in `mlflow.db` and `mlruns/` in the repository root (both
gitignored). To browse and compare them:

```bash
uv run mlflow ui     # then open http://localhost:5000
```

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

The tests run on small synthetic files shaped like LAMP data, so they need no
download and no GPU. CI runs lint and tests on every pull request and on each
push to `main`.

## Project layout

```
src/prep_data.py              downloads LAMP data into data/
src/astrotrain/train_lstm.py  features, targets, model and training
src/astrotrain/evaluate.py    error against the schedule by stops ahead and route
tests/                        pytest suite on synthetic LAMP data
data/                         downloaded parquet files (gitignored)
models/countdown_lstm/        saved model and preprocessing (gitignored)
```
