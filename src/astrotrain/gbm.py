"""
Gradient-boosted trees (LightGBM) on the same information as the LSTM, laid out
as a table: one row per departure and number of stops ahead.

Phase 2 of the model roadmap. Comparing this with the LSTM shows whether the
LSTM's sequence architecture adds anything over a strong tabular model, and
comparing it with and without the train-ahead features shows how much knowing
about the train ahead is worth.

It reuses the LSTM's preprocessing (targets, typical times, fill values, train
ahead) and predicts exactly the same targets: arrival 1-10 stops ahead, from
departures after a train's first sequence_length - 1 stops, so its results line
up one-to-one with evaluate.evaluate() and work with summarize() and the MBTA
head-to-head.
"""

from typing import List, Optional

import lightgbm as lgb
import pandas as pd

from astrotrain.network import AHEAD_COLUMNS
from astrotrain.train_lstm import CountdownClockLSTM

# The last few legs, like the LSTM's window: this stop (lag 0) and the 3 before
LAGGED_COLUMNS = ["travel_time_seconds", "dwell_time_seconds", "leg_excess"]
LAGS = 3

BASE_FEATURES = [
    "stops_ahead",
    "typical_time",
    "scheduled",
    "position",
    "travel_time_missing",
    "headway_trunk_seconds",
    "headway_branch_seconds",
    "scheduled_headway_trunk",
    "scheduled_travel_time",
    "route_encoded",
    "stop_encoded",
    "direction_binary",
    "hour_of_day",
    "departure_hour",
    "day_of_week",
    "stop_count",
    *LAGGED_COLUMNS,
    *(f"{column}_lag{lag}" for lag in range(1, LAGS + 1) for column in LAGGED_COLUMNS),
]

# Route and station are categories, not quantities: trees can split on them
# directly, where the LSTM only sees arbitrary label numbers
CATEGORICAL_FEATURES = ["route_encoded", "stop_encoded", "direction_binary"]

DEFAULT_PARAMS = {
    "objective": "l1",  # the metric we report is mean absolute error
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


class CountdownClockGBM:
    def __init__(
        self,
        use_train_ahead: bool,
        sequence_length: int = 5,
        prediction_horizon: int = 10,
        seed: int = 0,
        params: Optional[dict] = None,
    ):
        """
        Args:
            use_train_ahead: Include the train-ahead features (network.py)
            sequence_length: Predict only from a train's sequence_length-th stop
                             on, as the LSTM does, so both cover the same targets
            prediction_horizon: Number of stops ahead to predict arrival at
            params: LightGBM parameters, over DEFAULT_PARAMS
        """
        self.prep = CountdownClockLSTM(sequence_length, prediction_horizon, seed)
        self.use_train_ahead = use_train_ahead
        self.params = {**DEFAULT_PARAMS, "seed": seed, **(params or {})}
        self.booster: Optional[lgb.Booster] = None

    @property
    def feature_columns(self) -> List[str]:
        return BASE_FEATURES + (AHEAD_COLUMNS if self.use_train_ahead else [])

    def table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One row per departure and stop ahead with a known target, from
        preprocess_data output: the features, the actual seconds, and what's
        needed to line results up (trip, target stop, departure time).
        """
        prep = self.prep
        trips = df.groupby(prep.TRIP_KEY)
        df = df.assign(
            position=trips.cumcount(),
            # How much slower than typical the leg into this stop was
            leg_excess=df["travel_time_seconds"] - trips["typical_time_1"].shift(1),
        )
        trips = df.groupby(prep.TRIP_KEY)
        for lag in range(1, LAGS + 1):
            for column in LAGGED_COLUMNS:
                df[f"{column}_lag{lag}"] = trips[column].shift(lag)
        for k in range(1, prep.prediction_horizon + 1):
            df[f"target_stop_{k}"] = trips["stop_id"].shift(-k)
        df = df[df["position"] >= prep.sequence_length - 1]

        shared = [c for c in self.feature_columns if c not in _PER_HORIZON]
        parts = []
        for k in range(1, prep.prediction_horizon + 1):
            known = df[f"target_{k}"].notna()
            rows = df[known]
            parts.append(
                rows[
                    ["service_date", "trip_id", "route_id", "departure_time", *shared]
                ].assign(
                    stop_id=rows[f"target_stop_{k}"],
                    stops_ahead=k,
                    typical_time=rows[f"typical_time_{k}"],
                    scheduled=rows[f"target_scheduled_{k}"],
                    actual=rows[f"target_{k}"],
                )
            )
        return pd.concat(parts, ignore_index=True)

    def _tables(self, files: List[str], file_batch_size: int) -> pd.DataFrame:
        return pd.concat(
            [
                self.table(
                    self.prep.preprocess_data(
                        self.prep.load_data(files[i : i + file_batch_size])
                    )
                )
                for i in range(0, len(files), file_batch_size)
            ],
            ignore_index=True,
        )

    def _dataset(self, table: pd.DataFrame, reference=None) -> lgb.Dataset:
        return lgb.Dataset(
            table[self.feature_columns],
            table["actual"],
            categorical_feature=CATEGORICAL_FEATURES,
            reference=reference,
            free_raw_data=False,
        )

    def train(
        self,
        parquet_files: List[str],
        num_boost_round: int = 3000,
        early_stopping_rounds: int = 50,
        file_batch_size: int = 5,
    ) -> lgb.Booster:
        """
        Fit the preprocessing as train() does for the LSTM (encoders on all files,
        typical times on the training files, fill values on a sample), then train
        on the first 80% of files with early stopping on the last 20%.
        """
        split = int(len(parquet_files) * 0.8)
        train_files, val_files = parquet_files[:split], parquet_files[split:]
        self.prep.fit_encoders(parquet_files)
        self.prep.fit_typical_times(train_files, file_batch_size)
        self.prep.fit_fill_values(self.prep.load_data(parquet_files[:10]))

        train_table = self._tables(train_files, file_batch_size)
        train_set = self._dataset(train_table)
        val_set = self._dataset(
            self._tables(val_files, file_batch_size), reference=train_set
        )
        features = len(self.feature_columns)
        print(f"Training on {len(train_table):,} rows, {features} features")
        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(100),
            ],
        )
        return self.booster

    def results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict raw LAMP data, in the same format as evaluate.evaluate(): one row
        per known target with its trip, route, stop, stops ahead, departure time,
        and the actual, predicted and scheduled seconds
        """
        table = self.table(self.prep.preprocess_data(df))
        table["predicted"] = self.booster.predict(table[self.feature_columns])
        return table[
            [
                "service_date",
                "trip_id",
                "route_id",
                "stop_id",
                "stops_ahead",
                "departure_time",
                "actual",
                "predicted",
                "scheduled",
            ]
        ]


# Features that differ by stops ahead, filled per horizon in table()
_PER_HORIZON = {"stops_ahead", "typical_time", "scheduled"}
