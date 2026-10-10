"""
LSTM Model for MBTA Countdown Clock Predictions
Trains on historical LAMP data to predict vehicle arrival times: from the moment a
train leaves a stop, how long until it arrives at each of the next several stops
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import keras
from keras import layers
import pickle
from typing import List, Optional, Tuple
from glob import glob

# Configure TensorFlow for better memory management
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth must be set before GPUs are used: {e}")

# For CPU: limit the number of threads to prevent overallocation
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


def _fill_unknown(y_true, y_pred):
    """
    Replace missing (NaN) targets with the prediction, so they add no error and no
    gradient. Windows near a trip's end have no target for stops past it.
    """
    known = keras.ops.logical_not(keras.ops.isnan(y_true))
    return keras.ops.where(known, y_true, y_pred), known


@keras.saving.register_keras_serializable(package="astrotrain")
def masked_huber(y_true, y_pred):
    """
    Huber loss over known targets. Huber is squared error for small misses and
    linear past 1 std, so the rare real delays of 10+ minutes don't dominate.
    """
    y_true, _ = _fill_unknown(y_true, y_pred)
    return keras.losses.huber(y_true, y_pred, delta=1.0)


@keras.saving.register_keras_serializable(package="astrotrain")
def masked_mae(y_true, y_pred):
    """Mean absolute error over known targets (in scaled units)"""
    y_true, known = _fill_unknown(y_true, y_pred)
    n_known = keras.ops.sum(keras.ops.cast(known, y_pred.dtype), axis=-1)
    total = keras.ops.sum(keras.ops.abs(y_true - y_pred), axis=-1)
    return total / keras.ops.maximum(n_known, 1.0)


class CountdownClockLSTM:
    """LSTM model for predicting MBTA countdown clock times"""

    # Identifies one run of a trip. Scheduled trip IDs repeat on every day with the
    # same timetable, and some Green Line trips appear under two branches.
    TRIP_KEY = ["service_date", "route_id", "trip_id"]

    # Travel times above this between adjacent stops are recording errors (mostly
    # trains sitting at a terminal, logged as hours-long trips), not real delays
    MAX_TRAVEL_TIME_SECONDS = 30 * 60

    # Numeric columns filled with their training median when missing
    FILLED_COLUMNS = [
        "dwell_time_seconds",
        "headway_trunk_seconds",
        "headway_branch_seconds",
        "scheduled_travel_time",
        "scheduled_headway_trunk",
        "scheduled_headway_branch",
    ]

    def __init__(
        self,
        sequence_length: int = 5,
        prediction_horizon: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Args:
            sequence_length: Number of historical stops to use as input. A train
                             gets no prediction until it has passed this many, so
                             shorter windows cover more of each trip
            prediction_horizon: Number of stops ahead to predict arrival at; the
                                model has one output per stop (1, 2, ... ahead)
            seed: Seed for shuffling training data (None for a different
                  order every run)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.rng = np.random.default_rng(seed)
        self.model = keras.Sequential()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.route_encoder = LabelEncoder()
        self.stop_encoder = LabelEncoder()
        self.fill_values = {}
        self.feature_columns = []

    @property
    def target_columns(self) -> List[str]:
        """Seconds from leaving a stop to arriving 1, 2, ... stops ahead"""
        return [f"target_{k}" for k in range(1, self.prediction_horizon + 1)]

    @property
    def scheduled_target_columns(self) -> List[str]:
        """The schedule's estimate of each target, for evaluating against"""
        return [f"target_scheduled_{k}" for k in range(1, self.prediction_horizon + 1)]

    def load_data(self, parquet_files: List[str]) -> pd.DataFrame:
        """Load and combine multiple parquet files"""
        dfs = []
        for file_path in parquet_files:
            print(f"Loading {file_path}...")
            df = pd.read_parquet(file_path)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total records: {len(combined_df)}")
        return combined_df

    def fit_encoders(self, parquet_files: List[str]):
        """
        Fit categorical encoders on every route and station across all files,
        so each one maps to the same code in every file batch.
        """
        routes, stations = set(), set()
        for file_path in parquet_files:
            df = pd.read_parquet(file_path, columns=["route_id", "parent_station"])
            routes.update(df["route_id"].unique())
            stations.update(df["parent_station"].unique())

        self.route_encoder.fit(list(routes))
        self.stop_encoder.fit(list(stations))
        print(f"Encoded {len(routes)} routes and {len(stations)} stations")

    def fit_fill_values(self, df: pd.DataFrame):
        """
        Learn the value used for each missing numeric column from raw training data,
        so every batch (and prediction) is filled the same way. A column that's empty
        in the training data is filled with 0.
        """
        medians = df[self.FILLED_COLUMNS].median()
        self.fill_values = medians.fillna(0).to_dict()

    def sequence_generator(
        self,
        parquet_files: List[str],
        file_batch_size: int = 5,
        mini_batch_size: int = 64,
        shuffle: bool = False,
    ):
        """
        Memory-efficient generator that yields mini-batches of sequences from files.

        Args:
            parquet_files: List of parquet file paths
            file_batch_size: Number of files to load at once
            mini_batch_size: Size of mini-batches to yield for training
            shuffle: Shuffle the file order, and the sequences within each file
                     batch, on every pass

        Yields:
            Tuple of (X_batch, y_batch) arrays of size mini_batch_size
        """
        if shuffle:
            order = self.rng.permutation(len(parquet_files))
            parquet_files = [parquet_files[k] for k in order]

        for i in range(0, len(parquet_files), file_batch_size):
            file_batch = parquet_files[i : i + file_batch_size]
            print(
                f"\nProcessing files {i + 1} to "
                f"{min(i + file_batch_size, len(parquet_files))} of "
                f"{len(parquet_files)}"
            )

            # Load and process this batch of files
            df = self.load_data(file_batch)
            df = self.preprocess_data(df)
            X, y = self.create_sequences(df)

            # Scale features
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = self.feature_scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
            X_scaled = X_scaled.astype(np.float32)

            # Scale targets
            y_scaled = self.target_scaler.transform(y).astype(np.float32)

            # Sequences come out grouped by trip, so mix them before batching
            if shuffle:
                order = self.rng.permutation(n_samples)
                X_scaled, y_scaled = X_scaled[order], y_scaled[order]

            # Yield mini-batches from this data
            for j in range(0, n_samples, mini_batch_size):
                end_idx = min(j + mini_batch_size, n_samples)
                yield X_scaled[j:end_idx], y_scaled[j:end_idx]

            # Clear memory
            del df, X, y, X_reshaped, X_scaled, y_scaled

    def _add_targets(self, df: pd.DataFrame):
        """
        Add target_k: seconds from the train leaving each stop to arriving k stops
        later, for k = 1 to prediction_horizon. Rows are kept even when a target
        is unknown (past the trip's end, or a recording error on the way), so
        trips stay intact; create_sequences skips windows with no known target.

        Also adds target_scheduled_k, the schedule's estimate of the same time.
        """
        # Arrival-to-arrival time into each stop. A leg longer than the limit is a
        # recording error (e.g. a train parked at a terminal), so any target that
        # passes through one is left unknown.
        df["arrival_leg"] = df.groupby(self.TRIP_KEY)["stop_timestamp"].diff()
        trips = df.groupby(self.TRIP_KEY)

        # The train leaves when it starts moving to the next stop, or, if that
        # wasn't recorded, after its dwell here
        departure = trips["move_timestamp"].shift(-1)
        departure = departure.fillna(df["stop_timestamp"] + df["dwell_time_seconds"])
        dwell_done = departure - df["stop_timestamp"]

        broken = pd.Series(False, index=df.index)
        scheduled = pd.Series(0.0, index=df.index)
        for k in range(1, self.prediction_horizon + 1):
            target = trips["stop_timestamp"].shift(-k) - departure
            if k == 1:
                broken = target > self.MAX_TRAVEL_TIME_SECONDS
            else:
                broken |= trips["arrival_leg"].shift(-k) > self.MAX_TRAVEL_TIME_SECONDS
            df[f"target_{k}"] = target.where(~broken & (target > 0))

            # Scheduled times run arrival to arrival, so the schedule's estimate
            # from departure is the sum of the legs minus the dwell already done
            scheduled = scheduled + trips["scheduled_travel_time"].shift(-k)
            df[f"target_scheduled_{k}"] = scheduled - dwell_done

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for modeling"""
        if not self.fill_values:
            raise RuntimeError(
                "Fill values aren't fit yet: call fit_fill_values() or load a model"
            )

        # Sort by trip and stop sequence
        df = df.sort_values([*self.TRIP_KEY, "stop_sequence"]).reset_index(drop=True)

        self._add_targets(df)

        # Treat impossible travel times as missing
        df["travel_time_seconds"] = df["travel_time_seconds"].where(
            df["travel_time_seconds"] <= self.MAX_TRAVEL_TIME_SECONDS
        )

        # Flag missing travel times (usually the first stop, plus recording gaps)
        # and fill them with the scheduled time, or 0 when there isn't one
        df["travel_time_missing"] = df["travel_time_seconds"].isna().astype(int)
        df["travel_time_seconds"] = (
            df["travel_time_seconds"].fillna(df["scheduled_travel_time"]).fillna(0)
        )

        # Fill the other numeric columns with their training medians
        df = df.fillna(self.fill_values)

        # Encode categorical variables (encoders are fit once in fit_encoders)
        df["route_encoded"] = self.route_encoder.transform(df["route_id"])
        df["stop_encoded"] = self.stop_encoder.transform(df["parent_station"])

        # Create time-based features
        df["hour_of_day"] = (df["start_time"] // 3600) % 24
        df["day_of_week"] = pd.to_datetime(
            df["service_date"], format="%Y%m%d"
        ).dt.dayofweek

        # Binary direction
        df["direction_binary"] = df["direction_id"].astype(int)

        return df

    def create_sequences(
        self, df: pd.DataFrame, return_index: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """
        Create time series sequences for LSTM

        Args:
            df: Output of preprocess_data
            return_index: Also return the df index of each window's last row,
                          to match predictions back to their trip and stop
        """
        # Define features for the model
        self.feature_columns = [
            "stop_sequence",
            "travel_time_seconds",
            "travel_time_missing",
            "dwell_time_seconds",
            "headway_trunk_seconds",
            "headway_branch_seconds",
            "scheduled_travel_time",
            "scheduled_headway_trunk",
            "route_encoded",
            "stop_encoded",
            "hour_of_day",
            "day_of_week",
            "direction_binary",
            "stop_count",
        ]

        length = self.sequence_length

        # Put each trip's rows together, trips in key order and stops in their
        # existing order (preprocess_data already sorts this way). Rows with a
        # missing trip key get no trip number and are left out, as groupby does.
        trip = df.groupby(self.TRIP_KEY).ngroup().to_numpy()
        order = np.argsort(trip, kind="stable")
        trip = trip[order]
        features = df[self.feature_columns].to_numpy(dtype=float)[order]
        targets = df[self.target_columns].to_numpy(dtype=float)[order]
        index = df.index.to_numpy()[order]

        # Every run of consecutive rows, as a view without copying. A window is
        # kept if it starts and ends in the same trip (so it's all one trip,
        # since a trip's rows are together) and its last stop has a known target.
        n_starts = max(len(df) - length + 1, 0)
        starts = np.arange(n_starts)
        ends = starts + length - 1
        keep = (
            (trip[starts] == trip[ends])
            & ~np.isnan(trip[starts])
            & ~np.isnan(targets[ends]).all(axis=1)
        )
        if n_starts:
            windows = sliding_window_view(features, (length, features.shape[1]))
            X = windows[:, 0][starts[keep]]
        else:
            X = np.empty((0, length, len(self.feature_columns)))
        y = targets[ends[keep]]

        print(f"Created {len(X)} sequences of shape {X.shape}")
        if return_index:
            return X, y, index[ends[keep]]
        return X, y

    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM neural network"""
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                # First LSTM layer with dropout
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.2),
                # Second LSTM layer
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.2),
                # Third LSTM layer
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.2),
                # Dense layers
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),
                # Output layer: time to arrive at each of the next stops
                layers.Dense(self.prediction_horizon, activation="linear"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=masked_huber,
            metrics=[masked_mae],
        )

        self.model = model
        return model

    def train(
        self,
        parquet_files: List[str],
        epochs: int = 50,
        batch_size: int = 64,
        file_batch_size: int = 30,
    ):
        """
        Complete training pipeline with memory-efficient file batching.

        Args:
            parquet_files: List of parquet file paths
            epochs: Number of training epochs
            batch_size: Mini-batch size for SGD
            file_batch_size: Number of parquet files to load at once
                           (adjust based on RAM)
        """
        # Fit encoders on all files, then scalers on a sample of data
        print("Fitting encoders on all files...")
        self.fit_encoders(parquet_files)

        print("Fitting fill values and scalers on initial data sample...")
        sample_files = parquet_files[: min(10, len(parquet_files))]
        df_sample = self.load_data(sample_files)
        self.fit_fill_values(df_sample)
        df_sample = self.preprocess_data(df_sample)
        X_sample, y_sample = self.create_sequences(df_sample)

        # A horizon no sample trip reaches can't be scaled or learned
        never_known = np.isnan(y_sample).all(axis=0)
        if never_known.any():
            raise ValueError(
                f"No trips in the sample reach {np.argmax(never_known) + 1} stops "
                f"ahead; lower prediction_horizon (now {self.prediction_horizon})"
            )

        # Fit scalers
        n_samples, n_timesteps, n_features = X_sample.shape
        X_reshaped = X_sample.reshape(-1, n_features)
        self.feature_scaler.fit(X_reshaped)
        self.target_scaler.fit(y_sample)

        print(f"Feature shape: ({n_timesteps}, {n_features})")

        # Build model
        self.build_model(input_shape=(n_timesteps, n_features))
        print(self.model.summary())

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
        )

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath="temp_best_weights.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )

        # Split files into train/val sets (time-based split)
        split_idx = int(len(parquet_files) * 0.8)
        train_files = parquet_files[:split_idx]
        val_files = parquet_files[split_idx:]

        print(
            f"\nTraining on {len(train_files)} files, "
            f"validating on {len(val_files)} files"
        )

        # Create datasets. from_generator takes a callable, so the generator is
        # restarted every epoch and each epoch is one full pass over the files.
        output_signature = (
            tf.TensorSpec(shape=(None, n_timesteps, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.prediction_horizon), dtype=tf.float32),
        )
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self.sequence_generator(
                train_files, file_batch_size, batch_size, shuffle=True
            ),
            output_signature=output_signature,
        )
        val_dataset = tf.data.Dataset.from_generator(
            lambda: self.sequence_generator(val_files, file_batch_size, batch_size),
            output_signature=output_signature,
        )

        # Train using fit with callbacks
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            # Keras can't shuffle a generator dataset; sequence_generator does it
            shuffle=False,
            verbose=1,
        )

        print("\nTraining complete!")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and inverse transform to original scale"""
        # Scale input
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

        # Predict
        y_pred_scaled = self.model.predict(X_scaled)

        # Inverse transform
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)

        return y_pred

    def save(self, path: str = "models/countdown_lstm"):
        """Save model and preprocessing objects"""
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(f"{path}/model.keras")

        # Save scalers and encoders
        with open(f"{path}/scalers.pkl", "wb") as f:
            pickle.dump(
                {
                    "feature_scaler": self.feature_scaler,
                    "target_scaler": self.target_scaler,
                    "route_encoder": self.route_encoder,
                    "stop_encoder": self.stop_encoder,
                    "fill_values": self.fill_values,
                    "feature_columns": self.feature_columns,
                    "sequence_length": self.sequence_length,
                    "prediction_horizon": self.prediction_horizon,
                },
                f,
            )

        print(f"Model saved to {path}")

    def load(self, path: str = "models/countdown_lstm"):
        """Load model and preprocessing objects"""
        # Load model
        self.model = keras.models.load_model(f"{path}/model.keras")

        # Load scalers and encoders
        with open(f"{path}/scalers.pkl", "rb") as f:
            saved_objects = pickle.load(f)
            self.feature_scaler = saved_objects["feature_scaler"]
            self.target_scaler = saved_objects["target_scaler"]
            self.route_encoder = saved_objects["route_encoder"]
            self.stop_encoder = saved_objects["stop_encoder"]
            self.fill_values = saved_objects["fill_values"]
            self.feature_columns = saved_objects["feature_columns"]
            self.sequence_length = saved_objects["sequence_length"]
            self.prediction_horizon = saved_objects["prediction_horizon"]

        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage with multiple files

    # Get all parquet files (assumes daily files named like YYYY-MM-DD.parquet)
    parquet_files = sorted(glob("data/*.parquet"))

    # Or single file for testing
    # parquet_files = ['MBTA_LAMP.parquet']

    print(f"Found {len(parquet_files)} parquet files")

    # Initialize model
    lstm = CountdownClockLSTM(sequence_length=5, prediction_horizon=10)

    # Train with file batching
    # Adjust file_batch_size based on your RAM
    # (5 files should be much more memory-friendly)
    history = lstm.train(parquet_files, epochs=50, batch_size=64, file_batch_size=5)

    # Save
    lstm.save("models/countdown_lstm")
