"""
LSTM Model for MBTA Countdown Clock Predictions
Trains on historical LAMP data to predict vehicle arrival times
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from typing import List, Tuple
from glob import glob


class CountdownClockLSTM:
    """LSTM model for predicting MBTA countdown clock times"""

    def __init__(self, sequence_length: int = 10, prediction_horizon: int = 1):
        """
        Args:
            sequence_length: Number of historical stops to use as input
            prediction_horizon: How many stops ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.route_encoder = LabelEncoder()
        self.stop_encoder = LabelEncoder()
        self.feature_columns = []

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

    def sequence_generator(self, parquet_files: List[str], batch_size: int = 32):
        """
        Memory-efficient generator that yields batches of sequences from files.

        Args:
            parquet_files: List of parquet file paths
            batch_size: Number of files to load at once

        Yields:
            Tuple of (X_batch, y_batch) arrays
        """
        for i in range(0, len(parquet_files), batch_size):
            file_batch = parquet_files[i:i + batch_size]
            print(f"\nProcessing files {i+1} to {min(i+batch_size, len(parquet_files))} of {len(parquet_files)}")

            # Load and process this batch of files
            df = self.load_data(file_batch)
            df = self.preprocess_data(df)
            X, y = self.create_sequences(df)

            # Scale features
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = self.feature_scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

            # Scale targets
            y_scaled = self.target_scaler.transform(y)

            yield X_scaled, y_scaled

            # Clear memory
            del df, X, y, X_reshaped, X_scaled, y_scaled

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for modeling"""
        # Sort by trip and stop sequence
        df = df.sort_values(['trip_id', 'stop_sequence']).reset_index(drop=True)

        # Create target variable: time until arrival (for countdown clock)
        # We'll predict travel_time_seconds to the next stop
        df['target_travel_time'] = df.groupby('trip_id')['travel_time_seconds'].shift(-self.prediction_horizon)

        # Remove rows without target
        df = df.dropna(subset=['target_travel_time'])

        # Fill missing values with median for numeric columns
        numeric_cols = ['travel_time_seconds', 'dwell_time_seconds',
                       'headway_trunk_seconds', 'headway_branch_seconds',
                       'scheduled_travel_time', 'scheduled_headway_trunk',
                       'scheduled_headway_branch']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Encode categorical variables
        df['route_encoded'] = self.route_encoder.fit_transform(df['route_id'])
        df['stop_encoded'] = self.stop_encoder.fit_transform(df['parent_station'])

        # Create time-based features
        df['hour_of_day'] = (df['start_time'] // 3600) % 24
        df['day_of_week'] = pd.to_datetime(df['service_date'], format='%Y%m%d').dt.dayofweek

        # Binary direction
        df['direction_binary'] = df['direction_id'].astype(int)

        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series sequences for LSTM"""
        # Define features for the model
        self.feature_columns = [
            'stop_sequence', 'travel_time_seconds', 'dwell_time_seconds',
            'headway_trunk_seconds', 'headway_branch_seconds',
            'scheduled_travel_time', 'scheduled_headway_trunk',
            'route_encoded', 'stop_encoded', 'hour_of_day',
            'day_of_week', 'direction_binary', 'stop_count'
        ]

        X_sequences = []
        y_targets = []

        # Group by trip to maintain temporal continuity
        for trip_id, trip_df in df.groupby('trip_id'):
            if len(trip_df) < self.sequence_length + 1:
                continue

            trip_features = trip_df[self.feature_columns].values
            trip_targets = trip_df['target_travel_time'].values

            # Create sliding windows
            for i in range(len(trip_df) - self.sequence_length):
                X_sequences.append(trip_features[i:i + self.sequence_length])
                y_targets.append(trip_targets[i + self.sequence_length - 1])

        X = np.array(X_sequences)
        y = np.array(y_targets).reshape(-1, 1)

        print(f"Created {len(X)} sequences of shape {X.shape}")
        return X, y

    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM neural network"""
        model = keras.Sequential([
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
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),

            # Output layer (predicting travel time in seconds)
            layers.Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        return model

    def train(self, parquet_files: List[str], epochs: int = 50, batch_size: int = 64,
              file_batch_size: int = 30):
        """
        Complete training pipeline with memory-efficient file batching.

        Args:
            parquet_files: List of parquet file paths
            epochs: Number of training epochs
            batch_size: Mini-batch size for SGD
            file_batch_size: Number of parquet files to load at once (adjust based on RAM)
        """
        # First pass: fit scalers and encoders on a sample of data
        print("Fitting preprocessors on initial data sample...")
        sample_files = parquet_files[:min(10, len(parquet_files))]
        df_sample = self.load_data(sample_files)
        df_sample = self.preprocess_data(df_sample)
        X_sample, y_sample = self.create_sequences(df_sample)

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
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        # Split files into train/val sets (time-based split)
        split_idx = int(len(parquet_files) * 0.8)
        train_files = parquet_files[:split_idx]
        val_files = parquet_files[split_idx:]

        print(f"\nTraining on {len(train_files)} files, validating on {len(val_files)} files")

        # Train using generator
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")

            # Training phase
            train_losses = []
            train_maes = []
            for X_batch, y_batch in self.sequence_generator(train_files, file_batch_size):
                history = self.model.fit(
                    X_batch, y_batch,
                    epochs=1,
                    batch_size=batch_size,
                    verbose=0
                )
                train_losses.append(history.history['loss'][0])
                train_maes.append(history.history['mae'][0])

            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)

            # Validation phase
            val_losses = []
            val_maes = []
            for X_batch, y_batch in self.sequence_generator(val_files, file_batch_size):
                val_loss, val_mae, _ = self.model.evaluate(X_batch, y_batch, verbose=0)
                val_losses.append(val_loss)
                val_maes.append(val_mae)

            avg_val_loss = np.mean(val_losses)
            avg_val_mae = np.mean(val_maes)

            print(f"Epoch {epoch + 1} - loss: {avg_train_loss:.4f} - mae: {avg_train_mae:.2f} - "
                  f"val_loss: {avg_val_loss:.4f} - val_mae: {avg_val_mae:.2f}")

            # Check early stopping (simplified)
            if epoch > 0 and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.model.save_weights('temp_best_weights.weights.h5')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("\nEarly stopping triggered")
                    self.model.load_weights('temp_best_weights.weights.h5')
                    break

            if epoch == 0:
                best_val_loss = avg_val_loss
                patience_counter = 0

        print("\nTraining complete!")
        return None

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

    def save(self, path: str = 'models/countdown_lstm'):
        """Save model and preprocessing objects"""
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(f"{path}/model.keras")

        # Save scalers and encoders
        with open(f"{path}/scalers.pkl", 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'route_encoder': self.route_encoder,
                'stop_encoder': self.stop_encoder,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }, f)

        print(f"Model saved to {path}")

    def load(self, path: str = 'models/countdown_lstm'):
        """Load model and preprocessing objects"""
        # Load model
        self.model = keras.models.load_model(f"{path}/model.keras")

        # Load scalers and encoders
        with open(f"{path}/scalers.pkl", 'rb') as f:
            saved_objects = pickle.load(f)
            self.feature_scaler = saved_objects['feature_scaler']
            self.target_scaler = saved_objects['target_scaler']
            self.route_encoder = saved_objects['route_encoder']
            self.stop_encoder = saved_objects['stop_encoder']
            self.feature_columns = saved_objects['feature_columns']
            self.sequence_length = saved_objects['sequence_length']
            self.prediction_horizon = saved_objects['prediction_horizon']

        print(f"Model loaded from {path}")


if __name__ == '__main__':
    # Example usage with multiple files

    # Get all parquet files (assumes daily files named like YYYY-MM-DD.parquet)
    parquet_files = sorted(glob('data/*.parquet'))

    # Or single file for testing
    # parquet_files = ['MBTA_LAMP.parquet']

    print(f"Found {len(parquet_files)} parquet files")

    # Initialize model
    lstm = CountdownClockLSTM(sequence_length=10, prediction_horizon=1)

    # Train with file batching
    # Adjust file_batch_size based on your RAM (30 files ~= 1-2GB depending on file size)
    history = lstm.train(
        parquet_files,
        epochs=50,
        batch_size=64,
        file_batch_size=30
    )

    # Save
    lstm.save('models/countdown_lstm')
