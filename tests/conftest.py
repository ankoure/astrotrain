import pytest

from astrotrain.train_lstm import CountdownClockLSTM
from tests.lamp_data import write_daily_files


@pytest.fixture
def parquet_files(tmp_path):
    return write_daily_files(tmp_path / "data")


@pytest.fixture
def lstm(parquet_files):
    """Small model with encoders and fill values fit on parquet_files"""
    model = CountdownClockLSTM(sequence_length=3, prediction_horizon=1, seed=0)
    model.fit_encoders(parquet_files)
    model.fit_fill_values(model.load_data(parquet_files))
    return model


@pytest.fixture
def built_lstm(lstm, parquet_files):
    """lstm with scalers fit and an untrained model, ready to predict or save"""
    df = lstm.preprocess_data(lstm.load_data(parquet_files))
    X, y = lstm.create_sequences(df)
    lstm.feature_scaler.fit(X.reshape(-1, X.shape[-1]))
    lstm.target_scaler.fit(y)
    lstm.build_model(input_shape=X.shape[1:])
    return lstm
