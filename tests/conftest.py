import pytest

from astrotrain.train_lstm import CountdownClockLSTM
from tests.lamp_data import write_daily_files


@pytest.fixture
def parquet_files(tmp_path):
    return write_daily_files(tmp_path / "data")


def fitted_model(parquet_files, prediction_horizon):
    """Small model with encoders and fill values fit on parquet_files"""
    model = CountdownClockLSTM(
        sequence_length=3, prediction_horizon=prediction_horizon, seed=0
    )
    model.fit_encoders(parquet_files)
    model.fit_fill_values(model.load_data(parquet_files))
    return model


@pytest.fixture
def lstm(parquet_files):
    """Predicts 1 stop ahead, so target_1 is the next stop's travel time"""
    return fitted_model(parquet_files, prediction_horizon=1)


@pytest.fixture
def lstm3(parquet_files):
    """Predicts arrival at each of the next 3 stops"""
    return fitted_model(parquet_files, prediction_horizon=3)


@pytest.fixture
def built_lstm(lstm3, parquet_files):
    """lstm3 with scalers fit and an untrained model, ready to predict or save"""
    df = lstm3.preprocess_data(lstm3.load_data(parquet_files))
    X, y = lstm3.create_sequences(df)
    lstm3.feature_scaler.fit(X.reshape(-1, X.shape[-1]))
    lstm3.target_scaler.fit(y)
    lstm3.build_model(input_shape=X.shape[1:])
    return lstm3
