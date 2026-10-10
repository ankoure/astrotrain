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
