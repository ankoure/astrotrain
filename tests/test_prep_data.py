import re
from pathlib import Path

import pandas as pd
import pytest

import prep_data

BASE_URL = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/"
PARQUET_NAMES = [
    "2025-09-01-subway-on-time-performance-v1.parquet",
    "2025-09-02-subway-on-time-performance-v1.parquet",
]
INDEX_CSV = "file_url\n" + "".join(f"{BASE_URL}{name}\n" for name in PARQUET_NAMES)


class FakeDownload:
    """Stands in for urlretrieve: records each call and writes `content`"""

    def __init__(self):
        self.calls = []
        self.content = INDEX_CSV

    def __call__(self, url, filename):
        self.calls.append((url, str(filename)))
        Path(filename).write_text(self.content)


@pytest.fixture
def fake_download(monkeypatch):
    fake = FakeDownload()
    monkeypatch.setattr(prep_data.urllib.request, "urlretrieve", fake)
    return fake


def test_download_index_saves_dated_and_latest_copies(tmp_path, fake_download):
    data_dir = tmp_path / "data"

    path = prep_data.download_index_csv(data_dir=data_dir)

    latest = data_dir / "mbta_subway_ontime_performance_latest.csv"
    assert len(fake_download.calls) == 1  # latest is copied, not downloaded again
    assert path.parent == data_dir
    assert re.fullmatch(r"mbta_subway_ontime_performance_\d{8}\.csv", path.name)
    assert path.read_text() == latest.read_text() == INDEX_CSV


def test_bad_download_is_removed_and_previous_latest_kept(tmp_path, fake_download):
    prep_data.download_index_csv(data_dir=tmp_path)
    fake_download.content = ""  # an empty response fails CSV validation

    with pytest.raises(pd.errors.EmptyDataError):
        prep_data.download_index_csv(data_dir=tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == [
        "mbta_subway_ontime_performance_latest.csv"
    ]
    assert (tmp_path / "mbta_subway_ontime_performance_latest.csv").read_text() == (
        INDEX_CSV
    )


def test_download_parquet_files_listed_in_index(tmp_path, monkeypatch, fake_download):
    monkeypatch.chdir(tmp_path)  # this function uses paths relative to the cwd
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "mbta_subway_ontime_performance_latest.csv").write_text(
        INDEX_CSV
    )

    prep_data.download_parquet_from_csv()

    assert fake_download.calls == [
        (f"{BASE_URL}{name}", f"data/{name}") for name in PARQUET_NAMES
    ]
