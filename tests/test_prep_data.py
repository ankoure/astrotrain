import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import prep_data

BASE_URL = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/"
PARQUET_BODY = "parquet bytes"


def parquet_name(service_date):
    return f"{service_date}-subway-on-time-performance-v1.parquet"


def index_csv(dates, size=len(PARQUET_BODY)):
    """An index like the real one, listing one file per service date"""
    rows = "".join(
        f"{size},2026-10-09 00:00:00+00:00,{d},{BASE_URL}{parquet_name(d)}\n"
        for d in dates
    )
    return "size_bytes,last_modified,service_date,file_url\n" + rows


INDEX_CSV = index_csv(["2025-09-01", "2025-09-02"])


class FakeDownload:
    """
    Stands in for urlretrieve: records each call and writes `content`. URLs in
    `fail` get half their content written and then raise, like a dropped connection.
    """

    def __init__(self):
        self.calls = []
        self.content = INDEX_CSV
        self.fail = set()

    def __call__(self, url, filename):
        self.calls.append((url, str(filename)))
        if url in self.fail:
            Path(filename).write_text(self.content[: len(self.content) // 2])
            raise OSError("connection reset")
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


@pytest.fixture
def data_dir(tmp_path, fake_download):
    """A data folder holding a latest index of 2025-09-01 to 2025-09-05"""
    dates = [f"2025-09-0{d}" for d in range(1, 6)]
    (tmp_path / "mbta_subway_ontime_performance_latest.csv").write_text(
        index_csv(dates)
    )
    fake_download.content = PARQUET_BODY
    return tmp_path


def test_downloads_only_files_in_date_range(data_dir, fake_download):
    paths = prep_data.download_parquet_files(
        date(2025, 9, 2), date(2025, 9, 4), data_dir=data_dir
    )

    expected = [parquet_name(f"2025-09-0{d}") for d in (2, 3, 4)]
    assert [p.name for p in paths] == expected
    assert [url for url, _ in fake_download.calls] == [BASE_URL + n for n in expected]
    assert all((data_dir / n).read_text() == PARQUET_BODY for n in expected)


def test_default_range_is_the_newest_days_in_the_index(data_dir):
    paths = prep_data.download_parquet_files(days=2, data_dir=data_dir)

    assert [p.name for p in paths] == [
        parquet_name("2025-09-04"),
        parquet_name("2025-09-05"),
    ]


def test_skips_complete_files_and_redownloads_incomplete_ones(data_dir):
    (data_dir / parquet_name("2025-09-01")).write_text(PARQUET_BODY)
    (data_dir / parquet_name("2025-09-02")).write_text(PARQUET_BODY[:3])

    paths = prep_data.download_parquet_files(
        date(2025, 9, 1), date(2025, 9, 3), data_dir=data_dir
    )

    assert [p.name for p in paths] == [
        parquet_name("2025-09-02"),
        parquet_name("2025-09-03"),
    ]


def test_failed_download_leaves_no_partial_file_and_others_continue(
    data_dir, fake_download
):
    fake_download.fail = {BASE_URL + parquet_name("2025-09-02")}

    with pytest.raises(RuntimeError, match="1 downloads failed"):
        prep_data.download_parquet_files(
            date(2025, 9, 1), date(2025, 9, 3), data_dir=data_dir
        )

    assert sorted(p.name for p in data_dir.glob("*.parquet*")) == [
        parquet_name("2025-09-01"),
        parquet_name("2025-09-03"),
    ]
