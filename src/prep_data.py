import argparse
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
import logging
import shutil
import urllib.request
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data folder in the project root (parent of src)
DATA_DIR = Path(__file__).parent.parent / "data"


def download_index_csv(data_dir: Path = DATA_DIR):
    """
    Download MBTA subway on-time performance data from the official API.
    Saves the data to the data folder in the project root.
    """
    # URL for MBTA subway on-time performance data
    url = (
        "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/index.csv"
    )

    # Ensure data directory exists
    data_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"mbta_subway_ontime_performance_{timestamp}.csv"
    filepath = data_dir / filename

    # Also save as latest version
    latest_filepath = data_dir / "mbta_subway_ontime_performance_latest.csv"

    try:
        logger.info(f"Downloading MBTA data from: {url}")
        urllib.request.urlretrieve(url, filepath)
        logger.info(f"Successfully downloaded data to: {filepath}")

        # Validate the downloaded data
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data validation successful. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            # Remove invalid file, leaving any previous latest version intact
            if filepath.exists():
                filepath.unlink()
            raise

        shutil.copyfile(filepath, latest_filepath)
        logger.info(f"Also saved as latest version: {latest_filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise


def download_parquet_files(
    start: Optional[date] = None,
    end: Optional[date] = None,
    days: int = 180,
    data_dir: Path = DATA_DIR,
) -> List[Path]:
    """
    Download the daily parquet files listed in the latest index for service dates
    from start to end, inclusive. end defaults to the newest date in the index and
    start to `days` days before it. Files already downloaded at the size the index
    lists are skipped, so re-running only fetches what's new or incomplete.
    """
    index = pd.read_csv(
        data_dir / "mbta_subway_ontime_performance_latest.csv",
        parse_dates=["service_date"],
    )
    end = end or index["service_date"].max().date()
    start = start or end - timedelta(days=days - 1)
    wanted = index[index["service_date"].dt.date.between(start, end)]
    logger.info(f"{len(wanted)} daily files from {start} to {end}")

    downloaded, skipped, failed = [], 0, []
    for row in wanted.itertuples():
        path = data_dir / row.file_url.rsplit("/", 1)[-1]
        if path.exists() and path.stat().st_size == row.size_bytes:
            skipped += 1
            continue

        # Download under a temporary name so an interrupted download never leaves
        # a partial .parquet file for training to pick up
        partial = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(row.file_url, partial)
            partial.replace(path)
            downloaded.append(path)
        except Exception as e:
            logger.error(f"Failed to download {row.file_url}: {e}")
            partial.unlink(missing_ok=True)
            failed.append(row.file_url)

    logger.info(f"Downloaded {len(downloaded)}, skipped {skipped} already present")
    if failed:
        raise RuntimeError(f"{len(failed)} downloads failed: {failed}")
    return downloaded


# The MBTA's weekly Rapid Transit and Bus Prediction Accuracy Data, from the
# MassDOT/MBTA open data portal (mbta-massdot.opendata.arcgis.com)
PREDICTION_ACCURACY_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "155ab68df00145cabddfb90377201b0e/data"
)
PREDICTION_ACCURACY_COLUMNS = {
    "weekly",
    "mode",
    "route_id",
    "bin",
    "arrival_departure",
    "num_predictions",
    "num_accurate_predictions",
}


def download_prediction_accuracy(data_dir: Path = DATA_DIR) -> Path:
    """
    Download how accurate the MBTA's own arrival predictions were each week, by
    line and how far ahead, to compare our model against (evaluate.py
    --mbta-accuracy). The previous copy is kept if the download isn't valid.
    """
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "mbta_prediction_accuracy.csv"
    partial = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(PREDICTION_ACCURACY_URL, partial)
    try:
        df = pd.read_csv(partial)
        missing = PREDICTION_ACCURACY_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Prediction accuracy data is missing {sorted(missing)}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    logger.info(f"Prediction accuracy through week {df['weekly'].max()}: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Download MBTA LAMP subway on-time performance data into data/"
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="First service date, YYYY-MM-DD (default: --days before --end)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        help="Last service date, YYYY-MM-DD (default: the newest in the index)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Days to download when --start isn't given (default: 180)",
    )
    args = parser.parse_args()

    download_index_csv()
    download_parquet_files(args.start, args.end, args.days)
    download_prediction_accuracy()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error downloading MBTA data: {e}")
        exit(1)
