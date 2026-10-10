import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import shutil
import urllib.request

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


def download_parquet_from_csv():
    """
    Downloads the Parquet data from the MBTA LAMP data source.
    """
    filepath = r"data/mbta_subway_ontime_performance_latest.csv"
    df = pd.read_csv(filepath)
    files = df["file_url"].values.tolist()
    for file in files:
        filename = file.replace(
            "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/",
            "data/",
        )
        urllib.request.urlretrieve(file, filename)


if __name__ == "__main__":
    try:
        download_index_csv()
        # download_parquet_from_csv()
    except Exception as e:
        print(f"Error downloading MBTA data: {e}")
        exit(1)
