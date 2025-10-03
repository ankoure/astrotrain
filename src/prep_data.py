import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import urllib.request

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_index_csv():
    """
    Download MBTA subway on-time performance data from the official API.
    Saves the data to the data folder in the project root.
    """
    # URL for MBTA subway on-time performance data
    url = (
        "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/index.csv"
    )

    # Get the project root directory (parent of src)
    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    data_dir = project_root / "data"

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

        logger.info(f"Successfully downloaded data to: {filepath}")
        urllib.request.urlretrieve(url, filepath)
        logger.info(f"Also saved as latest version: {latest_filepath}")
        urllib.request.urlretrieve(url, latest_filepath)

        # Validate the downloaded data
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data validation successful. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            return filepath

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            # Remove invalid file
            if filepath.exists():
                filepath.unlink()
            if latest_filepath.exists():
                latest_filepath.unlink()
            raise

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
