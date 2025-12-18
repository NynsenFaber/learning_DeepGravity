"""
OpenStreetMap Italy PBF Downloader.

This script automates the downloading of OpenStreetMap protocol buffer (PBF) files
for all Italian provinces. It fetches the official list of provinces from OpenPolis,
constructs the download URLs for the `osmit-estratti` service, and downloads
the files in parallel.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import requests

# --- Configuration & Constants ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Network Constants
BASE_DOWNLOAD_URL = "https://osmit-estratti.wmcloud.org/output/pbf/province"
PROVINCES_METADATA_URL = (
    "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_provinces.geojson"
)
REQUEST_TIMEOUT = 30  # seconds
CHUNK_SIZE = 8192  # bytes

# File System Constants
# Using .resolve() ensures absolute paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (BASE_DIR / "../data/pbf_provinces").resolve()
MAX_WORKERS = 4


def sanitize_province_name(name: str) -> str:
    """
    Sanitizes a province name to match the OSM server's naming convention.

    Rules:
    - Replace spaces with underscores.
    - Replace apostrophes with underscores.
    - Replace slashes (often found in bilingual names) with underscores.

    Args:
        name (str): The raw province name (e.g., "Valle d'Aosta").

    Returns:
        str: The sanitized name (e.g., "Valle_d_Aosta").
    """
    return name.replace(" ", "_").replace("'", "_").replace("/", "_")


def fetch_province_metadata() -> List[Dict[str, any]]:
    """
    Fetches the official list of Italian provinces from the OpenPolis repository.

    Returns:
        List[Dict]: A list of dictionaries containing province codes and names.
                    Example: [{'code': 1, 'name': 'Torino'}, ...]
    """
    logger.info("Fetching province list from OpenPolis...")
    try:
        response = requests.get(PROVINCES_METADATA_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        provinces = []
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            provinces.append({
                'code': props.get('prov_istat_code_num'),
                'name': props.get('prov_name')
            })

        logger.info(f"Successfully retrieved metadata for {len(provinces)} provinces.")
        return provinces

    except requests.RequestException as e:
        logger.error(f"Failed to fetch province metadata: {e}")
        return []


def download_province_pbf(province_info: Dict[str, any]) -> None:
    """
    Downloads a single PBF file for a specific province.

    Args:
        province_info (Dict): Dictionary with 'code' and 'name'.
    """
    # 1. Prepare Filenames and Paths
    code_str = f"{province_info['code']:03d}"  # Ensure 3 digits (e.g., 001)
    safe_name = sanitize_province_name(province_info['name'])

    filename = f"{code_str}_{safe_name}.osm.pbf"
    file_path = OUTPUT_DIR / filename
    download_url = f"{BASE_DOWNLOAD_URL}/{filename}"

    # 2. Skip if already exists
    if file_path.exists():
        # Optional: You could check file size here to ensure it's not empty
        if file_path.stat().st_size > 0:
            logger.info(f"Skipping {filename} (already exists).")
            return
        else:
            logger.warning(f"File {filename} exists but is empty. Re-downloading.")

    # 3. Download Logic
    try:
        # HEAD request to check if file exists on server before downloading
        head_response = requests.head(download_url, timeout=REQUEST_TIMEOUT)

        if head_response.status_code != 200:
            logger.warning(f"File not found on server: {filename} (URL: {download_url})")
            return

        logger.info(f"Downloading {filename}...")

        with requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()

            # Write to disk in chunks to handle large files memory-efficiently
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)

        logger.info(f"✓ Successfully saved {filename}")

    except requests.RequestException as e:
        logger.error(f"✗ Network error downloading {filename}: {e}")
    except IOError as e:
        logger.error(f"✗ File system error saving {filename}: {e}")


def main():
    """
    Main execution flow.
    """
    # 1. Setup Output Directory
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory confirmed: {OUTPUT_DIR}")
    except IOError as e:
        logger.critical(f"Could not create output directory: {e}")
        return

    # 2. Get Data
    provinces = fetch_province_metadata()

    if not provinces:
        logger.error("No province data found. Exiting.")
        return

    # 3. Parallel Execution
    logger.info(f"Starting download with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_province_pbf, provinces)

    logger.info("All download tasks completed.")


if __name__ == "__main__":
    main()