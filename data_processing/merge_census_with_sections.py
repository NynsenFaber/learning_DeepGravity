"""
Census Sections Data Processor.

This script merges ISTAT census data (demographics) with ISTAT territorial bases 
(geometry/shapefiles). It processes 20 Italian regions, cleans the data, 
and aggregates it into a single Geospatial Parquet file.

Output:
    - GeoDataFrame indexed by 'SEZ2011' (Census Section ID)
    - Columns: ['geometry', 'P1' (Total Population)]
    - CRS: EPSG:4326 (WGS84)
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

# --- Configuration ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
# Adjust these relatives paths as needed based on your actual file structure
ISTAT_BASE = (BASE_DIR / "../../data/ISTAT").resolve()
CENSUS_DIR = ISTAT_BASE / "dati-cpa_2011/Sezioni di Censimento"
SHAPE_DIR = ISTAT_BASE / "basi territoriali"
OUTPUT_PATH = (BASE_DIR / "../data/processed_data/sections_dataset.parquet").resolve()

# Constants
TARGET_CRS = "EPSG:4326"  # WGS84
CSV_ENCODING = "latin-1"
CSV_SEP = ";"


def get_file_paths(region_id: int) -> tuple[Path, Path]:
    """
    Generates the specific CSV and Shapefile paths for a given region ID.

    Args:
        region_id (int): The numeric ID of the region (1-20).

    Returns:
        tuple: (Path to CSV, Path to Shapefile)
    """
    # Format ID with leading zero, e.g., 1 -> "01"
    rid_str = f"{region_id:02d}"

    # Construct paths based on the standard ISTAT naming convention
    csv_path = CENSUS_DIR / f"R{rid_str}_indicatori_2011_sezioni.csv"

    # Shapefiles are often nested in folders like R01_11_WGS84
    shp_folder = SHAPE_DIR / f"R{rid_str}_11_WGS84"
    shp_path = shp_folder / f"R{rid_str}_11_WGS84.shp"

    return csv_path, shp_path


def process_region(region_id: int) -> Optional[gpd.GeoDataFrame]:
    """
    Loads and merges census and geometry data for a single region.

    Args:
        region_id (int): Region number.

    Returns:
        gpd.GeoDataFrame: Processed regional data or None if files are missing.
    """
    csv_path, shp_path = get_file_paths(region_id)

    # Validate file existence
    if not csv_path.exists():
        logger.warning(f"CSV missing for Region {region_id}: {csv_path}")
        return None
    if not shp_path.exists():
        logger.warning(f"Shapefile missing for Region {region_id}: {shp_path}")
        return None

    logger.info(f"Processing Region {region_id:02d}...")

    try:
        # 1. Load Data
        df_census = pd.read_csv(csv_path, sep=CSV_SEP, encoding=CSV_ENCODING)
        gdf_shapes = gpd.read_file(shp_path)

        # 2. Standardize Join Keys
        # Ensure SEZ2011 is treated as a string to preserve leading zeros and matching
        df_census['SEZ2011'] = df_census['SEZ2011'].astype(str)
        gdf_shapes['SEZ2011'] = gdf_shapes['SEZ2011'].astype(str)

        # Set Indices
        df_census.set_index('SEZ2011', inplace=True)
        gdf_shapes.set_index('SEZ2011', inplace=True)

        # 3. Merge
        # We perform a Left Join on the Geometry to ensure we only keep records 
        # that have a physical location.
        merged_gdf = pd.merge(
            gdf_shapes[['geometry']],
            df_census[['P1']],
            left_index=True,
            right_index=True,
            how='left'
        )

        # 4. Clean Data
        # Fill missing population data with 0
        merged_gdf['P1'] = merged_gdf['P1'].fillna(0)

        # 5. Coordinate Reference System (CRS) Standardization
        # The input folder name implies WGS84, but we ensure it here.
        if merged_gdf.crs != TARGET_CRS:
            merged_gdf = merged_gdf.to_crs(TARGET_CRS)

        return merged_gdf

    except Exception as e:
        logger.error(f"Error processing Region {region_id}: {e}")
        return None


def main():
    """
    Main execution flow.
    """
    regional_datasets = []

    # Loop through all 20 Italian regions
    for region_id in range(1, 21):
        gdf = process_region(region_id)
        if gdf is not None:
            regional_datasets.append(gdf)

    if not regional_datasets:
        logger.error("No datasets were processed. Exiting.")
        return

    # Combine all regions
    logger.info("Concatenating all regions...")
    full_dataset = pd.concat(regional_datasets)

    # Ensure it's still a GeoDataFrame after concat
    full_dataset = gpd.GeoDataFrame(full_dataset, crs=TARGET_CRS)

    # Save Output
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving final dataset to {OUTPUT_PATH}...")

    # Parquet is efficient for large geospatial data
    full_dataset.to_parquet(OUTPUT_PATH, compression='brotli')

    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    main()