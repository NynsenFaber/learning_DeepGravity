"""
Italy Tessellation Generator.

This script generates a regular grid (tessellation) of 25km x 25km cells covering
the geographical extent of Italy. It filters the grid to strictly retain cells
that intersect with the Italian landmass (census sections).

Methodology:
1. Load Italian census sections.
2. Project to Web Mercator (EPSG:3857) to ensure accurate metric units (meters).
3. Generate a grid of square polygons based on the total bounds.
4. Filter grid cells using a Spatial Join (sjoin) for performance.
5. Save the result as a Parquet file.

"""

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

# Suppress specific future warnings from libraries if necessary
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = (BASE_DIR / "../data/processed_data/sections_dataset.parquet").resolve()
OUTPUT_PATH = (BASE_DIR / "../data/processed_data/tessellation_25km.parquet").resolve()

# Constants
METRIC_CRS = "EPSG:3857"  # Web Mercator (Unit: Meters)
OUTPUT_CRS = "EPSG:4326"  # WGS84 (Unit: Degrees)
GRID_SIZE_METERS = 25_000.0  # 25km


def create_tessellation_grid(gdf: gpd.GeoDataFrame, cell_size: float) -> gpd.GeoDataFrame:
    """
    Generates a square grid covering the bounding box of the input GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): Source data to define the bounds.
        cell_size (float): The width/height of the square cells in CRS units.

    Returns:
        gpd.GeoDataFrame: A grid of Polygon geometries.
    """
    logger.info(f"Generating grid with cell size: {cell_size} units...")

    # 1. Get total bounds (min_x, min_y, max_x, max_y)
    minx, miny, maxx, maxy = gdf.total_bounds

    # 2. Generate coordinate ranges
    # We use arange to create steps of `cell_size`.
    # We subtract/add a buffer to ensure full coverage of the edges.
    x_coords = np.arange(minx, maxx + cell_size, cell_size)
    y_coords = np.arange(miny, maxy + cell_size, cell_size)

    # 3. Create grid polygons
    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            # Create a box: (minx, miny, maxx, maxy)
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))

    logger.info(f"Created initial grid with {len(grid_cells)} cells.")

    return gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf.crs)


def filter_grid_by_intersection(grid: gpd.GeoDataFrame, target: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Retains only the grid cells that spatially intersect with the target geometry.

    Uses a Spatial Join (sjoin) which is significantly faster than
    iterating through intersections or creating a Unary Union.
    """
    logger.info("Filtering grid cells that do not intersect with Italy...")

    # Perform Spatial Join.
    # 'inner' join keeps only records from 'grid' that match 'target'.
    # 'predicate=intersects' checks if they touch/overlap.
    matching_cells = gpd.sjoin(grid, target, how="inner", predicate="intersects")

    # The join might duplicate grid cells if they touch multiple sections,
    # so we filter the original grid by the indices that matched.
    filtered_grid = grid.loc[matching_cells.index.unique()].copy()

    return filtered_grid.reset_index(drop=True)


def main():
    """Main execution flow."""
    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return

    # 1. Load Data
    logger.info(f"Loading data from {INPUT_PATH}...")
    gdf_italy = gpd.read_parquet(INPUT_PATH)

    # 2. Reproject to Metric CRS
    # We must use a projected CRS (meters) to define a "25km" grid.
    if gdf_italy.crs != METRIC_CRS:
        logger.info(f"Reprojecting input data to {METRIC_CRS}...")
        gdf_italy = gdf_italy.to_crs(METRIC_CRS)

    # 3. Create Tessellation
    tessellation = create_tessellation_grid(gdf_italy, cell_size=GRID_SIZE_METERS)
    initial_count = len(tessellation)

    # 4. Filter Empty Cells
    # Remove grid cells that fall entirely into the sea (do not touch land)
    tessellation = filter_grid_by_intersection(tessellation, gdf_italy)

    discarded_count = initial_count - len(tessellation)
    logger.info(f"Filtering complete. Discarded {discarded_count} ocean/empty cells.")
    logger.info(f"Final grid count: {len(tessellation)} cells.")

    # 5. Finalize and Save
    # Convert back to WGS84 (standard lat/lon) for storage/sharing
    logger.info(f"Reprojecting output to {OUTPUT_CRS}...")
    tessellation = tessellation.to_crs(OUTPUT_CRS)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {OUTPUT_PATH}...")
    tessellation.to_parquet(OUTPUT_PATH, compression='brotli')
    logger.info("Process completed successfully.")


if __name__ == "__main__":
    main()