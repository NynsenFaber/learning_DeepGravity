"""
Census Sections Subsetter.

This script creates spatial subsets of the Italian Census Sections dataset
corresponding to each Province. It uses OpenStreetMap (PBF) data to determine
the province boundaries and filters the census sections accordingly.

Usage:
    python create_subsets.py

"""

import gc
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pyrosm
from shapely.geometry import box, Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---

@dataclass(frozen=True)
class Config:
    """Configuration settings for file paths and CRS."""
    BASE_DIR: Path = Path(__file__).resolve().parent
    SECTIONS_PATH: Path = (BASE_DIR / "../data/processed_data/sections_dataset.parquet").resolve()
    PBF_DIR: Path = (BASE_DIR / "../data/pbf_provinces").resolve()
    OUTPUT_DIR: Path = (BASE_DIR / "../data/processed_data/sections").resolve()

    # CRS Constants
    CRS_SOURCE: int = 4326   # WGS84 (OSM standard)
    CRS_METRIC: int = 3003   # Italy TM (Metric, accurate for Italy)

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configures logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("subsetter")
    logger.setLevel(logging.INFO)

    # File Handler (Detailed logs)
    file_handler = logging.FileHandler(log_dir / "processing.log", mode='w')
    file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_fmt)

    # Console Handler (Cleaner output, rely on tqdm for progress)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING) # Only show warnings/errors in console
    console_fmt = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_fmt)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# --- Core Logic Functions ---

def load_census_sections(path: Path, target_crs: int) -> gpd.GeoDataFrame:
    """Loads and projects the master census sections dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Census dataset not found at: {path}")

    logging.info("Loading census sections into memory...")
    gdf = gpd.read_parquet(path)

    if gdf.crs is None:
        raise ValueError("Census dataset is missing CRS information.")

    if gdf.crs.to_epsg() != target_crs:
        gdf = gdf.to_crs(epsg=target_crs)

    return gdf

def get_province_geometry(osm_path: Path, target_crs: int) -> Optional[Polygon]:
    """
    Extracts the province geometry from PBF.
    Tries administrative boundaries first, falls back to road network bounding box.
    """
    try:
        osm = pyrosm.OSM(str(osm_path))

        # Strategy 1: Administrative Boundaries
        boundaries = osm.get_boundaries()
        if boundaries is not None and not boundaries.empty:
            if boundaries.crs.to_epsg() != target_crs:
                boundaries = boundaries.to_crs(epsg=target_crs)
            # make_valid to fix any geometry issues
            boundaries['geometry'] = boundaries['geometry'].apply(make_valid)
            # Use convex hull to simplify complex multipolygons and cover enclaves
            return boundaries.union_all().convex_hull

        # Strategy 2: Fallback to Road Network Bounding Box
        logging.info(f"Boundary missing for {osm_path.name}, falling back to road network.")
        roads = osm.get_network(network_type="driving")
        if roads is not None and not roads.empty:
            # Roads are usually in WGS84 (4326)
            minx, miny, maxx, maxy = roads.total_bounds
            bbox = box(minx, miny, maxx, maxy)

            # Project the bbox to target metric CRS
            bbox_gs = gpd.GeoSeries([bbox], crs="EPSG:4326")
            bbox_projected = bbox_gs.to_crs(epsg=target_crs)
            return bbox_projected.iloc[0]

    except Exception as e:
        logging.error(f"Failed to extract geometry from {osm_path.name}: {e}")

    return None

def process_province(
    pbf_path: Path,
    sections_gdf: gpd.GeoDataFrame,
    config: Config,
    logger: logging.Logger
) -> None:
    """
    Processes a single province PBF file: extracts geometry, subsets sections, and saves.
    """
    output_file = config.OUTPUT_DIR / f"sections_subset_{pbf_path.stem}.parquet"

    # Skip if already exists
    if output_file.exists():
        return

    try:
        # 1. Get Province Geometry
        province_geom = get_province_geometry(pbf_path, config.CRS_METRIC)

        if province_geom is None:
            logger.warning(f"Skipping {pbf_path.name}: Could not determine spatial boundary.")
            return

        # 2. Spatial Filter
        # intersects is generally faster than within for this initial culling
        subset = sections_gdf[sections_gdf.intersects(province_geom)].copy()

        if subset.empty:
            logger.warning(f"Skipping {pbf_path.name}: No census sections found in boundary.")
            return

        # 3. Save
        subset.to_parquet(output_file, compression='brotli')
        logger.info(f"Saved {pbf_path.name}: {len(subset)} sections.")

    except Exception as e:
        logger.error(f"Critical error processing {pbf_path.name}: {e}")
        # Save explicit error log for debugging later
        (config.OUTPUT_DIR / f"ERROR_{pbf_path.stem}.txt").write_text(str(e))

    finally:
        # Explicit garbage collection after heavy PBF processing
        gc.collect()

# --- Main Execution ---

def main():
    config = Config()
    logger = setup_logging(config.OUTPUT_DIR)

    # 1. Validation
    if not config.PBF_DIR.exists():
        logger.critical(f"PBF Directory not found: {config.PBF_DIR}")
        return

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pbf_files = list(config.PBF_DIR.glob("*.pbf"))
    if not pbf_files:
        logger.critical("No PBF files found in directory.")
        return

    # 2. Load Data
    try:
        sections_gdf = load_census_sections(config.SECTIONS_PATH, config.CRS_METRIC)
    except Exception as e:
        logger.critical(f"Failed to load sections dataset: {e}")
        return

    print(f"🚀 Starting processing for {len(pbf_files)} provinces.")
    print(f"📂 Output directory: {config.OUTPUT_DIR}")

    # 3. Processing Loop
    # using tqdm for a progress bar is much cleaner than manual screen clearing
    for pbf_path in tqdm(pbf_files, unit="prov", desc="Processing"):
        process_province(pbf_path, sections_gdf, config, logger)

    print("\n✨ Processing Complete!")

if __name__ == "__main__":
    main()