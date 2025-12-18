"""
Feature Extraction Pipeline with Resume Capability.

This script enriches census section geometries with physical features.
It updates the files IN-PLACE.
It includes a 'features_extracted' flag to skip already completed files.

Usage:
    python extract_features.py
"""

import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import pyrosm
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---

@dataclass(frozen=True)
class Config:
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    SECTIONS_DIR: Path = (BASE_DIR / "../data/processed_data/sections").resolve()
    PBF_DIR: Path = (BASE_DIR / "../data/pbf_provinces").resolve()
    PROCESSED_DIR: Path = (BASE_DIR / "../data/processed_data").resolve()

    # CRS
    CRS_METRIC: int = 3003  # Italy TM

    # CONTROL FLAGS
    FORCE_UPDATE: bool = False  # Set to True to overwrite existing features

    # Feature Columns to Initialize
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        'landuse_residential', 'landuse_commercial', 'landuse_industrial',
        'landuse_retail', 'landuse_natural',
        'road_residential', 'road_main', 'road_other',
        'transport_poi', 'transport_building',
        'health_poi', 'health_building',
        'food_poi', 'food_building',
        'education_poi', 'education_building',
        'retail_poi', 'retail_building'
    ])

    # OSM Filter Tags
    OSM_TAGS: Dict[str, Dict] = field(default_factory=lambda: {
        "landuse": {
            "residential": ["residential", "housing"],
            "commercial": ["commercial", "office"],
            "industrial": ["industrial", "railway", "port"],
            "retail": ["retail"],
        },
        "roads": {
            "main": ["motorway", "trunk", "primary", "secondary", "motorway_link",
                     "primary_link"],
            "residential": ["residential", "living_street"],
        },
        "poi_categories": {
            "transport": {"amenity": ["bus_station", "parking", "taxi"],
                          "railway": ["station"], "highway": ["bus_stop"]},
            "food": {"amenity": ["restaurant", "cafe", "bar", "fast_food"],
                     "shop": ["supermarket", "bakery"]},
            "health": {"amenity": ["hospital", "pharmacy", "doctors"],
                       "building": ["hospital"]},
            "education": {"amenity": ["school", "university", "kindergarten"],
                          "building": ["school"]},
            "retail": {"shop": True, "building": ["retail"]}
        }
    })

# --- Helper Functions ---
def update_dataframe(target_df: pd.DataFrame, updates: pd.DataFrame) -> None:
    """Updates target_df in-place with values from updates."""
    if updates.empty: return
    cols = [c for c in updates.columns if c in target_df.columns]
    if cols:
        target_df.loc[updates.index, cols] += updates[cols]

# --- Classifiers ---

def classify_landuse(row: pd.Series, mapping: Dict[str, Dict]) -> str | None:
    lu_tag = str(row.get('landuse', '')).lower()

    for category, values in mapping['landuse'].items():
        if lu_tag in values:
            return f"landuse_{category}"

    return None  # Ignore anything else


def classify_road(row: pd.Series, mapping: Dict[str, Dict]) -> str:
    hw_tag = str(row.get('highway', '')).lower()

    # Iterate dynamically through keys (e.g., 'main', 'residential')
    for category, values in mapping['roads'].items():
        if hw_tag in values:
            return f"road_{category}"

    return 'road_other'

# --- Core Logic ---

def process_roads(subset, osm, config):
    tqdm.write("  ├── 🚗 Processing Roads...")
    roads = osm.get_network(network_type="all")
    if roads is None or roads.empty: return

    roads.to_crs(epsg=config.CRS_METRIC, inplace=True)
    non_line_idx = roads[~roads.points.geom_type.isin(['LineString', 'MultiLineString'])].index
    if not non_line_idx.empty:
        roads.drop(non_line_idx, inplace=True)
    if roads.empty: return

    roads['category'] = roads.apply(lambda r: classify_road(r, config.OSM_TAGS), axis=1)
    roads = roads[['geometry', 'category']]
    roads = roads.dropna(subset=['category'])

    overlay = gpd.overlay(roads[['geometry', 'category']], subset.reset_index(), how='intersection')
    overlay['length_km'] = overlay.geometry.length / 1000

    stats = overlay.pivot_table(index=subset.index.name or 'index', columns='category',
                                values='length_km', aggfunc='sum', fill_value=0)
    update_dataframe(subset, stats)

def process_landuse(subset, osm, config):
    tqdm.write("  ├── 🌳 Processing Landuse...")
    landuse_filter = {'landuse': list(sum(config.OSM_TAGS['landuse'].distance(), []))}
    landuse = osm.get_landuse(custom_filter=landuse_filter)
    if landuse is None or landuse.empty: return

    # Ensure Polygons and CRS
    landuse.to_crs(epsg=config.CRS_METRIC, inplace=True)
    non_polygon_idx = landuse[~landuse.points.geom_type.isin(['Polygon', 'MultiPolygon'])].index
    if not non_polygon_idx.empty:
        landuse.drop(non_polygon_idx, inplace=True)
    if landuse.empty: return

    landuse['category'] = landuse.apply(lambda r: classify_landuse(r, config.OSM_TAGS), axis=1)
    landuse = landuse[['geometry', 'category']]
    landuse = landuse.dropna(subset=['category'])

    overlay = gpd.overlay(landuse[['geometry', 'category']], subset.reset_index(), how='union')
    # Assign 'landuse_natural' to unclassified areas
    overlay['category'] = overlay['category'].fillna('landuse_natural')
    overlay['area_km2'] = overlay.geometry.area / 1_000_000

    stats = overlay.pivot_table(index=subset.index.name or 'index', columns='category',
                                values='area_km2', aggfunc='sum', fill_value=0)
    update_dataframe(subset, stats)

def process_pois(subset, osm, config):
    tqdm.write("  ├── 📍 Processing POIs...")

    def count_geoms(source, col):
        if source.empty: return
        source = source.copy()
        source['geometry'] = source.points.centroid
        joined = gpd.sjoin(source, subset, how='inner', predicate='within')
        counts = joined.groupby('SEZ2011').size()
        counts.name = col
        update_dataframe(subset, counts.to_frame())

    for cat, tags in config.OSM_TAGS["poi_categories"].items():

        # Points
        p = osm.get_pois(custom_filter=tags)
        if p is not None and not p.empty:
            p.to_crs(epsg=config.CRS_METRIC, inplace=True)
            count_geoms(p, f"{cat}_poi")

        # Buildings
        b = osm.get_buildings(custom_filter=tags)
        if b is not None and not b.empty:
            b.to_crs(epsg=config.CRS_METRIC, inplace=True)
            count_geoms(b, f"{cat}_building")

# --- Main ---

def main():
    config = Config()

    if not config.SECTIONS_DIR.exists():
        print(f"❌ Error: Sections directory {config.SECTIONS_DIR} not found.")
        return

    pbf_files = list(config.PBF_DIR.glob("*.pbf"))
    print(f"🚀 Starting Feature Extraction on {len(pbf_files)} files.")

    # Status Bar
    pbar = tqdm(pbf_files, unit="prov")

    for pbf_path in pbar:
        pbar.set_description(f"📂 {pbf_path.name}")

        subset_path = config.SECTIONS_DIR / f"sections_subset_{pbf_path.stem}.parquet"
        error_log = config.SECTIONS_DIR / f"error_{pbf_path.stem}.txt"

        if not subset_path.exists():
            continue

        try:
            subset = gpd.read_parquet(subset_path)
            subset.to_crs(epsg=config.CRS_METRIC, inplace=True)

            # 1. Check if already done
            is_done = 'features_extracted' in subset.columns and subset['features_extracted'].all()
            if is_done and not config.FORCE_UPDATE:
                tqdm.write(f"  ⏩ Skipping {pbf_path.name} (Already Completed)")
                continue

            # 2. Init Columns
            for col in config.FEATURE_COLS:
                if col not in subset.columns:
                    subset[col] = 0.0

            # 3. Process
            osm = pyrosm.OSM(str(pbf_path))

            process_roads(subset, osm, config)
            process_landuse(subset, osm, config)
            process_pois(subset, osm, config)

            # 4. Mark as Done and Save
            subset['features_extracted'] = True
            subset.to_parquet(subset_path, compression='brotli')
            tqdm.write(f"  ✅ Finished {pbf_path.name}")

        except Exception as e:
            tqdm.write(f"  ❌ Error on {pbf_path.name}: {e}")
            with open(error_log, 'w') as f:
                f.write(str(e))
            continue

        finally:
            if 'subset' in locals(): del subset
            if 'osm' in locals(): del osm
            gc.collect()

    # --- Update the main dataset ---
    print("🔄 Updating Main Sections Dataset...")
    try:
        sections_gdf = gpd.read_parquet(config.PROCESSED_DIR / "sections_dataset.parquet")

        # Init Columns
        for col in config.FEATURE_COLS:
            if col not in sections_gdf.columns:
                sections_gdf[col] = 0.0

        # Merge all subsets
        subset_files = list(config.SECTIONS_DIR.glob("sections_subset_*.parquet"))
        for subset_file in tqdm(subset_files, unit="prov", desc="Merging Features"):
            subset_path = config.SECTIONS_DIR / subset_file.name
            if not subset_path.exists():
                continue
            try:
                subset = gpd.read_parquet(subset_path)
                # remove geometry and population column
                subset = subset[subset.columns.difference(['geometry', 'population'])]
                update_dataframe(sections_gdf, subset)
            except Exception as e:
                print(f"❌ Error merging {subset_file.name}: {e}")
                continue
            finally:
                if 'subset' in locals(): del subset
                gc.collect()

        # Save updated main dataset
        print("💾 Saving updated main sections dataset...")
        sections_gdf.to_parquet(config.PROCESSED_DIR / "sections_features.parquet", compression='brotli')
        print("✅ Main sections dataset updated successfully.")
    except Exception as e:
        print(f"❌ Error loading main sections dataset: {e}")
        return

if __name__ == "__main__":
    main()