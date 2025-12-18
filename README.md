# Deep Gravity 
Implementation of the Deep Gravity model from 
*Simini, F., Barlacchi, G., Luca, M. et al. A Deep Gravity model for mobility flows generation. Nat Commun 12, 6576 (2021). https://doi.org/10.1038/s41467-021-26752-4*

This served as the base code for my PhD course project at the University of Padova in Deep Learning.

## Data Processing
### Commuting Flows
From ISTAT website, download: https://www.istat.it/storage/cartografia/matrici_pendolarismo/matrici-pendolarismo-sezione-censimento-2011.zip

The file contains commuting flows between Italian census sections in 2011. The script to process the file is in `data_processing/get_commutes.py`. This script will generate a nested dictionary

```python
{
    origin_section_id: {
        destination_section_id: number_of_commuters,
        ...
    },
    ...
}
```

saved as a pickle file `data/commutes.pkl`. The `id` of each section is computed as `PROCOM` + `SEZ2011`, where `PROCOM` is the code of the municipality and `SEZ2011` is the code of the census section within the municipality.
The code of the municipality is padded with leading zeros the get 7 digits.

### Population data
From ISTAT website, download: https://www.istat.it/storage/cartografia/variabili-censuarie/dati-cpa_2011.zip to get census data at the section level.
Then, at the webpage https://www.istat.it/notizia/basi-territoriali-e-variabili-censuarie/ download the file for each region for the year 2011.

The script `merge_census_data.py` merges the data from all regions and extract the population of each census section. 
The output is a parquet file `data_processing/sections_dataset.parquet` containing a GeoDataframe file with the following schem:
```
Output:
    - GeoDataFrame indexed by 'SEZ2011' (Census Section ID as str)
    - Columns: ['geometry', 'P1' (Total Population)]
    - CRS: EPSG:4326 (WGS84)
```

### Tesselation

The script `create_tessellation.py` generates a regular spatial grid covering the geographical extent of Italy. To ensure accurate metric dimensions, the data is temporarily projected to Web Mercator (EPSG:3857) where a grid of **25km x 25km** square cells is created.

The grid is then filtered using a spatial join to retain only those cells that intersect with the Italian landmass (census sections), discarding empty ocean tiles. The result is reprojected back to WGS84 and saved.

The output is a parquet file `data/processed_data/tessellation_25km.parquet` containing a GeoDataFrame with the following schema:

```
Output:
    - GeoDataFrame (RangeIndex)
    - Columns: ['geometry'] (Polygon of the 25km cell)
    - CRS: EPSG:4326 (WGS84)
```

### ProtocolBuffer download

The script `download_pbf_provinces.py` automates the retrieval of OpenStreetMap data for all Italian provinces. It first fetches the official list of provinces (codes and names) from the OpenPolis repository to ensure accuracy. Then, it constructs the specific URLs for the `osmit-estratti` service (e.g., `001_Torino.osm.pbf`) and downloads the Protocol Buffer (PBF) files in parallel.

The output is a collection of `.osm.pbf` files stored in `data/processed_data/pbf_provinces/`, containing the vector map data for each province.

**Attention:** some provinces are not downloadable using the script `download_pbf_provinces.pbf`. In this case, the script will log a warning message and skip those provinces. Please, download them manually from https://osmit-estratti.wmcloud.org

### Data Features Extraction
This is a procedure that is highly memory consuming, thus I designed this pipeline:
1. Create a `processed_data/sections` folder containing the sections intersecting the boundary of each province. Run the script `data_processing/get_sections_per_province.py` to generate the files.
2. For each province, analyze the relative `PBF`file with `pyrosm` to extract the following features for each census sections:
   - Road Network Length (main, residential, other) in km
   - Landuse types area (residential, commercial, industrial, natural, retail) in km²
   - Number of POIs and buildings by type (health, education, retail, food, transport)
   Run the script `data_processing/extract_features.py` to generate the features. The output files will be saved as `data/processed_data/sections_features.parquet`.

## Mobility Data Manager

The `MobilityDataManager` class (located in `model/mobility_data_manager.py`) serves as the primary PyTorch `Dataset` for training the Deep Gravity model. It handles the complex logic of loading, cleaning, normalizing, and spatially indexing the processed datasets.

**Key Optimizations:**
To handle the massive number of potential origin-destination pairs (Italy has ~400k census sections), the manager implements several memory optimizations:
* **Spatial Filtering:** It performs a spatial join to retain only sections that are spatially contained within the generated 25km tessellation tiles.
* **Minimal Perfect Hashing (BBHash):** Pairwise distances between sections within the same tile are pre-computed and stored using a static Minimal Perfect Hash function. This avoids the creation of massive dense matrices or memory-heavy Python dictionaries.
* **Half-Precision:** Tensors and distance arrays are stored using `float16` to minimize RAM footprint.

**Processing Pipeline:**
1.  **Feature Scaling:** Applies `MinMaxScaler` to all extracted section features (road network, land use, POIs) to normalize inputs between 0 and 1.
2.  **Commute Normalization:** Converts raw commuter counts into probability distributions ($P_{ij} = \frac{T_{ij}}{\sum_k T_{ik}}$). Flows are constrained locally: we only consider destinations that fall within the same tessellation tile as the origin.

**Output Schema:**
The dataset is designed to work with a custom collate function. Each item retrieved (`__getitem__`) represents a **single origin** and all its candidate destinations within the tile:

```
Output Tuple (X, y):

X : torch.Tensor 
    Shape: (N_destinations, 2 * N_features + 1)
    Structure: [Origin_Features | Destination_Features | Distance]
    # Origin features are repeated N times to match the destinations.

y : torch.Tensor
    Shape: (N_destinations,)
    Values: Probability of commuting (Sum over N_destinations = 1.0)
```

## Training the Deep Gravity Model (To be added)

