# Deep Gravity 
Implementation of the Deep Gravity model from 
*Simini, F., Barlacchi, G., Luca, M. et al. A Deep Gravity model for mobility flows generation. Nat Commun 12, 6576 (2021). https://doi.org/10.1038/s41467-021-26752-4*

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
   