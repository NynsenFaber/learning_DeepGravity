"""
ISTAT Commute Data Processor.

This script processes commute data from ISTAT (Pendolarismo per sezione di censimento).
It reads zipped CSV files, extracts origin-destination flows, and serializes the
result into a dictionary structure saved as a pickle file.

The output structure is:
    {
        origin_section_id: {
            destination_section_id: number_of_commuters,
            ...
        },
        ...
    }
"""

import logging
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# --- Configuration & Constants ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Input/Output Paths
# Using .resolve() ensures we have absolute paths, minimizing relative path errors.
BASE_DIR = Path(__file__).resolve().parent
DATA_INPUT_DIR = (BASE_DIR / "../../data/ISTAT/Pendolarismo_per_sezione_di_censimento").resolve()
DATA_OUTPUT_PATH = (BASE_DIR / "../data/processed_data/commute_data.pickle").resolve()

# Data Schema Constants
CSV_ENCODING = 'ISO-8859-1'
CSV_SEPARATOR = ';'
REQUIRED_COLUMNS = ['PROCOM_ORIG', 'NSEZ_ORIG', 'PROCOM_DEST', 'NSEZ_DEST', 'TOTALE']


def process_commute_data(data_dir: Path) -> pd.DataFrame:
    """
    Extracts and processes commute data from a directory of zip files.

    Iterates through all .zip files in the specified directory, extracts the
    contained CSVs, standardizes the section IDs (SEZ2011), and concatenates
    them into a single DataFrame.

    Args:
        data_dir (Path): The directory containing ISTAT zip files.

    Returns:
        pd.DataFrame: A combined DataFrame containing origin, destination,
                      and commuter counts. Returns empty DataFrame on failure.
    """
    if not data_dir.exists():
        logger.error(f"Input directory not found: {data_dir}")
        return pd.DataFrame()

    dataframes_list: List[pd.DataFrame] = []
    zip_files = list(data_dir.glob("*.zip"))

    logger.info(f"Starting extraction. Found {len(zip_files)} zip files in {data_dir}")

    for zip_path in zip_files:
        _process_single_zip(zip_path, dataframes_list)

    if not dataframes_list:
        logger.warning("No data was processed successfully.")
        return pd.DataFrame()

    # Combine all processed chunks
    logger.info("Concatenating dataframes...")
    combined_df = pd.concat(dataframes_list, ignore_index=True)

    logger.info(f"Data extraction complete. Total records: {len(combined_df)}")
    return combined_df


def _process_single_zip(zip_path: Path, data_accumulator: List[pd.DataFrame]) -> None:
    """
    Helper function to process a single zip file and append results to the accumulator.

    Args:
        zip_path (Path): Path to the specific zip file.
        data_accumulator (List[pd.DataFrame]): List to append the processed DataFrame to.
    """
    temp_extract_dir = zip_path.parent / zip_path.stem
    csv_filename = f"{zip_path.stem}.csv"
    csv_path = temp_extract_dir / csv_filename

    try:
        logger.info(f"Processing: {zip_path.name}")

        # 1. Extract Zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(temp_extract_dir)

        if not csv_path.exists():
            logger.warning(f"Expected CSV not found: {csv_path}")
            return

        # 2. Read CSV
        df = pd.read_csv(
            csv_path,
            low_memory=False,
            encoding=CSV_ENCODING,
            sep=CSV_SEPARATOR,
            usecols=REQUIRED_COLUMNS
        )

        # 3. Preprocess Data
        # Ensure section numbers are 7 digits (zero-padded)
        df['NSEZ_ORIG'] = df['NSEZ_ORIG'].astype(str).str.zfill(7)
        df['NSEZ_DEST'] = df['NSEZ_DEST'].astype(str).str.zfill(7)

        # Create unique 'SEZ2011' ID: Municipality Code (PROCOM) + Section Number (NSEZ)
        df['SEZ2011_ORIG'] = df['PROCOM_ORIG'].astype(str) + df['NSEZ_ORIG']
        df['SEZ2011_DEST'] = df['PROCOM_DEST'].astype(str) + df['NSEZ_DEST']

        # Clean up columns
        df_clean = df[['SEZ2011_ORIG', 'SEZ2011_DEST', 'TOTALE']]

        data_accumulator.append(df_clean)
        logger.debug(f"  - Parsed {len(df_clean)} records from {zip_path.name}")

    except Exception as e:
        logger.error(f"Failed to process {zip_path.name}: {e}")

    finally:
        # 4. Clean up temporary files
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)


def convert_df_to_nested_dict(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Converts the DataFrame into a nested dictionary for fast lookups.

    Structure: { Origin_ID: { Destination_ID: Total_Commuters } }

    Args:
        df (pd.DataFrame): The processed DataFrame with columns
                           ['SEZ2011_ORIG', 'SEZ2011_DEST', 'TOTALE'].

    Returns:
        Dict: The nested dictionary.
    """
    logger.info("Converting DataFrame to nested dictionary structure...")

    # Using a dictionary comprehension with groupby is efficient here
    result_dict = {
        origin: dict(zip(group['SEZ2011_DEST'], group['TOTALE']))
        for origin, group in df.groupby('SEZ2011_ORIG')
    }
    return result_dict


def save_pickle(data: object, output_path: Path) -> None:
    """
    Safely saves a Python object to a pickle file, ensuring directories exist.
    """
    try:
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving data to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info("Save successful.")

    except IOError as e:
        logger.error(f"Failed to save pickle file to {output_path}: {e}")


def main():
    """
    Main execution flow.
    """
    # 1. Process Raw Data
    commute_df = process_commute_data(DATA_INPUT_DIR)

    if commute_df.empty:
        logger.error("Aborting: No data to save.")
        return

    # 2. Transform Data Structure
    commute_dictionary = convert_df_to_nested_dict(commute_df)

    # 3. Save to Disk
    save_pickle(commute_dictionary, DATA_OUTPUT_PATH)


if __name__ == "__main__":
    main()
