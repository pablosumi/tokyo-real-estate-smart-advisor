import os
import time
import logging
from datetime import date
from pathlib import Path
from typing import List, Dict, Any

import requests
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
PREF_CODE = "13"  # Tokyo
START_YEAR = 2010

def get_api_key() -> str:
    """
    Loads and validates the API key.
    """
    load_dotenv()
    key = os.getenv('MLIT_API_KEY')
    if not key:
        logger.error("MLIT_API_KEY not found in environment variables.")
        raise ValueError("Missing API Key")
    return key

def fetch_data(api_key: str, start_year: int, end_year: int) -> List[Dict[str, Any]]:
    """
    Iterates through years and fetches real estate data from MLIT endpoint.
    """
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    all_records = []

    logger.info(f"Starting ingestion from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        params = {
            "area": PREF_CODE,
            "year": year,
            "language": "en"
        }

        try:
            response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
            response.raise_for_status()  # Raises error for 4xx/5xx status codes
            
            data = response.json().get("data", [])
            all_records.extend(data)
            
            logger.info(f"Fetched {year}: {len(data)} records found.")
            
            # Be polite to the API server
            time.sleep(1) 

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for year {year}: {e}")
            # Decision: Do we stop or continue? 
            # For production, we might want to continue and log the failure.
            continue

    return all_records

def save_to_parquet(data: List[Dict[str, Any]], output_path: Path):
    """
    Converts list of dicts to PyArrow Table and writes to Parquet.
    """
    if not data:
        logger.warning("No data to save.")
        return

    try:
        # Convert Python list of dicts directly to a PyArrow Table
        # This is much faster and lighter than creating a DataFrame
        table = pa.Table.from_pylist(data)
        
        pq.write_table(table, output_path)
        logger.info(f"Successfully wrote {table.num_rows} rows to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to write parquet file: {e}")
        raise

def main():
    # 1. Setup Paths (Robust method)
    # Finds the script's folder, goes up one level to root, then into data
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    output_file = data_dir / "tokyo.parquet"

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # 2. Configuration
    api_key = get_api_key()
    current_year = date.today().year

    # 3. Execution
    records = fetch_data(api_key, START_YEAR, current_year)
    save_to_parquet(records, output_file)

if __name__ == "__main__":
    main()