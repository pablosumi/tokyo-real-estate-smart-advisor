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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
PREF_CODE = "13"
START_YEAR = 2010

def get_api_key() -> str:
    """
    Loads and validates the API key.
    """
    load_dotenv()
    key = os.getenv('MLIT_API_KEY')
    if not key:
        raise ValueError("Missing API Key")
    return key

def fetch_year_data(api_key: str, year: int) -> List[Dict[str, Any]]:
    """
    Fetches a single year of data.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    params = {
        "area": PREF_CODE,
        "year": year,
        "language": "en"
    }
    
    try:
        response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {year}: {e}")
        return []

def main():
    # setup paths (find script's folder, go up one level to root, then into data directory)
    base_dir = Path(__file__).resolve().parent.parent
    output_file = base_dir / "data" / "tokyo.parquet"

    # ensure directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # config
    api_key = get_api_key()
    current_year = date.today().year
    writer = None
    total_rows = 0

    logger.info(f"Starting streamed ingestion to {output_file}...")

    # execute
    try:
        for year in range(START_YEAR, current_year + 1):
            
            # 1. Fetch only a single year
            records = fetch_year_data(api_key, year)
            
            if not records:
                logger.warning(f"No records found for {year}, skipping...")
                continue

            # 2. Convert immediately to Arrow Table (Lightweight)
            batch_table = pa.Table.from_pylist(records)
            
            # 3. Initialize Writer (Only on the first batch)
            # We need the first batch to determine the columns/schema for the file
            if writer is None:
                writer = pq.ParquetWriter(output_file, batch_table.schema)
            
            # 4. Write this batch to disk
            try:
                writer.write_table(batch_table)
                total_rows += batch_table.num_rows
                logger.info(f"Wrote {year}: {batch_table.num_rows} rows. (Total: {total_rows})")
            except ValueError as e:
                 # This happens if the API changes columns between years (Schema mismatch)
                 logger.error(f"Schema mismatch error for year {year}: {e}")
            
            # 5. Clear memory (Python does this automatically, but being explicit)
            del records
            del batch_table
            
            # 6. Avoid rate limiting by MLIT API
            time.sleep(1)

    finally:
        # 6. Close the file properly, even if errors occur
        if writer:
            writer.close()
            logger.info("Writer closed.")

if __name__ == "__main__":
    main()