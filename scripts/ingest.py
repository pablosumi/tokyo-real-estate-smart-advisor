import sys
import time
import logging
from datetime import date
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# --- Path Setup ---
# Add the project root to sys.path so we can import from 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import START_YEAR
from src.api import get_api_key, fetch_year_data

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Setup output path relative to project root
    output_file = project_root / "data" / "tokyo.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    api_key = get_api_key()
    current_year = date.today().year
    
    # Writer initialized lazily (needs first batch to determine schema)
    writer = None
    total_rows = 0

    logger.info(f"Starting streamed ingestion to {output_file}...")

    try:

        # Query data one year at a time
        for year in range(START_YEAR, current_year + 1):
            
            # Fetch one year of data from MLIT api endpoint
            records = fetch_year_data(api_key, year)
            
            if not records:
                logger.warning(f"No records found for {year}, skipping...")
                continue

            # Convert to Arrow Table (efficient in-memory format)
            batch_table = pa.Table.from_pylist(records)
            
            # Initialize writer using schema from the first valid batch
            if writer is None:
                writer = pq.ParquetWriter(output_file, batch_table.schema)
            
            try:
                writer.write_table(batch_table)
                total_rows += batch_table.num_rows
                logger.info(f"Wrote {year}: {batch_table.num_rows} rows. (Total: {total_rows})")
            except ValueError as e:
                 # Catches API schema changes between years
                 logger.error(f"Schema mismatch error for year {year}: {e}")
            
            del records, batch_table
            time.sleep(1) # Rate limiting

    finally:
        # Ensure file is properly closed/finalized, even on error
        if writer:
            writer.close()
            logger.info("Writer closed.")

if __name__ == "__main__":
    main()