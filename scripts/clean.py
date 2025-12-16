import sys
import logging
from pathlib import Path
import pandas as pd

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the processing functions
from src.processing import (
    initial_clean,
    filter_residential,
    map_municipalities,
    convert_types,
    remove_outliers,
    handle_special_flags,
    parse_periods
)

# --- Logging Setup ---
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "clean.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    input_file = project_root / "data" / "tokyo.parquet"
    output_file = project_root / "data" / "tokyo-clean.parquet"
    
    if not input_file.exists():
        logger.error(f"Input file not found at: {input_file}")
        logger.error("Please run scripts/ingest.py first.")
        return

    logger.info("Starting data cleaning pipeline...")
    
    try:
        # 1. Load Data
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

        # 2. Initial Setup
        df = initial_clean(df)
        df = filter_residential(df)
        logger.info(f"Filtered to residential/condos: {len(df)} rows")

        # 3. Mappings & Conversions
        df = map_municipalities(df)
        df = convert_types(df)
        
        # 4. Outlier Removal
        df = remove_outliers(df)
        logger.info(f"After outlier removal: {len(df)} rows")

        # 5. Feature Engineering
        df = handle_special_flags(df)
        df = parse_periods(df)

        # 6. Save Data
        df.to_parquet(output_file, index=False)
        logger.info(f"Successfully wrote cleaned data to {output_file}")
        logger.info(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

    except Exception as e:
        logger.exception(f"Data cleaning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()