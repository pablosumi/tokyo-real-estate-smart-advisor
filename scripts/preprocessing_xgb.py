import pandas as pd
import numpy as np
import sys
import os
import logging

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import add_basic_features, parse_floor_plan, impute_missing_categoricals

# Constants
INPUT_PATH = 'data/tokyo-clean.parquet'
OUTPUT_PATH = 'data/tokyo-preprocessed.parquet'
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'preprocessing_xgb.log')

# --- CONFIGURE LOGGING ---
# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1. File Handler (Writes to logs/preprocessing_xgb.log)
file_handler = logging.FileHandler(LOG_FILE, mode='w') # mode='w' overwrites each run. Use 'a' to append.
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 2. Stream Handler (Writes to Console)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def main():
    logger.info("Starting Preprocessing Pipeline...")
    logger.info(f"Writing logs to: {LOG_FILE}")

    # 1. Load Data
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Input file not found: {INPUT_PATH}")
        return

    logger.info(f"Loading data from {INPUT_PATH}...")
    try:
        data = pd.read_parquet(INPUT_PATH)
        logger.info(f"Initial shape: {data.shape}")
    except Exception as e:
        logger.critical(f"Failed to load data: {e}")
        return

    # 2. Feature Engineering
    logger.info("Adding basic features (Ward Flag, Building Age)...")
    data = add_basic_features(data)

    logger.info("Parsing Floor Plans (LDK)...")
    data = parse_floor_plan(data, col_name='FloorPlan')
    
    # Drop original FloorPlan as it's no longer needed for training
    if 'FloorPlan' in data.columns:
        data.drop('FloorPlan', axis=1, inplace=True)

    # 3. Handle Missing Values
    logger.info("Imputing missing categorical values...")
    data = impute_missing_categoricals(data)

    # 4. Target Transformation
    logger.info("Transforming Target (Log Price)...")
    if 'TradePriceYen' in data.columns:
        # Check for non-positive prices just in case
        min_price = data['TradePriceYen'].min()
        logger.info(f"Minimum trade price: ¥{min_price:,.0f}")
        
        # Avoid log(0) or log(negative) errors
        if min_price <= 0:
            logger.warning("Found non-positive trade prices! Filtering them out before log transform.")
            data = data[data['TradePriceYen'] > 0].copy()

        data['LogTradePriceYen'] = np.log(data['TradePriceYen'])
    else:
        logger.warning("TradePriceYen column not found. Skipping target transformation.")

    # 5. Save Data
    logger.info(f"Saving processed data to {OUTPUT_PATH}...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        data.to_parquet(OUTPUT_PATH, index=False)
        logger.info(f"✅ Success! Wrote {data.shape[0]} rows and {data.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Failed to write parquet file: {e}")

if __name__ == "__main__":
    main()