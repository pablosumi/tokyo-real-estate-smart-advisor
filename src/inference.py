import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.features import add_basic_features, parse_floor_plan, impute_missing_categoricals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_prediction(user_input_dict, artifacts_path='models/tokyo_mass_market_xgb.pkl'):
    """
    Takes a dictionary of raw inputs, processes them, and returns a price prediction.
    """
    
    # 1. Load Artifacts
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Model artifacts not found at {artifacts_path}")
    
    artifacts = joblib.load(artifacts_path)
    model = artifacts['model']
    encoder = artifacts['encoder']
    feature_order = artifacts['features']
    
    # 2. Convert raw input to DataFrame
    df_raw = pd.DataFrame([user_input_dict])
    
    # 3. Apply Feature Engineering Logic
    try:
        df_processed = add_basic_features(df_raw)
        
        if 'FloorPlan' in df_processed.columns:
            df_processed = parse_floor_plan(df_processed, col_name='FloorPlan')
            df_processed = df_processed.drop(columns=['FloorPlan'], errors='ignore')
            
        df_processed = impute_missing_categoricals(df_processed)
        
    except Exception as e:
        logger.error(f"Error during feature processing: {e}")
        raise

    # 4. ALIGN COLUMNS (Refactored to avoid Dtype Warning)
    # Initialize with None to force 'object' dtype, avoiding float64 vs string conflicts
    X_full = pd.DataFrame(index=[0], columns=feature_order).astype(object)
    
    # Fill in the columns we have from processing
    for col in df_processed.columns:
        if col in X_full.columns:
            X_full.at[0, col] = df_processed.at[0, col]
    
    # Ensure order is exactly as expected by the model
    X_full = X_full[feature_order]

    # 5. FIX DATA TYPES
    for col in X_full.columns:
        try:
            # Attempt to convert numeric-like strings/objects to actual floats/ints
            X_full[col] = pd.to_numeric(X_full[col], errors='raise')
        except (ValueError, TypeError):
            # Keep as object/string if it's categorical (e.g., Municipality)
            continue

    # 6. Encode Categorical Features
    try:
        X_encoded = encoder.transform(X_full)
    except Exception as e:
        logger.error(f"Error during encoding: {e}")
        raise

    # 7. Final numeric enforcement for XGBoost
    # The encoder outputs numeric values, but we ensure the DataFrame reflects this
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')

    # 8. Predict and Reverse Log Transform
    log_pred = model.predict(X_encoded)
    prediction_yen = np.exp(log_pred)[0]

    return prediction_yen

if __name__ == "__main__":
    # Test Case
    test_input = {
        'Type': 'Pre-owned Condominiums, etc.',
        'Municipality': '新宿区 (Shinjuku Ward)',
        'Area': 40,
        'Frontage': 7,
        'TotalFloorArea': 110,
        'FloorPlan': '1LDK',
        'BuildingYear': 2002,
        'Structure': 'RC',
        'Renovation': None,
        'TransactionYear': 2025
    }
    
    try:
        price = make_prediction(test_input)
        print(f"\n✅ Prediction Successful!")
        print(f"Predicted Price: ¥{price:,.0f}")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")