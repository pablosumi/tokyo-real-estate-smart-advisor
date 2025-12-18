import pandas as pd

# Define lists here so they are accessible to both Training and Inference
CAT_COLS_TO_FILL = [
    'Municipality', 'DistrictName', 'Use', 'Structure', 'LandShape', 
    'Renovation', 'Purpose', 'Type', 'Region', 'CityPlanning', 
    'Classification', 'RoadDirection', 'Remarks'
]

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features like Ward flags and Building Age.
    """
    df = df.copy()
    
    # 1. Create Ward Flag
    # Robustness: We use na=False to handle potential missing values safely
    if 'Municipality' in df.columns:
        df['Is_Ward'] = df['Municipality'].str.contains('区', case=False, na=False).astype(int)
    
    # 2. Calculate Building Age
    # Note: Logic assumes 'TransactionYear' exists. 
    # In live inference, you might calculate this against the current year.
    if 'TransactionYear' in df.columns and 'BuildingYear' in df.columns:
        df['BuildingAge'] = df['TransactionYear'] - df['BuildingYear']
        
    return df

def parse_floor_plan(df: pd.DataFrame, col_name: str = 'FloorPlan') -> pd.DataFrame:
    """
    Parses Japanese real estate 'LDK' strings into numerical features.
    Example: '2LDK+S' -> RoomCount:2, L:1, D:1, K:1, S:1
    """
    df = df.copy()
    
    if col_name not in df.columns:
        return df

    # 1. Standardize special text cases
    # We map 'Studio'/'Open' to '1R' so regex can catch the '1'
    df['TempPlan'] = df[col_name].replace({
        'Studio Apartment': '1R',
        'Open Floor': '1R',
        'Duplex': '2LDK',
        'None': '0R'
    })

    # 2. Extract Room Count
    df['RoomCount'] = df['TempPlan'].str.extract(r'^(\d+)').fillna(0).astype(int)

    # 3. Create Boolean Flags
    df['Has_L'] = df['TempPlan'].str.contains('L', case=False, na=False).astype(int)
    df['Has_D'] = df['TempPlan'].str.contains('D', case=False, na=False).astype(int)
    df['Has_K'] = df['TempPlan'].str.contains('K', case=False, na=False).astype(int)
    df['Has_S'] = df['TempPlan'].str.contains('S', case=False, na=False).astype(int)

    # 4. Clean up temporary column
    df.drop('TempPlan', axis=1, inplace=True)
    
    return df

def impute_missing_categoricals(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Fills missing categorical values with 'Unknown'.
    This captures implicit negatives (e.g., No Remarks, No Renovation info).
    """
    if cols is None:
        cols = CAT_COLS_TO_FILL
    
    # Only fill columns that actually exist in the dataframe
    target_cols = [c for c in cols if c in df.columns]
    
    df = df.copy()
    df[target_cols] = df[target_cols].fillna('Unknown')
    return df