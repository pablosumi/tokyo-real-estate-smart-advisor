import pandas as pd
import numpy as np

# --- Constants ---

MUNICIPALITY_MAPPING = {
    'Chiyoda Ward': '千代田区 (Chiyoda Ward)',
    'Chuo Ward': '中央区 (Chuo Ward)',
    'Minato Ward': '港区 (Minato Ward)',
    'Shinjuku Ward': '新宿区 (Shinjuku Ward)',
    'Bunkyo Ward': '文京区 (Bunkyo Ward)',
    'Taito Ward': '台東区 (Taito Ward)',
    'Sumida Ward': '墨田区 (Sumida Ward)',
    'Koto Ward': '江東区 (Koto Ward)',
    'Shinagawa Ward': '品川区 (Shinagawa Ward)',
    'Meguro Ward': '目黒区 (Meguro Ward)',
    'Ota Ward': '大田区 (Ota Ward)',
    'Setagaya Ward': '世田谷区 (Setagaya Ward)',
    'Shibuya Ward': '渋谷区 (Shibuya Ward)',
    'Nakano Ward': '中野区 (Nakano Ward)',
    'Suginami Ward': '杉並区 (Suginami Ward)',
    'Toshima Ward': '豊島区 (Toshima Ward)',
    'Kita Ward': '北区 (Kita Ward)',
    'Arakawa Ward': '荒川区 (Arakawa Ward)',
    'Itabashi Ward': '板橋区 (Itabashi Ward)',
    'Nerima Ward': '練馬区 (Nerima Ward)',
    'Adachi Ward': '足立区 (Adachi Ward)',
    'Katsushika Ward': '葛飾区 (Katsushika Ward)',
    'Edogawa Ward': '江戸川区 (Edogawa Ward)',
    'Hachioji City': '八王子市 (Hachioji City)',
    'Tachikawa City': '立川市 (Tachikawa City)',
    'Musashino City': '武蔵野市 (Musashino City)',
    'Mitaka City': '三鷹市 (Mitaka City)',
    'Oume City': '青梅市 (Oume City)',
    'Fuchu City': '府中市 (Fuchu City)',
    'Akishima City': '昭島市 (Akishima City)',
    'Chofu City': '調布市 (Chofu City)',
    'Machida City': '町田市 (Machida City)',
    'Koganei City': '小金井市 (Koganei City)',
    'Kodaira City': '小平市 (Kodaira City)',
    'Hino City': '日野市 (Hino City)',
    'Higashimurayama City': '東村山市 (Higashimurayama City)',
    'Kokubunji City': '国分寺市 (Kokubunji City)',
    'Kunitachi City': '国立市 (Kunitachi City)',
    'Fussa City': '福生市 (Fussa City)',
    'Komae City': '狛江市 (Komae City)',
    'Higashiyamato City': '東大和市 (Higashiyamato City)',
    'Kiyose City': '清瀬市 (Kiyose City)',
    'Higashikurume City': '東久留米市 (Higashikurume City)',
    'Musashimurayama City': '武蔵村山市 (Musashimurayama City)',
    'Tama City': '多摩市 (Tama City)',
    'Inagi City': '稲城市 (Inagi City)',
    'Hamura City': '羽村市 (Hamura City)',
    'Akiruno City': 'あきる野市 (Akiruno City)',
    'Nishitokyo City': '西東京市 (Nishitokyo City)',
    'Mizuho Town, Nishitama County': '瑞穂町 (Mizuho Town, Nishitama County)',
    'Hinode Town, Nishitama County': '日の出町 (Hinode Town, Nishitama County)',
    'Hinohara Village, Nishitama County': '檜原村 (Hinohara Village, Nishitama County)',
    'Okutama Town, Nishitama County': '奥多摩町 (Okutama Town, Nishitama County)',
    'Oshima Town': '大島町 (Oshima Town)',
    'Niijima Village': '新島村 (Niijima Village)',
    'Miyake Village': '三宅村 (Miyake Village)',
    'Hachijo Town': '八丈町 (Hachijo Town)',
    'Ogasawara Village': '小笠原村 (Ogasawara Village)',
    'Kozushima Village': '神津島村 (Kozushima Village)',
}

QUARTER_END_dates = {
    1: (3, 31),
    2: (6, 30),
    3: (9, 30),
    4: (12, 31)
}

# --- Functions ---

def initial_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unused columns and renames key columns."""
    drop_cols = [
        'MunicipalityCode', 'DistrictCode', 'PriceCategory', 
        'PricePerUnit', 'UnitPrice', 'Prefecture'
    ]
    # Only drop columns that actually exist to avoid errors
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    df = df.rename(columns={
        'TradePrice': 'TradePriceYen',
        'Direction': 'RoadDirection'
    })
    return df

def filter_residential(df: pd.DataFrame) -> pd.DataFrame:
    """Filters dataset to only include Residential Land and Pre-owned Condos."""
    target_types = ['Residential Land(Land and Building)', 'Pre-owned Condominiums, etc.']
    df = df[df['Type'].isin(target_types)].copy()
    return df.reset_index(drop=True)

def map_municipalities(df: pd.DataFrame) -> pd.DataFrame:
    """Maps English municipality names to the Japanese/English combo format."""
    df['Municipality'] = df['Municipality'].map(MUNICIPALITY_MAPPING)
    return df

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Handles standard type conversions and empty string handling."""
    df = df.replace('', np.nan).copy()
    
    # Safe conversion helpers
    df['TradePriceYen'] = df['TradePriceYen'].astype(float).astype('Int64')
    df['Area'] = df['Area'].astype(float).astype('Int64')
    df['Frontage'] = df['Frontage'].astype(float)
    df['TotalFloorArea'] = df['TotalFloorArea'].astype(float).astype('Int64')
    df['CoverageRatio'] = df['CoverageRatio'].astype(float).astype('Int64')
    df['FloorAreaRatio'] = df['FloorAreaRatio'].astype(float).astype('Int64')
    df['Breadth'] = df['Breadth'].astype(float)
    
    return df

def remove_outliers(df: pd.DataFrame, lower_q: float = 0.005, upper_q: float = 0.995) -> pd.DataFrame:
    """Removes price outliers and massive area plots."""
    # Price filtering
    low = df['TradePriceYen'].quantile(lower_q)
    high = df['TradePriceYen'].quantile(upper_q)
    df = df[(df['TradePriceYen'] >= low) & (df['TradePriceYen'] <= high)].copy()
    
    # Area filtering
    df = df[df['Area'] < 9999].copy()
    
    return df.reset_index(drop=True)

def handle_special_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles specific MLIT quirks:
    1. 'before the war' in BuildingYear
    2. Capped values (9999.9) in Frontage/FloorArea
    """
    # 1. Building Year
    df['BuildingYearFloored'] = df['BuildingYear'].apply(lambda x: 1 if x == 'before the war' else 0)
    df['BuildingYear'] = df['BuildingYear'].apply(lambda x: 1945 if x == 'before the war' else x)
    df['BuildingYear'] = df['BuildingYear'].astype(float).astype('Int64')

    # 2. Capped Flags
    # We use 1/0 integers explicitly for XGBoost compatibility
    df['FrontageCapped'] = df['Frontage'].apply(lambda x: 1 if x == 9999.9 else 0)
    df['TotalFloorAreaCapped'] = df['TotalFloorArea'].apply(lambda x: 1 if x == 9999 else 0)
    
    return df

def parse_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Parses '2nd quarter 2010' into Quarter, Year, and Date objects."""
    
    # Extract Year and Quarter
    df['TransactionYear'] = df['Period'].apply(lambda x: int(str(x)[-4:]))
    df['TransactionQuarter'] = df['Period'].apply(lambda x: int(str(x)[0]))
    
    # Create Timestamp
    def get_end_date(row):
        month, day = QUARTER_END_dates[row['TransactionQuarter']]
        return pd.Timestamp(year=row['TransactionYear'], month=month, day=day)

    df['TransactionQuarterEndDate'] = df.apply(get_end_date, axis=1)
    
    return df.drop(columns=['Period'])