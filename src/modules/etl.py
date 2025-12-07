import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
def import_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV
    """

    try:
        df = pd.read_csv(filepath)
        logger.info('Data loaded to Pandas dataframe')
        return df
    
    except Exception as e:
        logger.info(f"Failed to load data: {e}")

def format_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """

    df_processed = df.copy()

    # format column names (lowercase, remove spaces, remove special characters)
    df_processed.columns = df_processed.columns.str.lower()
    df_processed.columns = df_processed.columns.str.replace(' ', '').str.replace('(', '_').str.replace(')', '')

    # convert column types to relevant types (str, float, datetime)
    df_processed['delivery_person_age'] = df_processed['delivery_person_age'].astype(float)
    df_processed['delivery_person_ratings'] = df_processed['delivery_person_ratings'].astype(float)
    df_processed['order_date'] = pd.to_datetime(df_processed['order_date'], errors='coerce').dt.strftime("%Y-%m-%d")
    df_processed['time_orderd'] = pd.to_datetime(df_processed['time_orderd'], errors='coerce').dt.strftime("%H:%M:%S")
    df_processed['time_order_picked'] = pd.to_datetime(df_processed['time_order_picked'], errors='coerce').dt.strftime("%H:%M:%S")
    df_processed['multiple_deliveries'] = df_processed['multiple_deliveries'].astype(float)
    df_processed['time_taken_min'] = df_processed['time_taken_min'].str.replace('(min) ', '')
    df_processed['time_taken_min'] = df_processed['time_taken_min'].astype(int)
    
    # remove spaces
    for col in df_processed.select_dtypes(object):
        df_processed[col] = df_processed[col].str.strip()

    # replace "NaN" with np.nan
    df_processed = df_processed.replace('NaN', np.nan, regex=False)

    # lower case and remove spaces in values
    lower_cols = df_processed.select_dtypes(object).drop(['id', 'delivery_person_id', 'order_date', 'time_orderd', 'time_order_picked'], axis=1).columns.tolist()
    df_processed[lower_cols] = df_processed[lower_cols].apply(lambda x: x.str.lower().str.strip().str.replace(' ', '').str.replace('-', '_'))

    # log non NaN row (%)
    logger.info(f"Non Missing Rows: {(len(df_processed.dropna())/len(df_processed)) * 100:.2f}%")

    # log outliers
    Q1 = df_processed['time_taken_min'].quantile(0.25) 
    Q3 = df_processed['time_taken_min'].quantile(0.75)
    IQR = Q3-Q1

    logger.info(f"Outlier Upper: {len(df_processed[df_processed['time_taken_min'] < Q1 - 1.5*IQR])/len(df_processed):.2f}%")
    logger.info(f"Outlier Lower: {len(df_processed[df_processed['time_taken_min'] > Q3 + 1.5*IQR].describe())/len(df_processed):.2f}%")

    return df_processed

def validate_schema(df: pd.DataFrame, expected_columns: dict) -> bool:
    """Check if all expected columns exist"""

    missing = set(expected_columns) - set(df.columns)
    missing_inverse = set(df.columns) - set(expected_columns)
    
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if missing_inverse:
        logger.info('Extra column in dataframe')
        
    return True

def validate_dtypes(df: pd.DataFrame, expected_dtypes: dict) -> bool:
    """Check data types match expected"""

    for col, expected_type in expected_dtypes.items():
        if df[col].dtype != expected_type:
            raise TypeError(f"{col}: expected {expected_type}, got {df[col].dtype}")
        
    return True