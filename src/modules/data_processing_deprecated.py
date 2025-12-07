import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging

logger = logging.getLogger(__name__)
def train_test_split_time_based(df: pd.DataFrame, train_percent: float) -> pd.DataFrame:
    """
    
    """

    df_processed = df.copy()
    df_processed = df_processed.sort_values(['order_date', 'time_orderd']).reset_index(drop=True)

    # Train-Test Split
    df_train = df_processed[0:int(train_percent*len(df_processed))].copy()
    df_test = df_processed[int(train_percent*len(df_processed)):].copy()

    assert len(df_train) > 0
    assert len(df_test) > 0

    return df_train, df_test

def impute_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, mean_impute_cols: list, median_impute_cols: list, mode_impute_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    """

    df_train_imputed = df_train.copy()
    df_test_imputed = df_test.copy()

    for col in mean_impute_cols+median_impute_cols+mode_impute_cols:
        if df_train_imputed[col].isna().all():
            logger.info(f'{col}: All values are Nan')

    for col in mean_impute_cols:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_train_imputed[col] = imp_mean.fit_transform(df_train_imputed[[col]]).ravel()
        df_test_imputed[col] = imp_mean.transform(df_test_imputed[[col]]).ravel()

    for col in median_impute_cols:
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        df_train_imputed[col] = imp_median.fit_transform(df_train_imputed[[col]]).ravel()
        df_test_imputed[col] = imp_median.transform(df_test_imputed[[col]]).ravel()

    for col in mode_impute_cols:
        imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df_train_imputed[col] = imp_mode.fit_transform(df_train_imputed[[col]]).ravel()
        df_test_imputed[col] = imp_mode.transform(df_test_imputed[[col]]).ravel()

    assert df_train_imputed.isna().sum().sum() == 0, "Missing values in train data"
    assert df_test_imputed.isna().sum().sum() == 0, "Missing values in test data"

    return df_train_imputed, df_test_imputed

def straight_line_distance(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    """

    R = 6371  # Earth radius in km

    dfs = []
    for df in [train.copy(), test.copy()]:
        lat1, lon1, lat2, lon2 = map(np.radians, [df['restaurant_latitude'], df['restaurant_longitude'], df['delivery_location_latitude'], df['delivery_location_longitude']])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        df['distance'] = R * c
        df = df.drop(['restaurant_latitude', 'restaurant_longitude', 'delivery_location_latitude', 'delivery_location_longitude'], axis=1)

        dfs.append(df)
        
    return dfs[0], dfs[1]

def date_time_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    dfs = []
    for df in [train.copy(), test.copy()]:
        df['day_of_week'] = pd.to_datetime(df['order_date']).dt.day_of_week
        df['hour_of_day'] = pd.to_datetime(df['time_orderd']).dt.hour
        df = df.drop(['order_date', 'time_orderd', 'time_order_picked'], axis=1)
        dfs.append(df)
    
    return dfs[0], dfs[1]

def driver_statistics(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    
    """

    df_output = train.copy()

    df_driver_speed = df_output.groupby(['delivery_person_id'], as_index=False)[['distance', 'time_taken_min']].sum()
    df_driver_speed['driver_speed'] = df_driver_speed['distance'] / df_driver_speed['time_taken_min']
    df_driver_speed = df_driver_speed[['delivery_person_id', 'driver_speed']]

    df_avg_rating = df_output.groupby('delivery_person_id', as_index=False)[['delivery_person_ratings']].mean()
    df_avg_rating = df_avg_rating.rename(columns={'delivery_person_ratings': 'avg_driver_rating'})

    dfs = []
    for df in [train.copy(), test.copy()]:

        df = pd.merge(df, df_driver_speed, how='left', on='delivery_person_id')
        df = pd.merge(df, df_avg_rating, how='left', on='delivery_person_id')

        df = df.drop(['delivery_person_ratings'], axis=1)

        dfs.append(df)

    return dfs[0], dfs[1]

def fit_transformers(train: pd.DataFrame, ARTIFACTS_DIR) -> tuple:
    """Fit and save scaler and encoder"""
    
    # Fit scaler
    scaler = StandardScaler()
    scaled_cols = train.drop(['time_taken_min'], axis=1).select_dtypes([float, int]).columns.tolist()
    scaler.fit(train[scaled_cols])
    
    # Fit encoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cols = train.drop(['id', 'delivery_person_id'], axis=1).select_dtypes(object).columns.tolist()
    encoder.fit(train[encoded_cols])
    
    # Save artifacts
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.pkl")
    joblib.dump(encoder, f"{ARTIFACTS_DIR}/encoder.pkl")
    joblib.dump(scaled_cols, f"{ARTIFACTS_DIR}/scaled_cols.pkl")
    joblib.dump(encoded_cols, f"{ARTIFACTS_DIR}/encoded_cols.pkl")
    
    return scaler, encoder, scaled_cols, encoded_cols


def transform_data(df: pd.DataFrame, scaler, encoder, scaled_cols, encoded_cols) -> pd.DataFrame:
    """Transform data using fitted transformers"""
    
    df_transformed = df.copy()
    
    # Scale
    df_transformed[scaled_cols] = scaler.transform(df_transformed[scaled_cols])
    
    # Encode
    encoded_array = encoder.transform(df_transformed[encoded_cols])
    features_names = encoder.get_feature_names_out()
    df_encoded = pd.DataFrame(columns=features_names, data=encoded_array)
    df_transformed = pd.concat([df_transformed.drop(encoded_cols, axis=1), df_encoded], axis=1)
    
    return df_transformed

def feature_target_split(train: pd.DataFrame, test: pd.DataFrame, features: list, target: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Output: train_x, train_y, test_x, test_y
    """

    dfs = []

    for df in [train.copy(), test.copy()]:

        df_x = df[features]
        df_y = df[target]
        df_id = df[['id']]

        dfs.append([df_x, df_y, df_id])

    return dfs[0][0], dfs[0][1], dfs[0][2], dfs[1][0], dfs[1][1], dfs[1][2]

def save_data_to_csv(df, path):

    df.to_csv(path)