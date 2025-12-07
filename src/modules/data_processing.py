import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FoodDeliveryPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, mean_impute_cols, median_impute_cols, mode_impute_cols):
        self.mean_impute_cols = mean_impute_cols
        self.median_impute_cols = median_impute_cols
        self.mode_impute_cols = mode_impute_cols
        self.driver_speed = None
        self.avg_driver_rating = None
        self.scaler = None
        self.encoder = None
        self.scaled_cols = None
        self.encoded_cols = None
    
    def fit(self, X, y=None):
        
        X_copy = X.copy()
        
        # Store fitted imputers
        self.imputers = {}
        for col in self.mean_impute_cols:
            imp = SimpleImputer(strategy='mean')
            imp.fit(X_copy[[col]])
            self.imputers[col] = imp
            
        for col in self.median_impute_cols:
            imp = SimpleImputer(strategy='median')
            imp.fit(X_copy[[col]])
            self.imputers[col] = imp
            
        for col in self.mode_impute_cols:
            imp = SimpleImputer(strategy='most_frequent')
            imp.fit(X_copy[[col]])
            self.imputers[col] = imp
        
        # Apply imputation for driver stats calculation
        for col, imp in self.imputers.items():
            X_copy[col] = imp.transform(X_copy[[col]]).ravel()
        
        # Apply feature engineering
        X_copy = self._distance(X_copy)
        X_copy = self._datetime(X_copy)
        
        # Fit driver statistics
        df_driver_speed = X_copy.groupby('delivery_person_id')[['distance', 'time_taken_min']].sum()
        df_driver_speed['driver_speed'] = df_driver_speed['distance'] / df_driver_speed['time_taken_min']
        self.driver_speed = df_driver_speed[['driver_speed']]
        
        self.avg_driver_rating = X_copy.groupby('delivery_person_id')['delivery_person_ratings'].mean().to_frame('avg_driver_rating')
        
        # Apply driver stats
        X_copy = self._driver_stats(X_copy)
        
        # Fit scaler and encoder
        
        self.scaled_cols = X_copy.drop(['time_taken_min'], axis=1).select_dtypes([float, int]).columns.tolist()
        self.scaler = StandardScaler()
        self.scaler.fit(X_copy[self.scaled_cols])
        
        self.encoded_cols = X_copy.drop(['id', 'delivery_person_id'], axis=1).select_dtypes(object).columns.tolist()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(X_copy[self.encoded_cols])
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Apply imputation
        for col, imp in self.imputers.items():
            X_copy[col] = imp.transform(X_copy[[col]]).ravel()
        
        # Feature engineering
        X_copy = self._distance(X_copy)
        X_copy = self._datetime(X_copy)
        X_copy = self._driver_stats(X_copy)
        
        # Scale and encode
        X_copy[self.scaled_cols] = self.scaler.transform(X_copy[self.scaled_cols])
        encoded_array = self.encoder.transform(X_copy[self.encoded_cols])
        df_encoded = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out())
        X_copy = pd.concat([X_copy.drop(self.encoded_cols, axis=1).reset_index(drop=True), df_encoded], axis=1)
        
        return X_copy
    
    def _distance(self, df):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [df['restaurant_latitude'], df['restaurant_longitude'], 
                                                     df['delivery_location_latitude'], df['delivery_location_longitude']])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        df['distance'] = R * 2 * np.arcsin(np.sqrt(a))
        return df.drop(['restaurant_latitude', 'restaurant_longitude', 'delivery_location_latitude', 'delivery_location_longitude'], axis=1)
    
    def _datetime(self, df):
        df['day_of_week'] = pd.to_datetime(df['order_date']).dt.day_of_week
        df['hour_of_day'] = pd.to_datetime(df['time_orderd']).dt.hour
        return df.drop(['order_date', 'time_orderd', 'time_order_picked'], axis=1)
    
    def _driver_stats(self, df):
        df = df.merge(self.driver_speed, on='delivery_person_id', how='left')
        df = df.merge(self.avg_driver_rating, on='delivery_person_id', how='left')
        return df.drop('delivery_person_ratings', axis=1)

def save_data_to_csv(df, path):

    df.to_csv(path)

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