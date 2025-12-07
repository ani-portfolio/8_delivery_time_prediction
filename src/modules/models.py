import pandas as pd
import lightgbm as lgb

class BaselineModel:
    def __init__(self):
        self.avg_time_city = None
        self.overall_mean = None
    
    def fit(self, X: pd.DataFrame, y=None) -> 'BaselineModel':
        self.avg_time_city = X.groupby('city', as_index=False)[['time_taken_min']].mean().rename(columns={'time_taken_min': 'baseline_time_taken_min'})
        self.overall_mean = X['time_taken_min'].mean()

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.merge(self.avg_time_city, how='left', on='city')
        result['baseline_predictions'] = result['baseline_time_taken_min'].fillna(self.overall_mean)

        return result[['id', 'baseline_predictions']]
    
class GradientBoostedDecisionTree:
    def __init__(self, params=None):
        self.params = params or {'objective': 'regression', 'metric': 'rmse'}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'GradientBoostedDecisionTree':
        lgbm_train = lgb.Dataset(X, y)
        self.model = lgb.train(self.params,lgbm_train)
        return self

    def predict(self, X: pd.DataFrame, df_id: pd.DataFrame) -> pd.DataFrame:
        predictions = df_id[['id']].copy()
        predictions['GradientBoostedDecisionTree_predictions'] = self.model.predict(X)

        return predictions