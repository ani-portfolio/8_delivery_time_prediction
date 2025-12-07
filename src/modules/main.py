from config import *
from etl import *
from data_processing import *
from model_training import *
from models import *
from utils import *

import time
from datetime import timedelta
from logging_config import setup_logger
import tracemalloc


def main():
    logger = setup_logger()
    tracemalloc.start()
    logger.info("Starting process...")

    logger.info('Import Raw Data')
    df_raw_data = import_data(order_data_path_csv)

    # Validate Raw Data
    validate_schema(df_raw_data, raw_data_expected_columns)

    validate_dtypes(df_raw_data, raw_data_expected_columns)

    # ETL
    df_etl_data = format_raw_data(df_raw_data)

    # Data Processing
    df_train, df_test = train_test_split_time_based(df_etl_data, 0.8)

    df_train_imputed, df_test_imputed = impute_missing_values(df_train, df_test, mean_impute_columns, median_impute_columns, mode_impute_columns)

    df_train_imputed, df_test_imputed = straight_line_distance(df_train_imputed, df_test_imputed)

    df_train_imputed, df_test_imputed = date_time_features(df_train_imputed, df_test_imputed)

    df_train_imputed, df_test_imputed = driver_statistics(df_train_imputed, df_test_imputed)

    df_train_imputed, df_test_imputed = feature_scaling_encoding(df_train_imputed, df_test_imputed)

    # Model
    start_time = time.time()

    df_train_x, df_train_y, df_train_id, df_test_x, df_test_y, df_test_id = feature_target_split(df_train_imputed, df_test_imputed, features, target)

    baseline_model = BaselineModel()
    baseline_model.fit(df_train)
    baseline_predictions = baseline_model.predict(df_test)

    decision_tree_model = GradientBoostedDecisionTree()
    best_params = hyperparameter_tuning(df_train_x, df_train_y, gbdt_n_splits, gbdt_n_trials, gbdt_params, decision_tree_model)

    decision_tree_model = GradientBoostedDecisionTree(params=best_params)
    decision_tree_model.fit(X=df_train_x, y=df_train_y)
    model_predictions = decision_tree_model.predict(X=df_test_x, df_id=df_test_id)
    end_time = time.time()

    training_time = end_time - start_time
    logger.info(f"Training completed in {str(timedelta(seconds=training_time))}")

    logger.info(f'Save predictions to {results_path}/results.csv')

    model_predictions.to_csv(f'{results_path}/results.csv')

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    logger.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
if __name__ == '__main__':
    main()