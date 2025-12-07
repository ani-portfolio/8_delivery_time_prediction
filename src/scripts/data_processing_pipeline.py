import os
from src.config import *
from src.modules.etl import *
from src.modules.data_processing import *
from src.modules.model_training import *
from src.modules.models import *
from src.modules.utils import *
import joblib

import time
from datetime import timedelta
from src.modules.logging_config import setup_logger
import tracemalloc

def main():

    script_name = os.path.basename(__file__).replace('.py', '')
    logger = setup_logger(script_name)
    tracemalloc.start()
    logger.info("Starting data processing...")

    start_time = time.time()

    # ETL
    df_raw_data = import_data(order_data_path_csv)
    validate_schema(df_raw_data, raw_data_expected_columns)
    validate_dtypes(df_raw_data, raw_data_expected_columns)
    etl_data = format_raw_data(df_raw_data)

    # Train-Test Split
    df_train, df_test = train_test_split_time_based(etl_data, 0.8)
    df_full = etl_data.copy()

    # Data Processing
    preprocessor = FoodDeliveryPreprocessor(mean_impute_columns, median_impute_columns, mode_impute_columns)
    preprocessor.fit(df_train)
    train_processed = preprocessor.transform(df_train)
    test_processed = preprocessor.transform(df_test)
    full_processed = preprocessor.transform(df_full)

    # Save CSV
    save_data_to_csv(train_processed, processed_train_data_path)
    save_data_to_csv(test_processed, processed_test_data_path)
    save_data_to_csv(full_processed, processed_full_data_path)
    
    # Save artifact
    joblib.dump(preprocessor, f'{ARTIFACTS_DIR}/preprocessor.pkl')

    end_time = time.time()

    training_time = end_time - start_time
    logger.info(f"Data processing completed in {str(timedelta(seconds=training_time))}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    logger.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
if __name__ == '__main__':
    main()