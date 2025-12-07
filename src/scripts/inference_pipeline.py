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
    logger.info("Starting inference...")

    start_time = time.time()

    df_raw_data = import_data(order_data_path_csv)
    validate_schema(df_raw_data, raw_data_expected_columns)
    validate_dtypes(df_raw_data, raw_data_expected_columns)
    etl_data = format_raw_data(df_raw_data)

    model = joblib.load(f'{ARTIFACTS_DIR}/trained_model.pkl')
    preprocessor = joblib.load(f'{ARTIFACTS_DIR}/preprocessor.pkl')

    # DUMMY inference data 
    inference_processed = preprocessor.transform(etl_data)  # No splitting happens
    df_full_x, df_full_y, df_train_id, df_test_x, df_test_y, df_test_id = feature_target_split(inference_processed, inference_processed, features, target)

    predictions = model.predict(df_full_x, df_test_id)

    save_data_to_csv(predictions, f'{results_path}/save_preds.csv')


    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Inference completed in {str(timedelta(seconds=training_time))}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    logger.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()