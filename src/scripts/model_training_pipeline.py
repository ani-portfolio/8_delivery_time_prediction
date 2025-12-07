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
    logger.info("Model Training Started...")

    start_time = time.time()

    train_processed = pd.read_csv(processed_train_data_path)
    test_processed = pd.read_csv(processed_test_data_path)
    full_processed = pd.read_csv(processed_full_data_path)

    df_train_x, df_train_y, df_train_id, df_test_x, df_test_y, df_test_id = feature_target_split(train_processed, test_processed, features, target)
    df_full_x, df_full_y, df_train_id, df_test_x, df_test_y, df_test_id = feature_target_split(full_processed, full_processed, features, target)

    # baseline_model = BaselineModel()
    # baseline_model.fit(df_train)
    # baseline_predictions = baseline_model.predict(df_test)

    decision_tree_model = GradientBoostedDecisionTree()
    best_params = hyperparameter_tuning(df_train_x, df_train_y, gbdt_n_splits, gbdt_n_trials, gbdt_params, decision_tree_model)

    decision_tree_model = GradientBoostedDecisionTree(params=best_params)
    decision_tree_model.fit(X=df_full_x, y=df_full_y)
    joblib.dump(decision_tree_model, f"{ARTIFACTS_DIR}/trained_model.pkl")
    end_time = time.time()

    training_time = end_time - start_time
    logger.info(f"Training completed in {str(timedelta(seconds=training_time))}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    logger.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()