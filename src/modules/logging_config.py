import logging
from datetime import datetime
import os

def setup_logger(script_name):
    # Create logs directory if it doesn't exist
    os.makedirs('/Users/ani/Projects/8_DELIVERY_TIME_PREDICTION/logs', exist_ok=True)
    
    # Setup logging with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'/Users/ani/Projects/8_DELIVERY_TIME_PREDICTION/logs/{script_name}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)