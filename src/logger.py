import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR,exist_ok=True)

# log_2023-10-01.log
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")


logging.basicConfig(
    filename=LOG_FILE,
    
    # time - level (INFO, WARNING, ERROR) - message
    # e.g. 2023-10-01 12:00:00 - INFO - This is a log message
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger