import logging
import os
import sys
from datetime import datetime

log_file_name = f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
log_dir = "logs"

os.makedirs(log_dir, exist_ok= True)
log_file_path = os.path.join(log_dir, log_file_name)

logging.basicConfig(
    filename=log_file_path,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger