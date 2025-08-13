import os
import logging
import sys
from datetime import datetime

logs_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
current_time = datetime.now().strftime("%Y-%m-%d")
log_file_name = f"log_{current_time}.log"

log_dir = 'logs'
log_file_path = os.path.join(log_dir, log_file_name)

# Create the logs directory if it doesn't exist 
if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
    os.makedirs(log_dir , exist_ok=True)

# Configure the logging    
logging.basicConfig(
    level=logging.INFO,
    format= logs_format,
    handlers=[
        logging.StreamHandler(sys.stdout),  # print the logs to terminal 
        logging.FileHandler(log_file_path)       # File handler
    ]
)
  
