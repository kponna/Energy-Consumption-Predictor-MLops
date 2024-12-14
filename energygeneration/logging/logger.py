import logging
import os
from datetime import datetime

LOGS_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
 
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
 

LOGS_FILE_PATH = os.path.join(logs_dir,LOGS_FILE)

logging.basicConfig(
    filename= LOGS_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO,
)