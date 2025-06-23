import logging 
import os
from datetime import datetime


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# creating log folder if it doesnt alr exist
logs_path =os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

# giving fullpath

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


