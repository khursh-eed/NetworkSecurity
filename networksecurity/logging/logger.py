import logging 
import os
from datetime import datetime

# format of logging the file
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

# there are multiple inbult log levels in python 
# Level	    Value	Meaning
# DEBUG	    10	    Detailed info (dev only)
# INFO	    20	    Normal operation
# WARNING	30	    Something suspicious
# ERROR	    40	    Something failed
# CRITICAL	50	    System may crash

# so now our log level is INFO- it logs INFO, WARNING, ERROR ,CRITICAL (info and everything ab) it ignores DEBUG


