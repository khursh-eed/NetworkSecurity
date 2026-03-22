import sys
from networksecurity.logging import logger
# made a custom exception cuz python errors can be too simple or too messy (if asked to traceback then) shows : Traceback (most recent call last):

# so instead this class takes the rrror message, and gets information about which fine and which line the error occured, and shows us in a particualr format which is very readble and understanble

# sys module used here 
class NetworkSecurityException(Exception):
    def __init__ (self, error_message, error_details:sys):
        self.error_message = error_message

        # exc_info gives 3 imp things -type, value and traceback- we ignore the first two and take the third one
        _,_,exc_tb = error_details.exc_info()

        # and from traceback we extract the line and filename
        self.lineo=exc_tb.tb_lineno
        self.filename=exc_tb.tb_frame.f_code.co_filename


        # then we present it in a simple language 
    def __str__ (self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format (self.filename,self.lineo, str(self.error_message))
    
if __name__ == '__main__':
    try: 
        logger.logging.info("enter the try block")
        a=1/0
        print("this will not be printed",a)
    except Exception as e:
        raise NetworkSecurityException(e,sys)

