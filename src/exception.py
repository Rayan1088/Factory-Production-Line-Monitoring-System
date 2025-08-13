import sys
from src.logger import logging 

logger = logging.getLogger("exception")

class CustomException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg
        
        _,_, exc_tp = error_detail.exc_info()
        
        if exc_tp:
            filename = exc_tp.tb_frame.f_code.co_filename
            lineno = exc_tp.tb_lineno
            msg = str(error_msg)
    
            self.error_msg = f"Error occurred in script: [{filename}], line: [{lineno}], message: [{msg}]"
        else:
            self.error_msg = str(error_msg)  
            
    def __str__(self):
        return self.error_msg

    def __repr__(self):
        return self.error_msg

 
    