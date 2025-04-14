import traceback
import sys

class ExceptionsHandling(Exception):
    
    def __init__(self, error_message: str, error_details: str):
        super().__init__(error_message)
        self.error_message = self.get_detailed_of_error(error_message, error_details)


    @staticmethod
    def get_detailed_of_error(error_message:sys, error_details:sys):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        
        error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with error message: [{error_message}]"
        return error_message
    
    def __str__(self):
        return self.error_message
           