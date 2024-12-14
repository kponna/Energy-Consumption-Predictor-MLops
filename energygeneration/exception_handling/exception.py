import sys

class EnergyGenerationException(Exception):
    """
    Custom exception class for handling errors in the Energy Generation Prediction project.

    This class captures the error message, file name, and line number where an exception occurred.
    It is useful for debugging and identifying the source of the error in a structured way.
    """
    def __init__(self,error_message, error_details:sys):
        """
        Initialize the EnergyGenerationException instance.

        Args:
            error_message (str): The error message describing the exception.
            error_details (sys): The sys module, used to extract exception details.

        Attributes:
            error_message (str): Stores the original error message.
            lineno (int): The line number where the exception occurred.
            file_name (str): The file name where the exception occurred.
        """
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        """
        Return the formatted error message.

        Returns:
            str: A string containing the file name, line number, and error message.
        """
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name,self.lineno, str(self.error_message))
    
if __name__=="__main__":
    try: 
        a = 1/0  # This will raise a ZeroDivisionError
        print("This will not be printed",a)
    except Exception as e:
        raise EnergyGenerationException(e,sys) 