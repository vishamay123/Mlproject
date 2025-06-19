import sys
import logging

def get_error_message(error , error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "error occured in the python script[{0}] line number[{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_details:sys):
        super().__init__(get_error_message(error, error_details))
        self.error_message = get_error_message(error, error_details)
    def __str__(self):
        return self.error_message



if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e, sys) from e
    