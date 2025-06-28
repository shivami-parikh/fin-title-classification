import logging
import os
from datetime import datetime
from config import LOGS_DIR, LOGGING_LEVEL

def setup_project_logger(module_name):
    """
    Sets up a logger for the project, outputting to a date-stamped file
    in 'data/logs' and to the console.

    Args:
        module_name (str): The name of the module using the logger (e.g., __name__).

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Define the log directory
    os.makedirs(LOGS_DIR, exist_ok=True) # Ensure the log directory exists

    # Define the log file path with a date stamp
    today_date = datetime.now().strftime("%Y-%m-%d")
    LOG_FILE_PATH = os.path.join(LOGS_DIR, f"{module_name}_{today_date}_{datetime.now().strftime('%H-%M-%S')}.log")

    # Get a logger instance
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG) # Set the minimum level for this logger

    # Clear existing handlers to prevent duplicate messages if script is run multiple times in a session
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Create a file handler for detailed logs
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(LOGGING_LEVEL) # Log all messages (DEBUG and above) to file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create a console handler for general info/errors (e.g., what was previously 'print')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Only show INFO and above on console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger