import asyncio
import pandas as pd
from datetime import date
from classifier_excel_data_extractor import ExcelProcessor
from classifier_data_processor import DataProcessor
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from classifier_config import PURPOSE, classifier_worksheet, COLUMN_NAMES
# from project_logger import setup_project_logger

# logger = setup_project_logger("classifier_app")

def extract_data():
    processor = ExcelProcessor()
    processor.process_all_excel_files_in_directory()
    
def process_data(is_test=False, is_base_model=False):
    processor = DataProcessor()
    # Use limit_data for quick testing on a subset of data
    limit_data = 1000 if PURPOSE == "demo" else 200
    asyncio.run(processor.run(is_test=is_test, is_base_model=is_base_model, limit_data=limit_data)) # For production, remove limit_data

def save_to_database(data) -> bool:
    """
    Save the classified titles and selection to a database.
    
    Args:
        pandas DataFrame: Data to be saved in the database.
            Expected format: [title, user_selection(optional), ai_selection]
        
    Returns:
        bool: True if saving was successful, False otherwise.
    """
    data["Date"] = date.today().strftime("%Y-%m-%d")
    
    try:
        existing_data = read_from_database()
        before_len = len(existing_data) + len(data)
        if existing_data.empty:
            combined_df = data
        else:
            titles_to_drop = data[COLUMN_NAMES[0]][data[COLUMN_NAMES[0]].isin(existing_data[COLUMN_NAMES[0]])].unique()
            existing_data = existing_data[~existing_data[COLUMN_NAMES[0]].isin(titles_to_drop)]
            combined_df = pd.concat([existing_data, data], ignore_index=True)
        
        after_len = len(combined_df)
        if before_len != after_len:
            print(f"Removed {before_len - after_len} duplicate rows.")
        set_with_dataframe(classifier_worksheet, combined_df, resize=True)
        # logger.info(f"Successfully appended {len(data)} rows to the Google Sheet.")
        return True
    except Exception as e:
        # logger.error(f"Error saving data to Google Sheet: {e}")
        return False

def read_from_database() -> pd.DataFrame:
    """
    Read data from the database.
    
    Returns:
        pd.DataFrame: The data read from the database.
    """
    df = get_as_dataframe(classifier_worksheet, evaluate_formulas=True).dropna(how="all")
    if isinstance(df, pd.DataFrame):
        return df
    else:
        return pd.DataFrame([df])

if __name__ == "__main__":
    extract_data()
    process_data(is_test=True, is_base_model=True)