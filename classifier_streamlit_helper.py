import streamlit as st
import pandas as pd
import os
import shutil
import asyncio
import io

# Assuming these imports are available in the environment
from classifier_app import extract_data, process_data, read_from_database, save_to_database
from classifier_config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_CSV_NAME, ENVIRONMENT,
    MODEL_OUTPUT_DIR, FT_MODEL_OUTPUT_CSV_NAME, DEFAULT_MODEL_OUTPUT_CSV_NAME
)
from project_logger import setup_project_logger

# Initialize logger for the helper
logger = setup_project_logger("streamlit_helper")
    
def clear_data_directory():
    """
    Deletes all files from the RAW_DATA_DIR.
    """
    try:
        if ENVIRONMENT=="prod":
            if os.path.exists(RAW_DATA_DIR):
                for f in os.listdir(RAW_DATA_DIR):
                    file_path = os.path.join(RAW_DATA_DIR, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleared all files from {RAW_DATA_DIR}")
            else:
                logger.info(f"RAW_DATA_DIR {RAW_DATA_DIR} does not exist. No files to clear.")
        else:
            logger.info("Environment is not 'prod'. No action taken for clearing raw data directory.")
    except Exception as e:
        logger.error(f"Error clearing RAW_DATA_DIR: {e}")
        st.error(f"Error clearing raw data directory: {e}")
        
    try:
        if ENVIRONMENT=="prod":
            if os.path.exists(MODEL_OUTPUT_DIR):
                for f in os.listdir(MODEL_OUTPUT_DIR):
                    file_path = os.path.join(MODEL_OUTPUT_DIR, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleared all files from {MODEL_OUTPUT_DIR}")
            else:
                logger.info(f"MODEL_OUTPUT_DIR {MODEL_OUTPUT_DIR} does not exist. No files to clear.")
        else:
            logger.info("Environment is not 'prod'. No action taken for clearing model output data directory.")
    except Exception as e:
        logger.error(f"Error clearing MODEL_OUTPUT_DIR: {e}")
        st.error(f"Error clearing model output data directory: {e}")

def reset_app_state():
    clear_data_directory()
    st.session_state['processed_df'] = pd.DataFrame()
    st.session_state['edited_df'] = pd.DataFrame()
    st.session_state['extracted_file_name'] = None
    st.session_state['is_test_selected'] = False
    st.session_state['file_uploader_key'] += 1 # Increment to refresh file uploader
    st.session_state['classification_completed'] = False
    st.session_state['reset_triggered'] = True # Set the reset flag for the next run
    st.session_state['is_saved'] = False
   
def save_uploaded_file_and_extract(uploaded_file):
    """
    Saves the uploaded Streamlit file to RAW_DATA_DIR and then
    triggers the data extraction process.
    """
    if uploaded_file is not None:
        # Create RAW_DATA_DIR if it doesn't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        for f in os.listdir(RAW_DATA_DIR):
            os.remove(os.path.join(RAW_DATA_DIR, f))
        logger.info("Cleared existing raw files.")

        # Define the path where the file will be saved
        file_path = os.path.join(RAW_DATA_DIR, uploaded_file.name)

        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Saved uploaded file to {file_path}")

        # Trigger the extraction process
        try:
            extract_data() # This will process all excel files in RAW_DATA_DIR
            return True
        except Exception as e:
            st.error(f"Error during data extraction: {e}")
            logger.exception("Error during data extraction")
            return False
    return False

def run_classification_and_load_output(is_test, is_base_model):
    """
    Runs the data classification process and loads the appropriate output CSV.
    """
    try:
        # Ensure model output directory exists and is clear for new outputs
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        with st.spinner("Running title classification... This may take a moment."):
            process_data(is_test=is_test, is_base_model=is_base_model)
        st.success("Title classification completed!")

        output_csv_name = DEFAULT_MODEL_OUTPUT_CSV_NAME if is_base_model else FT_MODEL_OUTPUT_CSV_NAME
        output_file_path = os.path.join(MODEL_OUTPUT_DIR, output_csv_name)

        if os.path.exists(output_file_path):
            df_output = pd.read_csv(output_file_path)
            return df_output
        else:
            st.warning(f"Output file '{output_csv_name}' not found after processing. Please check logs for errors.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error during data classification: {e}")
        logger.exception("Error during data classification")
        return pd.DataFrame()

def to_excel_bytes(df):
    """Converts a pandas DataFrame to an in-memory Excel file (bytes)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def read_from_db():
    with st.spinner("Loading history from database..."):
        df = read_from_database()
        if df.empty:
            st.warning("No history found in the database.")
        else:
            st.subheader("Titles from Database:")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("---")
            
def handle_save_button_click():
    with st.spinner("Saving selected title..."):
        success = save_to_database(st.session_state['edited_df'])
        if success:
            st.success("Titles saved successfully!")
            st.session_state['is_saved'] = True
            reset_app_state()
        else:
            st.error("Failed to save the title. Try again later.")
            st.session_state['is_saved'] = False
