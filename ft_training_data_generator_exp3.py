import pandas as pd
import os
import json
import random
from project_logger import setup_project_logger

# Initialize the logger for this module
logger = setup_project_logger("finetuning_data_generator_exp3")

def _format_to_json(title, selected):
    """
    Formats a single (title, selected) pair into the specified JSON structure
    for fine-tuning, as a single record.

    Args:
        title (str): The news article title.
        selected (str): The selection status ("Selected" or "Not Selected").

    Returns:
        dict: A dictionary representing the JSON structure for a single fine-tuning message.
              Returns None if the title is empty.
    """
    title_str = str(title).strip() if title is not None else ""
    if not title_str: # Skip empty titles
        return None

    # Construct the user message content with a single title
    user_message_content = (
        "You are an analyst with a deep understanding of the banking business and you support financial institutions in scanning the universe of disruptive financial innovations and places them in context, to drive strategy. Your job is to go through news titles and decide whether it should be selected or not for your target audience - banker or strategy executive. You are given a news title and you need to decide if it should be selected or not. The input will be under Title section and the output should be under Output section. The output must be one of the two values: Selected or Not Selected. "
        f"Title: {title_str}"
    )

    # Construct the assistant message content with a single output
    assistant_message_content = selected.strip()

    return {
        "messages": [
            {"role": "user", "content": user_message_content},
            {"role": "assistant", "content": assistant_message_content}
        ]
    }

def process_csv_for_finetuning(input_csv_path, all_train_data, all_test_data, all_validate_data):
    """
    Processes a single CSV file, balances the "Selected" and "Not Selected" groups,
    and then splits the combined data into train, test, and validation sets.
    Each record in the output will represent a single title.

    Args:
        input_csv_path (str): Path to the input CSV file.
        all_train_data (list): List to accumulate all training data.
        all_test_data (list): List to accumulate all testing data.
        all_validate_data (list): List to accumulate all validation data.
    """
    logger.info(f"Starting data processing for fine-tuning from '{input_csv_path}' (Experiment 3)")

    try:
        df = pd.read_csv(input_csv_path)
        logger.info(f"Successfully loaded CSV with {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"Error: Input CSV file not found at '{input_csv_path}'.")
        return
    except pd.errors.EmptyDataError:
        logger.warning(f"Warning: Input CSV file at '{input_csv_path}' is empty.")
        return
    except Exception as e:
        logger.error(f"Error reading CSV file '{input_csv_path}': {e}", exc_info=True)
        return

    # Ensure required columns exist
    required_columns = ['title', 'selected']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Error: CSV must contain all of the following columns: {required_columns}. Found: {df.columns.tolist()}")
        return

    # Drop rows where 'title' or 'selected' are null, as they are essential for the task
    df.dropna(subset=['title', 'selected'], inplace=True)
    if df.empty:
        logger.warning("No valid 'title' and 'selected' data found after dropping NaNs. Exiting.")
        return
    logger.info(f"Remaining rows after dropping NaNs in 'title'/'selected': {len(df)}")

    # Step 1 & 2: Take all rows, and group by the 'selected' column
    selected_group = df[df['selected'] == 'Selected']
    not_selected_group = df[df['selected'] == 'Not Selected']

    logger.info(f"Found {len(selected_group)} 'Selected' records and {len(not_selected_group)} 'Not Selected' records.")

    # Step 3: Count the number of records in "Selected" group, and randomly pick that many
    # records from the "Not Selected" group.
    num_selected = len(selected_group)
    
    if num_selected == 0:
        logger.warning("No 'Selected' records found. Cannot balance data. Skipping this CSV.")
        return
        
    # Sample 'Not Selected' data, ensuring we don't request more than available
    num_to_sample = min(num_selected, len(not_selected_group))
    sampled_not_selected_group = not_selected_group.sample(n=num_to_sample, random_state=42) # Using a fixed seed for reproducibility
    
    logger.info(f"Randomly picked {len(sampled_not_selected_group)} 'Not Selected' records to balance.")

    # Step 4: Combine the "Selected" records with the sampled "Not Selected" records
    balanced_df = pd.concat([selected_group, sampled_not_selected_group]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Combined data now has {len(balanced_df)} records after balancing.")

    # Convert to list of (title, selected) tuples for further processing
    all_titles_and_selected = [(row['title'], row['selected']) for idx, row in balanced_df.iterrows()
                               if pd.notna(row['title']) and str(row['title']).strip()]

    if not all_titles_and_selected:
        logger.warning("No valid titles found after balancing and cleaning. Skipping this CSV.")
        return

    # Step 5: Create a 60-20-20 train-test-validate split randomly for this combined data
    random.shuffle(all_titles_and_selected) # Shuffle the combined data for random splitting

    total_samples = len(all_titles_and_selected)
    train_end = int(0.6 * total_samples)
    test_end = train_end + int(0.2 * total_samples)

    train_data_combined = all_titles_and_selected[:train_end]
    test_data_combined = all_titles_and_selected[train_end:test_end]
    validate_data_combined = all_titles_and_selected[test_end:]

    logger.debug(f"Combined data split: Train={len(train_data_combined)}, Test={len(test_data_combined)}, Validate={len(validate_data_combined)}")

    # Step 5 (cont.): Create the train, test and validate lists in the specified JSON format
    for title, selected in train_data_combined:
        formatted_entry = _format_to_json(title, selected)
        if formatted_entry:
            all_train_data.append(formatted_entry)
    
    for title, selected in test_data_combined:
        formatted_entry = _format_to_json(title, selected)
        if formatted_entry:
            all_test_data.append(formatted_entry)
    
    for title, selected in validate_data_combined:
        formatted_entry = _format_to_json(title, selected)
        if formatted_entry:
            all_validate_data.append(formatted_entry)

    logger.info(f"Finished processing data from '{input_csv_path}'.")


def generate_finetuning_data_for_directory(titles_and_urls_folder, output_finetuning_dir):
    """
    Processes all CSV files in the specified input directory to generate fine-tuning data
    for Experiment 3, and then writes the combined train, test, and validate files.

    Args:
        titles_and_urls_folder (str): Path to the folder containing input CSV files.
        output_finetuning_dir (str): Directory where the output JSONL files will be saved.
    """
    # Ensure parent directories exist
    os.makedirs(titles_and_urls_folder, exist_ok=True)
    os.makedirs(output_finetuning_dir, exist_ok=True)

    csv_files_in_dir = [f for f in os.listdir(titles_and_urls_folder) if f.endswith('.csv')]

    if not csv_files_in_dir:
        logger.warning(f"No CSV files found in '{titles_and_urls_folder}' to process.")
        return # Exit if no files to process

    # Initialize lists to accumulate all data across all CSVs
    all_train_data = []
    all_test_data = []
    all_validate_data = []

    for csv_file_name in csv_files_in_dir:
        input_csv_file_path = os.path.join(titles_and_urls_folder, csv_file_name)
        logger.info(f"\n--- Processing CSV for Exp3: '{csv_file_name}' ---")
        # Pass the accumulating lists to the processing function
        process_csv_for_finetuning(input_csv_file_path, all_train_data, all_test_data, all_validate_data)
        logger.info(f"--- Finished processing CSV for Exp3: '{csv_file_name}' ---\n")

    logger.info("All relevant CSV files processed for fine-tuning data generation (Experiment 3).")

    # Step 6: Write the combined data to single files with _exp3 suffix
    output_train_path = os.path.join(output_finetuning_dir, "train_exp3.jsonl")
    output_test_path = os.path.join(output_finetuning_dir, "test_exp3.jsonl")
    output_validate_path = os.path.join(output_finetuning_dir, "validate_exp3.jsonl")

    # Helper to write final combined JSONL files
    def _write_combined_jsonl(data_list, file_path, name):
        if data_list:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for entry in data_list:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                logger.info(f"Successfully wrote {len(data_list)} combined {name} entries to '{file_path}'")
            except Exception as e:
                logger.error(f"Error writing combined {name} data to '{file_path}': {e}", exc_info=True)
        else:
            logger.warning(f"No combined {name} data generated. Skipping file '{file_path}'.")

    _write_combined_jsonl(all_train_data, output_train_path, "train")
    _write_combined_jsonl(all_test_data, output_test_path, "test")
    _write_combined_jsonl(all_validate_data, output_validate_path, "validate")


if __name__ == "__main__":
    titles_and_urls_folder = os.path.join('data', 'titles_and_urls')
    output_finetuning_dir = os.path.join('data', 'finetuning_data')
    
    generate_finetuning_data_for_directory(titles_and_urls_folder, output_finetuning_dir)