import pandas as pd
import os
import json
import random
from project_logger import setup_project_logger

# Initialize the logger for this module
logger = setup_project_logger("finetuning_data_generator")

def _format_to_json(titles_and_selected_pairs):
    """
    Formats a list of (title, selected) pairs into the specified JSON structure
    for fine-tuning.

    Args:
        titles_and_selected_pairs (list): A list of tuples, where each tuple
                                         is (title, selected_status).

    Returns:
        dict: A dictionary representing the JSON structure for the fine-tuning message.
    """
    if not titles_and_selected_pairs:
        return None

    user_content_titles = []
    assistant_content_outputs = []

    for i, (title, selected) in enumerate(titles_and_selected_pairs):
        # Ensure title is a string; replace None with empty string if necessary
        title_str = str(title).strip() if title is not None else ""
        if not title_str: # Skip empty titles
            continue
        user_content_titles.append(f"{i+1}: {title_str}")
        assistant_content_outputs.append(f"{i+1}: {selected.strip()}")

    if not user_content_titles: # If all titles were empty/invalid
        return None

    user_message_content = (
        "You are an analyst with a deep understanding of the banking business and you support financial institutions in scanning the universe of disruptive financial innovations and places them in context, to drive strategy. Your job is to go through news titles and decide whether it should be selected or not for your target audience - banker or strategy executive. You are given a news title and you need to decide if it should be selected or not. The input will be under Title section and the output should be under Output section. The output must be one of the two values: Selected or Not Selected.\n"
        f"Titles:\n{'\n'.join(user_content_titles)}\n"
    )

    assistant_message_content = f"Output:\n{'\n'.join(assistant_content_outputs)}"

    return {
        "messages": [
            {"role": "user", "content": user_message_content},
            {"role": "assistant", "content": assistant_message_content}
        ]
    }

def process_csv_for_finetuning(input_csv_path, all_train_data, all_test_data, all_validate_data):
    """
    Processes a single CSV file, splits its data into train, test, and validation sets,
    and appends the formatted data to the provided lists.

    Args:
        input_csv_path (str): Path to the input CSV file.
        all_train_data (list): List to accumulate all training data.
        all_test_data (list): List to accumulate all testing data.
        all_validate_data (list): List to accumulate all validation data.
    """
    logger.info(f"Starting data processing for fine-tuning from '{input_csv_path}'")

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
    required_columns = ['title', 'selected', 'date']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Error: CSV must contain all of the following columns: {required_columns}. Found: {df.columns.tolist()}")
        return

    # Drop rows where 'title' or 'selected' are null, as they are essential for the task
    df.dropna(subset=['title', 'selected'], inplace=True)
    if df.empty:
        logger.warning("No valid 'title' and 'selected' data found after dropping NaNs. Exiting.")
        return
    logger.info(f"Remaining rows after dropping NaNs in 'title'/'selected': {len(df)}")

    # Step 1: Group by date column
    grouped_by_date = df.groupby('date')
    logger.info(f"Grouped data by {len(grouped_by_date)} unique dates.")

    for date, group in grouped_by_date:
        logger.debug(f"Processing group for date: {date} with {len(group)} entries.")
        # Step 2: Read 'title' and 'selected' columns for the current group
        titles_and_selected = [(row['title'], row['selected']) for idx, row in group.iterrows()
                               if pd.notna(row['title']) and str(row['title']).strip()]
        
        if not titles_and_selected:
            logger.warning(f"No valid titles found for date '{date}'. Skipping this group.")
            continue

        # Step 3: Randomly split into sub-groups of train, test, and validate data
        random.shuffle(titles_and_selected) # Shuffle the data for random splitting

        total_samples = len(titles_and_selected)
        train_end = int(0.6 * total_samples)
        test_end = train_end + int(0.2 * total_samples)

        train_data = titles_and_selected[:train_end]
        test_data = titles_and_selected[train_end:test_end]
        validate_data = titles_and_selected[test_end:]

        logger.debug(f"  Split for date '{date}': Train={len(train_data)}, Test={len(test_data)}, Validate={len(validate_data)}")

        # Step 4: Convert each subgroup into the specified JSON format
        # Step 5: Append to the main lists
        if train_data:
            formatted_train = _format_to_json(train_data)
            if formatted_train:
                all_train_data.append(formatted_train)
        if test_data:
            formatted_test = _format_to_json(test_data)
            if formatted_test:
                all_test_data.append(formatted_test)
        if validate_data:
            formatted_validate = _format_to_json(validate_data)
            if formatted_validate:
                all_validate_data.append(formatted_validate)

    logger.info(f"Finished processing data from '{input_csv_path}'.")


def generate_finetuning_data_for_directory(titles_and_urls_folder, output_finetuning_dir):
    """
    Processes all CSV files in the specified input directory to generate fine-tuning data,
    and then writes the combined train, test, and validate files.

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
        logger.info(f"\n--- Processing CSV: '{csv_file_name}' ---")
        # Pass the accumulating lists to the processing function
        process_csv_for_finetuning(input_csv_file_path, all_train_data, all_test_data, all_validate_data)
        logger.info(f"--- Finished processing CSV: '{csv_file_name}' ---\n")

    logger.info("All relevant CSV files processed for fine-tuning data generation.")

    # Write the combined data to single files
    output_train_path = os.path.join(output_finetuning_dir, "train_exp1.jsonl")
    output_test_path = os.path.join(output_finetuning_dir, "test_exp1.jsonl")
    output_validate_path = os.path.join(output_finetuning_dir, "validate_exp1.jsonl")

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