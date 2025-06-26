import json
import re
import csv
import os
import time
import asyncio
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError # Import AsyncOpenAI
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Initialize the logger for this module
from project_logger import setup_project_logger
logger = setup_project_logger("finetuning_data_generator")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def extract_from_jsonl(file_path):
    """
    Extracts the title from user content and the full content from assistant content
    in a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'user_content' and 'assistant_response'.
    """
    extracted_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    user_content = None
                    assistant_content = None

                    # Find user and assistant messages
                    for message in data.get('messages', []):
                        if message.get('role') == 'user':
                            user_content = message.get('content', '')
                        elif message.get('role') == 'assistant':
                            assistant_content = message.get('content', '')

                    if not user_content:
                        user_content = "User content not found"

                    if assistant_content is None:
                        assistant_content = "Assistant content not found"

                    extracted_data.append({
                        'user_content': user_content,
                        'assistant_response': assistant_content
                    })

                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON on line {line_num}: {e} - Line content: {line.strip()}")
                except KeyError as e:
                    logger.error(f"Key error on line {line_num}: Missing expected key {e} - Line content: {line.strip()}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred on line {line_num}: {e} - Line content: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
        return []

    return extracted_data

async def chat(message, model):
    """
    Sends a message to the OpenAI chat model asynchronously and returns the response.
    Includes robust error handling and retry mechanism for API calls.

    Args:
        message (str): The input message for the chat model.

    Returns:
        str or None: The model's response string if successful, None otherwise.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model,
                stream=True,
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                timeout=30
            )
            answer = ""
            async for chunk in response:
                res = chunk.choices[0].delta.content or ""
                answer = answer + res
            return answer
        except RateLimitError as e:
            logger.warning(f"OpenAI API Rate Limit Exceeded (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for RateLimitError. Giving up.")
                return None
        except APITimeoutError as e:
            logger.warning(f"OpenAI API Timeout Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for APITimeoutError. Giving up.")
                return None
        except APIError as e:
            logger.warning(f"OpenAI API Error (Code: {e.status_code}, Type: {e.type}) (Attempt {attempt + 1}/{MAX_RETRIES}): {e.message}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for APIError. Giving up.")
                return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during API call (Attempt {attempt + 1}/{MAX_RETRIES}): {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for unexpected error. Giving up.")
                return None
    return None

def save_processed_data(data, output_file_path):
    """
    Saves the processed data (list of dictionaries) to a CSV file.
    If the file exists, it appends to it; otherwise, it creates a new file.

    Args:
        data (list): A list of dictionaries containing processed data.
        output_file_path (str): The path to the output CSV file.
    """
    if not data:
        logger.warning(f"No data to save to '{output_file_path}'.")
        return

    try:
        file_exists = os.path.exists(output_file_path)
        fieldnames = list(data[0].keys())

        mode = 'a' if file_exists else 'w'
        with open(output_file_path, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data)
        logger.info(f"Processed data (checkpoint) successfully written/appended to '{output_file_path}'.")
    except IOError as e:
        logger.error(f"Error writing processed data to '{output_file_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving processed data: {e}", exc_info=True)

def load_existing_results(output_csv_file):
    """
    Loads already processed results from the output CSV file to enable resumption.

    Args:
        output_csv_file (str): The path to the output CSV file.

    Returns:
        list: A list of dictionaries containing already processed data.
    """
    existing_data = []
    if os.path.exists(output_csv_file):
        try:
            with open(output_csv_file, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert row to a dictionary, ensuring 'input' key exists for comparison
                    if 'input' in row:
                        existing_data.append(row)
                    else:
                        logger.warning(f"Skipping row in existing CSV without 'input' key: {row}")
            logger.info(f"Loaded {len(existing_data)} existing records from '{output_csv_file}'.")
        except Exception as e:
            logger.error(f"Error loading existing results from '{output_csv_file}': {e}", exc_info=True)
            return []
    return existing_data

async def run_tests(model, title_selected_dict, output_csv_file):
    """
    Runs tests by calling the chat model for each item concurrently and collecting results.
    Includes checkpointing to save data periodically and resumption logic.

    Args:
        title_selected_dict (list): List of dictionaries with 'user_content' and 'assistant_response'.
        output_csv_file (str): The path to the CSV file where results will be written.

    Returns:
        list: The combined output after processing.
    """
    existing_results = load_existing_results(output_csv_file)
    processed_inputs = {item['input'] for item in existing_results if 'input' in item} # Ensure 'input' key exists

    # Filter out already processed items
    items_to_process = [item for item in title_selected_dict if item['user_content'] not in processed_inputs]

    if not items_to_process:
        logger.info("All items already processed or no new items to process. Exiting.")
        return existing_results # Return all existing data

    total_new_items = len(items_to_process)
    logger.info(f"Starting to process {total_new_items} new items (total items in source: {len(title_selected_dict)}).")

    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # Function to process a single item asynchronously
    async def process_single_item(item, original_index):
        async with semaphore:
            input_content = item['user_content']
            actual_output = item['assistant_response']

            logger.info(f"Processing item {original_index + 1}/{len(title_selected_dict)}...")
            model_output = await chat(input_content, model)

            if model_output is None:
                logger.error(f"API call failed for item {original_index + 1} after all retries.")
                return None # Indicate failure for this item
            else:
                logger.info(f"Successfully processed item {original_index + 1}.")
                return {
                    "input": input_content,
                    "actual_output": actual_output,
                    "model_output": model_output
                }

    # Group items into batches for concurrent processing and periodic saving
    # The list(enumerate()) is useful to keep track of original index for logging.
    all_processed_results = list(existing_results) # Start with existing results

    # The actual_start_index tracks the index in the original title_selected_dict
    for i in range(0, total_new_items, BATCH_SIZE):
        batch_items = items_to_process[i:i + BATCH_SIZE]
        
        # Create a list of coroutines for the current batch
        # Map back to original index for comprehensive logging
        coroutines = []
        for batch_item in batch_items:
            original_index = title_selected_dict.index(batch_item)
            coroutines.append(process_single_item(batch_item, original_index))

        logger.info(f"Processing batch {i // BATCH_SIZE + 1}. Number of items in this batch: {len(batch_items)}")
        
        # Run all coroutines in the current batch concurrently
        # gather will run tasks concurrently and return results in the order of coroutines
        batch_results = await asyncio.gather(*coroutines) 
        
        # Add successful results from the batch to the overall list
        successful_batch_results = [res for res in batch_results if res is not None]
        all_processed_results.extend(successful_batch_results)

        # Save the successful results of this batch to CSV
        if successful_batch_results:
            save_processed_data(successful_batch_results, output_csv_file)
        else:
            logger.warning(f"No successful results in batch {i // BATCH_SIZE + 1} to save.")

    return all_processed_results

def read_csv_file(file_path)-> list:
    """
    Reads a CSV file and returns its content as a list of dictionaries.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries representing the rows in the CSV file.
    """
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        return []
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV file: {e}")
        return []

def generate_classification_metrics_report(output_csv_file, experiment_number=2):
    # Step 1: Load the CSV file
    
    df = pd.DataFrame()
    
    if experiment_number == 1:
        rows = read_csv_file(output_csv_file)
        for x, row in enumerate(rows):
            actual_output_list = row.get('actual_output', '').strip().split('\n')
            model_output_list = row.get('model_output', '').strip().split('\n')
            
            for y, (actual_output_text, model_output_text) in enumerate(zip(actual_output_list, model_output_list)):
                
                if ":" not in actual_output_text or ":" not in model_output_text:
                    continue
                
                actual_title = actual_output_text.rsplit(":", 1)[0].strip()
                actual_output = actual_output_text.rsplit(":", 1)[1].strip()
                model_title = model_output_text.rsplit(":", 1)[0].strip()
                model_output = model_output_text.rsplit(":", 1)[1].strip()
                
                if actual_title == model_title:
                    new_df_row = pd.DataFrame({
                        'input': actual_title,
                        'actual_output': actual_output,
                        'model_output': model_output
                    }, index=[0])
                    df = pd.concat([df, new_df_row], ignore_index=True)
                else:
                    logger.warning(f"Title mismatch in row {x+1}, title {y+1}")
            
    if df.empty:
        df = pd.read_csv(output_csv_file)
    
    # Normalize labels
    before_count = len(df)
    df = df[df['actual_output'].isin(["Selected", "Not Selected"])]
    df = df[df['model_output'].isin(["Selected", "Not Selected"])]
    after_count = len(df)
    logger.info(f"Filtered data: {before_count} -> {after_count} rows after filtering for 'Selected' and 'Not Selected' labels.")
    
    y_true = df['actual_output'].str.strip()
    y_pred = df['model_output'].str.strip()
    
    # Step 3: Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Selected')
    recall = recall_score(y_true, y_pred, pos_label='Selected')
    f1 = f1_score(y_true, y_pred, pos_label='Selected')
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['Selected', 'Not Selected'])
    class_report = classification_report(y_true, y_pred, labels=['Selected', 'Not Selected'])

    # Step 4: Print metrics
    logger.info("Classification Metrics:")
    logger.info(f"Accuracy     : {accuracy}")
    logger.info(f"Precision    : {precision}")
    logger.info(f"Recall       : {recall}")
    logger.info(f"F1 Score     : {f1}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"\nClassification Report:\n{class_report}")

def main(experinent_number: int, 
         run_experiment_flag:bool=False,
         generate_metrics_flag:bool=False
        ) -> bool:
    """Main function to execute the script."""
    logger.info("Script started execution.")
    
    if experinent_number not in [1, 2, 3]:
        logger.error(f"Invalid experiment number: {experinent_number}. Must be 1, 2, or 3.")
        return False
    
    if experinent_number == 1:
        model = "ft:gpt-4o-mini-2024-07-18:utilizeai:title-classification-bulk:BmdwSpca"
        file_name = "data/finetuning_data/test_exp1.jsonl"
        csv_output_file = "data/testing_data/test_exp1_output.csv"
    elif experinent_number == 2:
        model = "ft:gpt-4o-mini-2024-07-18:utilizeai:title-classification:BmLI1fnt"
        file_name = "data/finetuning_data/test_exp2.jsonl"
        csv_output_file = "data/testing_data/test_exp2_output.csv"
    elif experinent_number == 3:
        model = "ft:gpt-4o-mini-2024-07-18:utilizeai:title-classification-balanced:Bmcz4fNK"
        file_name = "data/finetuning_data/test_exp3.jsonl"
        csv_output_file = "data/testing_data/test_exp3_output.csv"

    if run_experiment_flag:
        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(csv_output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: '{output_dir}'")
        # Load the JSONL file and extract data
        title_selected_dict = extract_from_jsonl(file_name)
        if not title_selected_dict:
            logger.error(f"No data extracted from '{file_name}'. Please check the file content or path.")
            exit(1)
        logger.info(f"Total items to consider from source: {len(title_selected_dict)}")
        
        # Run the asynchronous api call function    
        logger.info(f"Running tests with model '{model}' on {len(title_selected_dict)} items.")
        asyncio.run(run_tests(model, title_selected_dict, csv_output_file))
        final_count_after_run = len(load_existing_results(csv_output_file))
        logger.info(f"Total items processed and saved in '{csv_output_file}': {final_count_after_run}")
    
    if generate_metrics_flag:
        # Generate classification metrics report
        generate_classification_metrics_report(csv_output_file, experiment_number)
        logger.info("Classification metrics report generated.")
    
    logger.info("Script finished execution.")
    
# --- How to implement and run the code ---
if __name__ == "__main__":
    
    # Variable to control the experiment
    experiment_number = 1
    run_experiment_flag = False
    generate_metrics_flag = True
    
    # Run the main function
    if run_experiment_flag:
        # Initialize AsyncOpenAI client
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0) # Set client-level timeout

        # Configuration for batch processing and retries
        BATCH_SIZE = 100  # Number of records to process before saving to CSV
        CONCURRENT_REQUESTS = 10 # Number of API requests to make concurrently within a batch
        MAX_RETRIES = 5   # Maximum number of retries for an API call per request
        RETRY_DELAY_SECONDS = 5 # Initial delay for retries (will increase exponentially)
    
    main(experiment_number, run_experiment_flag, generate_metrics_flag)
    
    pass