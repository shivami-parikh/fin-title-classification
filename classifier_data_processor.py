import asyncio
import os
import csv
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

# Import configurations and logger setup
from classifier_config import (
    MAX_RETRIES, RETRY_DELAY_SECONDS, BATCH_SIZE, CONCURRENT_REQUESTS,
    MODEL_OUTPUT_DIR, OPENAI_API_KEY, PROCESSED_DATA_DIR,
    PROCESSED_CSV_NAME, FT_MODEL_OUTPUT_CSV_NAME, DEFAULT_MODEL_OUTPUT_CSV_NAME,
    FT_MODEL, BASE_MODEL, FT_PROMPT_TEMPLATE, BASE_PROMPT_TEMPLATE, COLUMN_NAMES
)
from project_logger import setup_project_logger

logger = setup_project_logger("classifier_data_processor") # Module-specific logger

class DataLoader:
    """
    Handles all CSV file reading and writing operations.
    Adheres to SRP by centralizing data persistence logic.
    """
    def __init__(self, logger):
        self.logger = logger
        self.fieldnames = COLUMN_NAMES # Standard fields for processed data

    def read_input_csv(self, file_path):
        """
        Reads a CSV file with headers: source_website, title, url, selected, date.
        Extracts 'title' for model input and 'selected' for user (actual label).

        Args:
            file_path (str): The path to the input CSV file.

        Returns:
            list: A list of dictionaries, where each dictionary contains
                  'Title' (for model input) and 'user' (actual label).
        """
        extracted_data = []
        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not all(col in reader.fieldnames for col in ['title', 'selected']):
                    self.logger.error(f"CSV file '{file_path}' must contain 'title' and 'selected' columns.")
                    return []
                for row_num, row in enumerate(reader, 1):
                    title = row.get('title', '').strip()
                    selected = row.get('selected', '').strip()

                    if not title:
                        self.logger.warning(f"Skipping row {row_num} due to missing 'title'.")
                        continue
                    
                    extracted_data.append({
                        'title': title,
                        'user': selected
                    })
            self.logger.info(f"Successfully extracted {len(extracted_data)} records from '{file_path}'.")
        except FileNotFoundError:
            self.logger.error(f"Error: The file '{file_path}' was not found.")
            return []
        except Exception as e:
            self.logger.error(f"An error occurred while reading the CSV file '{file_path}': {e}", exc_info=True)
            return []
        return extracted_data

    def load_existing_results(self, output_csv_file):
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
                        # Validate row structure
                        if all(key in row for key in self.fieldnames):
                            existing_data.append(row)
                        else:
                            self.logger.warning(f"Skipping malformed row in existing CSV: {row}")
                self.logger.info(f"Loaded {len(existing_data)} existing records from '{output_csv_file}'.")
            except Exception as e:
                self.logger.error(f"Error loading existing results from '{output_csv_file}': {e}", exc_info=True)
                return []
        return existing_data

    def save_processed_data(self, data, output_file_path):
        """
        Saves the processed data (list of dictionaries) to a CSV file.
        If the file exists, it appends to it; otherwise, it creates a new file.

        Args:
            data (list): A list of dictionaries containing processed data.
            output_file_path (str): The path to the output CSV file.
        """
        if not data:
            self.logger.warning(f"No data to save to '{output_file_path}'.")
            return

        try:
            file_exists = os.path.exists(output_file_path)
            mode = 'a' if file_exists else 'w'
            with open(output_file_path, mode=mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(data)
            self.logger.info(f"Processed data (checkpoint) successfully written/appended to '{output_file_path}'.")
        except IOError as e:
            self.logger.error(f"Error writing processed data to '{output_file_path}': {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while saving processed data: {e}", exc_info=True)

class APIManager:
    """
    Manages all interactions with the OpenAI API, including retry logic.
    Adheres to SRP by focusing solely on API communication.
    """
    def __init__(self, api_key, model, max_retries, retry_delay_seconds, logger):
        self.client = AsyncOpenAI(api_key=api_key, timeout=30.0)
        self.model = FT_MODEL
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.logger = logger

    async def _handle_api_error(self, e, attempt, error_type="API"):
        """Helper to log and handle API errors consistently."""
        log_func = self.logger.warning if isinstance(e, (RateLimitError, APITimeoutError)) else self.logger.error
        log_func(f"OpenAI {error_type} Error (Attempt {attempt + 1}/{self.max_retries}): {e}")
        if attempt < self.max_retries - 1:
            sleep_time = self.retry_delay_seconds * (2 ** attempt)
            self.logger.info(f"Retrying in {sleep_time:.2f} seconds...")
            await asyncio.sleep(sleep_time)
            return True # Indicate that a retry should happen
        else:
            self.logger.error(f"Max retries reached for {error_type} error. Giving up.")
            return False # Indicate that no more retries should happen

    async def chat_completion(self, message):
        """
        Sends a message to the OpenAI chat model asynchronously and returns the response.
        Includes robust error handling and exponential backoff retry mechanism.

        Args:
            message (str): The input message for the chat model.

        Returns:
            str or None: The model's response string if successful, None otherwise.
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    stream=True,
                    messages=[{"role": "user", "content": message}],
                    timeout=30 # Can also be configured
                )
                answer = ""
                async for chunk in response:
                    res = chunk.choices[0].delta.content or ""
                    answer += res
                return answer
            except RateLimitError as e:
                if not await self._handle_api_error(e, attempt, "RateLimit"): return None
            except APITimeoutError as e:
                if not await self._handle_api_error(e, attempt, "Timeout"): return None
            except APIError as e:
                self.logger.warning(f"OpenAI API Error (Code: {e.status_code}, Type: {e.type}) (Attempt {attempt + 1}/{self.max_retries}): {e.message}")
                if not await self._handle_api_error(e, attempt, "API"): return None
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during API call (Attempt {attempt + 1}/{self.max_retries}): {e}", exc_info=True)
                if not await self._handle_api_error(e, attempt, "Unexpected"): return None
        return None

class Classifier:
    """
    Orchestrates the classification process, fetching data, making API calls,
    and saving results.
    """
    def __init__(self, data_loader: DataLoader, api_manager: APIManager, logger, prompt_template, batch_size, concurrent_requests):
        self.data_loader = data_loader
        self.api_manager = api_manager
        self.logger = logger
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.concurrent_requests = concurrent_requests
        self.semaphore = asyncio.Semaphore(self.concurrent_requests)

    async def _process_single_item(self, item, original_index):
        """Helper to process a single item with API call."""
        async with self.semaphore:
            title_input = item['title']
            user = item['user']
            
            message = self.prompt_template + title_input

            self.logger.info(f"Processing item {original_index + 1}")
            model_output = await self.api_manager.chat_completion(message)

            if model_output is None:
                self.logger.error(f"API call failed for item {original_index + 1} after all retries.")
                return None
            else:
                self.logger.info(f"Successfully processed item {original_index + 1}.")
                return {
                    COLUMN_NAMES[0] : title_input,
                    COLUMN_NAMES[1] : user,
                    COLUMN_NAMES[2] : model_output.strip() # Strip whitespace from model output
                }

    async def classify_titles(self, input_data_list, output_csv_file):
        """
        Runs tests by calling the chat model for each item concurrently and collecting results.
        Includes checkpointing to save data periodically and resumption logic.

        Args:
            input_data_list (list): List of dictionaries with 'title' and 'user'.
            output_csv_file (str): The path to the CSV file where results will be written.

        Returns:
            list: The combined output after processing.
        """
        logger.info(f"Shivami:{output_csv_file}")
        existing_results = self.data_loader.load_existing_results(output_csv_file)
        processed_titles = {item['Title'] for item in existing_results if 'Title' in item}

        items_to_process = [item for item in input_data_list if item['title'] not in processed_titles]

        if not items_to_process:
            self.logger.info("All items already processed or no new items to process. Exiting.")
            return existing_results

        total_new_items = len(items_to_process)
        self.logger.info(f"Starting to process {total_new_items} new items (total items in source: {len(input_data_list)}).")

        all_processed_results = list(existing_results) # Start with existing results

        for i in range(0, total_new_items, self.batch_size):
            batch_items = items_to_process[i:i + self.batch_size]
            
            coroutines = []
            for batch_item in batch_items:
                try:
                    original_index = input_data_list.index(batch_item) # Find original index for logging context
                except ValueError:
                    original_index = -1
                coroutines.append(self._process_single_item(batch_item, original_index))

            self.logger.info(f"Processing batch {i // self.batch_size + 1}. Number of items in this batch: {len(batch_items)}")
            
            batch_results = await asyncio.gather(*coroutines)
            
            successful_batch_results = [res for res in batch_results if res is not None]
            all_processed_results.extend(successful_batch_results)

            if successful_batch_results:
                self.data_loader.save_processed_data(successful_batch_results, output_csv_file)
            else:
                self.logger.warning(f"No successful results in batch {i // self.batch_size + 1} to save.")

        return all_processed_results

class MetricsReporter:
    """
    Generates and reports classification metrics based on model output.
    Adheres to SRP by focusing solely on reporting.
    """
    def __init__(self, logger):
        self.logger = logger

    def generate_report(self, output_csv_file):
        self.logger.info("Generating classification metrics report.")
        
        try:
            df = pd.read_csv(output_csv_file)
        except FileNotFoundError:
            self.logger.error(f"Error: Output file '{output_csv_file}' not found for metrics generation.")
            return
        except Exception as e:
            self.logger.error(f"Error reading CSV for metrics generation from '{output_csv_file}': {e}", exc_info=True)
            return

        if COLUMN_NAMES[1] not in df.columns or COLUMN_NAMES[2] not in df.columns:
            self.logger.error(f"Required columns '{COLUMN_NAMES[1]}' and '{COLUMN_NAMES[2]}' not found in '{output_csv_file}'.")
            return

        # Normalize labels and filter for 'Selected' and 'Not Selected'
        df['user'] = df[COLUMN_NAMES[1]].str.strip()
        df['model'] = df[COLUMN_NAMES[2]].str.strip()
        
        before_count = len(df)
        df_filtered = df[
            (df['user'].isin(["Selected", "Not Selected"])) &
            (df['model'].isin(["Selected", "Not Selected"]))
        ].copy() # Use .copy() to avoid SettingWithCopyWarning
        after_count = len(df_filtered)
        
        self.logger.info(f"Filtered data: {before_count} -> {after_count} rows after filtering for 'Selected' and 'Not Selected' labels.")
        
        if after_count == 0:
            self.logger.warning("No valid data points remaining after filtering for 'Selected' and 'Not Selected' labels. Cannot generate metrics.")
            return

        y_true = df_filtered['user']
        y_pred = df_filtered['model']
        
        # Define target names and labels explicitly for consistent reporting
        target_names = ['Not Selected', 'Selected']
        labels = ['Not Selected', 'Selected']

        try:
            accuracy = accuracy_score(y_true, y_pred)
            # Use 'pos_label' for binary classification if 'Selected' is the positive class
            precision = precision_score(y_true, y_pred, pos_label='Selected', zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label='Selected', zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label='Selected', zero_division=0)
            conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
            class_report_str = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)

            self.logger.info("Classification Metrics:")
            self.logger.info(f"Accuracy     : {accuracy:.4f}")
            self.logger.info(f"Precision    : {precision:.4f}")
            self.logger.info(f"Recall       : {recall:.4f}")
            self.logger.info(f"F1 Score     : {f1:.4f}")
            self.logger.info(f"Confusion Matrix:\n{conf_matrix}")
            self.logger.info(f"\nClassification Report:\n{class_report_str}")
            
            # Optionally save the report to a file
            report_path = os.path.join(MODEL_OUTPUT_DIR, "classification_metrics_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("Classification Metrics:\n")
                f.write(f"Accuracy     : {accuracy:.4f}\n")
                f.write(f"Precision    : {precision:.4f}\n")
                f.write(f"Recall       : {recall:.4f}\n")
                f.write(f"F1 Score     : {f1:.4f}\n")
                f.write(f"Confusion Matrix:\n{conf_matrix}\n")
                f.write(f"\nClassification Report:\n{class_report_str}\n")
            self.logger.info(f"Metrics report saved to '{report_path}'.")

        except ValueError as e:
            self.logger.error(f"Error during metrics calculation. This often means a label is missing or inconsistent: {e}")
            self.logger.error(f"Unique true labels: {y_true.unique()}")
            self.logger.error(f"Unique predicted labels: {y_pred.unique()}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during metrics generation: {e}", exc_info=True)

class DataProcessor:

    """
    Main class to orchestrate the entire classification process.
    """
    def __init__(self):
        self.logger = setup_project_logger("data_processor")
        self.data_loader = DataLoader(self.logger)
        self.base_prompt_template = BASE_PROMPT_TEMPLATE
        self.ft_prompt_template = FT_PROMPT_TEMPLATE
        self.base_model = BASE_MODEL
        self.ft_model = FT_MODEL
        model = self.ft_model
        prompt_template = self.ft_prompt_template
        self.api_manager = APIManager(OPENAI_API_KEY, model, MAX_RETRIES, RETRY_DELAY_SECONDS, self.logger)
        self.classifier = Classifier(self.data_loader, self.api_manager, self.logger, prompt_template, BATCH_SIZE, CONCURRENT_REQUESTS)
        self.metrics_reporter = MetricsReporter(self.logger)
        self.input_file_path = os.path.join(PROCESSED_DATA_DIR, PROCESSED_CSV_NAME)
        self.ft_output_csv_file = os.path.join(MODEL_OUTPUT_DIR, FT_MODEL_OUTPUT_CSV_NAME)
        self.default_output_csv_file = os.path.join(MODEL_OUTPUT_DIR, DEFAULT_MODEL_OUTPUT_CSV_NAME)
        
        # Ensure model output directory exists
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


    async def run(self, is_test=False, is_base_model=False, limit_data=None):
        """
        Executes the main classification workflow.

        Args:
            is_test (bool): If True, generates a classification metrics report.
            limit_data (int, optional): If provided, limits the number of input items for testing.
        """
        self.logger.info("Script started execution.")

        if not os.path.exists(self.input_file_path):
            self.logger.error(f"Input file not found: '{self.input_file_path}'. Please ensure data is processed by excel_extractor.py.")
            return False

        input_data_list = self.data_loader.read_input_csv(self.input_file_path)
        
        if limit_data is not None:
            input_data_list = input_data_list[:limit_data]
            self.logger.info(f"Limiting input data to {limit_data} items for testing purposes.")
        
        if not input_data_list:
            self.logger.error(f"No data extracted from '{self.input_file_path}'. Please check the file content or path.")
            return False
            
        self.logger.info(f"Total items to consider from source: {len(input_data_list)}")
        
        if is_base_model:
            self.api_manager.model = self.base_model
            self.classifier.prompt_template = self.base_prompt_template
        else:
            self.api_manager.model = self.ft_model
            self.classifier.prompt_template = self.ft_prompt_template

        self.logger.info(f"Running classification with model '{self.api_manager.model}' on {len(input_data_list)} items.")
        
        # Pass input_data_list directly to classifier for processing
        self.output_csv_file = self.default_output_csv_file if is_base_model else self.ft_output_csv_file
        
        final_processed_data = await self.classifier.classify_titles(input_data_list, self.output_csv_file)
        self.logger.info(f"Total items processed and saved in '{self.output_csv_file}': {len(final_processed_data)}")
        
        if is_test:
            self.metrics_reporter.generate_report(self.output_csv_file)
            self.logger.info("Classification metrics report generated.")
        
        self.logger.info("Script finished execution.")
        return True

# --- How to use the code ---
if __name__ == "__main__":
    processor = DataProcessor()    # Run with is_test=True to generate metrics report
    # Use limit_data for quick testing on a subset of data
    asyncio.run(processor.run(is_test=True, limit_data=105)) # For production, remove limit_data