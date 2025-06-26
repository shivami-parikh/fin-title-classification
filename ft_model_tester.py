import json
import re
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
import csv
import os
from dotenv import load_dotenv
from project_logger import setup_project_logger

# Initialize the logger for this module
logger = setup_project_logger("finetuning_data_generator")
# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

def extract_from_jsonl(file_path):
    """
    Extracts the title from user content and the full content from assistant content
    in a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'extracted_title' and 'assistant_response'.
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

def chat(message):
    """
    Sends a message to the OpenAI chat model and returns the response.
    Includes robust error handling for API calls.

    Args:
        message (str): The input message for the chat model.

    Returns:
        str or None: The model's response string if successful, None otherwise.
    """
    try:
        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:utilizeai:title-classification:BmLI1fnt",
            # model="chatgpt-4o-latest", # Uncomment and use this if you want to test with a standard model
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ],
            timeout=30 # Add a timeout to prevent indefinite waiting
        )
        answer = ""
        for chunk in response:
            res = chunk.choices[0].delta.content or ""
            answer = answer + res
        return answer
    except RateLimitError as e:
        logger.error(f"OpenAI API Rate Limit Exceeded: {e}")
        return None
    except APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error: {e}")
        return None
    except APIError as e:
        logger.error(f"OpenAI API Error (Code: {e.status_code}, Type: {e.type}): {e.message}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call: {e}", exc_info=True)
        return None

def save_processed_data(data, output_file_path):
    """
    Saves the processed data (list of dictionaries) to a CSV file.

    Args:
        data (list): A list of dictionaries containing processed data.
        output_file_path (str): The path to the output CSV file.
    """
    if not data:
        logger.warning(f"No data to save to '{output_file_path}'.")
        return

    try:
        # Determine headers from the keys of the first dictionary
        fieldnames = data[0].keys()
        with open(output_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Processed data (checkpoint) successfully written to '{output_file_path}'.")
    except IOError as e:
        logger.error(f"Error writing processed data to '{output_file_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving processed data: {e}", exc_info=True)


def run_tests(title_selected_dict, output_csv_file):
    """
    Runs tests by calling the chat model for each item and collecting results.
    Includes checkpointing to save data on API failure.

    Args:
        title_selected_dict (list): List of dictionaries with 'user_content' and 'assistant_response'.
        output_csv_file (str): The path to the CSV file where results will be written.

    Returns:
        list: The combined output after processing.
    """
    combined_output = []
    total_items = len(title_selected_dict)

    for i, item in enumerate(title_selected_dict):
        logger.info(f"Processing item {i+1}/{total_items}...")
        input_content = item['user_content']
        actual_output = item['assistant_response']

        model_output = chat(input_content)

        if model_output is None:
            logger.error(f"API call failed for item {i+1}. Saving processed data and exiting.")
            # Save data processed so far
            save_processed_data(combined_output, output_csv_file)
            break # Exit the loop if API call failed
        else:
            combined_output.append({
                "input": input_content,
                "actual_output": actual_output,
                "model_output": model_output
            })
            logger.info(f"Successfully processed item {i+1}.")

    return combined_output


# --- How to implement and run the code ---
if __name__ == "__main__":
    file_name = "data/finetuning_data/test_exp2.jsonl"
    csv_output_file = "data/testing_data/test_exp2_output.csv"

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(csv_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: '{output_dir}'")

    title_selected_dict = extract_from_jsonl(file_name)

    if not title_selected_dict:
        logger.error(f"No data extracted from '{file_name}'. Please check the file content or path.")
        exit(1)

    logger.info(f"Starting API calls for {len(title_selected_dict)} items.")
    final_combined_output = run_tests(title_selected_dict, csv_output_file)

    # Save data after all items are processed (or after an early exit due to API error)
    # This call acts as the final save or a redundant save if an error occurred earlier.
    save_processed_data(final_combined_output, csv_output_file)

    logger.info("Script finished execution.")
