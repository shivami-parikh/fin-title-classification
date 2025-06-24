import os
from excel_data_extractor import process_files_in_directory
import ft_training_data_generator_exp1 as exp1
import ft_training_data_generator_exp2 as exp2
import ft_training_data_generator_exp3 as exp3

if __name__ == "__main__":
    
    extract_data_from_excel_files = True
    
    if extract_data_from_excel_files:
        daily_trackers_folder = 'data/daily_trackers'
        titles_and_urls_folder = 'data/titles_and_urls'
        process_files_in_directory(daily_trackers_folder, titles_and_urls_folder)
        
    titles_and_urls_folder = os.path.join('data', 'titles_and_urls')
    output_finetuning_dir = os.path.join('data', 'finetuning_data')
    
    exp1.generate_finetuning_data_for_directory(titles_and_urls_folder, output_finetuning_dir)
    exp2.generate_finetuning_data_for_directory(titles_and_urls_folder, output_finetuning_dir)
    exp3.generate_finetuning_data_for_directory(titles_and_urls_folder, output_finetuning_dir)