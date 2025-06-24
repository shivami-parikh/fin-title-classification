from excel_data_extractor import process_files_in_directory

if __name__ == "__main__":
    
    extract_data_from_excel_files = True
    
    if extract_data_from_excel_files:
        daily_trackers_folder = 'data/daily_trackers'
        titles_and_urls_folder = 'data/titles_and_urls'
        process_files_in_directory(daily_trackers_folder, titles_and_urls_folder)