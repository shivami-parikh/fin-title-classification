from excel_data_extractor import ExcelProcessor

def extract_data():
    processor = ExcelProcessor()
    processor.process_all_excel_files_in_directory()

if __name__ == "__main__":
    extract_data()