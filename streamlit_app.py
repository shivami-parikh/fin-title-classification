import streamlit as st
import pandas as pd
import io
import os
from classifier_streamlit_helper import (
    save_uploaded_file_and_extract, run_classification_and_load_output,
    to_excel_bytes, clear_data_directory, read_from_db,
    reset_app_state, handle_save_button_click)
from classifier_config import COLUMN_NAMES, MODEL_OUTPUT_DIR
import matplotlib.pyplot as plt
import re

def start_page():
    # Initialize session state for data persistence
    if 'processed_df' not in st.session_state:
        st.session_state['processed_df'] = pd.DataFrame()
    if 'edited_df' not in st.session_state:
        st.session_state['edited_df'] = pd.DataFrame()
    if 'extracted_file_name' not in st.session_state:
        st.session_state['extracted_file_name'] = None
    if 'is_test_selected' not in st.session_state:
        st.session_state['is_test_selected'] = False
    if 'file_uploader_key' not in st.session_state:
        st.session_state['file_uploader_key'] = 0
    if 'classification_completed' not in st.session_state:
        st.session_state['classification_completed'] = False
    if 'reset_triggered' not in st.session_state:
        clear_data_directory()
        st.session_state['reset_triggered'] = False
    if 'is_saved' not in st.session_state:
        st.session_state['is_saved'] = False

    st.title("AI-Powered Title Curation")    

    # --- 1. User uploads an Excel file and triggers extraction ---
    st.header("1. Upload Excel File")

    # Always call st.file_uploader to ensure it's rendered
    temp_uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        key=f"excel_uploader_{st.session_state['file_uploader_key']}"
    )

    uploaded_file = None
    if st.session_state['reset_triggered']:
        st.session_state['reset_triggered'] = False
    else:
        uploaded_file = temp_uploaded_file

    if uploaded_file is not None:
        if st.session_state['extracted_file_name'] != uploaded_file.name:
            if st.button(f"Extract '{uploaded_file.name}'",
                         key="extract_button_new",
                         disabled=st.session_state['classification_completed']):
                with st.spinner(f"Extracting titles from {uploaded_file.name}..."):
                    extraction_success = save_uploaded_file_and_extract(uploaded_file)
                    if extraction_success:
                        st.session_state['extracted_file_name'] = uploaded_file.name
                        st.session_state['processed_df'] = pd.DataFrame()
                        st.session_state['edited_df'] = pd.DataFrame()
                        st.success("You can now classify the titles.")
                    else:
                        st.error("Data extraction failed.")
        else:
            st.info(f"'{uploaded_file.name}' is ready for classification.")
            if st.button(f"Re-extract '{uploaded_file.name}'",
                         key="extract_button_re",
                         disabled=st.session_state['classification_completed']):
                with st.spinner(f"Re-extracting titles from {uploaded_file.name}..."):
                    extraction_success = save_uploaded_file_and_extract(uploaded_file)
                    if extraction_success:
                        st.session_state['extracted_file_name'] = uploaded_file.name
                        st.session_state['processed_df'] = pd.DataFrame()
                        st.session_state['edited_df'] = pd.DataFrame()
                        st.success("Data re-extraction complete. You can now configure and process the data.")
                    else:
                        st.error("Data re-extraction failed.")

def handle_input_and_classify():
    # --- 2. User selects model and report options, then triggers classification ---
    if st.session_state['extracted_file_name'] is not None and not st.session_state['reset_triggered']:
        st.header("2. Classify Titles")

        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.radio(
                "Select Model Type:",
                ("Open AI", "Open AI - Finnovate Research"),
                key="model_selection",
                index=1,
                help="Choose 'Open AI' or 'Open AI - Finnovate Research'",
                disabled=st.session_state['classification_completed']
            )
        with col2:
            generate_report = st.checkbox(
                "Evaluate",
                value=st.session_state['is_test_selected'],
                key="generate_report_checkbox",
                help="Check this if the uploaded file is a test file and you want a metrics report.",
                disabled=st.session_state['classification_completed']
            )

        is_base_model = (model_choice == "Open AI")
        st.session_state['is_test_selected'] = generate_report

        def on_classify_click():
            st.session_state['processed_df'] = run_classification_and_load_output(is_test=st.session_state['is_test_selected'], is_base_model=is_base_model)
            st.session_state['edited_df'] = st.session_state['processed_df'].copy()
            st.session_state['classification_completed'] = True

        if st.button("Classify", key="process_button",
                     on_click=on_classify_click,
                     disabled=st.session_state['classification_completed']):
            pass

def load_classification_metrics_report():
    report_path = MODEL_OUTPUT_DIR / "classification_metrics_report.txt"
    metrics = {}
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Parse Classification Metrics (existing code)
        metrics_start = content.find("Classification Metrics:")
        confusion_matrix_start = content.find("Confusion Matrix:")
        classification_report_start = content.find("Classification Report:")

        if metrics_start != -1 and confusion_matrix_start != -1:
            metrics_section = content[metrics_start : confusion_matrix_start].strip()
            metrics['metrics'] = metrics_section.replace("Classification Metrics:\n", "").strip()
        
        if confusion_matrix_start != -1 and classification_report_start != -1:
            confusion_matrix_section = content[confusion_matrix_start : classification_report_start].strip()
            metrics['confusion_matrix_raw'] = confusion_matrix_section.replace("Confusion Matrix:\n", "").strip()

            # --- Existing CODE FOR PARSING CONFUSION MATRIX ---
            lines = metrics['confusion_matrix_raw'].strip().split('\n')
            if len(lines) == 2:
                try:
                    row1_str = lines[0].strip('[] ').split()
                    row2_str = lines[1].strip('[] ').split()

                    if len(row1_str) == 2 and len(row2_str) == 2:
                        metrics['tn'] = int(row1_str[0])
                        metrics['fp'] = int(row1_str[1])
                        metrics['fn'] = int(row2_str[0])
                        metrics['tp'] = int(row2_str[1])
                except (ValueError, IndexError):
                    st.error("Error parsing confusion matrix values.")
                    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = None, None, None, None
            # --- END Existing CODE ---

        if classification_report_start != -1:
            classification_report_section_raw = content[classification_report_start:].replace("Classification Report:\n", "").strip()
            metrics['classification_report_raw'] = classification_report_section_raw 

            report_data = []
            lines = classification_report_section_raw.split('\n')
            
            class_pattern = re.compile(r'^\s*(\w+\s*\w*)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$')
            accuracy_pattern = re.compile(r'^\s*accuracy\s+([\d.]+)\s+(\d+)$')

            data_lines = [line for line in lines if line.strip() and not line.strip().startswith('precision')]

            for line in data_lines:
                match_class = class_pattern.match(line)
                if match_class:
                    label, precision, recall, f1, support = match_class.groups()
                    # --- MODIFIED: Exclude 'macro avg' and 'weighted avg' rows ---
                    if label.strip().lower() not in ['macro avg', 'weighted avg']:
                        report_data.append({
                            'Metric': label.strip(),
                            'Precision': float(precision),
                            'Recall': float(recall),
                            'F1-Score': float(f1),
                            'Support': int(support)
                        })
                else:
                    match_accuracy = accuracy_pattern.match(line)
                    if match_accuracy:
                        # --- MODIFIED: Skip 'Accuracy' row entirely ---
                        pass # Do not append the accuracy row to report_data
            
            if report_data:
                df_report = pd.DataFrame(report_data)
                for col in ['Precision', 'Recall', 'F1-Score']:
                    df_report[col] = pd.to_numeric(df_report[col], errors='coerce')
                df_report['Support'] = pd.to_numeric(df_report['Support'], errors='coerce', downcast='integer')
                
                metrics['classification_report_df'] = df_report
            else:
                metrics['classification_report_df'] = pd.DataFrame(columns=['Metric', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    return metrics
    
def display_results():
    # --- 3. Display, Filter, Search, and Edit Data ---
    if st.session_state['classification_completed'] and not st.session_state['edited_df'].empty and not st.session_state['reset_triggered']:
        st.header("3. Classification Result")

        # Display Classification Metrics in Accordions if "Evaluate" was selected
        if st.session_state['is_test_selected']:
            metrics_data = load_classification_metrics_report()
            if metrics_data:
                accordion_col, _= st.columns([2, 2])
                
                with accordion_col:
                    st.subheader("Performance Metrics")
                    with st.expander("Classification Metrics"):
                        st.code(metrics_data.get('metrics', 'Metrics not found.'), language='text')

                    with st.expander("Confusion Matrix"):
                        if all(k in metrics_data and metrics_data[k] is not None for k in ['tn', 'fp', 'fn', 'tp']):
                            tn = metrics_data['tn']
                            fp = metrics_data['fp']
                            fn = metrics_data['fn']
                            tp = metrics_data['tp']

                            # Create the confusion matrix array
                            # Structure: [[TN, FP], [FN, TP]]
                            confusion_matrix_array = [[tn, fp], [fn, tp]]

                            # Define class labels for the heatmap
                            class_labels = ['Not Selected', 'Selected']

                            # Create the heatmap using Matplotlib's imshow
                            fig_cm, ax_cm = plt.subplots(figsize=(5, 4), dpi=100) # Adjust figure size as needed
                            
                            # Use imshow for heatmap with a color map
                            im = ax_cm.imshow(confusion_matrix_array, cmap='Blues') # 'Blues' is a good sequential colormap

                            # Add text annotations directly on the heatmap cells
                            for i in range(len(class_labels)):
                                for j in range(len(class_labels)):
                                    text_color = "white" if confusion_matrix_array[i][j] > (max(tn,fp,fn,tp) / 2) else "black" # Simple logic for text color contrast
                                    ax_cm.text(j, i, confusion_matrix_array[i][j],
                                            ha="center", va="center", color=text_color, fontsize=12) # Adjust fontsize as needed

                            # Set ticks and labels for the axes
                            ax_cm.set_xticks(range(len(class_labels)))
                            ax_cm.set_yticks(range(len(class_labels)))
                            ax_cm.set_xticklabels(class_labels, fontsize=8)
                            ax_cm.set_yticklabels(class_labels, fontsize=8)

                            # Set axis labels and title
                            ax_cm.set_xlabel("AI Selection", fontsize=9)
                            ax_cm.set_ylabel("User Selection", fontsize=9)
                            # Adjust plot layout to prevent labels from being cut off
                            fig_cm.tight_layout()

                            st.pyplot(fig_cm)
                            plt.close(fig_cm) # Close the figure to free up memory

                        else:
                            st.code(metrics_data.get('confusion_matrix_raw', 'Confusion Matrix not found.'), language='text')
                            st.warning("Could not parse confusion matrix into numerical values for chart.")

                    with st.expander("Classification Report"):
                        if 'classification_report_df' in metrics_data and not metrics_data['classification_report_df'].empty:
                            st.dataframe(
                                metrics_data['classification_report_df'],
                                hide_index=True,        # Hides the default DataFrame index
                                use_container_width=True # Makes the table fill the column width
                            )
                        else:
                            # Fallback to displaying raw text if DataFrame parsing failed
                            st.code(metrics_data.get('classification_report_raw', 'Classification Report not found.'), language='text')
                            st.warning("Could not parse classification report into a tabular format.")
            else:
                st.warning("Classification metrics report not found or could not be parsed.")

        base_df = st.session_state['edited_df']

        st.subheader("Analyze and Edit Data")

        search_term = st.text_input("Search in Titles:", "")

        user_filter = []
        if st.session_state['is_test_selected']: # Only show user filter if "Evaluate" was selected
            user_filter = st.multiselect(
                "Filter by User Selection",
                options=base_df[COLUMN_NAMES[1]].dropna().unique().tolist()
                if COLUMN_NAMES[1] in base_df.columns else [],
                default=[]
            )

        ai_filter = st.multiselect(
            "Filter by AI Selection",
            options=base_df[COLUMN_NAMES[2]].dropna().unique().tolist()
            if COLUMN_NAMES[2] in base_df.columns else [],
            default=[]
        )

        view_df = base_df.copy()

        if search_term.strip():
            view_df = view_df[view_df[COLUMN_NAMES[0]].str.contains(search_term, case=False, na=False)]

        if user_filter:
            view_df = view_df[view_df[COLUMN_NAMES[1]].isin(user_filter)]

        if ai_filter:
            view_df = view_df[view_df[COLUMN_NAMES[2]].isin(ai_filter)]

        display_columns = [COLUMN_NAMES[0], COLUMN_NAMES[2]]
        if st.session_state['is_test_selected'] and COLUMN_NAMES[1] in view_df.columns:
            display_columns.insert(1, COLUMN_NAMES[1])

        st.subheader("Classified Titles")

        edited_view = st.data_editor(
            view_df[display_columns],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            key="data_editor_main"
        )

        if not edited_view.equals(view_df[display_columns]):
            for col in display_columns:
                st.session_state['edited_df'].loc[edited_view.index, col] = edited_view[col]

def save_data():    
    # --- 4. Save Edited Data ---
    if st.session_state['classification_completed'] and not st.session_state['edited_df'].empty and not st.session_state['reset_triggered']:
        st.header("4. Save Classified Titles")
        
        download_columns = [COLUMN_NAMES[0], COLUMN_NAMES[2]]
        
        if st.session_state['is_test_selected'] and COLUMN_NAMES[1] in st.session_state['edited_df'].columns:
            download_columns.insert(1, COLUMN_NAMES[1])
        else:
            st.session_state['edited_df'][COLUMN_NAMES[1]] = "Not Classified"
            
        if st.button("Save To Database", type="primary", on_click=handle_save_button_click):
            pass
        
        df_for_download = st.session_state['edited_df'][download_columns]
        excel_data_bytes = to_excel_bytes(df_for_download)
        
        st.download_button(
            label="Download as Excel",
            data=excel_data_bytes,
            file_name="classified_and_edited_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Click to download the edited titles and classified data as an Excel file."
        )

        st.markdown("---")
        st.subheader("Reset")
        if st.button("Reset", key="process_another_button", on_click=reset_app_state):
            pass


def classifier_app_logic():
    tab1, tab2 = st.tabs(["Title Curation", "Curated Titles"])
    
    with tab1:
        start_page()
        handle_input_and_classify()
        display_results()
        save_data()
        
    with tab2:
        st.header("Curated Titles")    
        read_from_db()
    
classifier_app_logic()