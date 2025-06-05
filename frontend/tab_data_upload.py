import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import json
from datetime import datetime

from api_utils import get_token, clear_index, upload_data, predict, accuracy_score, train_test_split, classification_report, confusion_matrix, filter_high_quality_classes

def render_data_upload_tab(api_url, username, password, collection=None):
    """Display the Data Upload and Quality Evaluation tab"""
    title = "Data Upload and Quality Evaluation"
    if collection:
        title += f" ({collection})"
    st.title(title)
    
    # Create directories for saving files and metrics if they don't exist
    data_dir = "uploaded_data"
    metrics_dir = os.path.join("metrics", collection or "prod")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # Upload file (CSV or Excel)
    uploaded_file = st.file_uploader(
        "Choose data file", type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file:
        # Determine file type and load accordingly
        file_extension = uploaded_file.name.split(".")[-1]
        
        try:
            if file_extension.lower() == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension.lower() == "xlsx":
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_extension.lower() == "xls":
                df = pd.read_excel(uploaded_file, engine='xlrd')
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return
                
            st.success(f"File {uploaded_file.name} uploaded successfully")

            # Save the file locally
            with open(os.path.join(data_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.expander("Data Preview", expanded=False):
                st.dataframe(df)
            
            # Select columns for processing
            st.subheader("Select columns for processing")
            
            # Get list of all columns + None option
            all_columns = df.columns.tolist()
            columns_with_none = ["None"] + all_columns
            
            # Select columns for ID, subject, and description
            col0, col1, col2 = st.columns(3)
            
            with col0:
                id_col = st.selectbox(
                    "Select request ID column:",
                    options=all_columns,
                    index=all_columns.index("id") if "id" in all_columns else 0
                )
            with col1:
                subject_col = st.selectbox(
                    "Select column for subject (or None):",
                    options=columns_with_none,
                    index=columns_with_none.index("None")
                )
            with col2:
                description_col = st.selectbox(
                    "Select column for description:",
                    options=all_columns,
                    index=all_columns.index("description") if "description" in all_columns else 0
                )
            
            # Select target variable (class) column
            target_col = st.selectbox(
                "Select target variable (class) column:",
                options=all_columns,
                index=all_columns.index("class") if "class" in all_columns else 0
            )
            df = df.dropna(subset=[target_col])
            
            # Compute statistics for the target
            with st.expander("Target Variable Statistics", expanded=False):
                target_counts = df[target_col].value_counts()
                
                st.write("Distribution of target variable values:")
                
                target_stats_df = pd.DataFrame({
                    'Value': target_counts.index,
                    'Count': target_counts.values,
                    'Percentage': (target_counts.values / target_counts.sum() * 100).round(2)
                })
                
                st.dataframe(target_stats_df)
                
                # Visualize target distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                top_n = min(20, len(target_counts))
                target_counts.head(top_n).plot(kind='bar', ax=ax)
                
                plt.title(f'Top-{top_n} target variable values')
                plt.xlabel('Value')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Option to handle rare values
            st.subheader("Handling Rare Values")
            
            filter_method = st.radio(
                "Select class filtering method:",
                ["By frequency", "By classification quality"]
            )
            # Create processed DataFrame with renamed columns
            df_processed = df[[id_col, target_col, description_col]].copy()
            df_processed = df_processed.rename(columns={
                id_col: "id",
                description_col: "description",
                target_col: "class"
            })
            # Handle subject column (could be None)
            if subject_col != "None":
                df_processed["subject"] = df[subject_col]
            else:
                df_processed["subject"] = None  # Set subject to None if not selected

            if filter_method == "By frequency":
                top_n_values = st.slider(
                    "Number of top frequent values to keep",
                    min_value=1,
                    max_value=min(50, len(target_counts)),
                    value=10
                )
                
                # Get top-N most frequent values
                top_values = target_counts.head(top_n_values).index.tolist()
                
                # Replace all other values with "Other"
                df_processed["class"] = df_processed["class"].apply(
                    lambda x: x if x in top_values else "Other Related/Others"
                )
                
            else:  # By classification quality
                min_samples = st.slider(
                    "Minimum number of samples to keep a class",
                    min_value=1,
                    max_value=50,
                    value=10
                )
                min_f1_score = st.slider(
                    "Minimum F1-score to keep a class",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05
                )
                
                if st.checkbox("Check if parameters selected", value=False):
                    df_high_quality, report_dict = filter_high_quality_classes(
                        df_processed,
                        min_samples=min_samples,
                        min_f1_score=min_f1_score,
                        api_url=api_url,
                        username=username,
                        password=password
                    )
                    
                    if df_high_quality is not None:
                        df_processed = df_high_quality
            
            if filter_method:
                # Show statistics after handling rare values
                with st.expander("Statistics After Handling Rare Values", expanded=False):
                    new_counts = df_processed["class"].value_counts()
                    
                    new_stats_df = pd.DataFrame({
                        'Value': new_counts.index,
                        'Count': new_counts.values,
                        'Percentage': (new_counts.values / new_counts.sum() * 100).round(2)
                    })
                    
                    st.dataframe(new_stats_df)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    new_counts.plot(kind='bar', ax=ax2)
                    plt.title('Distribution of values after processing')
                    plt.xlabel('Value')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig2)
                
                with st.expander("Processed Data", expanded=False):
                    st.dataframe(df_processed)
            
            # Option to clear index before uploading
            clear_index_flag = st.checkbox("Clear existing index before uploading", value=True)
            
            if st.button("Upload data and calculate metrics"):
                token = get_token(api_url, username, password)
                if token:
                    # Split into train and test sets
                    train_df, test_df = train_test_split(
                        df_processed, test_size=0.1, random_state=42, stratify=df_processed["class"]
                    )
                    
                    with st.spinner("Uploading data to the system..."):
                        if clear_index_flag:
                            st.info("Clearing existing index...")
                            clear_index(token, api_url, collection)

                        upload_data(train_df, token, api_url, collection)
                    
                    st.success("Data uploaded successfully")
                    
                    with st.spinner("Getting predictions for the test set..."):
                        preds = predict(test_df, token, api_url, collection)
                    
                    # Calculate metrics
                    y_true = test_df["class"].tolist()
                    y_pred = preds
                    
                    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
                    y_true_valid = [y_true[i] for i in valid_indices]
                    y_pred_valid = [y_pred[i] for i in valid_indices]
                    
                    acc = accuracy_score(y_true_valid, y_pred_valid)
                    
                    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True)
                    report_text = classification_report(y_true_valid, y_pred_valid)
                    
                    cm = confusion_matrix(y_true_valid, y_pred_valid)
                    
                    unique_classes = sorted(list(set(y_true_valid + y_pred_valid)))
                    
                    st.subheader("Evaluation Results")
                    st.write(f"Accuracy: {acc:.4f}")
                    
                    with st.expander("Classification Report", expanded=True):
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        columns_to_keep = ['precision', 'recall', 'f1-score', 'support']
                        report_df = report_df[[col for col in columns_to_keep if col in report_df.columns]]
                        
                        for col in ['precision', 'recall', 'f1-score']:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].round(2)
                        
                        if 'support' in report_df.columns:
                            report_df['support'] = report_df['support'].astype(int)
                        
                        st.dataframe(report_df)
                    
                    with st.expander("Visualization of Class Metrics", expanded=False):
                        classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                        
                        metrics_data = {
                            'Class': [],
                            'Metric': [],
                            'Value': []
                        }
                        
                        for cls in classes:
                            for metric in ['precision', 'recall', 'f1-score']:
                                metrics_data['Class'].append(cls)
                                metrics_data['Metric'].append(metric)
                                metrics_data['Value'].append(report_dict[cls][metric])
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
                        
                        palette = {
                            'precision': 'blue',
                            'recall': 'green',
                            'f1-score': 'red'
                        }
                        
                        sns.barplot(
                            x='Class',
                            y='Value',
                            hue='Metric',
                            data=metrics_df,
                            palette=palette,
                            ax=ax_metrics
                        )
                        
                        plt.title('Classification metrics by class')
                        plt.ylabel('Value')
                        plt.xlabel('Class')
                        plt.xticks(rotation=45, ha='right')
                        plt.legend(title='Metric')
                        plt.tight_layout()
                        
                        st.pyplot(fig_metrics)
                    
                    with st.expander("Confusion Matrix", expanded=False):
                        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=unique_classes,
                            yticklabels=unique_classes
                        )
                        ax_cm.set_xlabel("Predicted Class")
                        ax_cm.set_ylabel("True Class")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig_cm)
                    
                    with st.expander("Upload Test Set", expanded=False):
                        upload_test_data = st.checkbox("Upload test set to the database", value=True)
                        
                        if upload_test_data:
                            with st.spinner("Uploading test set to the system..."):
                                test_upload_result = upload_data(
                                    test_df,
                                    token,
                                    api_url,
                                    collection,
                                )
                                if test_upload_result:
                                    st.success(f"Test set uploaded successfully ({len(test_df)} records)")
                                else:
                                    st.error("Error uploading test set")
                    
                    # Save metrics
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metrics_filename = f"metrics_{timestamp}.json"
                    metrics_data = {
                        "filename": uploaded_file.name,
                        "timestamp": timestamp,
                        "accuracy": float(acc),
                        "classification_report_text": report_text,
                        "classification_report_dict": report_dict,
                        "confusion_matrix": cm.tolist(),
                        "classes": unique_classes
                    }
                    
                    with open(os.path.join(metrics_dir, metrics_filename), "w", encoding="utf-8") as f:
                        json.dump(metrics_data, f, ensure_ascii=False, indent=4)
                    
                    # Allow download of processed data
                    csv_data = df_processed.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download processed data",
                        csv_data,
                        f"processed_{uploaded_file.name.split('.')[0]}.csv",
                        "text/csv",
                    )
                    
                    # Save figures
                    fig_cm_path = os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png")
                    fig_cm.savefig(fig_cm_path)
                    
                    fig_metrics_path = os.path.join(metrics_dir, f"classification_metrics_{timestamp}.png")
                    fig_metrics.savefig(fig_metrics_path)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Display latest metrics
    with st.expander("Latest Metrics", expanded=True):
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')] if os.path.exists(metrics_dir) else []
        
        if metrics_files:
            metrics_files.sort(reverse=True)
            
            with open(os.path.join(metrics_dir, metrics_files[0]), "r", encoding="utf-8") as f:
                last_metrics = json.load(f)
            
            st.write(f"File: {last_metrics['filename']}")
            st.write(f"Upload date: {last_metrics['timestamp']}")
            st.write(f"Accuracy: {last_metrics['accuracy']:.4f}")
            
            if 'classification_report_dict' in last_metrics:
                st.subheader("Classification Report")
                
                report_df = pd.DataFrame(last_metrics['classification_report_dict']).transpose()
                
                columns_to_keep = ['precision', 'recall', 'f1-score', 'support']
                report_df = report_df[[col for col in columns_to_keep if col in report_df.columns]]
                
                for col in ['precision', 'recall', 'f1-score']:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].round(2)
                
                if 'support' in report_df.columns:
                    report_df['support'] = report_df['support'].astype(int)
                
                st.dataframe(report_df)
            else:
                st.text(last_metrics['classification_report_text'])
            
            cm_file = os.path.join(metrics_dir, f"confusion_matrix_{last_metrics['timestamp']}.png")
            if os.path.exists(cm_file):
                st.image(cm_file, caption="Confusion Matrix")
                
            metrics_file = os.path.join(metrics_dir, f"classification_metrics_{last_metrics['timestamp']}.png")
            if os.path.exists(metrics_file):
                st.image(metrics_file, caption="Classification metrics by class")
        else:
            st.info("No saved metrics. Please upload a file and calculate metrics.")
