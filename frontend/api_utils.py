import uuid
import requests
import streamlit as st
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_token(api_url, username, password):
    """Function to obtain an authorization token"""
    try:
        response = requests.post(
            f"{api_url}/token",
            data={
                "username": username,
                "password": password,
                "scope": "predict upload search",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            st.error(f"Authentication error: {response.text}")
            return None

        return response.json()["access_token"]
    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to API at {api_url}")
        return None

def classify_request(subject, description, token, api_url, collection=None):
    """Function to classify a request"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Generate an ID for this request
    item_id = str(uuid.uuid4())

    payload = {
        "id": item_id,  # Required field!
        "subject": subject if subject else "no_subject",
        "description": description if description else "no_description",
    }
    if collection:
        payload["collection"] = collection

    try:
        response = requests.post(f"{api_url}/predict", json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return None

def search_similar(subject, description, token, api_url, limit=10, collection=None):
    """Search for similar documents based on subject and description"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "id": str(uuid.uuid4()),
        "subject": subject if subject else "no_subject",
        "description": description if description else "no_description",
        "limit": limit,
    }
    if collection:
        payload["collection"] = collection

    try:
        response = requests.post(f"{api_url}/search", json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"Server response: {e.response.text}")
        return None

def clear_index(token, api_url, collection=None):
    """Clear index before uploading new data"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {"collection": collection} if collection else None

    try:
        logger.info("Sending request to clear index")
        response = requests.post(
            f"{api_url}/clear_index",
            headers=headers,
            json=payload,
        )
        
        if response.status_code != 200:
            logger.error(f"Error clearing index: {response.text}")
            st.error(f"Error clearing index: {response.text}")
            return False
        
        result = response.json()
        if result.get("success"):
            logger.info("Index cleared successfully")
            st.success("Index cleared successfully")
            return True
        else:
            logger.warning(f"API returned unexpected response when clearing index: {result}")
            st.warning(f"API returned unexpected response when clearing index: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        st.error(f"Error clearing index: {str(e)}")
        return False

def create_collection(token, api_url, collection):
    """Create new collection on the server."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"collection": collection}
    try:
        response = requests.post(f"{api_url}/create_collection", json=payload, headers=headers)
        response.raise_for_status()
        st.success("Collection created")
        return True
    except Exception as e:
        st.error(f"Error creating collection: {str(e)}")
        return False

def copy_collection(token, api_url, source, dest=None):
    """Copy one collection to another on the server."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"source": source}
    if dest:
        payload["dest"] = dest
    try:
        response = requests.post(f"{api_url}/copy_collection", json=payload, headers=headers)
        response.raise_for_status()
        st.success("Collection copied")
        return True
    except Exception as e:
        st.error(f"Error copying collection: {str(e)}")
        return False
def upload_data(data, token, api_url, collection=None):
    """Upload data to the system"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Split data into batches of 100 records
    batch_size = 100
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)

    uploaded_ids = []
    progress_bar = st.progress(0)

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data.iloc[start_idx:end_idx]

        # Prepare items for upload
        items = []
        for _, row in batch_data.iterrows():
            # Use original ID from data if present
            item_id = row.get("id", str(uuid.uuid4()))

            item = {
                "id": item_id,
                "subject": row["subject"] if not pd.isna(row["subject"]) else "no_subject",
                "description": row["description"] if not pd.isna(row["description"]) else "no_description",
                "class_name": row["class"],
            }

            # Add task if present in data
            if "task" in row and not pd.isna(row["task"]):
                item["task"] = row["task"]

            items.append(item)

        payload = {"items": items}
        if collection:
            payload["collection"] = collection

        try:
            # Send upload request
            response = requests.post(
                f"{api_url}/upload",
                json=payload,
                headers=headers,
            )

            if response.status_code != 200:
                st.error(f"Error uploading batch {i + 1}/{total_batches}: {response.text}")
                continue
                
            result = response.json()
            if result.get("success") and "ids" in result:
                uploaded_ids.extend(result["ids"])
            else:
                st.warning(f"Warning: API returned unexpected response for batch {i + 1}/{total_batches}: {result}")

        except Exception as e:
            st.error(f"Error uploading batch {i + 1}/{total_batches}: {str(e)}")

        # Update progress
        progress_value = min(1.0, (i + 1) / total_batches)
        progress_bar.progress(progress_value)

    st.success(f"Uploaded {len(uploaded_ids)} records")
    return uploaded_ids

def predict(data, token, api_url, collection=None):
    """Get predictions for test data"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    predictions = []
    progress_bar = st.progress(0)
    total_rows = len(data)

    for i, (index, row) in enumerate(data.iterrows()):
        try:
            payload = {
                "id": str(uuid.uuid4()),  # Added ID because it's required
                "subject": row["subject"] if not pd.isna(row["subject"]) else "no_subject",
                "description": row["description"] if not pd.isna(row["description"]) else "no_description",
            }
            if collection:
                payload["collection"] = collection

            logger.info(f"Sending request for record {index}")
            response = requests.post(
                f"{api_url}/predict",
                json=payload,
                headers=headers,
            )

            response.raise_for_status()

            result = response.json()
            logger.debug(f"Received response: {result}")

            if not isinstance(result, dict) or "predictions" not in result:
                raise ValueError("Unexpected API response format")

            predictions_list = result["predictions"]
            if not isinstance(predictions_list, list) or not predictions_list:
                raise ValueError("Predictions list is empty or has invalid format")
                
            top_prediction = predictions_list[0]["class_name"]
            predictions.append(top_prediction)
            logger.info(f"Prediction for record {index}: {top_prediction}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending request for record {index}: {str(e)}")
            st.error(f"Error sending request for record {index}: {str(e)}")
            predictions.append(None)
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error processing API response for record {index}: {str(e)}")
            st.error(f"Error processing API response for record {index}: {str(e)}")
            predictions.append(None)
        except Exception as e:
            logger.error(f"Unexpected error for record {index}: {str(e)}")
            st.error(f"Unexpected error for record {index}: {str(e)}")
            predictions.append(None)

        # Update progress
        progress_value = min(1.0, (i + 1) / total_rows)
        progress_bar.progress(progress_value)

    return predictions

def calculate_metrics(y_true, y_pred):
    """Compute classification metrics"""
    # Remove records where prediction was not obtained
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if not y_true_valid:
        logger.error("Error: No valid predictions for evaluation")
        return None, None, None, None

    # Compute metrics
    acc = accuracy_score(y_true_valid, y_pred_valid)
    
    # Get classification report as dictionary
    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True)
    
    # Also save text version for backward compatibility
    report_text = classification_report(y_true_valid, y_pred_valid)
    
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    
    # Get unique classes for confusion matrix display
    classes = sorted(list(set(y_true_valid + y_pred_valid)))
    
    return acc, report_dict, report_text, cm, classes

def get_classification_report_df(report_dict):
    """Convert classification_report dict to DataFrame"""
    # Convert dict to DataFrame and transpose for better display
    df = pd.DataFrame(report_dict).transpose()
    
    # Remove unnecessary columns if present
    columns_to_keep = ['precision', 'recall', 'f1-score', 'support']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    # Round numeric values for better readability
    for col in ['precision', 'recall', 'f1-score']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # Convert support to integer
    if 'support' in df.columns:
        df['support'] = df['support'].astype(int)
    
    return df

def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # If class names not provided, use unique values (limit to improve readability)
    if class_names is None and cm.shape[0] <= 20:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )
    
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def plot_classification_metrics(report_dict):
    """Create visualization of classification metrics"""
    # Extract data for visualization
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Create DataFrame for visualization
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
    
    df = pd.DataFrame(metrics_data)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use different colors for different metrics
    palette = {
        'precision': 'blue',
        'recall': 'green',
        'f1-score': 'red'
    }
    
    # Create grouped bar chart
    sns.barplot(
        x='Class',
        y='Value',
        hue='Metric',
        data=df,
        palette=palette,
        ax=ax
    )
    
    plt.title('Classification Metrics by Class')
    plt.ylabel('Value')
    plt.xlabel('Class')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    
    return fig

def save_metrics(metrics_dir, filename, acc, report_dict, report_text, cm, class_names=None):
    """Save metrics to file"""
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"metrics_{timestamp}.json"
    
    metrics_data = {
        "filename": filename,
        "timestamp": timestamp,
        "accuracy": float(acc),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "classes": class_names if class_names else []
    }
    
    with open(os.path.join(metrics_dir, metrics_filename), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=4)
    
    # Save visualizations
    fig_cm = plot_confusion_matrix(cm, class_names)
    fig_cm.savefig(os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png"))
    
    fig_metrics = plot_classification_metrics(report_dict)
    fig_metrics.savefig(os.path.join(metrics_dir, f"classification_metrics_{timestamp}.png"))

    return metrics_filename


def get_latest_metrics(collection=None):
    """Return metrics from the latest run for the given collection."""
    metrics_dir = os.path.join("metrics", collection or "prod")
    if not os.path.isdir(metrics_dir):
        return None

    metrics_files = [
        f for f in os.listdir(metrics_dir)
        if f.startswith("metrics_") and f.endswith(".json")
    ]
    if not metrics_files:
        return None

    metrics_files.sort(reverse=True)
    latest_path = os.path.join(metrics_dir, metrics_files[0])
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def filter_high_quality_classes(df, min_samples=10, min_f1_score=0.5, api_url=None, username=None, password=None):
    """
    Function to filter classes with high classification quality.

    Parameters:
    df (DataFrame): DataFrame with data to process
    min_samples (int): Minimum number of samples to keep a class
    min_f1_score (float): Minimum F1-score to keep a class
    api_url, username, password: API connection parameters

    Returns:
    DataFrame with filtered data and a dictionary with class metrics
    """
    st.subheader("Analysis of classification quality by class")
    st.write(f"""
    The idea is to keep only classes on which we already perform well.
    First, move very rare observations (fewer than {min_samples} samples) to a separate category.
    Then train a model on this data and keep only classes with quality above {min_f1_score}.
    Finally, we upload to the database only those classes with good performance.
    """)
    
    # Copy original DataFrame
    df_processed = df.copy()
    
    # Count samples per class
    class_counts = df_processed["class"].value_counts()
    
    # Identify rare classes (fewer than min_samples)
    rare_classes = class_counts[class_counts < min_samples].index.tolist()
    
    # Replace rare classes with "Other"
    df_processed["class"] = df_processed["class"].apply(
        lambda x: "Other" if x in rare_classes else x
    )
    
    # Show statistics after replacing rare classes
    with st.expander("Statistics after filtering rare classes", expanded=True):
        new_counts = df_processed["class"].value_counts()
        
        stats_df = pd.DataFrame({
            'Value': new_counts.index,
            'Count': new_counts.values,
            'Percentage': (new_counts.values / new_counts.sum() * 100).round(2)
        })
        
        st.dataframe(stats_df)
        
        # Visualize new distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        new_counts.plot(kind='bar', ax=ax)
        plt.title('Distribution of values after filtering rare classes')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        df_processed, test_size=0.2, random_state=42, stratify=df_processed["class"]
    )
    
    # Get token for API access
    token = get_token(api_url, username, password)
    if not token:
        st.error("Failed to obtain API token")
        return None, None
    
    # Clear index before upload
    with st.spinner("Clearing existing index..."):
        clear_index(token, api_url)
    
    # Upload training set
    with st.spinner("Uploading training data to system..."):
        upload_data(train_df, token, api_url)
    
    # Get predictions for test set
    with st.spinner("Getting predictions for test set..."):
        preds = predict(test_df, token, api_url)
    
    # Compute metrics
    y_true = test_df["class"].tolist()
    y_pred = preds
    
    # Remove None from predictions
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    # Get classification report as dictionary
    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True)
    
    # Filter classes based on F1-score
    high_quality_classes = []
    
    for class_name, metrics in report_dict.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            if metrics['f1-score'] >= min_f1_score:
                high_quality_classes.append(class_name)
    
    # Show class metrics
    with st.expander("Metrics by class", expanded=True):
        metrics_df = pd.DataFrame([
            {
                'Class': class_name,
                'Precision': report_dict[class_name]['precision'],
                'Recall': report_dict[class_name]['recall'],
                'F1-score': report_dict[class_name]['f1-score'],
                'Support': report_dict[class_name]['support'],
                'Quality': 'High' if class_name in high_quality_classes else 'Low'
            }
            for class_name in report_dict.keys()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        ])
        
        # Sort by F1-score
        metrics_df = metrics_df.sort_values('F1-score', ascending=False)
        
        # Round numeric values
        for col in ['Precision', 'Recall', 'F1-score']:
            metrics_df[col] = metrics_df[col].round(3)
        
        st.dataframe(metrics_df)
        
        # Visualize F1-score by class
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create color palette based on quality
        colors = metrics_df['Quality'].map({'High': 'green', 'Low': 'red'})
        
        bars = ax.bar(metrics_df['Class'], metrics_df['F1-score'], color=colors)
        
        # Add horizontal threshold line
        ax.axhline(y=min_f1_score, color='red', linestyle='--', alpha=0.7)
        ax.text(0, min_f1_score + 0.01, f'F1-score threshold: {min_f1_score}', color='red')
        
        plt.title('F1-score by class')
        plt.xlabel('Class')
        plt.ylabel('F1-score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Filter data to keep only high-quality classes
    df_processed["class"] = df_processed["class"].apply(
        lambda x: x if x in high_quality_classes else "Other"
    )
    df_high_quality = df_processed
    
    # Show statistics after filtering by quality
    with st.expander("Statistics after filtering by quality", expanded=True):
        high_quality_counts = df_high_quality["class"].value_counts()
        
        stats_df = pd.DataFrame({
            'Value': high_quality_counts.index,
            'Count': high_quality_counts.values,
            'Percentage': (high_quality_counts.values / high_quality_counts.sum() * 100).round(2)
        })
        
        st.dataframe(stats_df)
        
        # Visualize distribution after filtering by quality
        fig, ax = plt.subplots(figsize=(10, 6))
        high_quality_counts.plot(kind='bar', ax=ax)
        plt.title('Distribution of values after filtering by quality')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Display filtering results
    st.info(f"Original number of classes: {len(class_counts)}")
    st.info(f"Number of classes after filtering rare: {len(new_counts)}")
    st.info(f"Number of high-quality classes (F1-score >= {min_f1_score}): {len(high_quality_classes)}")
    st.info(f"Number of records in filtered set: {len(df_high_quality)} out of {len(df_processed)} ({len(df_high_quality)/len(df_processed)*100:.1f}%)")
    
    # Return filtered DataFrame and metrics
    return df_high_quality, report_dict

# UI rendering function for data upload
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
        "Choose data file", type=["csv", "xlsx", "xls"], key=str(collection)
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
