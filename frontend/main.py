import streamlit as st

from config import API_URL, USERNAME, PASSWORD
from tab_classification import render_classification_tab
from tab_similar_docs import render_similar_docs_tab
from tab_data_upload import render_data_upload_tab
from tab_test_env import render_test_env_tab

# Page configuration
st.set_page_config(
    page_title="Intent Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global variables for API settings
api_url = API_URL
username = USERNAME
password = PASSWORD

# Sidebar for API settings
with st.sidebar:
    st.title("API Settings")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Save Settings"):
        st.success("Settings saved")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Classification",
    "Similar Documents",
    "Data Upload",
    "Test Stand",
])

# Tab 1: Classification
with tab1:
    render_classification_tab(api_url, username, password)

# Tab 2: Similar Documents
with tab2:
    render_similar_docs_tab(api_url, username, password)

# Tab 3: Data Upload and Evaluation
with tab3:
    render_data_upload_tab(api_url, username, password)

# Tab 4: Test Stand
with tab4:
    render_test_env_tab(api_url, username, password)
