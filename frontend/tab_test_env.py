import os
import shutil
import streamlit as st
from api_utils import get_token, copy_collection
from tab_data_upload import render_data_upload_tab
from config import TEST_COLLECTION


def _copy_latest_metrics(collection: str) -> None:
    """Copy latest metrics from test collection to production."""
    src_dir = os.path.join("metrics", collection)
    dst_dir = os.path.join("metrics", "prod")
    if not os.path.isdir(src_dir):
        return
    files = [f for f in os.listdir(src_dir) if f.startswith("metrics_") and f.endswith(".json")]
    if not files:
        return
    files.sort(reverse=True)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(os.path.join(src_dir, files[0]), os.path.join(dst_dir, files[0]))

def render_test_env_tab(api_url, username, password):
    if "test_collection" not in st.session_state:
        st.session_state["test_collection"] = TEST_COLLECTION

    render_data_upload_tab(
        api_url, username, password, collection=TEST_COLLECTION
    )

    if st.button("Promote test to production"):
        token = get_token(api_url, username, password)
        if token:
            copy_collection(token, api_url, TEST_COLLECTION)
            _copy_latest_metrics(TEST_COLLECTION)
