import os
import shutil
import streamlit as st
from api_utils import (
    get_token,
    copy_collection,
    render_data_upload_tab,
    get_latest_metrics,
)
from config import TEST_COLLECTION


def _copy_latest_metrics(collection: str) -> None:
    """Copy latest metrics from test collection to production."""
    src_dir = os.path.join("metrics", collection)
    dst_dir = os.path.join("metrics", "prod")
    if not os.path.isdir(src_dir):
        return
    files = [f for f in os.listdir(src_dir) if f.startswith("metrics_") and f.endswith(".json") or (f.startswith('confusion_matrix_') or f.startswith('classification_metrics_')) and f.endswith('.png')]
    if not files:
        return
    files.sort(reverse=True)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(os.path.join(src_dir, files[0]), os.path.join(dst_dir, files[0]))


def _render_metrics_comparison(collection: str) -> None:
    """Show accuracy on prod and test collections."""
    prod_metrics = get_latest_metrics()
    test_metrics = get_latest_metrics(collection)

    st.subheader("Model Quality Comparison")
    col1, col2 = st.columns(2)
    with col1:
        if prod_metrics:
            st.metric("Production accuracy", f"{prod_metrics['accuracy']:.4f}")
        else:
            st.info("No production metrics")
    with col2:
        if test_metrics:
            delta = None
            if prod_metrics:
                delta = f"{test_metrics['accuracy'] - prod_metrics['accuracy']:+.4f}"
            st.metric(f"{collection} accuracy", f"{test_metrics['accuracy']:.4f}", delta=delta)
        else:
            st.info(f"No metrics for {collection}")

def render_test_env_tab(api_url, username, password):
    render_data_upload_tab(
        api_url, username, password, collection=TEST_COLLECTION
    )
    _render_metrics_comparison(TEST_COLLECTION)

    if st.button("Promote test to production"):
        token = get_token(api_url, username, password)
        if token:
            copy_collection(token, api_url, TEST_COLLECTION)
            _copy_latest_metrics(TEST_COLLECTION)


