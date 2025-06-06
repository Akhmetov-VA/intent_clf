import streamlit as st
from api_utils import get_token, create_collection, copy_collection
from tab_data_upload import render_data_upload_tab
from config import TEST_COLLECTION

def render_test_env_tab(api_url, username, password):

    render_data_upload_tab(
        api_url, username, password, collection=TEST_COLLECTION
    )

    if st.button("Promote test to production"):
        token = get_token(api_url, username, password)
        if token:
            copy_collection(token, api_url, TEST_COLLECTION)
