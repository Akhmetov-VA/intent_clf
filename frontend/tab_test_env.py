import streamlit as st
from api_utils import get_token, create_collection, copy_collection
from tab_data_upload import render_data_upload_tab
from config import TEST_COLLECTION

def render_test_env_tab(api_url, username, password):
    st.title("Test Environment")

    # Get collection name from session or use default
    default_name = st.session_state.get("test_collection", TEST_COLLECTION)
    collection_name = st.text_input("Test collection name", value=default_name)

    if "test_collection" not in st.session_state:
        st.session_state["test_collection"] = default_name
        token = get_token(api_url, username, password)
        if token:
            create_collection(token, api_url, default_name)

    if st.button("Save name"):
        st.session_state["test_collection"] = collection_name
        token = get_token(api_url, username, password)
        if token:
            create_collection(token, api_url, collection_name)
        st.success(f"Test collection set to {collection_name}")

    st.info(
        f"Current test collection: {st.session_state.get('test_collection', collection_name)}"
    )

    st.divider()

    render_data_upload_tab(
        api_url, username, password, collection=st.session_state["test_collection"]
    )

    if st.button("Promote test to production"):
        token = get_token(api_url, username, password)
        if token:
            copy_collection(token, api_url, st.session_state["test_collection"])
