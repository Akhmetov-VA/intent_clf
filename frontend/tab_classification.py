import pandas as pd
import streamlit as st

from api_utils import get_token, classify_request
from config import DEFAULT_EXAMPLES


def _show_predictions(title: str, predictions: list) -> None:
    """Render predictions inside an expander."""
    if not predictions:
        st.warning("No predictions returned")
        return
    with st.expander(title, expanded=True):
        df = pd.DataFrame(predictions)
        st.dataframe(df, use_container_width=True)
        st.write(
            f"Predicted: {predictions[0]['class_name']} "
            f"({predictions[0]['probability']:.2f})"
        )

def render_classification_tab(api_url, username, password):
    """Display the classification tab"""
    st.title("Intent Classification")
    
    # Select a sample request
    st.subheader("Choose Request")

    use_default = st.checkbox("Use default request")

    if use_default:
        example_index = st.selectbox(
            "Select a sample request:",
            options=range(len(DEFAULT_EXAMPLES)),
            format_func=lambda i: f"{DEFAULT_EXAMPLES[i]['id']} - {DEFAULT_EXAMPLES[i]['subject']}",
        )

        selected_example = DEFAULT_EXAMPLES[example_index]

        # Show details of the selected sample
        st.info(f"""
        **ID:** {selected_example["id"]}
        **Subject:** {selected_example["subject"]}
        **Description:** {selected_example["description"]}
        **Class:** {selected_example["class"]}
        **Task:** {selected_example["task"]}
        """)

        # Pre-fill form fields
        default_subject = selected_example["subject"]
        default_description = selected_example["description"]
    else:
        default_subject = ""
        default_description = ""

    # Input form for classification data
    st.subheader("Data for Classification")
    subject = st.text_input("Subject:", value=default_subject)
    description = st.text_area(
        "Description:", value=default_description, height=200
    )

    if st.button("Classify"):
        if not subject and not description:
            st.warning("Please enter a subject or description")
        else:
            with st.spinner("Getting token..."):
                token = get_token(api_url, username, password)

            if token:
                with st.spinner("Classifying request..."):
                    prod_result = classify_request(subject, description, token, api_url)
                    test_collection = st.session_state.get("test_collection")
                    test_result = None
                    if test_collection:
                        test_result = classify_request(
                            subject, description, token, api_url, collection=test_collection
                        )

                if prod_result and "predictions" in prod_result:
                    st.success("Request successfully classified!")
                    if test_result and "predictions" in test_result:
                        col1, col2 = st.columns(2)
                        with col1:
                            _show_predictions(
                                "Production Results",
                                prod_result["predictions"],
                            )
                        with col2:
                            _show_predictions(
                                f"Test Results ({test_collection})",
                                test_result["predictions"],
                            )
                    else:
                        _show_predictions(
                            "Production Results",
                            prod_result["predictions"],
                        )
                else:
                    st.error("Failed to get classification results")
