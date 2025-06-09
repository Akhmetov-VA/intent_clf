import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from collections import Counter

from api_utils import get_token, search_similar
from config import DEFAULT_EXAMPLES


def _results_dataframe(results):
    return pd.DataFrame(
        [
            {
                "Request ID": r["request_id"],
                "Subject": r["subject"],
                "Description": r["description"],
                "Class": r["class_name"],
                "Score": f"{r['score']:.4f}",
            }
            for r in results
        ]
    )


def _show_search_results(title: str, results: list) -> None:
    if not results:
        st.info("No results")
        return
    with st.expander(title, expanded=True):
        top3 = results[: min(3, len(results))]
        st.subheader("Top 3 Most Similar Documents")
        st.dataframe(_results_dataframe(top3), use_container_width=True)

        st.subheader("All Retrieved Documents")
        st.dataframe(_results_dataframe(results), use_container_width=True)

        class_counts = Counter([r["class_name"] for r in results])
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(classes, counts, color="skyblue")
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height}",
                    ha="center",
                    va="bottom",
                )
            plt.title("Document Distribution by Class")
            plt.xlabel("Class")
            plt.ylabel("Number of Documents")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

def render_similar_docs_tab(api_url, username, password):
    """Display the Similar Documents Search tab"""
    st.title("Similar Documents Search")

    # Choose a sample request for search
    use_default_search = st.checkbox("Use a default request for search")

    if use_default_search:
        example_index_search = st.selectbox(
            "Select a sample request for search:",
            options=range(len(DEFAULT_EXAMPLES)),
            format_func=lambda i: f"{DEFAULT_EXAMPLES[i]['id']} - {DEFAULT_EXAMPLES[i]['subject']}",
            key="search_example",
        )

        selected_example_search = DEFAULT_EXAMPLES[example_index_search]
        default_search_subject = selected_example_search["subject"]
        default_search_description = selected_example_search["description"]
    else:
        default_search_subject = ""
        default_search_description = ""

    # Split input into subject and description
    st.subheader("Search Data")
    search_subject = st.text_input("Search subject:", value=default_search_subject)
    search_description = st.text_area(
        "Search description:", value=default_search_description, height=150
    )

    limit = st.slider("Number of results", min_value=1, max_value=20, value=10)

    if st.button("Search"):
        if not search_subject and not search_description:
            st.warning("Please enter a subject or description for search")
        else:
            with st.spinner("Retrieving token..."):
                token = get_token(api_url, username, password)

            if token:
                with st.spinner("Searching for similar documents..."):
                    search_results = search_similar(
                        search_subject,
                        search_description,
                        token,
                        api_url,
                        limit,
                    )
                    test_collection = st.session_state.get("test_collection")
                    test_results = None
                    if test_collection:
                        test_results = search_similar(
                            search_subject,
                            search_description,
                            token,
                            api_url,
                            limit,
                            collection=test_collection,
                        )

                if search_results and "results" in search_results:
                    st.success(
                        f"Found {len(search_results['results'])} documents"
                    )
                    if test_results and "results" in test_results:
                        st.info(
                            f"Test collection {test_collection}: {len(test_results['results'])} results"
                        )

                    st.subheader("Your Query")
                    st.dataframe(
                        pd.DataFrame(
                            {"Subject": [search_subject], "Description": [search_description]}
                        ),
                        use_container_width=True,
                    )

                    _show_search_results("Production Results", search_results["results"])

                    if test_results and "results" in test_results:
                        _show_search_results(
                            f"Test Results ({test_collection})",
                            test_results["results"],
                        )

                else:
                    st.warning("No similar documents found")
