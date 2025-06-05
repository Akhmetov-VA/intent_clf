import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

from api_utils import get_token, search_similar
from config import DEFAULT_EXAMPLES

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
                    st.success(f"Found {len(search_results['results'])} documents")
                    if test_results and "results" in test_results:
                        st.info(
                            f"Test collection {test_collection}: {len(test_results['results'])} results"
                        )
                    
                    # Display user's query
                    st.subheader("Your Query")
                    query_data = {
                        "Subject": [search_subject],
                        "Description": [search_description]
                    }
                    st.dataframe(pd.DataFrame(query_data), use_container_width=True)
                    
                    # Display top-3 most similar documents
                    if len(search_results["results"]) > 0:
                        st.subheader("Top 3 Most Similar Documents (Prod)")
                        top3_results = search_results["results"][:min(3, len(search_results["results"]))]
                        
                        top3_data = []
                        for result in top3_results:
                            top3_data.append({
                                "Request ID": result['request_id'],
                                "Subject": result['subject'],
                                "Description": result['description'],
                                "Class": result['class_name'],
                                "Similarity Score": f"{result['score']:.4f}"
                            })
                        
                        st.dataframe(pd.DataFrame(top3_data), use_container_width=True)

                    if test_results and "results" in test_results and len(test_results["results"]) > 0:
                        st.subheader(f"Top 3 Most Similar Documents ({test_collection})")
                        top3_test = test_results["results"][:min(3, len(test_results["results"]))]
                        tdata = []
                        for r in top3_test:
                            tdata.append(
                                {
                                    "Request ID": r["request_id"],
                                    "Subject": r["subject"],
                                    "Description": r["description"],
                                    "Class": r["class_name"],
                                    "Similarity Score": f"{r['score']:.4f}",
                                }
                            )
                        st.dataframe(pd.DataFrame(tdata), use_container_width=True)
                    
                    # Display all results in a table
                    st.subheader("All Retrieved Documents (Prod)")
                    
                    # Prepare data for table
                    table_data = []
                    for i, result in enumerate(search_results["results"]):
                        table_data.append({
                            "Request ID": result['request_id'],
                            "Subject": result['subject'],
                            "Description": result['description'],
                            "Class": result['class_name'],
                            "Score": f"{result['score']:.4f}"
                        })
                    
                    results_df = pd.DataFrame(table_data)
                    st.dataframe(results_df, use_container_width=True)

                    if test_results and "results" in test_results:
                        st.subheader(f"All Retrieved Documents ({test_collection})")
                        ttable = [
                            {
                                "Request ID": r["request_id"],
                                "Subject": r["subject"],
                                "Description": r["description"],
                                "Class": r["class_name"],
                                "Score": f"{r['score']:.4f}",
                            }
                            for r in test_results["results"]
                        ]
                        tdf = pd.DataFrame(ttable)
                        st.dataframe(tdf, use_container_width=True)

                    # Visualize class distribution
                    if search_results["results"]:
                        class_counts = Counter([result["class_name"] for result in search_results["results"]])
                        
                        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                        classes = [cls for cls, _ in sorted_classes]
                        counts = [count for _, count in sorted_classes]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(classes, counts, color='skyblue')
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{height}', ha='center', va='bottom')
                        
                        plt.title('Document Distribution by Class')
                        plt.xlabel('Class')
                        plt.ylabel('Number of Documents')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Display percentage distribution as pie chart
                        if len(class_counts) > 1:
                            fig2, ax2 = plt.subplots(figsize=(8, 8))
                            wedges, texts, autotexts = ax2.pie(
                                counts, 
                                labels=classes, 
                                autopct='%1.1f%%',
                                textprops={'fontsize': 9}
                            )
                            plt.title('Class Percentage Distribution')
                            plt.tight_layout()
                            
                            st.pyplot(fig2)

                else:
                    st.warning("No similar documents found")
