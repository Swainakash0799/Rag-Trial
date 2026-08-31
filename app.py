import os
import tempfile

import streamlit as st

from pipeline import (
    run_ingestion,
    run_query,
    run_streaming_query
)

from ingestion import (
    list_documents,
    delete_document
)

import config


# =================================
# Page Configuration
# =================================

st.set_page_config(
    page_title="IntelliDocs-AI",
    page_icon="📚",
    layout="wide"
)


# =================================
# CSS
# =================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0b0d12;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #999;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #333;
    background-color: #11141b;
    margin-bottom: 20px;
}

.source {
    background-color: #172033;
    padding: 8px;
    border-radius: 6px;
    margin: 5px 0;
}

</style>
""",
    unsafe_allow_html=True
)


# =================================
# Session State
# =================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =================================
# Header
# =================================

st.markdown(
    """
<div class="main-title">
    IntelliDocs-AI
</div>

<div class="subtitle">
    Advanced RAG-Based Knowledge Assistant
</div>
""",
    unsafe_allow_html=True
)


# =================================
# Tabs
# =================================

knowledge_tab, chat_tab = st.tabs(
    [
        "📁 Knowledge Base",
        "💬 Ask Documents"
    ]
)


# =================================
# Knowledge Base
# =================================

with knowledge_tab:

    st.header(
        "Upload Documents"
    )


    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=[
            "pdf",
            "docx",
            "csv",
            "xlsx"
        ],
        accept_multiple_files=True
    )


    if st.button(
        "Add to Knowledge Base",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload a document first."
            )

        else:

            saved_files = []


            temp_dir = tempfile.mkdtemp()


            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    temp_dir,
                    uploaded_file.name
                )


                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )


                saved_files.append(
                    file_path
                )


            with st.spinner(
                "Processing documents..."
            ):

                try:

                    result = run_ingestion(
                        saved_files
                    )


                    if result["added"]:

                        st.success(
                            "Added: "
                            + ", ".join(
                                result["added"]
                            )
                        )


                    if result[
                        "skipped_duplicates"
                    ]:

                        st.info(
                            "Skipped duplicates: "
                            + ", ".join(
                                result[
                                    "skipped_duplicates"
                                ]
                            )
                        )


                except Exception as error:

                    st.error(
                        f"Error: {error}"
                    )


    # ---------------------------------
    # Document List
    # ---------------------------------

    st.header(
        "Knowledge Base"
    )


    documents = list_documents()


    if not documents:

        st.info(
            "No documents uploaded yet."
        )


    else:

        for document in documents:

            col1, col2 = st.columns(
                [5, 1]
            )


            with col1:

                st.markdown(
                    f"""
                    <div class="card">

                    📄 <b>
                    {document["filename"]}
                    </b>

                    <br>

                    Type:
                    {document["file_type"]}

                    <br>

                    Chunks:
                    {document["chunk_count"]}

                    <br>

                    Uploaded:
                    {document["upload_date"][:10]}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                if st.button(
                    "Delete",
                    key=document[
                        "document_id"
                    ]
                ):

                    delete_document(
                        document[
                            "document_id"
                        ]
                    )

                    st.rerun()


# =================================
# Chat
# =================================

with chat_tab:

    documents = list_documents()


    if not documents:

        st.warning(
            "Upload documents before asking questions."
        )


    # ---------------------------------
    # Previous Messages
    # ---------------------------------

    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message("user"):

            st.write(
                message["question"]
            )


        with st.chat_message("assistant"):

            st.write(
                message["answer"]
            )


    # ---------------------------------
    # Question
    # ---------------------------------

    question = st.chat_input(
        "Ask something about your documents..."
    )


    if question:

        with st.chat_message("user"):

            st.write(question)


        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching documents..."
                ):

                    result = run_query(
                        question,
                        chat_history=(
                            st.session_state
                            .chat_history
                        )
                    )


                st.write(
                    result["answer"]
                )


                # ---------------------------------
                # Sources
                # ---------------------------------

                if result["citations"]:

                    st.subheader(
                        "📚 Sources"
                    )


                    shown_sources = set()


                    for citation in result[
                        "citations"
                    ]:

                        source = (
                            citation["filename"],
                            citation["page"]
                        )


                        if source in shown_sources:

                            continue


                        shown_sources.add(
                            source
                        )


                        st.markdown(
                            f"""
                            <div class="source">

                            📄
                            {citation["filename"]}
                            · Page
                            {citation["page"]}

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # ---------------------------------
                # Statistics
                # ---------------------------------

                st.subheader(
                    "📊 Retrieval Information"
                )


                col1, col2, col3, col4 = (
                    st.columns(4)
                )


                col1.metric(
                    "Chunks",
                    result[
                        "chunks_used"
                    ]
                )


                col2.metric(
                    "Confidence",
                    str(
                        result[
                            "confidence_percent"
                        ]
                    ) + "%"
                )


                col3.metric(
                    "Retrieval",
                    str(
                        result[
                            "latencies"
                        ][
                            "retrieval_seconds"
                        ]
                    ) + "s"
                )


                col4.metric(
                    "Generation",
                    str(
                        result[
                            "latencies"
                        ][
                            "generation_seconds"
                        ]
                    ) + "s"
                )


                # ---------------------------------
                # Advanced Details
                # ---------------------------------

                with st.expander(
                    "🔍 Advanced Retrieval Details"
                ):

                    st.write(
                        "Rewritten Query:"
                    )

                    st.code(
                        result[
                            "search_query"
                        ]
                    )


                    st.write(
                        "Retrieval Scores:"
                    )

                    st.json(
                        result["debug"]
                    )


                # ---------------------------------
                # Save Chat
                # ---------------------------------

                st.session_state.chat_history.append(
                    {
                        "question":
                            question,

                        "answer":
                            result["answer"]
                    }
                )


            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )


# =================================
# Footer
# =================================

st.markdown(
    """
<div style="text-align:center; color:#666; margin-top:40px;">
IntelliDocs-AI · Hybrid RAG · BM25 · Vector Search · Reranking
</div>
""",
    unsafe_allow_html=True
)