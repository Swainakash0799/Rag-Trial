import time
import math

from tools import (
    get_vectorstore,
    get_bm25,
    hybrid_search,
    rerank_documents
)

from agents import (
    rewrite_query,
    generate_answer,
    stream_answer
)

from ingestion import ingest_files

import config


# --------------------------------
# Format Chat History
# --------------------------------

def format_chat_history(chat_history):

    if not chat_history:

        return "No previous conversation."


    history = ""


    # Last 3 conversations
    recent = chat_history[-3:]


    for item in recent:

        history += (
            f"User: {item['question']}\n"
        )

        history += (
            f"Assistant: {item['answer']}\n\n"
        )


    return history


# --------------------------------
# Source Information
# --------------------------------

def get_source(document):

    filename = document.metadata.get(
        "filename",
        "Unknown"
    )

    page = document.metadata.get(
        "page",
        "-"
    )


    return (
        f"[source: {filename}, page {page}]"
    )


# --------------------------------
# Confidence
# --------------------------------

def calculate_confidence(results):

    if not results:

        return 0


    scores = []


    for document, info in results:

        if "rerank_score" in info:

            scores.append(
                info["rerank_score"]
            )


    if not scores:

        return 0


    average = (
        sum(scores) / len(scores)
    )


    # Convert score into 0-100
    value = 1 / (
        1 + math.exp(-average)
    )


    return round(
        value * 100
    )


# --------------------------------
# Retrieve Documents
# --------------------------------

def retrieve_documents(
    question,
    tenant_id,
    chat_history
):

    start_time = time.time()


    # -----------------------------
    # Query rewriting
    # -----------------------------

    history = format_chat_history(
        chat_history
    )


    search_query = rewrite_query(
        question,
        history
    )


    # -----------------------------
    # Chroma
    # -----------------------------

    vectorstore = get_vectorstore(
        tenant_id
    )


    # -----------------------------
    # Cached BM25
    # -----------------------------

    bm25 = get_bm25(
        tenant_id
    )


    # -----------------------------
    # Hybrid Retrieval
    # -----------------------------

    if bm25:

        results = hybrid_search(
            vectorstore,
            bm25,
            search_query
        )

    else:

        results = []


    retrieval_time = round(
        time.time() - start_time,
        3
    )


    # -----------------------------
    # Reranking
    # -----------------------------

    start_time = time.time()


    ranked_results = rerank_documents(
        search_query,
        results
    )


    rerank_time = round(
        time.time() - start_time,
        3
    )


    return {

        "search_query":
            search_query,

        "results":
            ranked_results,

        "retrieval_time":
            retrieval_time,

        "rerank_time":
            rerank_time
    }


# --------------------------------
# Create Context
# --------------------------------

def create_context(results):

    context = ""


    for document, scores in results:

        context += (
            get_source(document)
            + "\n"
        )

        context += (
            document.page_content
            + "\n\n"
        )


    return context


# --------------------------------
# Create Citations
# --------------------------------

def create_citations(results):

    citations = []


    for document, scores in results:

        citations.append(
            {
                "filename":
                    document.metadata.get(
                        "filename",
                        "Unknown"
                    ),

                "page":
                    document.metadata.get(
                        "page",
                        "-"
                    )
            }
        )


    return citations


# --------------------------------
# Run RAG
# --------------------------------

def run_query(
    question,
    tenant_id="default",
    chat_history=None
):

    if chat_history is None:

        chat_history = []


    # -----------------------------
    # Retrieval
    # -----------------------------

    retrieval = retrieve_documents(
        question,
        tenant_id,
        chat_history
    )


    results = retrieval["results"]


    # -----------------------------
    # Context
    # -----------------------------

    context = create_context(
        results
    )


    # -----------------------------
    # Answer
    # -----------------------------

    history = format_chat_history(
        chat_history
    )


    start_time = time.time()


    answer = generate_answer(
        question,
        context,
        history
    )


    generation_time = round(
        time.time() - start_time,
        3
    )


    # -----------------------------
    # Final Result
    # -----------------------------

    return {

        "answer":
            answer,

        "search_query":
            retrieval["search_query"],

        "citations":
            create_citations(
                results
            ),

        "debug":
            [
                {
                    "source":
                        get_source(
                            document
                        ),

                    **scores
                }

                for document, scores
                in results
            ],

        "chunks_retrieved":
            len(
                results
            ),

        "chunks_used":
            len(
                results
            ),

        "confidence_percent":
            calculate_confidence(
                results
            ),

        "latencies":
            {
                "retrieval_seconds":
                    retrieval[
                        "retrieval_time"
                    ],

                "rerank_seconds":
                    retrieval[
                        "rerank_time"
                    ],

                "generation_seconds":
                    generation_time
            }
    }


# --------------------------------
# Streaming Query
# --------------------------------

def run_streaming_query(
    question,
    tenant_id="default",
    chat_history=None
):

    if chat_history is None:

        chat_history = []


    # Retrieval
    retrieval = retrieve_documents(
        question,
        tenant_id,
        chat_history
    )


    results = retrieval["results"]


    # Context
    context = create_context(
        results
    )


    history = format_chat_history(
        chat_history
    )


    # Return everything needed by UI
    return {

        "tokens":
            stream_answer(
                question,
                context,
                history
            ),

        "search_query":
            retrieval[
                "search_query"
            ],

        "citations":
            create_citations(
                results
            ),

        "debug":
            [
                {
                    "source":
                        get_source(
                            document
                        ),

                    **scores
                }

                for document, scores
                in results
            ],

        "chunks_retrieved":
            len(results),

        "chunks_used":
            len(results),

        "confidence_percent":
            calculate_confidence(
                results
            ),

        "retrieval_seconds":
            retrieval[
                "retrieval_time"
            ],

        "rerank_seconds":
            retrieval[
                "rerank_time"
            ]
    }


# --------------------------------
# Ingestion Wrapper
# --------------------------------

def run_ingestion(
    file_paths,
    tenant_id="default"
):

    return ingest_files(
        file_paths,
        tenant_id
    )