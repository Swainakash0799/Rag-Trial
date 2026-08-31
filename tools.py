from functools import lru_cache

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import config


# --------------------------------
# PDF Loader
# --------------------------------

def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


# --------------------------------
# DOCX Loader
# --------------------------------

def load_docx(file_path):

    loader = Docx2txtLoader(file_path)

    documents = loader.load()

    return documents


# --------------------------------
# CSV Loader
# --------------------------------

def load_csv(file_path):

    loader = CSVLoader(file_path)

    documents = loader.load()

    return documents


# --------------------------------
# Excel Loader
# --------------------------------

def load_excel(file_path):

    loader = UnstructuredExcelLoader(file_path)

    documents = loader.load()

    return documents


# --------------------------------
# Split Documents
# --------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# --------------------------------
# Collection Name
# --------------------------------

def get_collection_name(tenant_id):

    return f"tenant_{tenant_id}"


# --------------------------------
# ChromaDB
# --------------------------------

def get_vectorstore(tenant_id="default"):

    collection_name = get_collection_name(
        tenant_id
    )

    vectorstore = Chroma(
        persist_directory=config.CHROMA_DIR,
        collection_name=collection_name,
        embedding_function=config.embedding_model
    )

    return vectorstore


# --------------------------------
# Add Documents
# --------------------------------

def add_documents(vectorstore, documents):

    if not documents:
        return

    vectorstore.add_documents(
        documents
    )


# --------------------------------
# Delete Document
# --------------------------------

def delete_document_chunks(
    vectorstore,
    document_id
):

    vectorstore._collection.delete(
        where={
            "document_id": document_id
        }
    )


# --------------------------------
# BM25
# --------------------------------

def create_bm25(documents):

    retriever = BM25Retriever.from_documents(
        documents
    )

    retriever.k = config.TOP_K

    return retriever


# --------------------------------
# Cached BM25
# --------------------------------

@lru_cache(maxsize=10)
def get_bm25(tenant_id):

    vectorstore = get_vectorstore(
        tenant_id
    )

    data = vectorstore._collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = []

    for text, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):

        document = Document(
            page_content=text,
            metadata=metadata
        )

        documents.append(
            document
        )

    if not documents:

        return None

    return create_bm25(
        documents
    )


# --------------------------------
# Clear BM25 Cache
# --------------------------------

def clear_bm25_cache():

    get_bm25.cache_clear()


# --------------------------------
# Vector Search
# --------------------------------

def vector_search(
    vectorstore,
    query
):

    results = (
        vectorstore
        .similarity_search_with_relevance_scores(
            query,
            k=config.TOP_K
        )
    )

    return results


# --------------------------------
# Hybrid Search
# --------------------------------

def hybrid_search(
    vectorstore,
    bm25_retriever,
    query
):

    vector_results = vector_search(
        vectorstore,
        query
    )

    keyword_results = (
        bm25_retriever.invoke(
            query
        )
    )

    results = []

    seen = set()


    # Vector results
    for document, score in vector_results:

        text = document.page_content

        if text not in seen:

            seen.add(text)

            results.append(
                (
                    document,
                    {
                        "vector_score": round(
                            score,
                            4
                        ),
                        "bm25_score": None
                    }
                )
            )


    # BM25 results
    for document in keyword_results:

        text = document.page_content

        if text not in seen:

            seen.add(text)

            results.append(
                (
                    document,
                    {
                        "vector_score": None,
                        "bm25_score": "matched"
                    }
                )
            )


    return results


# --------------------------------
# Reranking
# --------------------------------

def rerank_documents(
    query,
    results
):

    if not results:

        return []


    pairs = []

    for document, scores in results:

        pairs.append(
            [
                query,
                document.page_content
            ]
        )


    scores = (
        config
        .reranker_model
        .predict(pairs)
    )


    ranked_results = []


    for item, score in zip(
        results,
        scores
    ):

        document = item[0]

        old_scores = item[1]

        new_scores = {
            **old_scores,
            "rerank_score": round(
                float(score),
                4
            )
        }

        ranked_results.append(
            (
                document,
                new_scores
            )
        )


    # Highest score first
    ranked_results.sort(
        key=lambda x:
            x[1]["rerank_score"],
        reverse=True
    )


    return ranked_results[
        :config.RERANK_TOP_K
    ]