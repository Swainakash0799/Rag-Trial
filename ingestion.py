import os
import json
import uuid
import hashlib
from datetime import datetime

from tools import (
    load_pdf,
    load_docx,
    load_csv,
    load_excel,
    split_documents,
    get_vectorstore,
    add_documents,
    delete_document_chunks,
    clear_bm25_cache
)

import config


# --------------------------------
# Load Any File
# --------------------------------

def load_file(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()


    if extension == ".pdf":

        return load_pdf(
            file_path
        )


    elif extension == ".docx":

        return load_docx(
            file_path
        )


    elif extension == ".csv":

        return load_csv(
            file_path
        )


    elif extension in [".xlsx", ".xls"]:

        return load_excel(
            file_path
        )


    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )


# --------------------------------
# File Hash
# --------------------------------

def get_file_hash(file_path):

    sha256 = hashlib.sha256()


    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            data = file.read(
                8192
            )

            if not data:

                break

            sha256.update(
                data
            )


    return sha256.hexdigest()


# --------------------------------
# Manifest
# --------------------------------

def load_manifest(
    tenant_id="default"
):

    path = config.get_manifest_path(
        tenant_id
    )


    if not os.path.exists(path):

        return {}


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def save_manifest(
    manifest,
    tenant_id="default"
):

    path = config.get_manifest_path(
        tenant_id
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )


# --------------------------------
# List Documents
# --------------------------------

def list_documents(
    tenant_id="default"
):

    manifest = load_manifest(
        tenant_id
    )

    return list(
        manifest.values()
    )


# --------------------------------
# Ingest Files
# --------------------------------

def ingest_files(
    file_paths,
    tenant_id="default"
):

    manifest = load_manifest(
        tenant_id
    )

    vectorstore = get_vectorstore(
        tenant_id
    )


    added_files = []

    duplicate_files = []


    for file_path in file_paths:

        filename = os.path.basename(
            file_path
        )


        # -------------------------
        # Check duplicate
        # -------------------------

        file_hash = get_file_hash(
            file_path
        )


        duplicate = False


        for document in manifest.values():

            if (
                document["file_hash"]
                == file_hash
            ):

                duplicate = True

                break


        if duplicate:

            duplicate_files.append(
                filename
            )

            config.logger.info(
                f"Duplicate skipped: {filename}"
            )

            continue


        # -------------------------
        # Document ID
        # -------------------------

        document_id = str(
            uuid.uuid4()
        )


        # -------------------------
        # File information
        # -------------------------

        file_type = (
            os.path.splitext(
                filename
            )[1]
            .lower()
            .replace(
                ".",
                ""
            )
        )


        upload_date = (
            datetime.now()
            .isoformat()
        )


        # -------------------------
        # Load document
        # -------------------------

        documents = load_file(
            file_path
        )


        # -------------------------
        # Split document
        # -------------------------

        chunks = split_documents(
            documents
        )


        # -------------------------
        # Add metadata
        # -------------------------

        for index, chunk in enumerate(
            chunks
        ):

            chunk.metadata.update(
                {
                    "filename":
                        filename,

                    "document_id":
                        document_id,

                    "tenant_id":
                        tenant_id,

                    "upload_date":
                        upload_date,

                    "chunk_index":
                        index,

                    "file_type":
                        file_type,

                    "page":
                        chunk.metadata.get(
                            "page",
                            "-"
                        )
                }
            )


        # -------------------------
        # Save to Chroma
        # -------------------------

        add_documents(
            vectorstore,
            chunks
        )


        # -------------------------
        # Save manifest
        # -------------------------

        manifest[document_id] = {

            "document_id":
                document_id,

            "filename":
                filename,

            "file_hash":
                file_hash,

            "upload_date":
                upload_date,

            "chunk_count":
                len(chunks),

            "file_type":
                file_type
        }


        added_files.append(
            filename
        )


        config.logger.info(
            f"Added {filename} "
            f"with {len(chunks)} chunks"
        )


    save_manifest(
        manifest,
        tenant_id
    )


    # Clear cached BM25 because
    # the document collection changed.
    clear_bm25_cache()


    return {
        "added":
            added_files,

        "skipped_duplicates":
            duplicate_files
    }


# --------------------------------
# Delete Document
# --------------------------------

def delete_document(
    document_id,
    tenant_id="default"
):

    manifest = load_manifest(
        tenant_id
    )


    if document_id not in manifest:

        return False


    vectorstore = get_vectorstore(
        tenant_id
    )


    delete_document_chunks(
        vectorstore,
        document_id
    )


    filename = manifest[
        document_id
    ]["filename"]


    del manifest[
        document_id
    ]


    save_manifest(
        manifest,
        tenant_id
    )


    # Clear cached BM25 because
    # the document collection changed.
    clear_bm25_cache()


    config.logger.info(
        f"Deleted document: {filename}"
    )


    return True