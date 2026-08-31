import os
import logging

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


# Load .env
load_dotenv()


# --------------------------------
# API Key
# --------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --------------------------------
# LLM
# --------------------------------

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)


# --------------------------------
# Embedding Model
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# Reranker Model
# --------------------------------

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------
# Storage
# --------------------------------

CHROMA_DIR = "chroma_db"
UPLOADS_DIR = "uploads"

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


# --------------------------------
# Tenant
# --------------------------------

DEFAULT_TENANT = "default"


def get_manifest_path(tenant_id):

    return os.path.join(
        CHROMA_DIR,
        f"manifest_{tenant_id}.json"
    )


# --------------------------------
# Retrieval Settings
# --------------------------------

TOP_K = 5
RERANK_TOP_K = 4


# --------------------------------
# Logging
# --------------------------------

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, "app.log")
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("IntelliDocs-AI")