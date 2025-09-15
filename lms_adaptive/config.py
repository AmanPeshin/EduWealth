import os

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Models
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Quiz & IRT policy
QUIZ_LENGTH = int(os.getenv("QUIZ_LENGTH", "10"))

# Cosine thresholds for per-attempt dedup
COSINE_THRESHOLD_HARD = float(os.getenv("COSINE_THRESHOLD_HARD", "0.90"))
COSINE_THRESHOLD_SOFT = float(os.getenv("COSINE_THRESHOLD_SOFT", "0.86"))

# Storage
DB_URL = os.getenv("DB_URL", "sqlite:///./lms.db")

# Vector stores
VECTOR_DIR = os.getenv("VECTOR_DIR", "./vectorstore")  # Chroma for RAG
FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_index")    # FAISS for bank ANN (optional)

# Checkpointers
CHECKPOINTER_BACKEND = os.getenv("CHECKPOINTER_BACKEND", "redis")  # redis | sqlite | memory
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SQLITE_CP_PATH = os.getenv("SQLITE_CP_PATH", "./graph_checkpoints.sqlite")

# Ingestion (company PDFs)
COMPANY_PDF_DIR = os.getenv("COMPANY_PDF_DIR", "./company_finance_pdfs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# IRT adaptive settings
INIT_THETA = float(os.getenv("INIT_THETA", "0.0"))
THETA_LR = float(os.getenv("THETA_LR", "0.25"))  # step for online update
