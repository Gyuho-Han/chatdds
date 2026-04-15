# ── Data Pipeline Paths ──
RAW_CSV_PATH = "creation_science_data.csv"
CLEANED_CSV_PATH = "cleaned_creation_science_data.csv"
RAG_CSV_PATH = "rag_preprocessed_data.csv"
RAG_JSON_PATH = "rag_preprocessed_data.json"

# ── Vector DB ──
CHROMA_DB_DIR = "./chroma_db"

# ── Models ──
EMBEDDING_MODEL = "qwen3-embedding:8b"
LLM_MODEL = "qwen2.5:14b"

# Reranker (16GB: "BAAI/bge-reranker-v2-m3", 32GB: "Qwen/Qwen3-Reranker-4B")
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N = 5

# ── Retriever Parameters ──
BM25_K = 10 # keyword search
VECTOR_K = 10 # context(vector) search
ENSEMBLE_WEIGHTS = [0.5, 0.5] # [BM25, Vector]

# ── LLM Parameters ──
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_REPEAT_PENALTY = 1.15
LLM_STOP_TOKENS = ["<|im_end|>", "User:", "Question:"]

# ── Chunking Parameters ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
