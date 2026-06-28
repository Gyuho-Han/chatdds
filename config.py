import os
from dotenv import load_dotenv

load_dotenv()

# ── Data Pipeline Paths ──
CLEANED_CSV_PATH = "cleaned_dataset.csv"
RAG_CSV_PATH = "rag_preprocessed_data.csv"
RAG_JSON_PATH = "rag_preprocessed_data.json"

# ── Vector DB ──
CHROMA_DB_DIR = "./chroma_db"

# ── Redis 캐시 ──
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "604800"))  # 캐시 유효기간(초), 기본 7일

# ── Models ──
# Local Ollama embedding (무료/로컬, qwen3-embedding:8b -> 4096-dim)
EMBEDDING_MODEL = "qwen3-embedding:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LLM_MODEL = "qwen3:14b"
# LLM_MODEL = "gemma4:12b"

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
# gemini-embedding-001 max 2048 tokens; Korean ~1.0-1.5 tokens/char.
# 800 chars / 100 overlap balances context vs. retrieval precision.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
