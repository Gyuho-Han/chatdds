import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import (
    RAG_JSON_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL,
    RERANKER_MODEL, RERANKER_TOP_N,
    BM25_K, VECTOR_K, ENSEMBLE_WEIGHTS,
)


def load_documents(path=RAG_JSON_PATH):
    """미리 전처리된 JSON 데이터를 읽어 LangChain Document 리스트로 변환합니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"데이터 파일({path})이 없습니다.")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for item in raw:
        content = f"제목: {item.get('title', '')}\n내용: {item.get('content_chunk', '')}"
        metadata = {
            "chunk_id": item.get("chunk_id", ""),
            "title": item.get("title", ""),
            "url": item.get("url", "")
        }
        docs.append(Document(page_content=content.strip(), metadata=metadata))
    return docs


def load_vectorstore(persist_directory=CHROMA_DB_DIR):
    """Chroma 벡터DB를 로드합니다."""
    if not Path(persist_directory).exists():
        raise FileNotFoundError("벡터DB가 아직 생성되지 않았습니다. 데이터를 먼저 임베딩하세요.")
    embed = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=persist_directory, embedding_function=embed)


def init_retrievers(docs, vector_db):
    """Hybrid (BM25 + Vector) 검색 + Cross-Encoder Reranker를 초기화합니다."""
    import torch
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = BM25_K
    vect = vector_db.as_retriever(search_kwargs={"k": VECTOR_K})
    hybrid = EnsembleRetriever(retrievers=[bm25, vect], weights=ENSEMBLE_WEIGHTS)

    if "Qwen" in RERANKER_MODEL:
        model_kwargs = {
            "automodel_args": {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
            }
        }
    else:
        model_kwargs = {"model_kwargs": {"torch_dtype": torch.float16}}

    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL, model_kwargs=model_kwargs)
    re_ranker = CrossEncoderReranker(model=model, top_n=RERANKER_TOP_N)
    return ContextualCompressionRetriever(base_compressor=re_ranker, base_retriever=hybrid)


def format_docs_and_extract_urls(docs):
    """검색된 문서를 텍스트화하고 고유 URL 하이퍼링크를 추출합니다."""
    context_parts = []
    unique_refs = {}

    for d in docs:
        title = d.metadata.get('title', '제목 없음')
        url = d.metadata.get('url', '')
        context_parts.append(f"[출처: {title}]\n{d.page_content}")
        if url and url not in unique_refs:
            unique_refs[url] = title

    context_str = "\n\n---\n\n".join(context_parts)
    ref_list = [f"- [{title}]({url})" for url, title in unique_refs.items()]
    refs_str = "\n".join(ref_list)

    return context_str, refs_str
