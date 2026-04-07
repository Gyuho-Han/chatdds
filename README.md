# Chat DDS Project

성경적 창조론과 젊은 지구 연대설을 바탕으로 질문에 답변하는 하이브리드 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 🛠 필수 환경 (Environment)

- **Python:** 3.9 이상
- **Ollama:** 로컬 LLM 및 임베딩 모델 실행을 위해 설치 필요
  - **LLM 모델:** `qwen2.5:14b` (Ollama)
  - **Embedding 모델:** `qwen3-embedding:8b` (Ollama)
- **Hugging Face Reranker:** `BAAI/bge-reranker-v2-m3` (HuggingFace에서 자동 다운로드)
  - *참고: Mac 메모리 최적화를 위해 16-bit (`float16`) 정밀도를 사용하며, Apple Silicon(MPS) 및 일반 CPU 환경에 최적화되어 있습니다.*

## 🏗 RAG 및 데이터베이스 구조 (Architecture)

### 1. 데이터베이스 (Database)
- **Vector Store:** ChromaDB를 사용하여 `chroma_db/` 폴더에 임베딩 데이터 저장.
- **Source Data:** `rag_preprocessed_data.json` 기반 (제목, 내용 청크, URL 포함).
- **Metadata:** `chunk_id`, `title`, `url` 정보를 포함하여 답변 시 출처 표기 가능.

### 2. 하이브리드 검색 (Hybrid Retrieval)
- **BM25 Retriever:** 키워드 기반의 전통적인 텍스트 검색 (Top 10).
- **Vector Retriever:** 의미적 유사성 기반의 벡터 검색 (Top 10).
- **Ensemble:** 두 검색 결과를 0.5:0.5 가중치로 결합하여 검색 정확도 극대화.

### 3. 재정렬 (Reranking)
- **Cross-Encoder:** `BAAI/bge-reranker-v2-m3` 모델을 사용하여 검색된 문서의 연관성 재평가.
- **Top-N:** 재정렬된 문서 중 가장 관련성이 높은 상위 5개 문서만 선별하여 컨텍스트로 사용.

### 4. 답변 생성 (Generation)
- **Chain of Thought:** `<think>` 태그를 활용한 단계별 논리적 사고 과정 출력.
- **Persona:** 창조과학 전문가 페르소나 적용 (젊은 지구 연대설, 노아의 홍수 기반 해석).

## 📦 패키지 설치 (Installation)

```bash
pip install -r requirements.txt
```

## 🚀 실행 순서 (Running Order)

1.  **데이터 정제 (Data Cleaning):**
    `creation_science_data.csv`를 정제하여 `cleaned_creation_science_data.csv`를 생성합니다.
    ```bash
    python data_cleaning.py
    ```

2.  **데이터 전처리 (Data Preprocessing):**
    정제된 CSV를 읽어 청크로 나누고 `rag_preprocessed_data.json`을 생성합니다.
    ```bash
    python data_preprocessing_for_RAG.py
    ```

3.  **벡터 DB 적재 (Vector DB Ingestion):**
    생성된 JSON 데이터를 `chroma_db` 폴더에 임베딩하여 저장합니다. (Ollama에서 `qwen3-embedding:8b` 모델이 실행 중이어야 합니다.)
    ```bash
    python ingest_vector_db.py
    ```

4.  **애플리케이션 실행 (Run App):**
    Streamlit 웹 인터페이스를 실행합니다.
    ```bash
    streamlit run app.py
    ```

## 📂 파일 구조 (File Structure)

- `app.py`: Streamlit 챗봇 UI, Hybrid Retriever 및 Reranker 로직
- `data_cleaning.py`: 중복 제거 및 노이즈 텍스트 정제
- `data_preprocessing_for_RAG.py`: RecursiveCharacterTextSplitter를 이용한 청킹
- `ingest_vector_db.py`: ChromaDB 생성 및 Ollama 기반 임베딩 적재
- `creation_science_data.csv`: 원본 데이터셋
- `chroma_db/`: 벡터 데이터베이스 저장 폴더
