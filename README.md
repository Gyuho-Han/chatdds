# Chat DDS Project

성경적 창조론과 젊은 지구 연대설을 바탕으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 🛠 필수 환경 (Environment)

- **Python:** 3.9 이상
- **Ollama:** 로컬 LLM 및 임베딩 모델 실행을 위해 설치 필요
  - **LLM 모델:** `gpt-oss:20b` (또는 `qwen2.5:14b`)
  - **Embedding 모델:** `bge-m3`

## 📦 패키지 설치 (Installation)

```bash
pip install -r requirements.txt
```

## 🚀 실행 순서 (Running Order)

1.  **데이터 전처리 (Data Preprocessing):**
    `creation_science_data.csv`를 읽어 청크로 나누고 `rag_preprocessed_data.json`을 생성합니다.
    ```bash
    python data_preprocessing_for_RAG.py
    ```

2.  **벡터 DB 적재 (Vector DB Ingestion):**
    생성된 JSON 데이터를 `chroma_db` 폴더에 임베딩하여 저장합니다. (Ollama에서 `bge-m3` 모델이 실행 중이어야 합니다.)
    ```bash
    python ingest_vector_db.py
    ```

3.  **애플리케이션 실행 (Run App):**
    Streamlit 웹 인터페이스를 실행합니다.
    ```bash
    streamlit run app.py
    ```

## 📂 파일 구조 (File Structure)

- `app.py`: Streamlit 기반 챗봇 UI 및 RAG 체인
- `data_preprocessing_for_RAG.py`: CSV 데이터 정제 및 청킹
- `ingest_vector_db.py`: ChromaDB 생성 및 데이터 적재
- `creation_science_data.csv`: 원본 데이터셋
- `chroma_db/`: 벡터 데이터베이스 저장 폴더
