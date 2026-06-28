# Chat DDS Project

성경적 창조론과 젊은 지구 연대설을 바탕으로 질문에 답변하는 하이브리드 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 🛠 필수 환경 (Environment)

- **Python:** 3.9 이상
- **Ollama:** 로컬 LLM 및 임베딩 모델 실행을 위해 설치 필요
  - **LLM 모델:** `qwen3:14b` (Ollama)
  - **Embedding 모델:** `qwen3-embedding:8b` (Ollama)
- **Hugging Face Reranker:** `BAAI/bge-reranker-v2-m3` (HuggingFace에서 자동 다운로드)
  - _참고: Mac 메모리 최적화를 위해 16-bit (`float16`) 정밀도를 사용하며, Apple Silicon(MPS) 및 일반 CPU 환경에 최적화되어 있습니다._
- **Redis:** 답변 캐싱용 (Docker로 실행 권장). 미실행 시에도 앱은 정상 동작하며 캐싱만 비활성화됩니다.

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

- **Tag Filtering:** 모델이 생성하는 불필요한 XML/HTML 태그(`<think>`, `<answer>` 등)를 자동 제거하여 깔끔한 답변 제공.
- **Persona:** 창조과학 전문가 페르소나 적용 (젊은 지구 연대설, 노아의 홍수 기반 해석, 확신에 찬 어조).
- **Hyperlink References:** 답변 마지막에 `[제목](URL)` 형식의 마크다운 하이퍼링크를 포함하는 🔗 **참고 자료** 섹션을 시스템이 자동으로 추가 (중복 URL 제거 로직 포함).

### 5. 답변 캐싱 (Redis Caching)

- **전체 답변 캐싱:** `질문(+대화 맥락) → 최종 답변`을 통째로 Redis에 저장. 동일 질문 재요청 시 검색·리랭킹·LLM 생성을 모두 건너뛰고 즉시 응답.
- **키 정규화:** 질문을 소문자화·문장부호 제거·공백 제거 후 SHA-256 해시로 변환. `"홍수 지질학이 뭐야?"`와 `"홍수지질학이 뭐야"`처럼 표현이 조금 달라도 같은 캐시로 처리 (의미가 다르면 구분).
- **품질 필터링:** 리랭킹 폴백이 발생한 답변은 캐싱하지 않음 (다음 요청 시 재시도).
- **TTL & 안전성:** 캐시는 기본 7일(`CACHE_TTL`) 후 자동 만료. Redis가 꺼져 있어도 앱은 정상 동작 (캐싱만 비활성화).

## 📦 패키지 설치 (Installation)

```bash
pip install -r requirements.txt
```

## ⚙️ Redis 캐시 설정 (Cache Setup)

Docker로 Redis를 실행합니다 (선택 사항이지만 권장).

```bash
docker compose up -d        # redis:7-alpine 컨테이너 실행
```

캐시 동작은 `.env`에서 조정할 수 있습니다.

```bash
REDIS_URL = redis://localhost:6379/0   # Redis 주소
CACHE_ENABLED = true                   # false로 두면 캐싱 비활성화
CACHE_TTL = 604800                     # 캐시 유효기간(초), 기본 7일
```

## 🚀 실행 순서 (Running Order)

1.  **데이터 정제 + 전처리 (Data Pipeline):**
    `creation_science_data.csv`를 정제하고, 청크로 나누어 `rag_preprocessed_data.json`을 생성합니다.

    ```bash
    python data_pipeline.py
    ```

2.  **벡터 DB 적재 (Vector DB Ingestion):**
    생성된 JSON 데이터를 `chroma_db` 폴더에 임베딩하여 저장합니다. (Ollama에서 `qwen3-embedding:8b` 모델이 실행 중이어야 합니다.)
    ```bash
    python ingest_vector_db.py
    ```

2.1 **Google Drive에서 chroma_db 파일 다운**
1,2 뛰어 넘어도 됨.
https://drive.google.com/file/d/1fJrSIh9gprEh8WU65v_vtT3LARoLQhuc/view?usp=drive_link

3.  **애플리케이션 실행 (Run App):**
    Streamlit 웹 인터페이스를 실행합니다.
    ```bash
    streamlit run app.py
    ```

## 📂 파일 구조 (File Structure)

- `config.py`: 모델명, 경로, 파라미터, Redis 캐시 설정값 중앙 관리
- `app.py`: Streamlit 챗봇 UI (캐시 조회/저장 연결)
- `cache.py`: Redis 답변 캐시 (키 정규화, get/set, graceful degradation)
- `text_utils.py`: 텍스트 클리닝 및 태그 분리 유틸리티
- `retriever.py`: 문서 로딩, 벡터DB, 하이브리드 검색 및 Reranker 로직
- `chain.py`: LLM 프롬프트 및 생성 체인
- `data_pipeline.py`: 데이터 정제 + RAG 전처리/청킹 통합 파이프라인
- `ingest_vector_db.py`: ChromaDB 생성 및 Ollama 기반 임베딩 적재
- `docker-compose.yml`: Redis 7 컨테이너 정의 (캐시 백엔드)
- `creation_science_data.csv`: 원본 데이터셋
- `chroma_db/`: 벡터 데이터베이스 저장 폴더
