# ChatDDS 버전 관리 문서 (Version History)

이 문서는 ChatDDS 프로젝트의 버전별 주요 특징, 모델 구성, DB 및 RAG 구조를 기록합니다.

---

## [v0.0.8] - 2026-06-27 (최신 버전)

### 🌟 주요 특징 (Features)

- **임베딩 로컬 전환 (비용 제거)**: 임베딩을 Google Gemini API(`models/gemini-embedding-001`)에서 로컬 Ollama 모델 `qwen3-embedding:8b`로 교체. 초기 인덱싱 및 RAG 질의 시 발생하던 API 요금을 완전히 제거.
- **Gemini 의존성 제거**: `GEMINI_API_KEY` 및 `langchain-google-genai` 의존성 제거. 임베딩 차원이 변경되므로(3072 → 4096) **기존 `chroma_db`는 폐기하고 재임베딩 필요** (`python ingest_vector_db.py`).

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen3:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama, 로컬)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용
- **Cache:** Redis 7 (via Docker, `redis-py` 클라이언트)

### 📂 변경된 파일 구조 (File Structure)

- `config.py` (수정): `EMBEDDING_MODEL`을 `qwen3-embedding:8b`로 변경, `OLLAMA_BASE_URL` 추가, `GEMINI_API_KEY`·Gemini task_type 설정 제거
- `ingest_vector_db.py` (수정): `OllamaEmbeddings`로 문서 임베딩 전환
- `retriever.py` (수정): `OllamaEmbeddings`로 질의 임베딩 전환
- `requirements.txt` (수정): `langchain-google-genai` 의존성 제거

---

## [v0.0.7] - 2026-06-27 (이전 버전)

### 🌟 주요 특징 (Features)

- **LLM 모델 업그레이드**: 생성 모델을 `qwen2.5:14b`에서 `qwen3:14b`로 교체.

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen3:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용
- **Cache:** Redis 7 (via Docker, `redis-py` 클라이언트)

### 📂 변경된 파일 구조 (File Structure)

- `config.py` (수정): `LLM_MODEL`을 `qwen3:14b`로 변경

---

## [v0.0.6] - 2026-06-21 (이전 버전)

### 🌟 주요 특징 (Features)

- **Redis 답변 캐싱 도입**: `질문(+대화 맥락) → 최종 답변`을 통째로 Redis에 캐싱하여, 동일 질문 재요청 시 하이브리드 검색·Cross-Encoder 리랭킹·LLM 생성을 모두 건너뛰고 즉시 응답.
- **캐시 키 정규화**: 질문을 소문자화·문장부호 제거·공백 제거 후 SHA-256 해시로 변환하여, `"홍수 지질학이 뭐야?"`와 `"홍수지질학이 뭐야"`처럼 사소한 표현 차이도 동일 캐시로 처리. (의미가 다른 질문은 그대로 구분)
- **맥락 인지 캐싱**: 캐시 키에 대화 기록(`chat_history`)을 포함하여, 같은 질문이라도 대화 맥락이 다르면 별도 답변을 캐싱.
- **품질 필터링**: 리랭킹 폴백(Fallback)이 발생한 답변은 품질이 낮을 수 있어 캐싱에서 제외. 다음 동일 질문 시 리랭킹을 재시도.
- **Graceful Degradation**: Redis가 꺼져 있거나 연결이 실패해도 캐시 함수가 조용히 우회되어 앱이 정상 동작 (매번 새로 생성).
- **TTL 자동 만료**: 캐시된 답변은 `CACHE_TTL`(기본 7일) 경과 시 자동 만료되어 오래된 답변이 영구히 남지 않음.
- **Docker 기반 Redis**: `docker-compose.yml`로 `redis:7-alpine`을 격리 실행 (AOF 영속성 + 헬스체크 포함). `docker compose up -d` 한 줄로 구동.

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen2.5:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용
- **Cache:** Redis 7 (via Docker, `redis-py` 클라이언트)

### 📂 변경된 파일 구조 (File Structure)

- `cache.py` (신규): Redis 연결 및 답변 캐시 get/set, 키 정규화, graceful degradation
- `docker-compose.yml` (신규): Redis 7 서비스 정의 (포트 6379, 영속 볼륨, 헬스체크)
- `config.py` (수정): `REDIS_URL`, `CACHE_ENABLED`, `CACHE_TTL` 설정 추가
- `app.py` (수정): 답변 생성 전 캐시 조회 → 미스 시 파이프라인 실행 후 캐시 저장 (리랭킹 폴백 시 저장 제외)
- `requirements.txt` (수정): `redis>=5.0` 의존성 추가
- `.env` (수정): `REDIS_URL`, `CACHE_ENABLED`, `CACHE_TTL` 항목 추가

---

## [v0.0.5] - 2026-04-15 (이전 버전)

### 🌟 주요 특징 (Features)

- **코드 모듈 분리**: 단일 `app.py`(294줄)를 `config.py`, `text_utils.py`, `retriever.py`, `chain.py`, `app.py`(120줄)로 분리하여 유지보수성 및 테스트 용이성 향상.
- **설정값 중앙 관리**: 모델명, 경로, 파라미터 등 하드코딩된 설정값을 `config.py`로 통합하여 한 곳에서 관리.
- **중복 코드 제거**: `app.py`와 `ingest_vector_db.py`에 중복 존재하던 `load_documents()` 함수를 `retriever.py`로 통합.
- **데이터 파이프라인 통합**: `data_cleaning.py`와 `data_preprocessing_for_RAG.py`를 `data_pipeline.py`로 통합하여 1,2단계를 한 번에 실행.
- **Reranker 모델 자동 전환**: `config.py`에서 `RERANKER_MODEL`만 변경하면 BAAI/Qwen 모델에 맞는 kwargs가 자동 적용.
- **의존성 버전 고정**: `requirements.txt`에 최소 버전 제약을 추가하여 환경 재현성 향상.
- **`.gitignore` 보강**: `__pycache__/`, `.DS_Store`, `.venv/` 등 표준 Python 제외 항목 추가.

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen2.5:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용

### 📂 변경된 파일 구조 (File Structure)

- `config.py` (신규): 설정값 중앙 관리
- `text_utils.py` (신규): 텍스트 클리닝 유틸리티
- `retriever.py` (신규): 문서 로딩, 검색, Reranker
- `chain.py` (신규): LLM 프롬프트 및 생성 체인
- `data_pipeline.py` (신규): 데이터 정제 + 전처리 통합
- `app.py` (리팩터링): UI 로직만 담당
- `ingest_vector_db.py` (리팩터링): 공통 모듈 임포트로 전환

---

## [v0.0.4] - 2026-04-13 (이전 버전)

### 🌟 주요 특징 (Features)

- **하이퍼링크 출처 자동 첨부**: 답변 끝에 `[제목](URL)` 형식의 마크다운 하이퍼링크를 자동으로 생성하여 가독성과 접근성 향상.
- **출처 중복 제거 로직**: 여러 청크에서 동일한 출처(URL)가 검색될 경우, 중복을 제거하고 고유한 링크만 노출되도록 개선.
- **프롬프트 지침 강화**: LLM 모델이 직접 출처를 생성하지 않도록 강력한 지침을 추가하여, 시스템이 생성하는 정확한 링크와 충돌하지 않도록 조정.
- **스트리밍 UI 및 상태 피드백 개선**: `st.status`를 도입하여 검색 및 재정렬 과정을 시각화하고, 실시간 스트리밍 답변 뒤에 출처가 안정적으로 결합되도록 UI 로직 개선.
- **Retriever 안정성 강화**: Reranking 과정에서 오류 발생 시 기본 검색 결과(k=5)로 자동 폴백(Fallback)하는 예외 처리 추가.

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen2.5:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용

### 🔍 RAG 구조 (RAG Architecture)

- **Hybrid Retriever**: `langchain-classic`의 `EnsembleRetriever` 사용.
- **Reranking**: `langchain-classic`의 `CrossEncoderReranker`를 통한 문서 재정렬.
- **Reference Injection**: 시스템 레벨에서 고유 URL을 추출하여 하이퍼링크 리스트 자동 생성.

---

## [v0.0.3] - 2026-04-08 (이전 버전)

### 🌟 주요 특징 (Features)

- **답변 정제 및 XML 태그 필터링**: 모델이 생성할 수 있는 불필요한 `<think>`, `<answer>`, `<response>` 등의 태그를 `clean_response` 함수를 통해 자동 제거하여 가독성 향상.
- **UI 간소화**: "AI의 사고 과정" expander를 비활성화하고 최종 답변에만 집중하도록 사용자 인터페이스 개선.
- **프롬프트 고도화**: 모든 형태의 XML/HTML 태그 사용을 금지하고, 출처(URL) 표기를 강제하는 강력한 페르소나 지침 적용.
- **패키지 업데이트**: `langchain-classic`을 도입하여 최신 Ensemble 및 Reranking 모듈 구조로 변경.

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen2.5:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용

### 🔍 RAG 구조 (RAG Architecture)

- **Hybrid Retriever**: `langchain-classic`의 `EnsembleRetriever` 사용.
- **Reranking**: `langchain-classic`의 `CrossEncoderReranker`를 통한 문서 재정렬.

---

## [v0.0.2] - 2026-04-07 (이전 버전)

### 🌟 주요 특징 (Features)

- **Mac 환경 최적화**: Apple Silicon(M1/M2/M3/M4) 및 메모리가 제한된 환경에서 원활하게 실행되도록 Reranker 모델 및 연산 정밀도 최적화.
- **Reranker 모델 교체**: 대규모 모델에서 경량 고성능 다국어 모델로 교체하여 추론 속도 향상.
- **메모리 효율화**: 16-bit Floating Point (`float16`) 연산을 적용하여 VRAM/RAM 사용량 절감.
- **Chain of Thought 도입**: `<think>` 태그를 통해 AI의 단계별 사고 과정을 시각화하여 답변의 신뢰도 향상.

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen2.5:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용

### 📂 데이터베이스 (Database)

- **Vector Database**: ChromaDB (로컬 스토리지 `chroma_db/`)
- **Indexing Data**: `rag_preprocessed_data.json`에서 추출한 컨텐츠 및 메타데이터.
- **Metadata Fields**:
  - `chunk_id`: 고유 청크 식별자
  - `title`: 원문 기사/자료 제목
  - `url`: 원본 자료 출처 링크

### 🔍 RAG 구조 (RAG Architecture)

- **Hybrid Retriever Strategy**:
  - **BM25 (Sparse)**: k=10, 텍스트 키워드 매칭 중심.
  - **Chroma (Dense)**: k=10, `qwen3-embedding:8b`를 이용한 의미 기반 검색.
  - **Ensemble**: BM25와 Chroma 결과를 0.5:0.5 가중치로 병합.
- **Contextual Compression (Reranking)**:
  - 병합된 검색 결과(20개)를 `BAAI/bge-reranker-v2-m3`로 재정렬.
  - 상위 5개(`top_n=5`)의 가장 연관성이 높은 문서만 최종 LLM 컨텍스트로 전달.
- **Generation Logic**:
  - 창조과학 전문가 페르소나 적용.
  - 세속 과학 연대를 노아의 홍수 기반 격변 연대론으로 자동 재해석.
  - 답변 마무리 단계에서 참고 자료(URL) 자동 생성.

---

## [v1.0.0] - 2026-04-07
- 초기 버전 (데이터베이스 및 기본 RAG 구조 수립)
