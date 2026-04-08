# ChatDDS 버전 관리 문서 (Version History)

이 문서는 ChatDDS 프로젝트의 버전별 주요 특징, 모델 구성, DB 및 RAG 구조를 기록합니다.

---

## [v0.0.3] - 2026-04-08 (최신 버전)

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
