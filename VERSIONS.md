# ChatDDS 버전 관리 문서 (Version History)

이 문서는 ChatDDS 프로젝트의 버전별 주요 특징, 모델 구성, DB 및 RAG 구조를 기록합니다.

---

## [v0.0.2] - 2026-04-07 (현재 버전)

### 🌟 주요 특징 (Features)

- **Mac 환경 최적화**: Apple Silicon(M1/M2/M3/M4) 및 메모리가 제한된 환경에서 원활하게 실행되도록 Reranker 모델 및 연산 정밀도 최적화.
- **Reranker 모델 교체**: 대규모 모델에서 경량 고성능 다국어 모델로 교체하여 추론 속도 향상.
- **메모리 효율화**: 16-bit Floating Point (`float16`) 연산을 적용하여 VRAM/RAM 사용량 절감.

### 🤖 모델 구성 (Models)

- **LLM (Generation):** `qwen2.5:14b` (via Ollama)
- **Embedding:** `qwen3-embedding:8b` (via Ollama)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (via HuggingFace CrossEncoder)
  - Optimization: `torch.float16` 적용

### 📂 데이터베이스 (Database)

- (v1.0.0과 동일)

### 🔍 RAG 구조 (RAG Architecture)

- (v1.0.0과 동일하되 Reranker 모델만 경량화된 버전으로 교체됨)

---

## [v1.0.0] - 2026-04-07
