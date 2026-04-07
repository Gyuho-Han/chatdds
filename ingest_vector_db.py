import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

# 로컬 임베딩을 위한 Ollama 라이브러리 임포트
from langchain_ollama import OllamaEmbeddings

# 진행 상황 표시를 위한 tqdm 임포트
from tqdm import tqdm


def load_documents(path: str) -> list[Document]:
    """
    미리 전처리된 창조과학 JSON 데이터를 읽어 LangChain Document 리스트로 변환합니다.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for item in raw:
        # 이미 청크 분할이 되어 있으므로 content_chunk를 메인 텍스트로 사용
        content = f"제목: {item.get('title', '')}\n내용: {item.get('content_chunk', '')}"
        
        # 메타데이터 구성 (출처 URL과 고유 ID 포함)
        metadata = {
            "chunk_id": item.get("chunk_id", ""),
            "title": item.get("title", ""),
            "url": item.get("url", "")
        }
        docs.append(Document(page_content=content.strip(), metadata=metadata))
    return docs


def ingest(json_path: str, persist_directory: str, batch_size: int = 100) -> None:
    source_path = Path(json_path)
    if not source_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")

    # 1. 문서 로드
    docs = load_documents(json_path)
    print(f"총 {len(docs)}개의 청크를 로드했습니다.")

    # 2. 문서 ID 추출
    ids = [doc.metadata.get("chunk_id") or f"doc_{i}" for i, doc in enumerate(docs)]

    # 3. 로컬 임베딩 모델 설정
    print("Ollama qwen3-embedding:8b 모델을 준비 중입니다...")
    # embedding = OllamaEmbeddings(model="bge-m3")
    embedding = OllamaEmbeddings(model="qwen3-embedding:8b")
    
    # 4. Chroma 벡터DB 객체 생성 (데이터는 아직 넣지 않음)
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # 5. 배치 단위로 잘라서 DB에 적재하며 진행 상황(tqdm) 표시
    print("\n본격적인 임베딩 및 DB 적재를 시작합니다:")
    
    # range(0, 전체개수, 배치크기)를 tqdm으로 감싸서 루프를 돕니다.
    for i in tqdm(range(0, len(docs), batch_size), desc="임베딩 진행률", unit="batch"):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        
        # 배치만큼 DB에 추가
        db.add_documents(documents=batch_docs, ids=batch_ids)

    print(f"\n✅ Ingest completed: 총 {len(docs)}개 청크 저장 완료 -> {persist_directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="전처리된 JSON 데이터를 Chroma 벡터DB에 적재하는 스크립트 (진행 상황 표시)"
    )
    parser.add_argument(
        "--json-path",
        default="rag_preprocessed_data.json",
        help="적재할 JSON 파일 경로 (기본값: rag_preprocessed_data.json)",
    )
    parser.add_argument(
        "--persist-directory",
        default="./chroma_db",
        help="Chroma DB 저장 경로 (기본값: ./chroma_db)",
    )
    # 배치 사이즈를 인자로 받을 수 있게 추가 (기본 100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="한 번에 처리할 문서의 개수 (기본값: 100)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(
        json_path=args.json_path, 
        persist_directory=args.persist_directory, 
        batch_size=args.batch_size
    )