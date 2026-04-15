import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

from config import EMBEDDING_MODEL, RAG_JSON_PATH, CHROMA_DB_DIR
from retriever import load_documents


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
    print(f"Ollama {EMBEDDING_MODEL} 모델을 준비 중입니다...")
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 4. Chroma 벡터DB 객체 생성
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # 5. 배치 단위로 잘라서 DB에 적재하며 진행 상황(tqdm) 표시
    print("\n본격적인 임베딩 및 DB 적재를 시작합니다:")
    for i in tqdm(range(0, len(docs), batch_size), desc="임베딩 진행률", unit="batch"):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        db.add_documents(documents=batch_docs, ids=batch_ids)

    print(f"\n✅ Ingest completed: 총 {len(docs)}개 청크 저장 완료 -> {persist_directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="전처리된 JSON 데이터를 Chroma 벡터DB에 적재하는 스크립트"
    )
    parser.add_argument(
        "--json-path",
        default=RAG_JSON_PATH,
        help=f"적재할 JSON 파일 경로 (기본값: {RAG_JSON_PATH})",
    )
    parser.add_argument(
        "--persist-directory",
        default=CHROMA_DB_DIR,
        help=f"Chroma DB 저장 경로 (기본값: {CHROMA_DB_DIR})",
    )
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
        batch_size=args.batch_size,
    )
