"""
데이터 파이프라인: 1단계(정제) + 2단계(RAG 전처리/청킹)를 통합 실행합니다.
벡터DB 적재(3단계)는 시간이 오래 걸리므로 ingest_vector_db.py에서 별도 실행합니다.
"""

import os
import re

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    RAW_CSV_PATH, CLEANED_CSV_PATH,
    RAG_CSV_PATH, RAG_JSON_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


# ── 1단계: 원본 CSV 데이터 정제 ──

def clean_data(input_file=RAW_CSV_PATH, output_file=CLEANED_CSV_PATH):
    if not os.path.exists(input_file):
        print(f"오류: {input_file} 파일이 현재 경로에 없습니다.")
        return None

    print(f"[{input_file}] 데이터 정제를 시작합니다...")

    df = pd.read_csv(input_file)
    initial_count = len(df)

    df = df.drop_duplicates(subset=['title', 'content'], keep='first')

    def _clean_content(text):
        if not isinstance(text, str):
            return ""
        noise_pattern = (
            r"(알림 뒤로|알림 설정|더보기 게시물|마이페이지|로그아웃|찾아오시는길"
            r"|자료실MAP|창조과학스쿨|E-Book|기도월력|문의게시판|후원기관"
            r"|전체보기|추천사이트|로그인이 필요합니다)"
        )
        text = re.sub(noise_pattern, "", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['content'] = df['content'].apply(_clean_content)
    df = df[df['content'].str.len() > 100]

    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    final_count = len(df)
    deleted_count = initial_count - final_count

    print("-" * 40)
    print(f"정제 작업이 완료되었습니다!")
    print(f"기존 행 개수: {initial_count:,}개")
    print(f"정제 후 행 개수: {final_count:,}개")
    print(f"삭제된 행 개수: {deleted_count:,}개 (약 {deleted_count/initial_count*100:.1f}% 감소)")
    print(f"최종 파일 저장 경로: {os.path.abspath(output_file)}")
    print("-" * 40)

    return df


# ── 2단계: 정제된 CSV를 청킹하여 RAG용 데이터 생성 ──

def preprocess_for_rag(input_file=CLEANED_CSV_PATH, output_csv=RAG_CSV_PATH, output_json=RAG_JSON_PATH):
    df = pd.read_csv(input_file)

    def _clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    df['cleaned_content'] = df['content'].apply(_clean_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    processed_data = []
    for index, row in df.iterrows():
        if not row['cleaned_content']:
            continue
        chunks = text_splitter.split_text(row['cleaned_content'])
        for i, chunk in enumerate(chunks):
            processed_data.append({
                "chunk_id": f"doc_{index}_chunk_{i}",
                "title": row['title'],
                "url": row['url'],
                "reference_urls": row['reference_urls'],
                "content_chunk": chunk,
            })

    final_df = pd.DataFrame(processed_data)
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    final_df.to_json(output_json, orient="records", force_ascii=False, indent=4)

    print(f"전처리 완료! 원본 문서 {len(df)}개가 {len(final_df)}개의 청크로 분할되어 저장되었습니다.")


# ── 통합 실행 ──

def main():
    result = clean_data()
    if result is not None:
        preprocess_for_rag()


if __name__ == "__main__":
    main()
