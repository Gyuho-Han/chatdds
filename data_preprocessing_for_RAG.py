import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 데이터 로드
file_path = "cleaned_creation_science_data.csv"
df = pd.read_csv(file_path)

# 2. 텍스트 정제 함수 (Cleaning)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 불필요한 이스케이프 문자나 과도한 공백, 개행문자 정제
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# content 컬럼 정제 적용
df['cleaned_content'] = df['content'].apply(clean_text)

# 3. 텍스트 청킹 (Chunking) 설정
# RecursiveCharacterTextSplitter는 문단 -> 문장 -> 단어 순으로 문맥이 끊기지 않게 분할해줍니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # 하나의 청크 크기 (글자 수 기준. 임베딩 모델에 따라 500~1000 권장)
    chunk_overlap=200,     # 청크 간 문맥이 끊기지 않도록 겹치게 할 글자 수
    length_function=len,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)

processed_data = []

# 4. 데이터프레임 순회하며 청킹 및 메타데이터 결합
for index, row in df.iterrows():
    # 정제된 텍스트가 없는 경우 건너뜀
    if not row['cleaned_content']:
        continue
        
    # 텍스트 분할
    chunks = text_splitter.split_text(row['cleaned_content'])
    
    for i, chunk in enumerate(chunks):
        # 각 청크 단위로 새로운 딕셔너리 생성 (메타데이터 보존)
        processed_data.append({
            "chunk_id": f"doc_{index}_chunk_{i}", # 고유 ID 부여
            "title": row['title'],
            "url": row['url'],
            "reference_urls": row['reference_urls'],
            "content_chunk": chunk # 분할된 텍스트
        })

# 5. 최종 데이터프레임 생성
final_df = pd.DataFrame(processed_data)

# 6. 최종 형태 저장 (CSV 및 JSON)
# 한글 깨짐 방지를 위해 utf-8-sig 인코딩 사용
final_df.to_csv("rag_preprocessed_data.csv", index=False, encoding='utf-8-sig')

# RAG 시스템에 따라 JSON 형태를 요구하는 경우가 많으므로 JSON으로도 저장
final_df.to_json("rag_preprocessed_data.json", orient="records", force_ascii=False, indent=4)

print(f"전처리 완료! 원본 문서 {len(df)}개가 {len(final_df)}개의 청크로 분할되어 저장되었습니다.")