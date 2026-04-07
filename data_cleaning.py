import pandas as pd
import re
import os

def run_cleaning_process():
    input_file = 'creation_science_data.csv'
    output_file = 'cleaned_creation_science_data.csv'

    if not os.path.exists(input_file):
        print(f"오류: {input_file} 파일이 현재 경로에 없습니다.")
        return

    print(f"[{input_file}] 데이터 정제를 시작합니다...")

    df = pd.read_csv(input_file)
    initial_count = len(df)

    df = df.drop_duplicates(subset=['title', 'content'], keep='first')

    def clean_content(text):
        if not isinstance(text, str):
            return ""
        
        # 수정 1: .* 를 제거하여 뒤에 오는 모든 텍스트가 삭제되는 것을 방지
        noise_pattern = r"(알림 뒤로|알림 설정|더보기 게시물|마이페이지|로그아웃|찾아오시는길|자료실MAP|창조과학스쿨|E-Book|기도월력|문의게시판|후원기관|전체보기|추천사이트|로그인이 필요합니다)"
        
        # 수정 2: flags=re.DOTALL 제거 (해당 키워드만 삭제하도록 변경)
        text = re.sub(noise_pattern, "", text)
        
        # 연속된 공백 및 줄바꿈 하나로 통일
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['content'] = df['content'].apply(clean_content)

    # 임베딩에 부적합한 짧은 텍스트 제거 (100자 미만)
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

if __name__ == "__main__":
    run_cleaning_process()