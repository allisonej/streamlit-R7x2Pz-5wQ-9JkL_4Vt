import streamlit as st
import pandas as pd
import requests
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"

@st.cache_data
def load_answer_key(url):
    # 구글 시트에서 정답 데이터를 CSV로 읽어오기
    response = requests.get(url)
    answer_key = pd.read_csv(StringIO(response.text))
    return answer_key

def process_files(best_file, current_file, answer_key):
    # 파일 읽기
    best_df = pd.read_csv(best_file)
    current_df = pd.read_csv(current_file)
    
    # 'target' 컬럼의 이름을 변경
    best_df.rename(columns={'target': 'target_best'}, inplace=True)
    current_df.rename(columns={'target': 'target_current'}, inplace=True)
    
    # 'ID' 기준으로 병합
    combined_df = pd.merge(best_df, current_df, on='ID', how='outer')
    
    # 정답지와 병합
    merged_df = pd.merge(combined_df, answer_key, on='ID', how='left')
    
    # 'target' 값이 'label'과 다른 행 필터링
    target_columns = ['target_best', 'target_current']
    conditions = [merged_df[col] != merged_df['label'] for col in target_columns]
    merged_df['target_mismatch'] = pd.concat(conditions, axis=1).any(axis=1)
    changed_df = merged_df[merged_df['target_mismatch']]
    
    # 결과 출력
    if not changed_df.empty:
        st.write("정답이 틀린 항목에 대한 분석표입니다.")
        st.write(f"총 {changed_df['ID'].nunique()}개의 ID가 틀린 예측을 포함하고 있습니다.")
        
        # 결과를 CSV로 저장
        csv_buffer = BytesIO()
        changed_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Result CSV",
            data=csv_buffer.getvalue(),
            file_name='compare_asr.csv'
        )
        
        # 각 label의 빈도수 출력
        st.write("각 label의 빈도수:")
        st.write(changed_df['label'].value_counts())
        
        # 그래프 그리기
        st.write("각 label별 target 값의 선 그래프:")
        plt.figure(figsize=(10, 6))
        
        for label in changed_df['label'].unique():
            subset = changed_df[changed_df['label'] == label]
            target_counts_best = subset['target_best'].value_counts().reindex(['target_best'], fill_value=0)
            target_counts_current = subset['target_current'].value_counts().reindex(['target_current'], fill_value=0)
            plt.plot(['Best File', 'Current File'], [target_counts_best.sum(), target_counts_current.sum()], marker='o', label=f'{label}')

        plt.xlabel('CSV 파일명')
        plt.ylabel('Target 값의 빈도수')
        plt.title('각 label별 target 값의 빈도수')
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("변경된 target 값이 없습니다.")

# Streamlit 앱의 레이아웃 설정
st.set_page_config(page_title="CSV File Grader and Analyzer", layout="wide")

# 사이드바에 메시지 추가
st.sidebar.write("우측 메뉴에서 wide mode를 적용해주세요.")

st.title("CSV File Grader and Analyzer")

st.write("업로드할 CSV 파일들을 선택하세요.")

# 파일 업로드
col1, col2 = st.columns(2)

with col1:
    best_file = st.file_uploader("Upload Best File (CSV)", type="csv")

with col2:
    current_file = st.file_uploader("Upload Current File (CSV)", type="csv")

# 처리 버튼 클릭 시 파일 처리
if best_file and current_file:
    if st.button("Process Files"):
        answer_key = load_answer_key(sheet_url)
        process_files(best_file, current_file, answer_key)
