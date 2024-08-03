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

def process_files(uploaded_files, answer_key):
    # 파일들 읽어오기 및 컬럼명 변경
    all_data = []
    file_names = []

    for i, file in enumerate(uploaded_files):
        user_df = pd.read_csv(file)
        # 파일 이름을 저장
        file_names.append(file.name)
        # 'target' 컬럼의 이름을 변경
        user_df.rename(columns={'target': f'target_{i+1}'}, inplace=True)
        all_data.append(user_df)
    
    # 모든 파일 데이터를 하나로 결합 (ID 기준)
    user_df_combined = pd.concat(all_data, axis=1, join='inner').drop_duplicates(subset='ID')
    
    # 정답지와 병합
    merged_df = pd.merge(user_df_combined, answer_key, on='ID', how='left')

    # 모든 target 컬럼과 label을 비교
    target_columns = [f'target_{i+1}' for i in range(len(uploaded_files))]
    conditions = [merged_df[col] != merged_df['label'] for col in target_columns]
    merged_df['target_mismatch'] = pd.concat(conditions, axis=1).any(axis=1)
    
    # target과 label이 다른 항목만 필터링
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
            for i in range(len(uploaded_files)):
                target_col = f'target_{i+1}'
                plt.plot(file_names, subset[target_col].value_counts().sort_index(), marker='o', label=f'{label} - {target_col}')

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

st.title("CSV Files Analyzer")

st.write("업로드할 CSV 파일들을 선택하세요.")

# 세션 상태에서 업로드된 파일 목록 관리
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# 파일 업로드 및 세션 상태 업데이트
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# 처리 버튼 클릭 시 파일 처리
if st.session_state.uploaded_files:
    if st.button("Process Files"):
        answer_key = load_answer_key(sheet_url)
        process_files(st.session_state.uploaded_files, answer_key)
