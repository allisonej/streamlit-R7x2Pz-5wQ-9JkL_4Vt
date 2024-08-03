import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"

@st.cache_data
def load_answer_key(url):
    # 구글 시트에서 정답 데이터를 CSV로 읽어오기
    response = requests.get(url)
    answer_key = pd.read_csv(StringIO(response.text))
    return answer_key

def process_files(uploaded_file, answer_key):
    # 업로드된 CSV 파일 읽기
    user_df = pd.read_csv(uploaded_file)

    # 데이터 처리
    merged_df = pd.merge(user_df, answer_key, on='ID')
    changed_df = merged_df[merged_df['target'] != merged_df['label']]
    
    # 분석 결과 출력
    if not changed_df.empty:
        st.write("ID가 변경된 target 값을 가진 항목들:")
        st.dataframe(changed_df[['ID', 'target', 'label']])
        
        st.write("target 열의 값 빈도수:")
        st.write(changed_df['target'].value_counts())

        st.write("label 열의 값 빈도수:")
        st.write(changed_df['label'].value_counts())

        st.write("[target, label] 조합의 빈도수:")
        pair_counts = changed_df.groupby(['target', 'label']).size().reset_index(name='Count')
        st.write(pair_counts)

        # 결과를 CSV로 저장
        changed_df.to_csv('compare_asr.csv', index=False)
        st.download_button(label="Download Result CSV", data=open('compare_asr.csv', 'rb'), file_name='compare_asr.csv')

        pair_counts.to_csv('pair.csv', index=False)
        st.download_button(label="Download Pair CSV", data=open('pair.csv', 'rb'), file_name='pair.csv')
    else:
        st.write("변경된 target 값이 없습니다.")

# Streamlit 앱의 레이아웃 설정
st.title("CSV File Grader and Analyzer")

st.write("업로드할 CSV 파일을 선택하세요.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    answer_key = load_answer_key(sheet_url)
    process_files(uploaded_file, answer_key)
