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

def process_files(uploaded_files, answer_key):
    # 업로드된 파일들을 읽어와서 하나의 DataFrame으로 결합
    all_data = []
    for file in uploaded_files:
        user_df = pd.read_csv(file)
        all_data.append(user_df)
    
    # 모든 파일 데이터를 하나로 결합
    user_df_combined = pd.concat(all_data, ignore_index=True)
    
    # 정답지와 병합
    merged_df = pd.merge(user_df_combined, answer_key, on='ID', how='left')
    
    # target과 label이 다른 항목만 필터링
    changed_df = merged_df[merged_df['target'] != merged_df['label']]
    
    # 분석 결과 출력
    if not changed_df.empty:
        st.write("정답이 틀린 항목에 대한 분석표입니다.")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("ID, 예측값, 정답:")
            st.dataframe(changed_df[['ID', 'target', 'label']])
            # 결과를 CSV로 저장
            changed_df.to_csv('compare_asr.csv', index=False)
            st.download_button(
                label="Download Result CSV",
                data=open('compare_asr.csv', 'rb').read(),
                file_name='compare_asr.csv'
            )
        
        with col2:
            st.write("틀린 예측값 빈도수:")
            st.write(changed_df['target'].value_counts())
        
        with col3:
            st.write("못 맞춘 정답 빈도수:")
            st.write(changed_df['label'].value_counts())
        
        with col4:
            st.write("[예측값, 정답] 조합의 빈도수:")
            pair_counts = changed_df.groupby(['target', 'label']).size().reset_index(name='Count')
            pair_counts_sorted = pair_counts.sort_values(by='Count', ascending=False)
            st.write(pair_counts_sorted)
            # 결과를 CSV로 저장
            pair_counts_sorted.to_csv('pair.csv', index=False)
            st.download_button(
                label="Download Pair CSV",
                data=open('pair.csv', 'rb').read(),
                file_name='pair.csv'
            )
    else:
        st.write("변경된 target 값이 없습니다.")

# Streamlit 앱의 레이아웃 설정
st.set_page_config(page_title="CSV File Grader and Analyzer", layout="wide")

# 사이드바에 메시지 추가
st.sidebar.write("우측 메뉴에서 wide mode를 적용해주세요.")

st.title("CSV File Grader and Analyzer")

st.write("업로드할 CSV 파일들을 선택하세요.")

uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    answer_key = load_answer_key(sheet_url)
    process_files(uploaded_files, answer_key)
