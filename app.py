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

    st.write("업로드 파일 ID: ", len(user_df), "\t정답 파일 ID: ", len(answer_key))
    st.write("na? : ", user_df.isna().sum())
    # 데이터 처리
    merged_df = pd.merge(user_df, answer_key, on='ID')
    changed_df = merged_df[merged_df['target'] != merged_df['label']]
    
    # 분석 결과 출력
    if not changed_df.empty:
        st.write("정답이 틀린 항목에 대한 분석표입니다.")
        st.write(f"총 틀린 항목 수: ",len(changed_df))  # 틀린 항목 수 추가
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
            st.write("틀린 예측값 빈도, 틀린 비율:")
            counts = changed_df['target'].value_counts()
            rates = counts / len(changed_df)
            result_df = pd.DataFrame({
                'count': counts,
                'wrong_rate': rates
            }) 
            st.write(result_df)
        
        with col3:
            st.write("못 맞춘 정답 빈도, 틀린 비율:")
            counts = changed_df['label'].value_counts()
            rates = counts / len(changed_df)
            result_df = pd.DataFrame({
                'count': counts,
                'wrong_rate': rates
            }) 
            st.write(result_df)
        
        with col4:
            st.write("[예측값, 정답] 조합의 빈도수:")
            pair_counts = changed_df.groupby(['target', 'label']).size().reset_index(name='count')
            pair_counts_sorted = pair_counts.sort_values(by='count', ascending=False)
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

st.write("업로드할 CSV 파일을 선택하세요.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    answer_key = load_answer_key(sheet_url)
    process_files(uploaded_file, answer_key)
