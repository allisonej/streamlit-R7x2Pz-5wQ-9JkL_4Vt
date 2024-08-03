import streamlit as st
import pandas as pd
import requests
from io import StringIO
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
    
    # 빈 칸인 값들의 수 계산
    missing_values_summary = changed_df[['target_best', 'target_current', 'label']].isna().sum()
    
    # 전체 행의 수 및 고유한 ID 수
    total_rows = len(changed_df)
    unique_ids = changed_df['ID'].nunique()
    
    # 분석 결과 출력
    if not changed_df.empty:
        st.write("정답이 틀린 항목에 대한 분석표입니다.")

        # 1. 병합되고 오답을 가진 행들로 이루어진 df의 출력
        st.write("1. 병합되고 오답을 가진 행들:")
        st.dataframe(changed_df[['ID', 'target_best', 'target_current', 'label']])
        
        # 빈 칸인 값들의 수 출력
        st.write("빈 칸인 값들의 수:")
        st.write(missing_values_summary)
        st.write("전체 행의 수:", total_rows)
        st.write("고유한 ID의 수:", unique_ids)
        
        # 2. 틀린 예측값 빈도수
        st.write("2. 틀린 예측값 빈도수:")
        target_counts_best = changed_df['target_best'].value_counts().sort_values(ascending=False)
        target_counts_current = changed_df['target_current'].value_counts().sort_values(ascending=False)
        common_targets = pd.DataFrame({
            'Best File Count': target_counts_best,
            'Current File Count': target_counts_current
        }).fillna(0).astype(int).sort_values(by='Best File Count', ascending=False)
        st.write(common_targets)
        
        # 3. 못 맞춘 정답 빈도수
        st.write("3. 못 맞춘 정답 빈도수:")
        label_counts_best = changed_df['label'].value_counts().sort_values(ascending=False)
        label_counts_current = changed_df['label'].value_counts().sort_values(ascending=False)
        common_labels = pd.DataFrame({
            'Best File Count': label_counts_best,
            'Current File Count': label_counts_current
        }).fillna(0).astype(int).sort_values(by='Best File Count', ascending=False)
        st.write(common_labels)
        
        # 4. target_best, target_current, label 조합의 빈도수
        st.write("4. target_best, target_current, label 조합의 빈도수:")
        pair_counts = changed_df.groupby(['target_best', 'target_current', 'label']).size().reset_index(name='Count')
        pair_counts_sorted = pair_counts.sort_values(by='Count', ascending=False)
        st.dataframe(pair_counts_sorted)
        
    #     # 그래프 그리기
    #     st.write("각 label별 target 값의 선 그래프:")
        
    #     plt.figure(figsize=(12, 6))
    #     labels = changed_df['label'].unique()
        
    #     for label in labels:
    #         subset = changed_df[changed_df['label'] == label]
    #         target_counts_best = subset['target_best'].notna().sum()
    #         target_counts_current = subset['target_current'].notna().sum()
            
    #         plt.plot(['Best File', 'Current File', 'Label'], 
    #                  [target_counts_best, target_counts_current, len(subset)], 
    #                  marker='o', 
    #                  label=f'Label {label}')
        
    #     plt.xlabel('File and Label')
    #     plt.ylabel('Count')
    #     plt.title('Target Value Count by Label')
    #     plt.legend(title='Labels')
    #     st.pyplot(plt)
    # else:
    #     st.write("변경된 target 값이 없습니다.")

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
