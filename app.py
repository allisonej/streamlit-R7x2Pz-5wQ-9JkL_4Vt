import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics import f1_score

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"
# meta_url = "https://docs.google.com/spreadsheets/d/1y-2ZLNxR7FzwqmCY5powZZkyYva7qOM2-Y1HnP2m248/export?format=csv"/

@st.cache_data
def load_key(url):
    # 구글 시트에서 정답 데이터를 CSV로 읽어오기
    response = requests.get(url)
    key = pd.read_csv(StringIO(response.text))
    return key

# def map_target_to_text(target_value, meta_key):
#     """Map target or label value to its corresponding text in the format 'target_value_translation'."""
#     mapping = meta_key.set_index('target')['translation'].to_dict()
#     translation = mapping.get(target_value, 'Unknown')
#     return f"{target_value}_{translation}"

def process_files(uploaded_file, answer_key):
    # 업로드된 CSV 파일 읽기
    user_df = pd.read_csv(uploaded_file)

    st.write("업로드 파일 ID 수: ", len(user_df), "\t정답 파일 ID 수: ", len(answer_key))
    st.write("target na? : ", user_df['target'].isna().sum())
    st.markdown("---")
    
    # 데이터 처리
    merged_df = pd.merge(user_df, answer_key, on='ID')
    changed_df = merged_df[merged_df['target'] != merged_df['label']]
    
    # Macro F1 Score 계산
    macro_f1 = f1_score(merged_df['label'], merged_df['target'], average='macro')
    st.markdown(f"Macro F1 Score: **:blue[{macro_f1:.4f}]**")

    # # meta 적용 for viewer
    # merged_df = merged_df.copy()  # 데이터프레임의 복사본을 생성합니다.
    # changed_df = changed_df.copy()  # 데이터프레임의 복사본을 생성합니다.
    
    # for column in ['target', 'label']:
    #     merged_df[f'{column}_text'] = merged_df[column].apply(lambda x: map_target_to_text(x, meta_key))
    #     changed_df[f'{column}_text'] = changed_df[column].apply(lambda x: map_target_to_text(x, meta_key))

    # 분석 결과 출력
    if not changed_df.empty:
        st.write("정답이 틀린 항목에 대한 분석표입니다.")
        st.write(f"총 틀린 항목 수: ", len(changed_df))
        col1, col2 = st.columns(2)

        with col1:
            st.write("ID, 예측값, 정답:")
            st.dataframe(changed_df[['ID', 'target', 'label']])
            # 결과를 CSV로 저장
            csv_data = changed_df.to_csv(index=False)
            st.download_button(
                label="Download Result CSV",
                data=csv_data,
                file_name='compare_asr.csv'
            )
        
        with col2:
            st.write("[예측값, 정답] 조합의 빈도수:")
            pair_counts = changed_df.groupby(['target', 'label']).size().reset_index(name='count')
            pair_counts_sorted = pair_counts.sort_values(by='count', ascending=False)
            st.write(pair_counts_sorted)
            # 결과를 CSV로 저장
            csv_data = pair_counts_sorted.to_csv(index=False)
            st.download_button(
                label="Download Pair CSV",
                data=csv_data,
                file_name='pair.csv'
            )

        col1, col2 = st.columns(2)
        with col1:
            st.write("틀린 예측값 빈도, 틀린 비율: target으로 잘못 예측한 수 / target으로 예측한 전체 수")
            wrong_counts = changed_df['target'].value_counts() # 예측값 틀린 것 셈
            total_counts = merged_df['target'].value_counts() # 예측값 전체 셈
            counts_combined = pd.concat([wrong_counts, total_counts], axis=1, sort=False).fillna(0)
            counts_combined.columns = ['wrong_count', 'total_count']
            counts_combined['rate'] = counts_combined['wrong_count'] / counts_combined['total_count']
            counts_combined['rate_view'] = counts_combined.apply(lambda row: f"{int(row['wrong_count'])} / {int(row['total_count'])}", axis=1)
            st.write(counts_combined[['wrong_count', 'rate_view', 'rate']])
        
        with col2:
            st.write("못 맞춘 정답 빈도, 틀린 비율: 잘못 예측한 실제값 수 / 전체 실제값 수")
            wrong_counts = changed_df['label'].value_counts() # 실제값 틀린 것 셈
            total_counts = merged_df['label'].value_counts() # 실제값 전체 셈
            counts_combined = pd.concat([wrong_counts, total_counts], axis=1, sort=False).fillna(0)
            counts_combined.columns = ['wrong_count', 'total_count']
            counts_combined['rate'] = counts_combined['wrong_count'] / counts_combined['total_count']
            counts_combined['rate_view'] = counts_combined.apply(lambda row: f"{int(row['wrong_count'])} / {int(row['total_count'])}", axis=1)
            st.write(counts_combined[['wrong_count', 'rate_view', 'rate']])

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
    answer_key = load_key(sheet_url)
    # meta_key = load_key(meta_url)
    process_files(uploaded_file, answer_key)
