import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback

# Adjust the width of the Streamlit page
# Streamlit 앱의 레이아웃 설정
st.set_page_config(page_title="CSV File Grader and Analyzer", layout="wide")

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"
meta_url = "https://docs.google.com/spreadsheets/d/1y-2ZLNxR7FzwqmCY5powZZkyYva7qOM2-Y1HnP2m248/export?format=csv"

@st.cache_data
def load_key(url):
    # 구글 시트에서 정답 데이터를 CSV로 읽어오기
    response = requests.get(url)
    key = pd.read_csv(StringIO(response.text))
    return key

def process_files(uploaded_file, answer_key, meta_key):
    try:
        # 업로드된 CSV 파일 읽기
        user_df = pd.read_csv(uploaded_file)

        st.write("업로드 파일 ID 수: ", len(user_df), "\t정답 파일 ID 수: ", len(answer_key))
        st.write("target na? : ", user_df['target'].isna().sum())
        st.markdown("---")

        # 데이터 처리 (join 사용)
        merged_df = user_df.set_index('ID').join(answer_key.set_index('ID'))
        changed_df = merged_df[merged_df['target'] != merged_df['label']]

        # Macro F1 Score 계산
        macro_f1 = f1_score(merged_df['label'], merged_df['target'], average='macro')
        st.markdown(f"Macro F1 Score: **:blue[{macro_f1:.4f}]**")

        # meta 데이터를 dictionary로 변환하여 매핑
        meta_dict = meta_key.set_index('target')['translation'].to_dict()

        # 분석 결과 출력
        if not changed_df.empty:
            st.write("정답이 틀린 항목에 대한 분석표입니다.")
            st.write(f"총 틀린 항목 수: ", len(changed_df))
            col1, col2 = st.columns(2)

            with col1:
                st.write("ID, 예측값, 정답:")
                st.dataframe(changed_df)

                # 결과를 CSV로 저장
                csv_data = changed_df.reset_index().to_csv(index=False) 
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
                # meta 정보를 활용하여 예측값 표시
                counts_combined_display = counts_combined.copy()
                # counts_combined_display.index = counts_combined_display.index.astype(str) + '_' + counts_combined_display.index.astype(str).map(meta_dict)
                st.write(counts_combined_display[['wrong_count', 'rate_view', 'rate']])

            with col2:
                st.write("못 맞춘 정답 빈도, 틀린 비율: 잘못 예측한 실제값 수 / 전체 실제값 수")
                wrong_counts = changed_df['label'].value_counts() # 실제값 틀린 것 셈
                total_counts = merged_df['label'].value_counts() # 실제값 전체 셈
                counts_combined = pd.concat([wrong_counts, total_counts], axis=1, sort=False).fillna(0)
                counts_combined.columns = ['wrong_count', 'total_count']
                counts_combined['rate'] = counts_combined['wrong_count'] / counts_combined['total_count']
                counts_combined['rate_view'] = counts_combined.apply(lambda row: f"{int(row['wrong_count'])} / {int(row['total_count'])}", axis=1)
                # meta 정보를 활용하여 정답 표시
                counts_combined_display = counts_combined.copy()
                # counts_combined_display.index = counts_combined_display.index.astype(str) + '_' + counts_combined_display.index.astype(str).map(meta_dict)
                st.write(counts_combined_display[['wrong_count', 'rate_view', 'rate']])

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(merged_df['label'], merged_df['target'])

                # 레이블 unique 값 확인 및 중복 제거 후 숫자형으로 변환하여 정렬
                unique_labels = np.unique(merged_df[['label', 'target']].values)
                unique_labels = np.sort(unique_labels.astype(int))

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                # 레이블 설정 (meta 정보를 활용하여 표시, unique_labels_wrong 순서 유지)
                ax.set_xticklabels(unique_labels)
                ax.set_yticklabels(unique_labels)
                st.pyplot(fig)

            # label별 평가 지표 출력
            st.markdown("---")
            st.subheader("Label별 평가 지표")
            report = classification_report(merged_df['label'], merged_df['target'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.round(4)

            # meta 정보를 활용하여 레이블 표시, unique_labels 순서 유지
            report_df_display = report_df.copy()
            # report_df_display.index = report_df_display.index.astype(str).map(
            #     lambda x: x + '_' + meta_dict.get(int(x), '') if x.isdigit() else x
            # )
            # # unique_labels 순서대로 정렬
            # report_df_display = report_df_display.loc[[str(label) + '_' + meta_dict.get(label, '') for label in unique_labels]]

            st.dataframe(report_df_display)

        else:
            st.write("변경된 target 값이 없습니다.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        st.write(traceback.format_exc()) 

# 사이드바에 메시지 추가
st.sidebar.write("우측 메뉴에서 wide mode를 적용해주세요.")

st.title("CSV File Grader and Analyzer")

st.write("업로드할 CSV 파일을 선택하세요.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    answer_key = load_key(sheet_url)
    meta_key = load_key(meta_url)
    process_files(uploaded_file, answer_key, meta_key)
