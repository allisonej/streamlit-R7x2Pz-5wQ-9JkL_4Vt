import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"

@st.cache
def load_answer_key(url):
    response = requests.get(url)
    return pd.read_csv(StringIO(response.text))

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
    
    # 업로드된 CSV의 ID 수와 정답지의 ID 수
    num_best_ids = best_df['ID'].nunique()
    num_current_ids = current_df['ID'].nunique()
    num_answer_key_ids = answer_key['ID'].nunique()
    
    st.write(f"업로드된 Best File의 ID 수: {num_best_ids}")
    st.write(f"업로드된 Current File의 ID 수: {num_current_ids}")
    st.write(f"정답지의 ID 수: {num_answer_key_ids}")

    # 'target' 값이 'label'과 다른 행 필터링
    target_columns = ['target_best', 'target_current']
    mismatch_conditions = [merged_df[col] != merged_df['label'] for col in target_columns]
    merged_df['target_mismatch'] = pd.concat(mismatch_conditions, axis=1).any(axis=1)
    changed_df = merged_df[merged_df['target_mismatch']]
    
    # 빈 칸인 값들의 수 계산
    missing_values_summary = changed_df[['target_best', 'target_current', 'label']].isna().sum()
    
    # 전체 행의 수 및 고유한 ID 수
    total_rows = len(changed_df)
    unique_ids = changed_df['ID'].nunique()
    
    # AUC 계산
    if 'target_best' in changed_df.columns and 'label' in changed_df.columns:
        y_true = (changed_df['label'] == 'positive').astype(int)  # 'positive'를 양성 클래스로 가정
        y_scores_best = (changed_df['target_best'] == 'positive').astype(int)
        y_scores_current = (changed_df['target_current'] == 'positive').astype(int)
        
        try:
            auc_best = roc_auc_score(y_true, y_scores_best)
            auc_current = roc_auc_score(y_true, y_scores_current)
        except ValueError:
            auc_best = None
            auc_current = None
        
        st.write(f"Best File의 AUC: {auc_best:.2f}" if auc_best is not None else "Best File의 AUC를 계산할 수 없습니다.")
        st.write(f"Current File의 AUC: {auc_current:.2f}" if auc_current is not None else "Current File의 AUC를 계산할 수 없습니다.")
    
    # Precision, Recall, F1-Score 계산
    if 'target_best' in changed_df.columns and 'label' in changed_df.columns:
        try:
            precision_best = precision_score(y_true, y_scores_best)
            recall_best = recall_score(y_true, y_scores_best)
            f1_best = f1_score(y_true, y_scores_best)
            
            precision_current = precision_score(y_true, y_scores_current)
            recall_current = recall_score(y_true, y_scores_current)
            f1_current = f1_score(y_true, y_scores_current)
        except ValueError:
            precision_best = recall_best = f1_best = precision_current = recall_current = f1_current = None
        
        st.write(f"Best File의 Precision: {precision_best:.2f}" if precision_best is not None else "Best File의 Precision을 계산할 수 없습니다.")
        st.write(f"Best File의 Recall: {recall_best:.2f}" if recall_best is not None else "Best File의 Recall을 계산할 수 없습니다.")
        st.write(f"Best File의 F1-Score: {f1_best:.2f}" if f1_best is not None else "Best File의 F1-Score를 계산할 수 없습니다.")
        
        st.write(f"Current File의 Precision: {precision_current:.2f}" if precision_current is not None else "Current File의 Precision을 계산할 수 없습니다.")
        st.write(f"Current File의 Recall: {recall_current:.2f}" if recall_current is not None else "Current File의 Recall을 계산할 수 없습니다.")
        st.write(f"Current File의 F1-Score: {f1_current:.2f}" if f1_current is not None else "Current File의 F1-Score를 계산할 수 없습니다.")
    
    # Confusion Matrix 계산
    if 'target_best' in changed_df.columns and 'label' in changed_df.columns:
        cm_best = confusion_matrix(y_true, y_scores_best, labels=[0, 1])
        cm_current = confusion_matrix(y_true, y_scores_current, labels=[0, 1])
        
        st.write("Confusion Matrix for Best File:")
        cm_display_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=['Negative', 'Positive'])
        st.pyplot(cm_display_best.plot(include_values=True, cmap='Blues').figure)
        
        st.write("Confusion Matrix for Current File:")
        cm_display_current = ConfusionMatrixDisplay(confusion_matrix=cm_current, display_labels=['Negative', 'Positive'])
        st.pyplot(cm_display_current.plot(include_values=True, cmap='Blues').figure)
    
    # 분석 결과 출력
    if not changed_df.empty:
        st.write("정답이 틀린 항목에 대한 분석표입니다.")

        # 1. 병합되고 오답을 가진 행들로 이루어진 df의 출력
        st.write("1. 병합되고 오답을 가진 행들:")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(changed_df[['ID', 'target_best', 'target_current', 'label']])
        with col2:
            # 빈 칸인 값들의 수 출력
            st.write("빈 칸인 값들의 수:")
            st.write(missing_values_summary)
            st.write("전체 행의 수:", total_rows)
            st.write("고유한 ID의 수:", unique_ids)
        
        # 2. 틀린 예측값 빈도수
        st.write("2. 틀린 예측값 빈도수:")
        col1, col2, col3 = st.columns(3)
        target_counts_best = changed_df['target_best'].value_counts().sort_values(ascending=False)
        target_counts_current = changed_df['target_current'].value_counts().sort_values(ascending=False)
        with col1:
            st.write("Best File에서의 빈도수:")
            st.write(target_counts_best)
        with col2:
            st.write("Current File에서의 빈도수:")
            st.write(target_counts_current)
        with col3:
            common_targets = pd.DataFrame({
                'Best File Count': target_counts_best,
                'Current File Count': target_counts_current
            }).astype(int).sort_values(by='Best File Count', ascending=False)
            st.write(common_targets)
        
        # 3. 못 맞춘 정답 빈도수
        st.write("3. 못 맞춘 정답 빈도수:")
        col1, col2, col3 = st.columns(3)
        with col1:
            # target_best에서 label과 다른 값들의 수
            wrong_label_best = changed_df[changed_df['target_best'] != changed_df['label']]['label'].value_counts().sort_values(ascending=False)
            st.write("Best File에서의 빈도수:")
            st.write(wrong_label_best)
        with col2:
            # target_current에서 label과 다른 값들의 수
            wrong_label_current = changed_df[changed_df['target_current'] != changed_df['label']]['label'].value_counts().sort_values(ascending=False)
            st.write("Current File에서의 빈도수:")
            st.write(wrong_label_current)
        with col3:
            # 전체적으로 틀린 label 수
            all_wrong_labels = changed_df['label'].value_counts().sort_values(ascending=False)
            st.write("전체적으로 틀린 label 수:")
            st.write(all_wrong_labels)
        
        # 4. target_best, target_current, label 조합의 빈도수
        st.write("4. target_best, target_current, label 조합의 빈도수:")
        pair_counts = changed_df.groupby(['target_best', 'target_current', 'label']).size().reset_index(name='Count')
        st.dataframe(pair_counts.sort_values(by='Count', ascending=False))
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
