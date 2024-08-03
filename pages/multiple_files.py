import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"

@st.cache_data
def load_answer_key(url):
    response = requests.get(url)
    return pd.read_csv(StringIO(response.text))

def read_files(best_file, current_file):
    """Read CSV files and rename 'target' columns."""
    best_df = pd.read_csv(best_file)
    current_df = pd.read_csv(current_file)
    
    best_df.rename(columns={'target': 'target_best'}, inplace=True)
    current_df.rename(columns={'target': 'target_current'}, inplace=True)
    
    return best_df, current_df

def merge_dataframes(best_df, current_df, answer_key):
    """Merge best and current dataframes and then merge with answer key."""
    combined_df = pd.merge(best_df, current_df, on='ID', how='outer')
    merged_df = pd.merge(combined_df, answer_key, on='ID', how='left')
    return merged_df

def calculate_mismatch(merged_df):
    """Identify rows where target values mismatch with the label."""
    target_columns = ['target_best', 'target_current']
    mismatch_conditions = [merged_df[col] != merged_df['label'] for col in target_columns]
    merged_df['target_mismatch'] = pd.concat(mismatch_conditions, axis=1).any(axis=1)
    changed_df = merged_df[merged_df['target_mismatch']]
    return changed_df

def calculate_missing_values(changed_df):
    """Calculate missing values in the changed dataframe."""
    return changed_df[['target_best', 'target_current', 'label']].isna().sum()

def calculate_statistics(changed_df):
    """Calculate and return statistics for the changed dataframe."""
    total_rows = len(changed_df)
    unique_ids = changed_df['ID'].nunique()
    return total_rows, unique_ids
def calculate_auc(y_true, y_scores_best, y_scores_current):
    """Calculate AUC scores for the best and current predictions."""
    try:
        auc_best = roc_auc_score(y_true, y_scores_best, multi_class='ovr')
        auc_current = roc_auc_score(y_true, y_scores_current, multi_class='ovr')
    except ValueError as e:
        auc_best = None
        auc_current = None
        st.write(f"AUC 계산 중 오류 발생: {e}")
    
    return auc_best, auc_current

def calculate_metrics(y_true, y_scores):
    """Calculate precision, recall, and F1-score for given predictions."""
    metrics = {}
    averages = ['macro', 'micro', 'weighted']
    
    for avg in averages:
        try:
            metrics[f'precision_{avg}'] = precision_score(y_true, y_scores, average=avg)
            metrics[f'recall_{avg}'] = recall_score(y_true, y_scores, average=avg)
            metrics[f'f1_{avg}'] = f1_score(y_true, y_scores, average=avg)
        except ValueError as e:
            metrics[f'precision_{avg}'] = None
            metrics[f'recall_{avg}'] = None
            metrics[f'f1_{avg}'] = None
            st.write(f"평가지표 계산 중 오류 발생: {e}")
    
    return metrics

def display_auc_results(auc_best, auc_current):
    """Display AUC results."""
    st.write(f"Best File의 AUC: {auc_best:.2f}" if auc_best is not None else "Best File의 AUC를 계산할 수 없습니다.")
    st.write(f"Current File의 AUC: {auc_current:.2f}" if auc_current is not None else "Current File의 AUC를 계산할 수 없습니다.")

def display_metrics_results(metrics_best, metrics_current):
    """Display precision, recall, and F1-score results."""
    for avg in ['macro', 'micro', 'weighted']:
        st.write(f"Best File의 {avg.capitalize()} Precision: {metrics_best[f'precision_{avg}']:.2f}" if metrics_best[f'precision_{avg}'] is not None else f"Best File의 {avg.capitalize()} Precision을 계산할 수 없습니다.")
        st.write(f"Best File의 {avg.capitalize()} Recall: {metrics_best[f'recall_{avg}']:.2f}" if metrics_best[f'recall_{avg}'] is not None else f"Best File의 {avg.capitalize()} Recall을 계산할 수 없습니다.")
        st.write(f"Best File의 {avg.capitalize()} F1-Score: {metrics_best[f'f1_{avg}']:.2f}" if metrics_best[f'f1_{avg}'] is not None else f"Best File의 {avg.capitalize()} F1-Score를 계산할 수 없습니다.")
        
        st.write(f"Current File의 {avg.capitalize()} Precision: {metrics_current[f'precision_{avg}']:.2f}" if metrics_current[f'precision_{avg}'] is not None else f"Current File의 {avg.capitalize()} Precision을 계산할 수 없습니다.")
        st.write(f"Current File의 {avg.capitalize()} Recall: {metrics_current[f'recall_{avg}']:.2f}" if metrics_current[f'recall_{avg}'] is not None else f"Current File의 {avg.capitalize()} Recall을 계산할 수 없습니다.")
        st.write(f"Current File의 {avg.capitalize()} F1-Score: {metrics_current[f'f1_{avg}']:.2f}" if metrics_current[f'f1_{avg}'] is not None else f"Current File의 {avg.capitalize()} F1-Score를 계산할 수 없습니다.")

def process_evaluation(changed_df):
    """Process evaluation metrics for given dataframe."""
    if 'target_best' in changed_df.columns and 'label' in changed_df.columns:
        # 실제 클래스와 예측 클래스
        y_true = changed_df['label'].astype(int)
        y_scores_best = changed_df['target_best'].astype(int)
        y_scores_current = changed_df['target_current'].astype(int)
        
        # AUC 계산
        auc_best, auc_current = calculate_auc(y_true, y_scores_best, y_scores_current)
        display_auc_results(auc_best, auc_current)
        
        # Precision, Recall, F1-Score 계산
        metrics_best = calculate_metrics(y_true, y_scores_best)
        metrics_current = calculate_metrics(y_true, y_scores_current)
        display_metrics_results(metrics_best, metrics_current)
def display_results(changed_df):
    """Display various results for the changed dataframe."""
    if not changed_df.empty:
        st.write("정답이 틀린 항목에 대한 분석표입니다.")

        # 1. 병합되고 오답을 가진 행들로 이루어진 df의 출력
        st.write("1. 병합되고 오답을 가진 행들:")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(changed_df[['ID', 'target_best', 'target_current', 'label']])
        with col2:
            # 빈 칸인 값들의 수 출력
            missing_values_summary = calculate_missing_values(changed_df)
            total_rows, unique_ids = calculate_statistics(changed_df)
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

def process_files(best_file, current_file, answer_key):
    best_df, current_df = read_files(best_file, current_file)
    merged_df = merge_dataframes(best_df, current_df, answer_key)
    changed_df = calculate_mismatch(merged_df)
    display_results(changed_df)

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
