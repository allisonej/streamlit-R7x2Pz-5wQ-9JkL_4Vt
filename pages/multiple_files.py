import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics import precision_score, recall_score, f1_score

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"

@st.cache_data
def load_answer_key(url):
    """Load answer key from Google Sheets."""
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

def calculate_statistics(changed_df, answer_df, current_df):
    """Calculate and return statistics for the changed dataframe."""
    total_rows = len(changed_df)
    unique_ids = changed_df['ID'].nunique()
    answer_ids = answer_df['ID'].nunique()  # 정답지의 고유 ID 수
    current_ids = current_df['ID'].nunique()  # 현재 파일의 고유 ID 수
    return total_rows, unique_ids, answer_ids, current_ids

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

def display_metrics_results(metrics_best, metrics_current):
    """Display precision, recall, and F1-score results in columns."""
    st.write("평가지표 결과:")

    # 3개의 컬럼으로 결과 표시
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Macro 평균**")
        st.markdown(f"""
        **Best File:**
        - Precision: {metrics_best['precision_macro']:.3f}  
        - Recall: {metrics_best['recall_macro']:.3f}  
        - **F1-Score: {metrics_best['f1_macro']:.3f}**  
        
        **Current File:**
        - Precision: {metrics_current['precision_macro']:.3f}  
        - Recall: {metrics_current['recall_macro']:.3f}  
        - **F1-Score: {metrics_current['f1_macro']:.3f}**  

        **Macro 평균**은 각 클래스의 지표를 개별적으로 계산한 후, 그 평균을 구하는 방법입니다.
        이 방법은 클래스 간 불균형을 무시하고 각 클래스의 중요성을 동등하게 고려합니다. 즉, 각 클래스가 동등한 가중치를 가지며 평가됩니다.
        
        **장점**:
        - 클래스 불균형이 있는 경우에도 각 클래스를 동등하게 평가합니다.
        - 각 클래스의 성능을 개별적으로 평가할 수 있습니다.
        
        **단점**:
        - 데이터 샘플 수가 적은 클래스의 성능이 과도하게 부각될 수 있습니다.
        """ if all(v is not None for v in metrics_best.values()) and all(v is not None for v in metrics_current.values()) else """
        **Best File:**
        - Precision: Precision 계산 불가  
        - Recall: Recall 계산 불가  
        - F1-Score: F1-Score 계산 불가  
        
        **Current File:**
        - Precision: Precision 계산 불가  
        - Recall: Recall 계산 불가  
        - F1-Score: F1-Score 계산 불가  
        """)

    with col2:
        st.write("**Micro 평균**")
        st.markdown(f"""
        **Best File:**
        - Precision: {metrics_best['precision_micro']:.3f}  
        - Recall: {metrics_best['recall_micro']:.3f}  
        - F1-Score: {metrics_best['f1_micro']:.3f}  
        
        **Current File:**
        - Precision: {metrics_current['precision_micro']:.3f}  
        - Recall: {metrics_current['recall_micro']:.3f}  
        - F1-Score: {metrics_current['f1_micro']:.3f}  

        **Micro 평균**은 전체 데이터의 지표를 계산하여, 각 클래스의 지표를 집계하는 방법입니다.
        이 방법은 데이터 샘플의 총합에 기반하여 지표를 계산하므로, 클래스 간의 불균형에 영향을 받지 않습니다.
        
        **장점**:
        - 모든 클래스의 데이터 샘플을 고려하여 평균을 계산하므로, 클래스 불균형 문제에 영향을 받지 않습니다.
        - 전체 데이터의 성능을 종합적으로 평가할 수 있습니다.
        
        **단점**:
        - 특정 클래스의 성능이 전체 성능에 덜 반영될 수 있습니다.
        """ if all(v is not None for v in metrics_best.values()) and all(v is not None for v in metrics_current.values()) else """
        **Best File:**
        - Precision: Precision 계산 불가  
        - Recall: Recall 계산 불가  
        - F1-Score: F1-Score 계산 불가  
        
        **Current File:**
        - Precision: Precision 계산 불가  
        - Recall: Recall 계산 불가  
        - F1-Score: F1-Score 계산 불가  
        """)

    with col3:
        st.write("**Weighted 평균**")
        st.markdown(f"""        
        **Best File:**
        - Precision: {metrics_best['precision_weighted']:.3f}  
        - Recall: {metrics_best['recall_weighted']:.3f}  
        - F1-Score: {metrics_best['f1_weighted']:.3f}  
        
        **Current File:**
        - Precision: {metrics_current['precision_weighted']:.3f}  
        - Recall: {metrics_current['recall_weighted']:.3f}  
        - F1-Score: {metrics_current['f1_weighted']:.3f}  

        **Weighted 평균**은 각 클래스의 지표를 계산한 후, 클래스의 샘플 수에 따라 가중 평균을 구하는 방법입니다.
        이 방법은 클래스의 샘플 수를 고려하여 지표를 조정합니다. 클래스 불균형을 반영하여, 더 많은 샘플을 가진 클래스에 더 많은 가중치를 부여합니다.
        
        **장점**:
        - 클래스 샘플 수에 비례하여 지표를 조정하므로, 클래스 불균형 문제를 완화할 수 있습니다.
        - 데이터의 실제 분포를 반영한 평가가 가능합니다.
        
        **단점**:
        - 샘플 수가 적은 클래스의 성능이 반영되지 않을 수 있습니다.
        """ if all(v is not None for v in metrics_best.values()) and all(v is not None for v in metrics_current.values()) else """
        **Best File:**
        - Precision: Precision 계산 불가  
        - Recall: Recall 계산 불가  
        - F1-Score: F1-Score 계산 불가  
        
        **Current File:**
        - Precision: Precision 계산 불가  
        - Recall: Recall 계산 불가  
        - F1-Score: F1-Score 계산 불가  
        """)

def process_evaluation(changed_df):
    """Process evaluation metrics for given dataframe."""
    if 'target_best' in changed_df.columns and 'label' in changed_df.columns:
        # 실제 클래스와 예측 클래스
        y_true = changed_df['label'].astype(int)
        y_scores_best = changed_df['target_best'].astype(int)
        y_scores_current = changed_df['target_current'].astype(int)
        
        # Precision, Recall, F1-Score 계산
        metrics_best = calculate_metrics(y_true, y_scores_best)
        metrics_current = calculate_metrics(y_true, y_scores_current)
        display_metrics_results(metrics_best, metrics_current)

def display_results(changed_df, answer_df, current_df):
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
            total_rows, unique_ids, answer_ids, current_ids = calculate_statistics(changed_df, answer_df, current_df)
            st.write("빈 칸인 값들의 수:")
            st.write(missing_values_summary)
            st.write("전체 행의 수:", total_rows)
            st.write("고유한 ID의 수:", unique_ids)
            st.write("정답지의 고유한 ID 수:", answer_ids)
            st.write("현재 파일의 고유한 ID 수:", current_ids)

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
            })
            st.write(common_targets)

            # NaN 값 포함하는 행 확인
            na_rows = common_targets[common_targets.isna().any(axis=1)]
            st.write("NaN 값을 포함하는 행:")
            st.write(na_rows)

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
    return changed_df, best_df, current_df

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

# 탭 생성
tabs = st.tabs(["평가지표", "통계표"])

if best_file and current_file:
    if st.button("Process Files"):
        answer_key = load_answer_key(sheet_url)
        changed_df, best_df, current_df = process_files(best_file, current_file, answer_key)

        with tabs[0]:
            st.header("평가지표")
            process_evaluation(changed_df)
        
        with tabs[1]:
            st.header("통계표")
            display_results(changed_df, answer_key, current_df)
