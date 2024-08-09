import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Google Sheets URL (공개 CSV 다운로드 링크)
sheet_url = "https://docs.google.com/spreadsheets/d/1xq_b1XDCdSTHLjaeg4Oy9WWMQDbBLM397BD8AaWmGU0/export?gid=1096947070&format=csv"
# meta_url = "https://docs.google.com/spreadsheets/d/1y-2ZLNxR7FzwqmCY5powZZkyYva7qOM2-Y1HnP2m248/export?format=csv"

@st.cache_data
def load_key(url):
    # 구글 시트에서 정답 데이터를 CSV로 읽어오기
    response = requests.get(url)
    key = pd.read_csv(StringIO(response.text))
    return key

# @st.cache_data
# def map_target_to_text(target_value, meta_key):
#     """Map target or label value to its corresponding text in the format 'target_value_translation'."""
#     mapping = meta_key.set_index('target')['translation'].to_dict()
#     translation = mapping.get(target_value, 'Unknown')
#     return f"{target_value}_{translation}"


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
        - Precision: {metrics_best['precision_macro']:.4f}  
        - Recall: {metrics_best['recall_macro']:.4f}  
        - **F1-Score: {metrics_best['f1_macro']:.4f}**  
        
        **Current File:**
        - Precision: {metrics_current['precision_macro']:.4f}  
        - Recall: {metrics_current['recall_macro']:.4f}  
        - **F1-Score: {metrics_current['f1_macro']:.4f}**  

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
        - Precision: {metrics_best['precision_micro']:.4f}  
        - Recall: {metrics_best['recall_micro']:.4f}  
        - F1-Score: {metrics_best['f1_micro']:.4f}  
        
        **Current File:**
        - Precision: {metrics_current['precision_micro']:.4f}  
        - Recall: {metrics_current['recall_micro']:.4f}  
        - F1-Score: {metrics_current['f1_micro']:.4f}  

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
        - Precision: {metrics_best['precision_weighted']:.4f}  
        - Recall: {metrics_best['recall_weighted']:.4f}  
        - F1-Score: {metrics_best['f1_weighted']:.4f}  
        
        **Current File:**
        - Precision: {metrics_current['precision_weighted']:.4f}  
        - Recall: {metrics_current['recall_weighted']:.4f}  
        - F1-Score: {metrics_current['f1_weighted']:.4f}  

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

def process_evaluation(merged_df):
    """Process evaluation metrics for given dataframe."""
    if 'target_best' in merged_df.columns and 'label' in merged_df.columns:
        # 실제 클래스와 예측 클래스
        y_true =  merged_df['label'].astype(int)
        y_scores_best = merged_df['target_best'].astype(int)
        y_scores_current = merged_df['target_current'].astype(int)
        
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
            missing_values_summary = changed_df[['target_best', 'target_current', 'label']].isna().sum()
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
            }).fillna(0).astype(int).sort_values(by='Best File Count', ascending=False) # 툭정 값으로 예상한것중 틀린게 없을 경우 0으로 채움 (오류 해결)
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
        pair_counts = changed_df.groupby(['target_best', 'target_current', 'label']).size().reset_index(name='count')
        st.dataframe(pair_counts.sort_values(by='count', ascending=False))
    else:
        st.write("변경된 target 값이 없습니다.")
    
# Streamlit 앱의 레이아웃 설정
st.set_page_config(page_title="CSV File Grader and Analyzer", layout="wide")

# 사이드바에 메시지 추가
st.sidebar.write("우측 메뉴에서 wide mode를 적용해주세요.")

st.title("CSV File Grader and Analyzer")

st.write("업로드할 CSV 파일들을 선택하세요.")

answer_key = load_key(sheet_url)
# meta_key = load_key(meta_url)

# 파일 업로드
col1, col2 = st.columns(2)

with col1:
    best_file = st.file_uploader("Upload Best File (CSV)", type="csv")
    if best_file:
        best_df = pd.read_csv(best_file)
        best_df.rename(columns={'target': 'target_best'}, inplace=True)
        merged_df = pd.merge(best_df, answer_key, on='ID', how='outer')
        mismatched_df = merged_df[merged_df['target_best'] != merged_df['label']]
        st.write("틀린 수 : ", len(mismatched_df))
with col2:
    current_file = st.file_uploader("Upload Current File (CSV)", type="csv")
    if current_file:
        current_df = pd.read_csv(current_file)
        current_df.rename(columns={'target': 'target_current'}, inplace=True)
        merged_df = pd.merge(current_df, answer_key, on='ID', how='outer')
        mismatched_df = merged_df[merged_df['target_current'] != merged_df['label']]
        st.write("틀린 수 : ", len(mismatched_df))

if best_file and current_file:

    combined_df = pd.merge(best_df, current_df, on='ID', how='outer')
    merged_df = pd.merge(combined_df, answer_key, on='ID', how='left')

    target_columns = ['target_best', 'target_current']
    mismatch_conditions = [merged_df[col] != merged_df['label'] for col in target_columns]
    merged_df['target_mismatch'] = pd.concat(mismatch_conditions, axis=1).any(axis=1)
    changed_df = merged_df[merged_df['target_mismatch']]

    # # meta_key를 적용하여 변환
    # for column in target_columns + ['label']:
    #     merged_df.loc[:, f'{column}_text'] = merged_df[column].apply(lambda x: map_target_to_text(x, meta_key))
    #     changed_df.loc[:, f'{column}_text'] = changed_df[column].apply(lambda x: map_target_to_text(x, meta_key))


    # 탭 생성
    tabs = st.tabs(["평가지표", "통계표", "데이터 시각화", "데이터 필터링", 'IDs'])

    with tabs[0]:
        st.header("평가지표")
        process_evaluation(merged_df)
    
    with tabs[1]:
        st.header("통계표")
        display_results(changed_df, answer_key, current_df)

    with tabs[2]:
        st.header("그래프")
        st.markdown(
            """
            ##### **그래프 설명:**

            - **target_best** (count wrong targets) : best file에서 틀린 항목들의 'target'별 수를 나타내는 빨간색 막대입니다.
            - **target_current** (count wrong targets) : current file에서 틀린 항목들의 'target'별 수를 나타내는 파란색 막대입니다.
            - **best_label** (count wrong labels) : best file에서 틀린 항목들의 'label' 분포를 나타내는 연보라색 막대입니다.
            - **current_label** (count wrong labels) : current file에서 틀린 항목들의 'label' 분포를 나타내는 하늘색 막대입니다.
            - **label** (count whole wrong ID' wrong labels) : 틀린 ID들의 'label' 값 전체 분포(두 파일의 합집합)를 나타내는 보라색 막대입니다.
            """
        )
        plt.figure(figsize=(14, 7))

        # 예시 데이터 생성
        labels = list(range(17))  # 0부터 16까지의 항목

        # 데이터 준비
        a = changed_df['target_best'].value_counts().reindex(labels, fill_value=0)
        b = changed_df['target_current'].value_counts().reindex(labels, fill_value=0)
        c = changed_df[changed_df['target_best'] != changed_df['label']]['label'].value_counts().reindex(labels, fill_value=0)
        d = changed_df[changed_df['target_current'] != changed_df['label']]['label'].value_counts().reindex(labels, fill_value=0)
        e = changed_df['label'].value_counts().reindex(labels, fill_value=0)

        # 막대의 위치와 너비 설정
        x = np.arange(len(labels))
        width = 0.15  # 막대 너비

        # 그룹화된 막대그래프를 그리기 위한 위치 설정
        fig, ax = plt.subplots(figsize=(14, 7))
        rects1 = ax.bar(x - width, a, width, label='target_best', color='red')
        rects2 = ax.bar(x, b, width, label='target_current', color='blue')
        rects3 = ax.bar(x + width, c, width, label='best_label', color='violet')
        rects4 = ax.bar(x + width*2, d, width, label='current_label', color='dodgerblue')
        rects5 = ax.bar(x + width*3, e, width, label='label', color='purple')

        # 레이블, 제목 및 범례 설정
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of target_best, target_current, and label')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(title='Category')

        # 막대그래프가 잘 보이도록 Y축 범위를 설정
        ax.set_ylim(0, max(max(a), max(b), max(c), max(d)) + 1)

        # 그래프 표시
        st.pyplot(fig)

    with tabs[3]:
        st.header("레이블 필터링")
        unique_labels = sorted(changed_df['label'].dropna().unique())
        selected_label = st.selectbox('Select Actual Label for Filtering', options=unique_labels)
        filtered_df = changed_df[changed_df['label'] == selected_label]
        filtered_best_df = filtered_df[filtered_df['label'] != filtered_df['target_best']]
        filtered_current_df = filtered_df[filtered_df['label'] != filtered_df['target_current']]
        st.write(f"Filtered data for label: {selected_label}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Filted data : **")
            st.dataframe(filtered_df[['ID', 'target_best', 'target_current', 'label']])
        with col2:
            st.markdown("**Filted best data : ** best가 틀린 데이터")
            st.dataframe(filtered_best_df[['ID', 'target_best', 'target_current', 'label']])
        with col3:
            st.markdown("**Filted current data : ** current가 틀린 데이터")
            st.dataframe(filtered_current_df[['ID', 'target_best', 'target_current', 'label']])

    with tabs[4]:
        st.header("IDs")

        # Table A: 'target_best' != 'label' and 'target_current' != 'label'
        table_a = changed_df[(changed_df['target_best'] != changed_df['label']) & 
                            (changed_df['target_current'] != changed_df['label'])]
        table_a_ids = table_a['ID'].tolist()
        st.markdown("""
        **테이블 A IDs (target_best와 target_current가 모두 label과 다른 데이터):**
        """)
        st.write(table_a)
        st.download_button(label="테이블 A IDs 복사", data="\n".join(map(str, table_a_ids)), file_name="table_a_ids.txt", mime="text/plain")

        # Table B: 'target_best' == 'label' or 'target_current' == 'label'
        table_b = changed_df[(changed_df['target_best'] == changed_df['label']) | 
                            (changed_df['target_current'] == changed_df['label'])]
        table_b_ids = table_b['ID'].tolist()
        st.markdown("""
        **테이블 B IDs (target_best 또는 target_current가 label과 같은 데이터):**
        """)
        st.write(table_b)
        st.download_button(label="테이블 B IDs 복사", data="\n".join(map(str, table_b_ids)), file_name="table_b_ids.txt", mime="text/plain")

        # Table C: 'target_best' == 'target_current'
        table_c = changed_df[changed_df['target_best'] == changed_df['target_current']]
        table_c_ids = table_c['ID'].tolist()
        st.markdown("""
        **테이블 C IDs (target_best와 target_current가 같은 데이터):**
        """)
        st.write(table_c)
        st.download_button(label="테이블 C IDs 복사", data="\n".join(map(str, table_c_ids)), file_name="table_c_ids.txt", mime="text/plain")
        
        
