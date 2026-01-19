import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi

# ---------------------------------------------------------
# 1. 데이터 다운로드 및 로드
# ---------------------------------------------------------

# Kaggle API 인증
api = KaggleApi()
api.authenticate()

# 데이터셋 다운로드
dataset_name = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
if not os.path.exists("ai4i2020.csv"):
    api.dataset_download_files(dataset_name, path="./", unzip=True)
    print("데이터셋 다운로드 완료.")
else:
    print("데이터셋 파일이 이미 존재합니다.")

# CSV 파일 읽기
data = pd.read_csv("ai4i2020.csv", encoding="utf-8")
data = data.drop(columns=["Product ID", "UDI", "Type"])

print("데이터 로드 완료. 데이터 크기:", data.shape)

# ---------------------------------------------------------
# 2. 데이터셋 분리 (고장 유형별)
# ---------------------------------------------------------

# 고장 유형
failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# 각 고장 유형별 데이터프레임
twf_df = data[data["TWF"] == 1]
hdf_df = data[data["HDF"] == 1] 
pwf_df = data[data["PWF"] == 1]
osf_df = data[data["OSF"] == 1]
rnf_df = data[data["RNF"] == 1]

# 전체 피처 리스트
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ---------------------------------------------------------
# 3. 수정된 시각화 함수 (스케일링 고정)
# ---------------------------------------------------------

def plot_features(df, title_prefix="", full_data=data):
    """
    df: 시각화할 데이터셋 (예: twf_df)
    title_prefix: 그래프 제목 접두사
    full_data: 전체 범위를 참조하기 위한 원본 데이터셋 (기본값: data)
    """
    # 데이터가 없는 경우 예외처리
    if len(df) == 0:
        print(f"[Skip] {title_prefix} - 데이터가 없습니다.")
        return

    plt.figure(figsize=(15, 8))

    for i, col in enumerate(FEATURES, start=1):
        if col not in df.columns:
            continue
            
        plt.subplot(2, 3, i)
        
        # 전체 데이터의 최소값, 최대값을 구함
        g_min = full_data[col].min()
        g_max = full_data[col].max()
        
        # range 파라미터를 사용하여 전체 범위 기준으로 히스토그램 빈(bin)을 생성
        df[col].hist(bins=30, edgecolor="black", range=(g_min, g_max))
        
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")
        
        # X축의 범위를 전체 데이터 기준으로 고정
        plt.xlim(g_min, g_max)

    plt.suptitle(f"{title_prefix} Feature Distributions (Scaled to Global Range)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ---------------------------------------------------------
# 4. 실행
# ---------------------------------------------------------

print("\n=== 1. TWF (Tool Wear Failure) ===")
plot_features(twf_df, title_prefix="TWF")

print("\n=== 2. HDF (Heat Dissipation Failure) ===")
plot_features(hdf_df, title_prefix="HDF")

print("\n=== 3. PWF (Power Failure) ===")
plot_features(pwf_df, title_prefix="PWF")

print("\n=== 4. OSF (Overstrain Failure) ===")
plot_features(osf_df, title_prefix="OSF")

print("\n=== 5. RNF (Random Failure) ===")
plot_features(rnf_df, title_prefix="RNF")