import pandas as pd
import numpy as np


df = pd.read_csv("ai4i2020.csv")
# Rotational speed는 1380 rpm에서 분포가 급변하며 상한선처럼 작동함
# 따라서 1380 rpm 미만을 HDF 조건 중 하나로 사용
df["HDF_rule"] = (
    (df["Process temperature [K]"] - df["Air temperature [K]"] < 8.6) &
    (df["Rotational speed [rpm]"] < 1380)
)
#수치형 컬럼 목록 만들기 (최소)
num_cols = df.select_dtypes(include="number").columns.tolist()

# 수치형 변수들 간의 상관관계 중 유의미한 관계만 추출하는 함수 정의
def meaningful_correlations(df, cols, threshold=0.4):
    """
    상관계수가 threshold 이상인 변수쌍만 반환
    """
    corr = df[cols].corr()
    
    result = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)): # 두 번째 변수 인덱스를 첫 번째 이후부터 반복하여 중복 계산을 방지함
            value = corr.iloc[i, j]
            if abs(value) >= threshold:  # 상관계수의 절댓값이 기준값 이상이면 유의미하다고 판단함
                result.append((cols[i], cols[j], round(value, 2)))
    
    return result # 유의미한 상관관계 목록을 반환함

meaningful_correlations(df, num_cols) # 수치형 변수들(num_cols) 중에서 유의미한 상관관계만 계산하여 출력함