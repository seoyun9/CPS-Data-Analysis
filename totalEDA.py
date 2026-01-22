import pandas as pd, numpy as np, os, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("ai4i2020.csv", encoding="utf-8")

# 파생변수 만들기 power랑 온도차이
df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power"] = (2*np.pi/60) * df["Rotational speed [rpm]"] * df["Torque [Nm]"]

# osf용 strain 변수 만들기 
df["Variant"] = df["Type"]
df["strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]  

labels = ["TWF","HDF","PWF","OSF","RNF"]

def plot_type_distribution(label_col):
    # Type(L/M/H) 분포: failure랑 non-failure
    fail = df[df[label_col] == 1]
    ok = df[df[label_col] == 0]
    cats = ["L","M","H"] # 카테고리

    fail_counts = fail["Type"].value_counts().reindex(cats).fillna(0)
    ok_counts = ok["Type"].value_counts().reindex(cats).fillna(0)

    x = np.arange(len(cats))
    width = 0.38 # 이게 젤 깔끔

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, ok_counts/len(ok), width, label="non-failure share")
    plt.bar(x + width/2, fail_counts/len(fail) if len(fail)>0 else 0, width, label="failure share")
    plt.xticks(x, cats)
    plt.ylabel("Proportion within group")
    plt.title(f"{label_col}: Type distribution (failure vs non-failure)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def plot_box(label_col, feature):
        # 피쳐 박스플롯: failure랑 non-failure
        fail = df[df[label_col] == 1][feature].astype(float)
        ok = df[df[label_col] == 0][feature].astype(float)

        plt.figure(figsize=(6,4))
        plt.boxplot([ok, fail], labels=["non-failure", "failure"], showfliers=False)
        plt.ylabel(feature)
        plt.title(f"{label_col}: {feature} (boxplot)")
        plt.tight_layout()
        plt.show()

    def plot_hist(label_col, feature, bins=35, density=True, logy=False):
        """feature 히스토그램: failure vs non-failure"""
        fail = df[df[label_col] == 1][feature].astype(float)
        ok = df[df[label_col] == 0][feature].astype(float)

        plt.figure(figsize=(6,4))
        plt.hist(ok, bins=bins, alpha=0.6, label="non-failure", density=density)
        plt.hist(fail, bins=bins, alpha=0.6, label="failure", density=density)
        plt.xlabel(feature)
        plt.ylabel("Density" if density else "Count")
        plt.title(f"{label_col}: {feature} (hist)")
        if logy:
            plt.yscale("log")
            plt.ylabel(("Density" if density else "Count") + " (log scale)")
            plt.title(f"{label_col}: {feature} (hist, log y)")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_failure_bundle(label_col, features, bins=35):
    """
    1) Type 분포
    2) features 각각에 대해 box + hist
    """
    plot_type_distribution(label_col)
    for f in features:
        plot_box(label_col, f)
        plot_hist(label_col, f, bins=bins, density=True, logy=False)


# 고장 별 시각화

# --- TWF: Tool wear 중심 ---
def plot_TWF(bins=35):
    plot_failure_bundle("TWF", ["Tool wear [min]", "Power"], bins=bins)

# --- HDF: Temp_diff + Torque ---
def plot_HDF(bins=35):
    plot_failure_bundle("HDF", ["Temp_diff [K]", "Torque [Nm]"], bins=bins)

# --- PWF: Torque + Power (요청 반영) ---
def plot_PWF(bins=35):
    plot_failure_bundle("PWF", ["Torque [Nm]", "Power"], bins=bins)

# --- RNF: 온도 쪽(설명력 약하지만 편향 확인용) ---
def plot_RNF(bins=35):
    plot_failure_bundle("RNF", ["Air temperature [K]", "Process temperature [K]"], bins=bins)


# OSF: strain + Variant별 비교(고장/비고장 같이)


def plot_OSF_basic(bins=35):
    # OSF 자체도 묶음으로 기본 비교
    plot_failure_bundle("OSF", ["strain", "Torque [Nm]"], bins=bins)

def plot_OSF_by_variant(variant, bins=40, logy=False):
    """
    Variant별로 나눠서, OSF 고장/비고장을 같이 비교 (strain)
    """
    ok = df[(df["Variant"]==variant) & (df["OSF"]==0)]["strain"].astype(float)
    fail = df[(df["Variant"]==variant) & (df["OSF"]==1)]["strain"].astype(float)

    # 히스토그램(겹치기)
    plt.figure(figsize=(7,5))
    plt.hist(ok, bins=bins, density=True, alpha=0.6, label="non-failure (OSF=0)")
    plt.hist(fail, bins=bins, density=True, alpha=0.6, label="failure (OSF=1)")
    if logy:
        plt.yscale("log")
        plt.ylabel("Density (log scale)")
        plt.title(f"Variant {variant}: strain (OSF=0 vs OSF=1, log y)")
    else:
        plt.ylabel("Density")
        plt.title(f"Variant {variant}: strain (OSF=0 vs OSF=1)")
    plt.xlabel("strain = Tool wear × Torque")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 박스플롯
    plt.figure(figsize=(6,4))
    plt.boxplot([ok, fail], labels=["non-failure", "failure"], showfliers=False)
    plt.ylabel("strain = Tool wear × Torque")
    plt.title(f"Variant {variant}: strain (boxplot)")
    plt.tight_layout()
    plt.show()

def plot_OSF_variants_all(bins=40, logy=False):
    for v in ["L","M","H"]:
        plot_OSF_by_variant(v, bins=bins, logy=logy)


# label alignment check

def check_label_alignment():
    """
    세부고장(OR)과 Machine failure 라벨이 완전히 일치하는지 확인
    """
    df_tmp = df.copy()
    df_tmp["Any_subfailure"] = (df_tmp[labels].sum(axis=1) > 0).astype(int)
    align_rate = (df_tmp["Any_subfailure"] == df_tmp["Machine failure"]).mean()
    mismatch = df_tmp[df_tmp["Any_subfailure"] != df_tmp["Machine failure"]][["Machine failure","Any_subfailure"]]
    print(f"Alignment rate (Any_subfailure == Machine failure): {align_rate:.4f}")
    print("Mismatch counts:")
    print(mismatch.value_counts())

# 실행 예시

# check_label_alignment()

# plot_TWF()
# plot_HDF()
# plot_PWF()
# plot_RNF()

# OSF 기본 비교 + Variant별 비교(로그축 옵션)
# plot_OSF_basic()
# plot_OSF_variants_all(logy=True)
