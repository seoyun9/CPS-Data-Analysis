import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# ==========================================
# 1. Load and Preprocess Data
# ==========================================
def load_and_preprocess_data():
    # Kaggle API Authentication and Download
    api = KaggleApi()
    api.authenticate()
    dataset_name = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"

    if not os.path.exists("ai4i2020.csv"):
        api.dataset_download_files(dataset_name, path="./", unzip=True)
        print("✅ Dataset downloaded.")
    else:
        print("✅ Dataset file already exists.")

    # Read CSV
    df = pd.read_csv("ai4i2020.csv")
    
    # Drop unnecessary identifier columns (Keep Type)
    df = df.drop(columns=["Product ID", "UDI"])
    
    # -- Feature Engineering --
    # 1. [For HDF] Temp diff
    df['Temp diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # 2. [For PWF] Power = Torque * Rotational speed (rad/s)
    df['Power [W]'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * (2 * np.pi / 60))
    
    # 3. [For OSF] Strain = Tool wear * Torque
    df['Strain [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']
    
    print(f"✅ Data load and feature engineering complete. (Shape: {df.shape})")
    return df

# ==========================================
# 2. Visualization Functions
# ==========================================

BASIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

def plot_general_failure_analysis(df, failure_code, additional_features=None):
    """
    Analysis for general failures (TWF, HDF, PWF, RNF) excluding OSF.
    Threshold visualization logic has been removed.
    """
    subset = df[df[failure_code] == 1]
    if len(subset) == 0:
        print(f"[Skip] No data for {failure_code}.")
        return

    # Combine features to plot
    features_to_plot = BASIC_FEATURES.copy()
    if additional_features:
        features_to_plot.extend(additional_features)
    
    # Graph Layout
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    for i, col in enumerate(features_to_plot, start=1):
        plt.subplot(n_rows, n_cols, i)
        
        # Fixed scale (based on entire dataset)
        g_min = df[col].min()
        g_max = df[col].max()
        
        # Histogram
        subset[col].hist(bins=30, range=(g_min, g_max), 
                         edgecolor='black', color='salmon', alpha=0.9, label=f'{failure_code}')
        
        plt.title(col)
        plt.xlim(g_min, g_max)
        plt.legend(loc='upper right')

    plt.suptitle(f"Failure Analysis: {failure_code}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_osf_full_analysis(df):
    """
    OSF (Overstrain Failure) specific analysis.
    Threshold visualization logic has been removed.
    """
    osf_data = df[df['OSF'] == 1]
    if len(osf_data) == 0:
        print("[Skip] No OSF data.")
        return
        
    # Layout: 2 rows, 4 columns
    fig = plt.figure(figsize=(24, 10))
    
    # --- Part 1: Basic 5 Features (All OSF data) ---
    for i, col in enumerate(BASIC_FEATURES, start=1):
        ax = fig.add_subplot(2, 4, i)
        
        g_min = df[col].min()
        g_max = df[col].max()
        
        osf_data[col].hist(bins=30, range=(g_min, g_max), 
                           edgecolor='black', color='salmon', alpha=0.9, ax=ax, label='OSF')
        
        ax.set_title(col)
        ax.set_xlim(g_min, g_max)
        ax.legend(loc='upper right')

    # --- Part 2: Strain by Type (L, M, H) ---
    types = ['L', 'M', 'H']
    
    # Unify Strain Axis
    s_min = df['Strain [minNm]'].min()
    s_max = df['Strain [minNm]'].max()

    for i, t in enumerate(types, start=6): # Positions 6, 7, 8
        ax = fig.add_subplot(2, 4, i)
        subset = osf_data[osf_data['Type'] == t]
        
        if len(subset) > 0:
            subset['Strain [minNm]'].hist(bins=20, range=(s_min, s_max), 
                                          edgecolor='black', color='salmon', alpha=0.9, ax=ax, label=f'OSF Type {t}')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
            
        ax.set_title(f"[Type {t}] Strain Analysis")
        ax.set_xlabel("Strain [minNm]")
        ax.set_xlim(s_min, s_max)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("OSF Analysis (Basic Features + Type-specific Strain)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# 3. Main Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Data Preparation
    data = load_and_preprocess_data()
    
    # 2. Analysis Execution (Threshold arguments removed)
    
    # (1) TWF
    print("\n=== 1. TWF Analysis ===")
    plot_general_failure_analysis(data, 'TWF')

    # (2) HDF
    print("\n=== 2. HDF Analysis ===")
    plot_general_failure_analysis(
        data, 'HDF', 
        additional_features=['Temp diff [K]']
    )

    # (3) PWF
    print("\n=== 3. PWF Analysis ===")
    plot_general_failure_analysis(
        data, 'PWF', 
        additional_features=['Power [W]']
    )

    # (4) OSF
    print("\n=== 4. OSF Analysis (Full Detail) ===")
    plot_osf_full_analysis(data)

    # (5) RNF
    print("\n=== 5. RNF Analysis ===")
    plot_general_failure_analysis(data, 'RNF')