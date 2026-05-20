import pandas as pd
import numpy as np

def main():
    # Load the ground truth exposures
    file_path = 'tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/ground.truth.syn.exposures.csv'
    
    try:
        df = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Make sure you are running this script from the root of the SigConfide repository.")
        return

    # Calculate prevalence (percentage of samples where signature is > 0)
    prevalence = (df > 0).mean(axis=1).sort_values(ascending=False)
    print("--- PREVALENCE (Percentage of samples with exposure > 0) ---")
    print(prevalence.to_string())
    print("\n")

    # Calculate Pearson correlation of raw exposures
    corr_exposure = df.T.corr(method='pearson')

    # Calculate Pearson correlation of occurrences (binary presence/absence)
    binary_df = (df > 0).astype(int)
    corr_occurrence = binary_df.T.corr(method='pearson')

    print("--- RAW EXPOSURE CORRELATIONS ---")
    for s1, s2 in [('SBS18', 'SBS26'), ('SBS22', 'SBS26')]:
        if s1 in df.index and s2 in df.index:
            print(f"Correlation between {s1} and {s2}: {corr_exposure.loc[s1, s2]:.4f}")
        else:
            print(f"One of {s1}, {s2} missing.")
            
    print("\n--- OCCURRENCE (BINARY) CORRELATIONS ---")
    for s1, s2 in [('SBS18', 'SBS26'), ('SBS22', 'SBS26')]:
        if s1 in df.index and s2 in df.index:
            print(f"Correlation between {s1} and {s2}: {corr_occurrence.loc[s1, s2]:.4f}")

    # Check the top correlated signatures for SBS26
    if 'SBS26' in corr_occurrence.index:
        print("\n--- TOP CORRELATIONS FOR SBS26 (BINARY) ---")
        print(corr_occurrence['SBS26'].sort_values(ascending=False).head())

if __name__ == '__main__':
    main()
