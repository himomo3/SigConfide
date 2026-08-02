import pandas as pd
import numpy as np
import os

csv_path = "comparison_output/synthetic2700_all/consensus_signatures.csv"
txt_path = "comparison_output/synthetic2700_all/consensus_signatures.txt"

# Read existing csv
df = pd.read_csv(csv_path, index_col=0)

# Clip negative values just in case
df = df.clip(lower=0.0)

# Normalize columns
col_sums = df.sum(axis=0)
col_sums[col_sums == 0] = 1.0
df = df / col_sums

# Save as tab-separated TXT which SigProfilerAssignment expects
df.to_csv(txt_path, sep='\t')
print(f"Fixed signatures and saved as tab-separated to {txt_path}.")
