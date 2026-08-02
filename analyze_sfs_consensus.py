import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_sbs96_mutation_types():
    std_types_list = []
    for base5 in ['A', 'C', 'G', 'T']:
        for mut in ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']:
            for base3 in ['A', 'C', 'G', 'T']:
                std_types_list.append(f"{base5}[{mut}]{base3}")
    return std_types_list

def analyze_sfs_consensus(output_dir="synthetic2700_all"):
    out_path = os.path.join("comparison_output", output_dir)
    global_data_path = os.path.join(out_path, "global_computed_data.npz")
    
    if not os.path.exists(global_data_path):
        print(f"Error: Data not found at {global_data_path}.")
        return

    print(f"Loading data from {global_data_path}...")
    data = np.load(global_data_path)
    
    P_original = data['P_original']
    P_sfs_whole_all = data['P_sfs_whole_all']
    sig_names = data['sig_names_filtered']

    print(f"Loaded signatures: {len(sig_names)}")
    print(f"P_sfs_whole_all shape: {P_sfs_whole_all.shape} (Features, Signatures, Iterations)")

    # 1. Compute consensus (average across all iterations)
    print("\nComputing consensus signatures across all SFS iterations...")
    P_consensus = np.mean(P_sfs_whole_all, axis=2)
    
    # Clip any potential negative values from floating point inaccuracies
    P_consensus = np.clip(P_consensus, 0.0, None)
    
    # Normalize columns so they sum to exactly 1.0 (expected by SigProfilerAssignment)
    col_sums = P_consensus.sum(axis=0)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1.0
    P_consensus = P_consensus / col_sums

    # 2. Compute Cosine Similarity between Original and Consensus
    cos_sims = []
    for i in range(P_original.shape[1]):
        p_orig = P_original[:, i]
        p_cons = P_consensus[:, i]
        
        norm_orig = np.linalg.norm(p_orig)
        norm_cons = np.linalg.norm(p_cons)
        
        if norm_orig == 0 or norm_cons == 0:
            sim = 0.0
        else:
            sim = np.dot(p_orig, p_cons) / (norm_orig * norm_cons)
            
        cos_sims.append(sim)

    # Save similarities to CSV
    sim_df = pd.DataFrame({
        "Signature": sig_names,
        "Cosine_Similarity": cos_sims
    })
    sim_csv_path = os.path.join(out_path, "consensus_vs_original_similarities.csv")
    sim_df.to_csv(sim_csv_path, index=False)
    print(f"Saved cosine similarities to {sim_csv_path}")

    print("\nCosine Similarities Summary:")
    print(sim_df.to_string(index=False))
    print(f"\nMean Cosine Similarity: {np.mean(cos_sims):.4f}")

    # 3. Save consensus signatures to CSV
    # We attempt to label the rows if it's SBS96
    if P_consensus.shape[0] == 96:
        row_labels = get_sbs96_mutation_types()
    else:
        row_labels = [f"Feature_{i}" for i in range(P_consensus.shape[0])]

    consensus_df = pd.DataFrame(P_consensus, index=row_labels, columns=sig_names)
    consensus_df.index.name = "Type"
    
    cons_csv_path = os.path.join(out_path, "consensus_signatures.txt")
    consensus_df.to_csv(cons_csv_path, sep='\t')
    print(f"Saved consensus signatures to {cons_csv_path}")

    # 4. Generate visual comparison (Difference Heatmap)
    print("Generating difference heatmap (Consensus - Original)...")
    P_diff = P_consensus - P_original
    
    fig, ax = plt.subplots(figsize=(12, 8))
    max_abs = np.max(np.abs(P_diff))
    if max_abs == 0:
        max_abs = 1.0
        
    im = ax.imshow(P_diff, cmap='coolwarm', vmin=-max_abs, vmax=max_abs, aspect='auto')
    ax.set_title("Difference Map: Consensus SFS - Original Signatures")
    ax.set_xlabel("Signatures")
    ax.set_ylabel("Mutation Types")
    
    ax.set_xticks(np.arange(len(sig_names)))
    ax.set_xticklabels(sig_names, rotation=90)
    
    # Only show some y-ticks if it's 96
    if P_diff.shape[0] == 96:
        step = 4
        ax.set_yticks(np.arange(0, 96, step))
        ax.set_yticklabels(row_labels[0::step], fontsize=8)
        
    fig.colorbar(im, ax=ax, label='Difference (Consensus - Original)')
    plt.tight_layout()
    
    diff_plot_path = os.path.join(out_path, "consensus_difference_heatmap.png")
    plt.savefig(diff_plot_path, dpi=200)
    plt.close()
    print(f"Saved difference heatmap to {diff_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find consensus modified signatures across all SFS iterations.")
    parser.add_argument("--output_dir", default="synthetic2700_all", help="Output directory containing the computed results")
    args = parser.parse_args()
    
    analyze_sfs_consensus(args.output_dir)
