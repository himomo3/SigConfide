import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualise_comparison_synthetic(output_dir="comparison_output"):
    out_path = os.path.join(output_dir, "synthetic2700_all")
    global_data_path = os.path.join(out_path, "global_computed_data.npz")
    
    if not os.path.exists(global_data_path):
        print(f"Data not found at {global_data_path}. Skipping.")
        return
        
    data = np.load(global_data_path)
    
    E_truth_whole_all = data['E_truth_whole_all']
    E_truth_sums = E_truth_whole_all.sum(axis=0)
    E_truth_whole_all = np.divide(E_truth_whole_all, E_truth_sums, out=np.zeros_like(E_truth_whole_all), where=E_truth_sums!=0)
    
    E_opt_whole_all = data['E_opt_whole_all']
    E_reg_whole_all = data['E_reg_whole_all']
    E_bs_whole_all = data['E_bs_whole_all']
    E_boot_pois_whole = data['E_boot_pois_whole']
    
    try:
        E_spa_whole_all = data['E_spa_whole_all']
        has_spa = True
    except KeyError:
        has_spa = False
    P_original = data['P_original']
    P_sfs_whole_all = data['P_sfs_whole_all']
    M_norm = data['M_norm']
    kl_errors_sfs_whole_all = data['kl_errors_sfs_whole_all']
    n_reg_sfs_whole = int(data['n_reg_sfs_whole'])
    sig_names_filtered = data['sig_names_filtered']

    # 1. Exposure Differences (E_other - E_truth)
    print("Generating global exposure difference boxplots...")
    
    num_sigs = len(sig_names_filtered)
    diff_boot_pois = (E_boot_pois_whole - E_truth_whole_all[:, :, np.newaxis]).reshape(num_sigs, -1)
    diff_sfs_reg = (E_reg_whole_all - E_truth_whole_all[:, :, np.newaxis]).reshape(num_sigs, -1)
    diff_sfs_bs = (E_bs_whole_all - E_truth_whole_all[:, :, np.newaxis]).reshape(num_sigs, -1)

    fig_diff_box, ax_diff_box = plt.subplots(figsize=(14, 6))
    x_bounds = np.arange(num_sigs)
    width = 0.25
    
    diff_boot_pois_data = [diff_boot_pois[i, :] for i in range(num_sigs)]
    diff_sfs_reg_data = [diff_sfs_reg[i, :] for i in range(num_sigs)]
    diff_sfs_bs_data = [diff_sfs_bs[i, :] for i in range(num_sigs)]

    diff_opt = E_opt_whole_all - E_truth_whole_all
    mean_diff_opt = np.mean(diff_opt, axis=1)

    if has_spa:
        diff_spa = E_spa_whole_all - E_truth_whole_all
        mean_diff_spa = np.mean(diff_spa, axis=1)

    flier_style = {'marker': ',', 'markersize': 0.5, 'alpha': 0.02, 'markeredgecolor': 'none', 'markerfacecolor': 'black'}
    bp_diff_boot_pois = ax_diff_box.boxplot(diff_boot_pois_data, positions=x_bounds - width, widths=width,
                                     patch_artist=True, showmeans=True, showfliers=True,
                                     meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':5},
                                     flierprops=flier_style)
    for patch in bp_diff_boot_pois['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
        
    bp_diff_sfs_reg = ax_diff_box.boxplot(diff_sfs_reg_data, positions=x_bounds, widths=width,
                                   patch_artist=True, showmeans=True, showfliers=True,
                                   meanprops={'marker':'o', 'markerfacecolor':'purple', 'markeredgecolor':'purple', 'markersize':5},
                                   flierprops=flier_style)
    for patch in bp_diff_sfs_reg['boxes']:
        patch.set_facecolor('plum')
        patch.set_alpha(0.7)

    bp_diff_sfs_bs = ax_diff_box.boxplot(diff_sfs_bs_data, positions=x_bounds + width, widths=width,
                                  patch_artist=True, showmeans=True, showfliers=True,
                                  meanprops={'marker':'o', 'markerfacecolor':'blue', 'markeredgecolor':'blue', 'markersize':5},
                                  flierprops=flier_style)
    for patch in bp_diff_sfs_bs['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
        
    ax_diff_box.plot(x_bounds - width/2, mean_diff_opt, '^', color='black', markersize=8, zorder=5)
    
    if has_spa:
        ax_diff_box.plot(x_bounds + width/2, mean_diff_spa, 's', color='green', markersize=8, zorder=5)
        
    ax_diff_box.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_diff_box.set_xticks(x_bounds)
    ax_diff_box.set_xticklabels(sig_names_filtered, rotation=45, ha='right')
    ax_diff_box.set_ylabel('Exposure Difference (E_other - E_truth)')
    ax_diff_box.set_title('All Samples: Exposure Difference Distribution vs Ground Truth')
    
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    legend_elements_global = [
        mpatches.Patch(facecolor='lightcoral', alpha=0.7, edgecolor='black', label='Poisson Bootstrap'),
        mpatches.Patch(facecolor='plum', alpha=0.7, edgecolor='black', label='Original SFS'),
        mpatches.Patch(facecolor='lightblue', alpha=0.7, edgecolor='black', label='Hybrid SFS (Bootstrap SFS)'),
        mlines.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, label='Optimal (QP Mean)')
    ]
    if has_spa:
        legend_elements_global.append(mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='SigProfilerAssignment'))
    
    ax_diff_box.legend(handles=legend_elements_global, loc='upper right', bbox_to_anchor=(1.15, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "global_exposure_diff_box.png"), dpi=200)
    plt.close()

    # 2. Global reconstruction error density plots
    print("Generating global reconstruction error density plots...")
    eps = 1e-10
    
    # Iteration 1 (index 0)
    idx_1 = 0
    P_1 = P_sfs_whole_all[..., idx_1]
    E_1 = E_reg_whole_all[..., idx_1]
    M_approx_1 = P_1 @ E_1
    kl_matrix_1 = M_norm * np.log((M_norm + eps) / (M_approx_1 + eps)) - M_norm + M_approx_1
    kl_errors_1 = np.sum(kl_matrix_1, axis=0)
    
    fig_dens1, ax_dens1 = plt.subplots(figsize=(8, 6))
    ax_dens1.hist(kl_errors_1, bins=100, density=True, color='skyblue', alpha=0.7, edgecolor='none')
    ax_dens1.set_xlabel('Reconstruction Error (KL Divergence)')
    ax_dens1.set_ylabel('Density')
    ax_dens1.set_title('Reconstruction Errors of M Columns (After 1 Iteration)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "global_recon_err_density_iter1.png"), dpi=200)
    plt.close()
    
    # Iteration 1000 (index 999 or n_reg_sfs_whole - 1)
    idx_1000 = min(999, n_reg_sfs_whole - 1)
    if n_reg_sfs_whole > 0:
        P_1000 = P_sfs_whole_all[..., idx_1000]
        E_1000 = E_reg_whole_all[..., idx_1000]
        M_approx_1000 = P_1000 @ E_1000
        kl_matrix_1000 = M_norm * np.log((M_norm + eps) / (M_approx_1000 + eps)) - M_norm + M_approx_1000
        kl_errors_1000 = np.sum(kl_matrix_1000, axis=0)
        
        fig_dens1000, ax_dens1000 = plt.subplots(figsize=(8, 6))
        ax_dens1000.hist(kl_errors_1000, bins=100, density=True, color='salmon', alpha=0.7, edgecolor='none')
        ax_dens1000.set_xlabel('Reconstruction Error (KL Divergence)')
        ax_dens1000.set_ylabel('Density')
        ax_dens1000.set_title('Reconstruction Errors of M Columns (After 1000 Iterations)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "global_recon_err_density_iter1000.png"), dpi=200)
        plt.close()

    # QP Reconstruction Error
    from sigconfide.estimates.standard import findSigExposures
    E_opt_norm, _ = findSigExposures(M_norm, P_1)
    M_approx_qp = P_1 @ E_opt_norm
    kl_matrix_qp = M_norm * np.log((M_norm + eps) / (M_approx_qp + eps)) - M_norm + M_approx_qp
    kl_errors_qp = np.sum(kl_matrix_qp, axis=0)
    
    fig_dens_qp, ax_dens_qp = plt.subplots(figsize=(8, 6))
    ax_dens_qp.hist(kl_errors_qp, bins=100, density=True, color='lightgreen', alpha=0.7, edgecolor='none')
    ax_dens_qp.set_xlabel('Reconstruction Error (KL Divergence)')
    ax_dens_qp.set_ylabel('Density')
    ax_dens_qp.set_title('Reconstruction Errors of M Columns (Original QP)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "global_recon_err_density_qp.png"), dpi=200)
    plt.close()

    # 3. Timeline of Modifications (P Cosine Similarity)
    print("Generating timeline of signature modifications...")
    P_original_flat = P_original.flatten()
    norm_P = np.linalg.norm(P_original_flat)
    
    num_elements = n_reg_sfs_whole
    cos_sims = []
    
    for i in range(num_elements):
        P_i = P_sfs_whole_all[..., i]
        P_i_flat = P_i.flatten()
        norm_P_i = np.linalg.norm(P_i_flat)
        
        if norm_P == 0 or norm_P_i == 0:
            sim = 0
        else:
            sim = np.dot(P_original_flat, P_i_flat) / (norm_P * norm_P_i)
        cos_sims.append(sim)
        
    fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
    ax_sim.plot(range(num_elements), cos_sims, color='purple', linewidth=2, label='Cosine Similarity')
    ax_sim.set_xlabel('SFS Iteration')
    ax_sim.set_ylabel('Cosine Similarity')
    ax_sim.set_title('Timeline of Modifications to Signatures')
    ax_sim.legend()
    ax_sim.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "P_cosine_similarity.png"), dpi=200)
    plt.close()
    
    # 4. Timeline of Modifications per Signature (Cosine Similarity)
    cos_sims_per_sig = {k: [] for k in range(P_original.shape[1])}
    
    for i in range(num_elements):
        P_i = P_sfs_whole_all[..., i]
        for k in range(P_original.shape[1]):
            P_orig_k = P_original[:, k]
            P_i_k = P_i[:, k]
            norm_orig_k = np.linalg.norm(P_orig_k)
            norm_i_k = np.linalg.norm(P_i_k)
            
            if norm_orig_k == 0 or norm_i_k == 0:
                sim = 0
            else:
                sim = np.dot(P_orig_k, P_i_k) / (norm_orig_k * norm_i_k)
            cos_sims_per_sig[k].append(sim)
            
    fig_sim_per_sig, ax_sim_per_sig = plt.subplots(figsize=(12, 7))
    
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, P_original.shape[1])]
    
    for k in range(P_original.shape[1]):
        ax_sim_per_sig.plot(range(num_elements), cos_sims_per_sig[k], color=colors[k], linewidth=1.5, label=sig_names_filtered[k])
        
    ax_sim_per_sig.set_xlabel('SFS Iteration')
    ax_sim_per_sig.set_ylabel('Cosine Similarity')
    ax_sim_per_sig.set_title('Timeline of Modifications to Signatures (Per Signature)')
    
    ax_sim_per_sig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_sim_per_sig.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "P_cosine_similarity_per_sig.png"), dpi=200)
    plt.close()
    
    # 5. Distribution of cosine similarities (per signature)
    fig_sim_box, ax_sim_box = plt.subplots(figsize=(12, 7))
    
    box_data = [cos_sims_per_sig[k] for k in range(P_original.shape[1])]
    
    bp_sim = ax_sim_box.boxplot(box_data, positions=np.arange(len(sig_names_filtered)), patch_artist=True)
    
    for patch, color in zip(bp_sim['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax_sim_box.set_xticks(np.arange(len(sig_names_filtered)))
    ax_sim_box.set_xticklabels(sig_names_filtered, rotation=45, ha='right')
    ax_sim_box.set_ylabel('Cosine Similarity')
    ax_sim_box.set_title('Distribution of cosine similarities (per signature)')
    ax_sim_box.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "P_cosine_similarity_dist.png"), dpi=200)
    plt.close()
    
    # 6. Pairwise Cosine Similarity Heatmaps
    norm_orig = np.linalg.norm(P_original, axis=0)
    norm_orig_safe = np.where(norm_orig == 0, 1.0, norm_orig)
    P_orig_norm = P_original / norm_orig_safe
    sim_orig = P_orig_norm.T @ P_orig_norm

    P_final = P_sfs_whole_all[..., n_reg_sfs_whole - 1]
    norm_final = np.linalg.norm(P_final, axis=0)
    norm_final_safe = np.where(norm_final == 0, 1.0, norm_final)
    P_final_norm = P_final / norm_final_safe
    sim_final = P_final_norm.T @ P_final_norm

    # --- Plot Original & Final side-by-side ---
    fig_heatmap_orig_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(sim_orig, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title("Original Signatures Pairwise Cosine Similarity")
    ax1.set_xticks(np.arange(len(sig_names_filtered)))
    ax1.set_yticks(np.arange(len(sig_names_filtered)))
    ax1.set_xticklabels(sig_names_filtered, rotation=90)
    ax1.set_yticklabels(sig_names_filtered)
    fig_heatmap_orig_final.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(sim_final, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title("Final SFS Signatures Pairwise Cosine Similarity")
    ax2.set_xticks(np.arange(len(sig_names_filtered)))
    ax2.set_yticks(np.arange(len(sig_names_filtered)))
    ax2.set_xticklabels(sig_names_filtered, rotation=90)
    ax2.set_yticklabels(sig_names_filtered)
    fig_heatmap_orig_final.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "pairwise_cosine_similarity_original_final.png"), dpi=200)
    plt.close()

    # --- Plot Difference ---
    fig_heatmap_diff, ax = plt.subplots(figsize=(10, 8))
    sim_diff = sim_final - sim_orig

    max_abs_diff = max(np.abs(np.min(sim_diff)), np.abs(np.max(sim_diff)))
    if max_abs_diff == 0:
        max_abs_diff = 1.0
    im = ax.imshow(sim_diff, cmap='coolwarm', vmin=-max_abs_diff, vmax=max_abs_diff)
    
    ax.set_title("Difference in Pairwise Cosine Similarity (Final - Original)")
    ax.set_xticks(np.arange(len(sig_names_filtered)))
    ax.set_yticks(np.arange(len(sig_names_filtered)))
    ax.set_xticklabels(sig_names_filtered, rotation=90)
    ax.set_yticklabels(sig_names_filtered)
    fig_heatmap_diff.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "pairwise_cosine_similarity_diff.png"), dpi=200)
    plt.close()
    
    print("Finished visualising synthetic data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise computed synthetic benchmark results.")
    parser.add_argument("--output_dir", default="comparison_output", help="Output directory containing the computed results")
    args = parser.parse_args()
    
    visualise_comparison_synthetic(args.output_dir)
