import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualise_comparison(patients, sample_file, sig_file="sigconfide/utils/data/COSMIC_v2_SBS_GRCh37.txt", bootstrap_type="regular", output_dir="brca560"):
    out_path = os.path.join("comparison_output", output_dir)
    
    global_diff_boot_reg = []
    global_diff_boot_pois = []
    global_diff_sfs_reg = []
    global_diff_sfs_bs = []
    global_sig_names = None
    
    for pt in patients:
        print(f"Visualising results for patient {pt}")
        pt_dir = os.path.join(out_path, pt)
        data_path = os.path.join(pt_dir, "computed_data.npz")
        
        if not os.path.exists(data_path):
            print(f"  Data not found at {data_path}. Skipping.")
            continue
            
        data = np.load(data_path)
        E_sfs = data['E_sfs']
        E_sfs_whole = data['E_sfs_whole']
        E_boot_reg = data['E_boot_reg'] if 'E_boot_reg' in data else data.get('E_boot')
        E_boot_pois = data['E_boot_pois'] if 'E_boot_pois' in data else data.get('E_boot')
        E_sfs_reg_whole = data['E_sfs_reg_whole'] if 'E_sfs_reg_whole' in data else E_sfs_whole[..., :int(data.get('n_reg_sfs_whole', 0))]
        E_sfs_bs_whole = data['E_sfs_bs_whole'] if 'E_sfs_bs_whole' in data else E_sfs_whole[..., int(data.get('n_reg_sfs_whole', 0)):]
        E_opt = data['E_opt']
        kl_errors_sfs = data['kl_errors_sfs']
        kl_errs_boot = data['kl_errs_boot_pois'] if 'kl_errs_boot_pois' in data else data.get('kl_errs_boot')
        sig_names_filtered = data['sig_names_filtered']
        P_sfs_whole = data['P_sfs_whole'] if 'P_sfs_whole' in data else None
        P_sfs = data['P_sfs'] if 'P_sfs' in data else None
        n_reg_sfs_whole = int(data['n_reg_sfs_whole']) if 'n_reg_sfs_whole' in data else 0
        E_spa = data['E_spa'] if 'E_spa' in data else None
        
        # 1. Element-Wise Bounds vs Confidence Intervals (Boxplots)
        fig_bounds, ax_bounds = plt.subplots(figsize=(14, 6))
        num_sigs = len(sig_names_filtered)
        x_bounds = np.arange(num_sigs)
        width = 0.2
        
        boot_reg_data = [E_boot_reg[i, :] for i in range(num_sigs)] if E_boot_reg is not None else []
        boot_pois_data = [E_boot_pois[i, :] for i in range(num_sigs)] if E_boot_pois is not None else []
        sfs_reg_data = [E_sfs_reg_whole[i, :] for i in range(num_sigs)]
        sfs_bs_data = [E_sfs_bs_whole[i, :] for i in range(num_sigs)] if E_sfs_bs_whole is not None else []
        
        # Regular Bootstrap Boxplots
        if E_boot_reg is not None and E_boot_reg.shape[-1] > 0:
            bp_boot_reg = ax_bounds.boxplot(boot_reg_data, positions=x_bounds - 1.5*width, widths=width,
                                            patch_artist=True, showmeans=True, showfliers=False,
                                            meanprops={'marker':'o', 'markerfacecolor':'green', 'markeredgecolor':'green', 'markersize':5})
            for patch in bp_boot_reg['boxes']:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.7)

        # Poisson Bootstrap Boxplots
        if E_boot_pois is not None and E_boot_pois.shape[-1] > 0:
            bp_boot_pois = ax_bounds.boxplot(boot_pois_data, positions=x_bounds - 0.5*width, widths=width,
                                             patch_artist=True, showmeans=True, showfliers=False,
                                             meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':5})
            for patch in bp_boot_pois['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
                
        # Original SFS Boxplots
        bp_sfs_reg = ax_bounds.boxplot(sfs_reg_data, positions=x_bounds + 0.5*width, widths=width,
                                       patch_artist=True, showmeans=True, showfliers=False,
                                       meanprops={'marker':'o', 'markerfacecolor':'purple', 'markeredgecolor':'purple', 'markersize':5})
        for patch in bp_sfs_reg['boxes']:
            patch.set_facecolor('plum')
            patch.set_alpha(0.7)

        # Hybrid SFS (Bootstrap SFS) Boxplots
        if E_sfs_bs_whole is not None and E_sfs_bs_whole.shape[-1] > 0:
            bp_sfs_bs = ax_bounds.boxplot(sfs_bs_data, positions=x_bounds + 1.5*width, widths=width,
                                          patch_artist=True, showmeans=True, showfliers=False,
                                          meanprops={'marker':'o', 'markerfacecolor':'blue', 'markeredgecolor':'blue', 'markersize':5})
            for patch in bp_sfs_bs['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
                
        ax_bounds.plot(x_bounds, E_opt, '*', color='black', markersize=8)
        
        if E_spa is not None:
            # Determine right-most position
            has_hybrid = E_sfs_bs_whole is not None and E_sfs_bs_whole.shape[-1] > 0
            has_pois = E_boot_pois is not None and E_boot_pois.shape[-1] > 0
            has_reg = E_boot_reg is not None and E_boot_reg.shape[-1] > 0
            
            # Logic: If hybrid, it's at +1.5w. Else, SFS is at +0.5w.
            pos_base = x_bounds + (1.5 * width if has_hybrid else 0.5 * width)
            pos_spa = pos_base + width
            ax_bounds.plot(pos_spa, E_spa.flatten() if E_spa.ndim > 0 else E_spa, 's', color='green', markersize=6, zorder=7)
        
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        legend_elements = [
            mpatches.Patch(facecolor='plum', alpha=0.7, edgecolor='black', label='Original SFS'),
            mlines.Line2D([0], [0], marker='o', color='w', label='Original SFS Mean', markerfacecolor='purple', markersize=8),
            mlines.Line2D([0], [0], marker='*', color='w', label='Original QP (E_opt)', markerfacecolor='black', markersize=12)
        ]
        if E_boot_reg is not None and E_boot_reg.shape[-1] > 0:
            legend_elements.insert(0, mpatches.Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='Regular Bootstrap'))
            legend_elements.insert(2, mlines.Line2D([0], [0], marker='o', color='w', label='Regular Boot Mean', markerfacecolor='green', markersize=8))
        if E_boot_pois is not None and E_boot_pois.shape[-1] > 0:
            legend_elements.insert(1 if (E_boot_reg is not None and E_boot_reg.shape[-1] > 0) else 0, mpatches.Patch(facecolor='lightcoral', alpha=0.7, edgecolor='black', label='Poisson Bootstrap'))
            legend_elements.insert(4 if (E_boot_reg is not None and E_boot_reg.shape[-1] > 0) else 2, mlines.Line2D([0], [0], marker='o', color='w', label='Poisson Boot Mean', markerfacecolor='red', markersize=8))
        if E_sfs_bs_whole is not None and E_sfs_bs_whole.shape[-1] > 0:
            idx1 = 3 if len(legend_elements) > 5 else 2
            idx2 = len(legend_elements) - 1
            legend_elements.insert(idx1, mpatches.Patch(facecolor='lightblue', alpha=0.7, edgecolor='black', label='Hybrid SFS (Bootstrap SFS)'))
            legend_elements.insert(idx2 + 1, mlines.Line2D([0], [0], marker='o', color='w', label='Hybrid SFS Mean', markerfacecolor='blue', markersize=8))
            
        if E_spa is not None:
            legend_elements.append(mlines.Line2D([0], [0], marker='s', color='w', label='SigProfilerAssignment', markerfacecolor='green', markersize=8))
        
        ax_bounds.set_xticks(x_bounds)
        ax_bounds.set_xticklabels(sig_names_filtered, rotation=45, ha='right')
        ax_bounds.set_ylabel('Exposure')
        ax_bounds.set_title(f'Patient {pt}: Multi-Method Exposure Distribution')
        ax_bounds.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.05))
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "element_bounds.png"), dpi=200)
        plt.close()
        
        # 2. Spatial Inclusion (PCA)
        X_sfs = E_sfs_whole.T
        if E_boot_pois is not None and E_boot_pois.shape[-1] > 0:
            X_boot = E_boot_pois.T
            X_all = np.vstack([X_sfs, X_boot])
        else:
            X_boot = np.array([])
            X_all = X_sfs
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all)
        X_sfs_pca = X_pca[:len(X_sfs)]
        if len(X_boot) > 0:
            X_boot_pca = X_pca[len(X_sfs):]
        else:
            X_boot_pca = np.array([])
        
        X_reg_sfs_pca = X_sfs_pca[:n_reg_sfs_whole]
        X_bs_sfs_pca = X_sfs_pca[n_reg_sfs_whole:]
        
        fig_pca, ax_pca = plt.subplots(figsize=(8, 8))
        if len(X_bs_sfs_pca) > 0:
            ax_pca.scatter(X_bs_sfs_pca[:, 0], X_bs_sfs_pca[:, 1], c='lightblue', label='Bootstrap SFS Samples', alpha=0.5, s=15, marker='o')
        if len(X_reg_sfs_pca) > 0:
            ax_pca.scatter(X_reg_sfs_pca[:, 0], X_reg_sfs_pca[:, 1], c='lime', label='Regular SFS Samples', alpha=0.8, s=20, marker='D', edgecolors='black', linewidths=0.5)
        if len(X_boot_pca) > 0:
            ax_pca.scatter(X_boot_pca[:, 0], X_boot_pca[:, 1], c='red', label='Bootstrap Samples', alpha=0.5, s=15, marker='x')
        ax_pca.set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax_pca.set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax_pca.set_title(f'Patient {pt}: Spatial Inclusion (PCA)')
        ax_pca.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "spatial_pca.png"), dpi=200)
        plt.close()
        
        # 3. Density of the Optimization Landscape
        fig_dens, ax_dens = plt.subplots(figsize=(10, 6))
        
        has_boot = kl_errs_boot is not None and len(kl_errs_boot) > 0
        min_err = np.min(kl_errors_sfs)
        max_err = np.max(kl_errors_sfs)
        if has_boot:
            min_err = min(np.min(kl_errs_boot), min_err)
            max_err = max(np.max(kl_errs_boot), max_err)
            
        bins = np.linspace(min_err, max_err, 100)
        
        if has_boot:
            hist_boot, _ = np.histogram(kl_errs_boot, bins=bins, density=True)
        else:
            hist_boot = np.zeros(len(bins)-1)
            
        hist_sfs, _ = np.histogram(kl_errors_sfs, bins=bins, density=True)
        
        bin_widths = np.diff(bins)
        bin_centers = bins[:-1] + bin_widths / 2
        
        mask_boot_taller = hist_boot > hist_sfs
        mask_sfs_taller = ~mask_boot_taller

        # Plot taller bars first (background)
        if np.any(mask_boot_taller):
            ax_dens.bar(bin_centers[mask_boot_taller], hist_boot[mask_boot_taller], width=bin_widths[mask_boot_taller], color='red', alpha=0.6, edgecolor='none', align='center')
        if np.any(mask_sfs_taller):
            ax_dens.bar(bin_centers[mask_sfs_taller], hist_sfs[mask_sfs_taller], width=bin_widths[mask_sfs_taller], color='lightblue', alpha=0.6, edgecolor='none', align='center')

        # Plot shorter bars second (foreground)
        if np.any(mask_boot_taller):
            ax_dens.bar(bin_centers[mask_boot_taller], hist_sfs[mask_boot_taller], width=bin_widths[mask_boot_taller], color='lightblue', alpha=0.9, edgecolor='none', align='center')
        if np.any(mask_sfs_taller):
            ax_dens.bar(bin_centers[mask_sfs_taller], hist_boot[mask_sfs_taller], width=bin_widths[mask_sfs_taller], color='red', alpha=0.9, edgecolor='none', align='center')

        legend_patches = [
            mpatches.Patch(color='lightblue', alpha=0.8, label='SFS (KL Div)')
        ]
        if has_boot:
            legend_patches.insert(0, mpatches.Patch(color='red', alpha=0.8, label='Bootstrap (KL Div)'))

        ax_dens.axvline(np.mean(kl_errors_sfs), color='blue', linestyle='dashed', linewidth=2, label='SFS Mean Error')
        if has_boot:
            ax_dens.axvline(np.mean(kl_errs_boot), color='darkred', linestyle='dotted', linewidth=2, label='Bootstrap Mean Error')
        if len(kl_errors_sfs) > 0:
            ax_dens.axvline(kl_errors_sfs[0], color='lime', linestyle='dashdot', linewidth=2, label='Regular SFS Error')
        
        handles, labels = ax_dens.get_legend_handles_labels()
        ax_dens.legend(handles=legend_patches + handles, labels=[p.get_label() for p in legend_patches] + labels)
        
        ax_dens.set_xlabel('Reconstruction Error (KL Divergence)')
        ax_dens.set_ylabel('Density')
        ax_dens.set_yscale('log')
        ax_dens.set_title(f'Patient {pt}: Error Landscape Density')
        ax_dens.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "error_density.png"), dpi=200)
        plt.close()
        
        # 4. Timeline of Modifications (P Cosine Similarity)
        from sigconfide.utils.utils import load_signatures_file
        import re
        
        if os.path.exists(sig_file):
            signatures, sig_names = load_signatures_file(sig_file)
            if sig_names[0] == 'Samples' or sig_names[0] == 'Type':
                sig_names = sig_names[1:]
                
            target_sigs_num = [1, 2, 3, 5, 6, 8, 9, 12, 13, 16, 17, 18, 20, 26, 30]
            target_indices = []
            for i, name in enumerate(sig_names):
                match = re.search(r'\d+', name)
                if match and int(match.group()) in target_sigs_num:
                    target_indices.append(i)
                    
            if not target_indices:
                target_indices = list(range(len(sig_names)))
                
            P_original = signatures[:, target_indices]
            P_original_flat = P_original.flatten()
            norm_P = np.linalg.norm(P_original_flat)
            
            num_elements = n_reg_sfs_whole if n_reg_sfs_whole > 0 else P_sfs_whole.shape[-1]
            cos_sims = []
            
            for i in range(num_elements):
                P_i = P_sfs_whole[..., i]
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
            plt.savefig(os.path.join(pt_dir, "P_cosine_similarity.png"), dpi=200)
            plt.close()
            
            # 5. Timeline of Modifications per Signature (Cosine Similarity)
            cos_sims_per_sig = {k: [] for k in range(P_original.shape[1])}
            
            for i in range(num_elements):
                P_i = P_sfs_whole[..., i]
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
            plt.savefig(os.path.join(pt_dir, "P_cosine_similarity_per_sig.png"), dpi=200)
            plt.close()
            
            # 5.5 Distribution of cosine similarities (per signature)
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
            plt.savefig(os.path.join(pt_dir, "P_cosine_similarity_dist.png"), dpi=200)
            plt.close()
            
            # 6. Pairwise Cosine Similarity Heatmaps
            # Original P
            norm_orig = np.linalg.norm(P_original, axis=0)
            norm_orig_safe = np.where(norm_orig == 0, 1.0, norm_orig)
            P_orig_norm = P_original / norm_orig_safe
            sim_orig = P_orig_norm.T @ P_orig_norm

            # Final P
            P_final = P_sfs[..., 0] if P_sfs is not None else P_sfs_whole[..., n_reg_sfs_whole - 1]
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
            plt.savefig(os.path.join(pt_dir, "pairwise_cosine_similarity_original_final.png"), dpi=200)
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
            plt.savefig(os.path.join(pt_dir, "pairwise_cosine_similarity_diff.png"), dpi=200)
            plt.close()
            
        else:
            print(f"  Warning: sig_file {sig_file} not found. Skipping P Cosine Similarity plot.")
            
        # 7. Exposure Differences (E_other - E_opt) Data Accumulation
        if E_boot_reg is not None and E_boot_reg.shape[-1] > 0:
            diff_boot_reg = E_boot_reg - E_opt[:, np.newaxis]
            global_diff_boot_reg.append(diff_boot_reg)
        if E_boot_pois is not None and E_boot_pois.shape[-1] > 0:
            diff_boot_pois = E_boot_pois - E_opt[:, np.newaxis]
            global_diff_boot_pois.append(diff_boot_pois)
            
        diff_sfs_reg = E_sfs_reg_whole - E_opt[:, np.newaxis]
        global_diff_sfs_reg.append(diff_sfs_reg)
        
        if E_sfs_bs_whole is not None and E_sfs_bs_whole.shape[-1] > 0:
            diff_sfs_bs = E_sfs_bs_whole - E_opt[:, np.newaxis]
            global_diff_sfs_bs.append(diff_sfs_bs)
            
        if E_spa is not None:
            if not hasattr(visualise_comparison, 'global_diff_spa'):
                visualise_comparison.global_diff_spa = []
            diff_spa = E_spa - E_opt
            visualise_comparison.global_diff_spa.append(diff_spa)
        
        if global_sig_names is None:
            global_sig_names = sig_names_filtered
            
        print(f"  Finished visualising {pt}")
        
    # Generate global reconstruction error density plots
    global_data_path = os.path.join(out_path, "global_computed_data.npz")
    if os.path.exists(global_data_path):
        print("Generating global reconstruction error density plots...")
        global_data = np.load(global_data_path)
        M_norm = global_data['M_norm']
        P_sfs_whole_all = global_data['P_sfs_whole_all']
        E_reg_whole_all = global_data['E_reg_whole_all']
        n_reg_sfs_whole = int(global_data['n_reg_sfs_whole'])
        
        eps = 1e-10
        
        # Iteration 1 (index 0)
        idx_1 = 0
        P_1 = P_sfs_whole_all[..., idx_1]
        E_1 = E_reg_whole_all[..., idx_1]
        M_approx_1 = P_1 @ E_1
        kl_matrix_1 = M_norm * np.log((M_norm + eps) / (M_approx_1 + eps)) - M_norm + M_approx_1
        kl_errors_1 = np.sum(kl_matrix_1, axis=0)
        
        # Plot 1: After 1 iteration
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
            
            # Plot 2: After 1000 iterations
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
        
        # Plot 3: QP
        fig_dens_qp, ax_dens_qp = plt.subplots(figsize=(8, 6))
        ax_dens_qp.hist(kl_errors_qp, bins=100, density=True, color='lightgreen', alpha=0.7, edgecolor='none')
        ax_dens_qp.set_xlabel('Reconstruction Error (KL Divergence)')
        ax_dens_qp.set_ylabel('Density')
        ax_dens_qp.set_title('Reconstruction Errors of M Columns (Original QP)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "global_recon_err_density_qp.png"), dpi=200)
        plt.close()

    # After the loop over all patients
    if len(global_diff_sfs_reg) > 0:
        print("Generating global exposure difference boxplots...")
        
        fig_diff_box, ax_diff_box = plt.subplots(figsize=(14, 6))
        num_sigs = len(global_sig_names)
        x_bounds = np.arange(num_sigs)
        width = 0.2
        
        if len(global_diff_boot_reg) > 0:
            global_diff_boot_reg_concat = np.concatenate(global_diff_boot_reg, axis=1)
            diff_boot_reg_data = [global_diff_boot_reg_concat[i, :] for i in range(num_sigs)]
            bp_diff_boot_reg = ax_diff_box.boxplot(diff_boot_reg_data, positions=x_bounds - 1.5*width, widths=width,
                                            patch_artist=True, showmeans=True, showfliers=False,
                                            meanprops={'marker':'o', 'markerfacecolor':'green', 'markeredgecolor':'green', 'markersize':5})
            for patch in bp_diff_boot_reg['boxes']:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.7)
                
        if len(global_diff_boot_pois) > 0:
            global_diff_boot_pois_concat = np.concatenate(global_diff_boot_pois, axis=1)
            diff_boot_pois_data = [global_diff_boot_pois_concat[i, :] for i in range(num_sigs)]
            bp_diff_boot_pois = ax_diff_box.boxplot(diff_boot_pois_data, positions=x_bounds - 0.5*width, widths=width,
                                             patch_artist=True, showmeans=True, showfliers=False,
                                             meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':5})
            for patch in bp_diff_boot_pois['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
                
        global_diff_sfs_reg_concat = np.concatenate(global_diff_sfs_reg, axis=1)
        diff_sfs_reg_data = [global_diff_sfs_reg_concat[i, :] for i in range(num_sigs)]
        bp_diff_sfs_reg = ax_diff_box.boxplot(diff_sfs_reg_data, positions=x_bounds + 0.5*width, widths=width,
                                       patch_artist=True, showmeans=True, showfliers=False,
                                       meanprops={'marker':'o', 'markerfacecolor':'purple', 'markeredgecolor':'purple', 'markersize':5})
        for patch in bp_diff_sfs_reg['boxes']:
            patch.set_facecolor('plum')
            patch.set_alpha(0.7)

        if len(global_diff_sfs_bs) > 0:
            global_diff_sfs_bs_concat = np.concatenate(global_diff_sfs_bs, axis=1)
            diff_sfs_bs_data = [global_diff_sfs_bs_concat[i, :] for i in range(num_sigs)]
            bp_diff_sfs_bs = ax_diff_box.boxplot(diff_sfs_bs_data, positions=x_bounds + 1.5*width, widths=width,
                                          patch_artist=True, showmeans=True, showfliers=False,
                                          meanprops={'marker':'o', 'markerfacecolor':'blue', 'markeredgecolor':'blue', 'markersize':5})
            for patch in bp_diff_sfs_bs['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
                
        if hasattr(visualise_comparison, 'global_diff_spa') and len(visualise_comparison.global_diff_spa) > 0:
            global_diff_spa_concat = np.column_stack(visualise_comparison.global_diff_spa)
            diff_spa_data = [global_diff_spa_concat[i, :] for i in range(num_sigs)]
            has_hybrid = len(global_diff_sfs_bs) > 0
            pos_spa_diff = x_bounds + (1.5 * width if has_hybrid else 0.5 * width) + width
            bp_diff_spa = ax_diff_box.boxplot(diff_spa_data, positions=pos_spa_diff, widths=width,
                                          patch_artist=True, showmeans=True, showfliers=False,
                                          meanprops={'marker':'s', 'markerfacecolor':'green', 'markeredgecolor':'green', 'markersize':5})
            for patch in bp_diff_spa['boxes']:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.7)
                
        ax_diff_box.axhline(0, color='black', linestyle='--', linewidth=1)
        ax_diff_box.set_xticks(x_bounds)
        ax_diff_box.set_xticklabels(global_sig_names, rotation=45, ha='right')
        ax_diff_box.set_ylabel('Exposure Difference (E_other - E_opt)')
        ax_diff_box.set_title('All Patients: Exposure Difference Distribution')
        
        import matplotlib.patches as mpatches
        legend_elements_global = [
            mpatches.Patch(facecolor='plum', alpha=0.7, edgecolor='black', label='Original SFS')
        ]
        if len(global_diff_boot_reg) > 0:
            legend_elements_global.insert(0, mpatches.Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='Regular Bootstrap'))
        if len(global_diff_boot_pois) > 0:
            legend_elements_global.insert(1 if len(global_diff_boot_reg) > 0 else 0, mpatches.Patch(facecolor='lightcoral', alpha=0.7, edgecolor='black', label='Poisson Bootstrap'))
        if len(global_diff_sfs_bs) > 0:
            legend_elements_global.append(mpatches.Patch(facecolor='lightblue', alpha=0.7, edgecolor='black', label='Hybrid SFS (Bootstrap SFS)'))
        if hasattr(visualise_comparison, 'global_diff_spa') and len(visualise_comparison.global_diff_spa) > 0:
            legend_elements_global.append(mpatches.Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='SigProfilerAssignment'))
        
        ax_diff_box.legend(handles=legend_elements_global, loc='upper right', bbox_to_anchor=(1.15, 1.05))
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "global_exposure_diff_box.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise computed SFS and Bootstrap signature exposures.")
    parser.add_argument("--patients", nargs="+", default=["PD24196", "PD8609", "PD13608"], help="List of patients")
    parser.add_argument("--sample_file", default="tests/data/tumorBRCA.txt", help="Path to sample file")
    parser.add_argument("--output_dir", default="brca560", help="Output directory inside comparison_output")
    parser.add_argument("--bootstrap_type", choices=["regular", "poisson", "all", "none"], default="poisson", help="Bootstrap type")
    parser.add_argument("--sig_file", default="sigconfide/utils/data/COSMIC_v2_SBS_GRCh37.txt", help="Path to signatures file")
    args = parser.parse_args()
    
    visualise_comparison(args.patients, args.sample_file, args.sig_file, args.bootstrap_type, args.output_dir)
