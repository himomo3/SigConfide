import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualise_comparison(patients, output_dir="comparison_output"):
    for pt in patients:
        print(f"Visualising results for patient {pt}")
        pt_dir = os.path.join(output_dir, pt)
        data_path = os.path.join(pt_dir, "computed_data.npz")
        
        if not os.path.exists(data_path):
            print(f"  Data not found at {data_path}. Skipping.")
            continue
            
        data = np.load(data_path)
        E_sfs = data['E_sfs']
        E_boot = data['E_boot']
        E_opt = data['E_opt']
        kl_errors_sfs = data['kl_errors_sfs']
        kl_errs_boot = data['kl_errs_boot']
        active_sigs = data['active_sigs']
        sig_names_filtered = data['sig_names_filtered']
        
        # 1. Element-Wise Bounds vs Confidence Intervals (Now Boxplots)
        fig_bounds, ax_bounds = plt.subplots(figsize=(10, 6))
        x_bounds = np.arange(len(active_sigs))
        width = 0.35
        
        sfs_data = [E_sfs[i, :] for i in active_sigs]
        boot_data = [E_boot[i, :] for i in active_sigs]
        
        # SFS Boxplots
        bp_sfs = ax_bounds.boxplot(sfs_data, positions=x_bounds - width/2, widths=width,
                                   patch_artist=True, showmeans=True, showfliers=False,
                                   meanprops={'marker':'o', 'markerfacecolor':'blue', 'markeredgecolor':'blue', 'markersize':6})
        for patch in bp_sfs['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
            
        # Bootstrap Boxplots
        bp_boot = ax_bounds.boxplot(boot_data, positions=x_bounds + width/2, widths=width,
                                    patch_artist=True, showmeans=True, showfliers=False,
                                    meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':6})
        for patch in bp_boot['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
            
        ax_bounds.plot(x_bounds, E_opt[active_sigs], '*', color='black', markersize=8)
        
        # Custom legend elements
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        legend_elements = [
            mpatches.Patch(facecolor='lightblue', alpha=0.7, edgecolor='black', label='SFS Distribution'),
            mpatches.Patch(facecolor='lightcoral', alpha=0.7, edgecolor='black', label='Bootstrap Distribution'),
            mlines.Line2D([0], [0], marker='o', color='w', label='SFS Mean', markerfacecolor='blue', markersize=8),
            mlines.Line2D([0], [0], marker='o', color='w', label='Bootstrap Mean', markerfacecolor='red', markersize=8),
            mlines.Line2D([0], [0], marker='*', color='w', label='Original QP (E_opt)', markerfacecolor='black', markersize=12)
        ]
        
        ax_bounds.set_xticks(x_bounds)
        ax_bounds.set_xticklabels(sig_names_filtered[active_sigs], rotation=45, ha='right')
        ax_bounds.set_ylabel('Exposure')
        ax_bounds.set_title(f'Patient {pt}: SFS vs Bootstrap Exposure Distribution')
        ax_bounds.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "element_bounds.png"), dpi=200)
        plt.close()
        
        # 2. Spatial Inclusion (PCA)
        X_sfs = E_sfs.T
        X_boot = E_boot.T
        X_all = np.vstack([X_sfs, X_boot])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all)
        X_sfs_pca = X_pca[:len(X_sfs)]
        X_boot_pca = X_pca[len(X_sfs):]
        
        fig_pca, ax_pca = plt.subplots(figsize=(8, 8))
        ax_pca.scatter(X_sfs_pca[:, 0], X_sfs_pca[:, 1], c='lightblue', label='SFS Samples', alpha=0.5, s=15, marker='o')
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
        min_err = min(np.min(kl_errs_boot), np.min(kl_errors_sfs))
        max_err = max(np.max(kl_errs_boot), np.max(kl_errors_sfs))
        bins = np.linspace(min_err, max_err, 100)
        
        hist_boot, _ = np.histogram(kl_errs_boot, bins=bins, density=True)
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
            mpatches.Patch(color='red', alpha=0.8, label='Bootstrap (KL Div)'),
            mpatches.Patch(color='lightblue', alpha=0.8, label='SFS (KL Div)')
        ]

        ax_dens.axvline(np.mean(kl_errors_sfs), color='blue', linestyle='dashed', linewidth=2, label='SFS Mean Error')
        ax_dens.axvline(np.mean(kl_errs_boot), color='darkred', linestyle='dotted', linewidth=2, label='Bootstrap Mean Error')
        
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
        print(f"  Finished visualising {pt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise computed SFS and Bootstrap signature exposures.")
    parser.add_argument("--patients", nargs="+", default=["PD24196", "PD8609", "PD13608"], help="List of patients")
    parser.add_argument("--output_dir", default="comparison_output", help="Output directory containing the computed results")
    args = parser.parse_args()
    
    visualise_comparison(args.patients, args.output_dir)
