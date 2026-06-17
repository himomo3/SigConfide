import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import argparse

def generate_patient_plots(output_dir="synthetic2700_all", sample_file=None, patient_list=None):
    out_path = os.path.join("comparison_output", output_dir)
    global_data_path = os.path.join(out_path, 'global_computed_data.npz')
    if not os.path.exists(global_data_path):
        print(f"Error: {global_data_path} not found.")
        return
        
    data = np.load(global_data_path)
    
    E_reg_whole_all = data['E_reg_whole_all']
    E_bs_whole_all = data['E_bs_whole_all']
    E_boot_pois_whole = data['E_boot_pois_whole']
    E_opt_whole_all = data['E_opt_whole_all']
    
    E_truth_whole_all = data['E_truth_whole_all']
    E_truth_sums = E_truth_whole_all.sum(axis=0)
    E_truth_whole_all = np.divide(E_truth_whole_all, E_truth_sums, out=np.zeros_like(E_truth_whole_all), where=E_truth_sums!=0)
    
    try:
        E_spa_whole_all = data['E_spa_whole_all']
        has_spa = True
    except KeyError:
        has_spa = False
    
    # Get sample names
    num_samples_data = E_truth_whole_all.shape[1]
    if 'patient_names' in data:
        sample_names = data['patient_names'].tolist()
    elif sample_file and os.path.exists(sample_file):
        import pandas as pd
        df = pd.read_csv(sample_file, sep='\t', index_col=0, nrows=0)
        sample_names = df.columns.tolist()
    else:
        # If no sample file, just use generic names
        sample_names = [f"Patient_{i}" for i in range(num_samples_data)]
        
    if patient_list is not None and patient_list.lower() != 'none':
        patient_list_path = os.path.join("comparison_output", patient_list)
        
        if not os.path.exists(patient_list_path):
            print(f"Error: patient list file {patient_list_path} not found.")
            return
            
        with open(patient_list_path, "r") as f:
            list_patients = [line.strip() for line in f if line.strip()]
            
        selected_indices = []
        missing_patients = []
        for p in list_patients:
            if p in sample_names:
                selected_indices.append(sample_names.index(p))
            else:
                missing_patients.append(p)
                
        if missing_patients:
            print(f"Error: The following patients from the list are not present in the data:\n" + "\n".join(missing_patients))
            return
    else:
        # Calculate L1 exposure error for each sample to pick big, moderate, and small error cases
        l1_errors = np.sum(np.abs(E_truth_whole_all - E_opt_whole_all), axis=0)
        sorted_indices = np.argsort(l1_errors)
        num_samples = len(sorted_indices)
        
        if num_samples >= 9:
            small_indices = sorted_indices[:3].tolist()
            median_idx = num_samples // 2
            moderate_indices = sorted_indices[median_idx-1:median_idx+2].tolist()
            big_indices = sorted_indices[-3:].tolist()
            
            # Combine in order: small, moderate, big, preserving uniqueness
            selected_indices = []
            for idx in small_indices + moderate_indices + big_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
        else:
            selected_indices = sorted_indices.tolist()

        # Save patient list to file in output_dir
        with open(os.path.join(out_path, "selected_patients.txt"), "w") as f:
            for idx in selected_indices:
                f.write(f"{sample_names[idx]}\n")
        
    # Replace '::' with '..' in sample names because '::' is invalid for Windows paths
    patients = [(i, sample_names[i].replace('::', '..')) for i in selected_indices]
    
    # Remove previous patient visualizations (subdirectories starting with Patient_ or SP.Syn)
    import shutil
    new_pt_names = set(pt_name for _, pt_name in patients)
    for item in os.listdir(out_path):
        item_path = os.path.join(out_path, item)
        if os.path.isdir(item_path):
            if (item.startswith("Patient_") or item.startswith("SP.Syn")) and item not in new_pt_names:
                print(f"Removing old visualization folder: {item}")
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error removing {item_path}: {e}")
    P_original = data['P_original']
    P_sfs_whole_all = data['P_sfs_whole_all']
    M_norm = data['M_norm']
    sig_names_filtered = data['sig_names_filtered']
    n_reg = int(data['n_reg_sfs_whole'])
    num_sigs = len(sig_names_filtered)
    eps = 1e-10
    
    for pt_idx, pt_name in patients:
        print(f"Processing {pt_name} (index {pt_idx})...")
        pt_dir = os.path.join(out_path, pt_name)
        os.makedirs(pt_dir, exist_ok=True)
        
        E_sfs_reg = E_reg_whole_all[:, pt_idx, :]
        E_sfs_bs = E_bs_whole_all[:, pt_idx, :]
        E_boot_pois = E_boot_pois_whole[:, pt_idx, :]
        E_opt = E_opt_whole_all[:, pt_idx]
        E_truth = E_truth_whole_all[:, pt_idx]
        
        # 1. Element Bounds
        fig_bounds, ax_bounds = plt.subplots(figsize=(14, 6))
        x_bounds = np.arange(num_sigs)
        width = 0.25
        
        boot_pois_data = [E_boot_pois[i, :] for i in range(num_sigs)]
        sfs_reg_data = [E_sfs_reg[i, :] for i in range(num_sigs)]
        sfs_bs_data = [E_sfs_bs[i, :] for i in range(num_sigs)]
        
        bp_boot_pois = ax_bounds.boxplot(boot_pois_data, positions=x_bounds - width, widths=width,
                                         patch_artist=True, showmeans=True, showfliers=False,
                                         meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':5})
        for patch in bp_boot_pois['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
            
        bp_sfs_reg = ax_bounds.boxplot(sfs_reg_data, positions=x_bounds, widths=width,
                                       patch_artist=True, showmeans=True, showfliers=False,
                                       meanprops={'marker':'o', 'markerfacecolor':'purple', 'markeredgecolor':'purple', 'markersize':5})
        for patch in bp_sfs_reg['boxes']:
            patch.set_facecolor('plum')
            patch.set_alpha(0.7)

        bp_sfs_bs = ax_bounds.boxplot(sfs_bs_data, positions=x_bounds + width, widths=width,
                                      patch_artist=True, showmeans=True, showfliers=False,
                                      meanprops={'marker':'o', 'markerfacecolor':'blue', 'markeredgecolor':'blue', 'markersize':5})
        for patch in bp_sfs_bs['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
            
        ax_bounds.plot(x_bounds - width, E_opt, '*', color='black', markersize=8, zorder=6)
        ax_bounds.plot(x_bounds, E_truth, 'D', color='gold', markersize=8, markeredgecolor='black', zorder=5)
        
        legend_elements = [
            mpatches.Patch(facecolor='lightcoral', alpha=0.7, edgecolor='black', label='Poisson Bootstrap'),
            mpatches.Patch(facecolor='plum', alpha=0.7, edgecolor='black', label='Original SFS'),
            mpatches.Patch(facecolor='lightblue', alpha=0.7, edgecolor='black', label='Hybrid SFS (Bootstrap SFS)'),
            mlines.Line2D([0], [0], marker='o', color='w', label='Poisson Boot Mean', markerfacecolor='red', markersize=8),
            mlines.Line2D([0], [0], marker='o', color='w', label='Original SFS Mean', markerfacecolor='purple', markersize=8),
            mlines.Line2D([0], [0], marker='o', color='w', label='Hybrid SFS Mean', markerfacecolor='blue', markersize=8),
            mlines.Line2D([0], [0], marker='*', color='w', label='Original QP (E_opt)', markerfacecolor='black', markersize=12),
            mlines.Line2D([0], [0], marker='D', color='w', label='Ground Truth (E_truth)', markerfacecolor='gold', markeredgecolor='black', markersize=10)
        ]
        
        if has_spa:
            E_spa = E_spa_whole_all[:, pt_idx]
            if E_spa.ndim == 1 or E_spa.shape[1] == 1:
                # Plot as singular green squares
                ax_bounds.plot(x_bounds + width, E_spa.flatten(), 's', color='green', markersize=8, zorder=7)
                legend_elements.append(mlines.Line2D([0], [0], marker='s', color='w', label='SigProfilerAssignment', markerfacecolor='green', markersize=10))
            else:
                # Plot as green box plots
                spa_data = [E_spa[i, :] for i in range(num_sigs)]
                bp_spa = ax_bounds.boxplot(spa_data, positions=x_bounds + width, widths=width,
                                           patch_artist=True, showmeans=True, showfliers=False,
                                           meanprops={'marker':'s', 'markerfacecolor':'green', 'markeredgecolor':'green', 'markersize':5})
                for patch in bp_spa['boxes']:
                    patch.set_facecolor('lightgreen')
                    patch.set_alpha(0.7)
                legend_elements.append(mpatches.Patch(facecolor='lightgreen', alpha=0.7, edgecolor='black', label='SigProfilerAssignment'))
        
        ax_bounds.set_xticks(x_bounds)
        ax_bounds.set_xticklabels(sig_names_filtered, rotation=45, ha='right')
        ax_bounds.set_ylabel('Exposure')
        ax_bounds.set_title(f'Patient {pt_name}: Multi-Method Exposure Distribution')
        ax_bounds.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.05))
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "element_bounds.png"), dpi=200)
        plt.close()
        
        # 2. Spatial PCA
        X_sfs_reg = E_sfs_reg.T
        X_sfs_bs = E_sfs_bs.T
        X_boot = E_boot_pois.T
        
        X_all = np.vstack([X_sfs_reg, X_sfs_bs, X_boot])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all)
        
        idx_reg_end = len(X_sfs_reg)
        idx_bs_end = idx_reg_end + len(X_sfs_bs)
        
        X_reg_sfs_pca = X_pca[:idx_reg_end]
        X_bs_sfs_pca = X_pca[idx_reg_end:idx_bs_end]
        X_boot_pca = X_pca[idx_bs_end:]
        
        fig_pca, ax_pca = plt.subplots(figsize=(8, 8))
        if len(X_bs_sfs_pca) > 0:
            ax_pca.scatter(X_bs_sfs_pca[:, 0], X_bs_sfs_pca[:, 1], c='lightblue', label='Bootstrap SFS Samples', alpha=0.5, s=15, marker='o')
        if len(X_reg_sfs_pca) > 0:
            ax_pca.scatter(X_reg_sfs_pca[:, 0], X_reg_sfs_pca[:, 1], c='lime', label='Regular SFS Samples', alpha=0.8, s=20, marker='D', edgecolors='black', linewidths=0.5)
        ax_pca.scatter(X_boot_pca[:, 0], X_boot_pca[:, 1], c='red', label='Bootstrap Samples', alpha=0.5, s=15, marker='x')
        ax_pca.set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax_pca.set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax_pca.set_title(f'Patient {pt_name}: Spatial Inclusion (PCA)')
        ax_pca.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "spatial_pca.png"), dpi=200)
        plt.close()
        
        # 3. Error density
        M_pt = M_norm[:, pt_idx]
        
        # Bootstrap error
        M_approx_boot = P_original @ E_boot_pois
        kl_boot = M_pt[:, None] * np.log((M_pt[:, None] + eps) / (M_approx_boot + eps)) - M_pt[:, None] + M_approx_boot
        kl_errs_boot = np.sum(kl_boot, axis=0)
        
        # SFS error
        E_sfs_all = np.hstack([E_sfs_reg, E_sfs_bs])
        M_approx_sfs = np.sum(P_sfs_whole_all * E_sfs_all[None, :, :], axis=1)
        kl_sfs = M_pt[:, None] * np.log((M_pt[:, None] + eps) / (M_approx_sfs + eps)) - M_pt[:, None] + M_approx_sfs
        kl_errors_sfs = np.sum(kl_sfs, axis=0)
        
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

        if np.any(mask_boot_taller):
            ax_dens.bar(bin_centers[mask_boot_taller], hist_boot[mask_boot_taller], width=bin_widths[mask_boot_taller], color='red', alpha=0.6, edgecolor='none', align='center')
        if np.any(mask_sfs_taller):
            ax_dens.bar(bin_centers[mask_sfs_taller], hist_sfs[mask_sfs_taller], width=bin_widths[mask_sfs_taller], color='lightblue', alpha=0.6, edgecolor='none', align='center')

        if np.any(mask_boot_taller):
            ax_dens.bar(bin_centers[mask_boot_taller], hist_sfs[mask_boot_taller], width=bin_widths[mask_boot_taller], color='lightblue', alpha=0.9, edgecolor='none', align='center')
        if np.any(mask_sfs_taller):
            ax_dens.bar(bin_centers[mask_sfs_taller], hist_boot[mask_sfs_taller], width=bin_widths[mask_sfs_taller], color='red', alpha=0.9, edgecolor='none', align='center')

        legend_patches = [
            mpatches.Patch(color='red', alpha=0.8, label='Poisson Bootstrap (KL Div)'),
            mpatches.Patch(color='lightblue', alpha=0.8, label='SFS (KL Div)')
        ]

        ax_dens.axvline(np.mean(kl_errors_sfs), color='blue', linestyle='dashed', linewidth=2, label='SFS Mean Error')
        ax_dens.axvline(np.mean(kl_errs_boot), color='darkred', linestyle='dotted', linewidth=2, label='Bootstrap Mean Error')
        if len(kl_errors_sfs) > 0:
            ax_dens.axvline(kl_errors_sfs[0], color='lime', linestyle='dashdot', linewidth=2, label='Regular SFS Error')
        
        handles, labels = ax_dens.get_legend_handles_labels()
        ax_dens.legend(handles=legend_patches + handles, labels=[p.get_label() for p in legend_patches] + labels)
        
        ax_dens.set_xlabel('Reconstruction Error (KL Divergence)')
        ax_dens.set_ylabel('Density')
        ax_dens.set_yscale('log')
        ax_dens.set_title(f'Patient {pt_name}: Error Landscape Density')
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "error_density.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic patient plots")
    parser.add_argument("--output_dir", default="comparison_output/synthetic2700_all", help="Output directory")
    parser.add_argument("--sample_file", default=None, help="Path to sample file to get names")
    parser.add_argument("--patient_list", default=None, help="Path to a text file containing a list of patients to plot")
    args = parser.parse_args()
    
    generate_patient_plots(args.output_dir, args.sample_file, args.patient_list)
