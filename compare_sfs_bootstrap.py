import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA

from sigconfide.utils.utils import load_samples_file, load_signatures_file, kl_divergence, FrobeniusNorm
from sigconfide.estimates.bootstrap import bootstrapSigExposures
from sigconfide.estimates.sfs import sample_sfs
from sigconfide.decompose.qp import decomposeQP

def run_comparison(sample_file, sig_file, patients, output_dir="comparison_output"):
    os.makedirs(output_dir, exist_ok=True)
    
    samples, patient_names = load_samples_file(sample_file)
    signatures, sig_names = load_signatures_file(sig_file)
    if sig_names[0] == 'Samples' or sig_names[0] == 'Type':
        sig_names = sig_names[1:]
    
    # Target subset of COSMIC signatures
    target_sigs_num = [1, 2, 3, 5, 6, 8, 9, 12, 13, 16, 17, 18, 20, 26, 30]
    
    # Map valid signatures from the input list based on text digits
    target_indices = []
    short_names = []
    for i, name in enumerate(sig_names):
        # Handle formats like "Signature 1", "SBS1", "Signature_1"
        import re
        match = re.search(r'\d+', name)
        if match:
            sig_num = int(match.group())
            if sig_num in target_sigs_num:
                target_indices.append(i)
                short_names.append(f"Sig {sig_num}")
                
    if not target_indices:
        print("Could not find the target subset in your signatures file formatting.")
        target_indices = list(range(len(sig_names)))
        short_names = [f"Sig {i+1}" for i in target_indices]
        
    P = signatures[:, target_indices]
    sig_names_filtered = np.array(short_names)
    
    for pt in patients:
        print(f"Processing patient {pt}")
        if pt not in patient_names:
            print(f"Patient {pt} not found!")
            continue
            
        pt_dir = os.path.join(output_dir, pt)
        os.makedirs(pt_dir, exist_ok=True)
            
        pt_idx = np.where(patient_names == pt)[0][0]
        m = samples[:, pt_idx]
        
        mutation_count = int(np.round(m.sum()))
        if mutation_count == 0:
            print(f"No mutations for {pt}")
            continue
            
        m_norm = m / mutation_count
        
        # Optimal
        E_opt = decomposeQP(m_norm, P)
        E_opt = E_opt / np.sum(E_opt)
        
        print(f"  Running Bootstrap...")
        R_boot = 1000
        E_boot, frob_errs_boot, kl_errs_boot = bootstrapSigExposures(
            m, P, R_boot, mutation_count=mutation_count, decomposition_method=decomposeQP
        )
        
        print(f"  Running SFS...")
        E_sfs, frob_errors_sfs, kl_errors_sfs = sample_sfs(m_norm, P, E_opt, max_iter=20000, check=500, eps=1e-8)
        
        if len(kl_errors_sfs) == 0:
            print(f"  SFS generated no samples for {pt}.")
            continue
            
        print(f"  Generated {E_sfs.shape[-1]} SFS samples.")
        
        # Calculate centroids
        wsfs_centroid = np.mean(E_sfs, axis=1)
        wboot_consensus = np.mean(E_boot, axis=1)
        
        # Opt errors for baseline relative calculations
        kl_opt = kl_divergence(m_norm, P @ E_opt)
        frob_opt = FrobeniusNorm(m_norm, P, E_opt)
        
        active_sigs = np.where((wsfs_centroid > 0.01) | (wboot_consensus > 0.01))[0]
        
        # === RESTORED ORIGINAL PLOTS ===
        
        # 1. Element-Wise Bounds vs Confidence Intervals
        fig_bounds, ax_bounds = plt.subplots(figsize=(10, 6))
        x_bounds = np.arange(len(active_sigs))
        sfs_min = np.min(E_sfs[active_sigs, :], axis=1)
        sfs_max = np.max(E_sfs[active_sigs, :], axis=1)
        boot_lower = np.percentile(E_boot[active_sigs, :], 2.5, axis=1)
        boot_upper = np.percentile(E_boot[active_sigs, :], 97.5, axis=1)
        
        ax_bounds.bar(x_bounds, sfs_max - sfs_min, bottom=sfs_min, color='lightblue', label='SFS Bounds', width=0.5, alpha=0.7)
        ax_bounds.plot(x_bounds, wsfs_centroid[active_sigs], 'o', color='blue', label='SFS Mean', markersize=6)
        ax_bounds.plot(x_bounds, E_opt[active_sigs], '*', color='black', label='Original QP (E_opt)', markersize=8)
        
        boot_err_lower = np.maximum(0, wboot_consensus[active_sigs] - boot_lower)
        boot_err_upper = np.maximum(0, boot_upper - wboot_consensus[active_sigs])
        
        ax_bounds.errorbar(x_bounds, wboot_consensus[active_sigs], yerr=[boot_err_lower, boot_err_upper], 
                           fmt='o', color='red', label='Bootstrap 95% CI & Mean', capsize=5)
        ax_bounds.set_xticks(x_bounds)
        ax_bounds.set_xticklabels(sig_names_filtered[active_sigs], rotation=45, ha='right')
        ax_bounds.set_ylabel('Exposure')
        ax_bounds.set_title(f'Patient {pt}: SFS Bounds vs Bootstrap CI')
        ax_bounds.legend()
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
        min_err = max(np.min(kl_errs_boot), 1e-12)
        max_err = max(np.max(kl_errs_boot), min_err * 10)
        bins = np.logspace(np.log10(min_err), np.log10(max_err), 50)
        ax_dens.hist(kl_errs_boot, bins=bins, alpha=0.5, color='red', label='Bootstrap (KL Div)', density=True, histtype='stepfilled')
        ax_dens.set_xscale('log')
        ax_dens.axvline(np.mean(kl_errors_sfs), color='blue', linestyle='dashed', linewidth=2, label='SFS Mean Error')
        ax_dens.axvline(np.mean(kl_errs_boot), color='darkred', linestyle='dotted', linewidth=2, label='Bootstrap Mean Error')
        ax_dens.set_xlabel('Reconstruction Error (KL Divergence)')
        ax_dens.set_ylabel('Density')
        ax_dens.set_title(f'Patient {pt}: Error Landscape Density')
        ax_dens.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "error_density.png"), dpi=200)
        plt.close()
        
        # === Plot A: Grouped Bar Chart (SFS vs Bootstrap) ===
        figA, axA = plt.subplots(figsize=(10, 5))
        x = np.arange(len(active_sigs))
        width = 0.35
        
        axA.bar(x - width/2, wsfs_centroid[active_sigs], width, label="SFS", color="blue", alpha=0.8)
        axA.bar(x + width/2, wboot_consensus[active_sigs], width, label="Bootstrap", color="green", alpha=0.8)
        
        axA.set_xticks(x)
        axA.set_xticklabels(sig_names_filtered[active_sigs], rotation=45, ha='right')
        axA.set_ylabel("Signature contribution")
        axA.set_title(f"Patient {pt}")
        axA.legend(title="Decomposition method")
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "plot_A_exposures.png"), dpi=200)
        plt.close()
        
        # === Plot B: Decomposition error (KL and Frobenius side-by-side or stacked) ===
        figB, (axB1, axB2) = plt.subplots(2, 1, figsize=(8, 6), sharey=False)
        colors = ['lightblue', 'lightgreen']
        
        # KL Divergence Boxplot
        box_data_kl = [kl_errors_sfs, kl_errs_boot]
        bplot_B1 = axB1.boxplot(box_data_kl, vert=False, tick_labels=["SFS", "Bootstrap"], patch_artist=True)
        for patch, color in zip(bplot_B1['boxes'], colors):
            patch.set_facecolor(color)
        axB1.set_xscale("log")
        axB1.set_xlabel("Decomposition error (KL divergence)")
        
        # Frobenius Norm Boxplot
        box_data_frob = [frob_errors_sfs, frob_errs_boot]
        bplot_B2 = axB2.boxplot(box_data_frob, vert=False, tick_labels=["SFS", "Bootstrap"], patch_artist=True)
        for patch, color in zip(bplot_B2['boxes'], colors):
            patch.set_facecolor(color)
        axB2.set_xscale("log")
        axB2.set_xlabel("Decomposition error (Frobenius norm)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "plot_B_error.png"), dpi=200)
        plt.close()
        
        # === Plot C: Relative Decomposition Error ===
        figC, (axC1, axC2) = plt.subplots(2, 1, figsize=(8, 6), sharey=False)
        
        # Relative KL
        rel_kl_sfs = np.array(kl_errors_sfs) / kl_opt * 100
        rel_kl_boot = np.array(kl_errs_boot) / kl_opt * 100
        bplot_C1 = axC1.boxplot([rel_kl_sfs, rel_kl_boot], vert=False, tick_labels=["SFS", "Bootstrap"], patch_artist=True)
        for patch, color in zip(bplot_C1['boxes'], colors):
            patch.set_facecolor(color)
        axC1.set_xscale("log")
        axC1.set_xlabel("KL divergence relative to QP [%]")
        axC1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}%'.format(y)))
        
        # Relative Frob
        rel_frob_sfs = np.array(frob_errors_sfs) / frob_opt * 100
        rel_frob_boot = np.array(frob_errs_boot) / frob_opt * 100
        bplot_C2 = axC2.boxplot([rel_frob_sfs, rel_frob_boot], vert=False, tick_labels=["SFS", "Bootstrap"], patch_artist=True)
        for patch, color in zip(bplot_C2['boxes'], colors):
            patch.set_facecolor(color)
        axC2.set_xscale("log")
        axC2.set_xlabel("Frobenius norm relative to QP [%]")
        axC2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}%'.format(y)))
        
        plt.tight_layout()
        plt.savefig(os.path.join(pt_dir, "plot_C_relative_error.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    sample_file = "tests/data/tumorBRCA.txt"
    sig_file = "sigconfide/utils/data/COSMIC_v2_SBS_GRCh37.txt"
    patients = ["PD24196", "PD8609", "PD13608"]
    
    run_comparison(sample_file, sig_file, patients, output_dir="comparison_output")
