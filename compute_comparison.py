import os
import sys
import time
import argparse
import numpy as np

from sigconfide.utils.utils import load_samples_file, load_signatures_file, kl_divergence, FrobeniusNorm
from sigconfide.estimates.bootstrap import bootstrapSigExposures, bootstrapPoissonSigExposures
from sigconfide.estimates.sfs import sample_sfs, bootstrap_sfs, bootstrap_poisson_sfs
from sigconfide.estimates.standard import findSigExposures
from sigconfide.decompose.qp import decomposeQP

def compute_comparison(sample_file, sig_file, patients, whole=True, output_dir="comparison_output", bootstrap_type="all"):
    base_sample = os.path.splitext(os.path.basename(sample_file))[0]
    parent_folder = f"{base_sample}_{bootstrap_type}BS"
    out_path = os.path.join(output_dir, parent_folder)
    
    start_time = time.time()    
    os.makedirs(out_path, exist_ok=True)
    
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
    
    if whole:
        print("Running computations on the whole matrix of all patients...")
        M = samples
        M_sums = M.sum(axis=0)
        
        mutation_counts_all = np.zeros(M.shape[1], dtype=int)
        M_norm = np.zeros_like(M, dtype=float)
        
        for i in range(M.shape[1]):
            m_sum = M_sums[i]
            if m_sum < 2:
                mutation_counts_all[i] = 4000 # assume 4000 mutations for samples with only ratios/probabilities
                M_norm[:, i] = M[:, i] / m_sum if m_sum > 0 else 0
            else:
                mutation_counts_all[i] = int(np.round(m_sum))
                M_norm[:, i] = M[:, i] / mutation_counts_all[i] if mutation_counts_all[i] > 0 else 0
                
        print("  Running Optimal...")
        E_opt_whole_all, frob_opt_whole_all = findSigExposures(M, P)
        
        print("  Running Bootstrap (Regular)...")
        t0_boot = time.time()
        R_boot = 1000
        E_boot_reg_whole, frob_errs_boot_reg_whole, kl_errs_boot_reg_whole = bootstrapSigExposures(
            M, P, R_boot, mutation_count=list(mutation_counts_all), decomposition_method=decomposeQP
        )
        print("  Running Bootstrap (Poisson)...")
        E_boot_pois_whole, frob_errs_boot_pois_whole, kl_errs_boot_pois_whole = bootstrapPoissonSigExposures(
            M, P, R_boot, mutation_count=list(mutation_counts_all), decomposition_method=decomposeQP
        )
        t1_boot = time.time()
        print(f"  [Bootstraps completed in {t1_boot - t0_boot:.2f} seconds]")
        
        print("  Running SFS...")
        t0_sfs = time.time()
        E_reg, E_reg_whole, P_reg, P_reg_whole, kl_reg, kl_reg_whole, _, _ = sample_sfs(
            M_norm, P, E_opt_whole_all, max_iter=20000, check=500, eps=1e-10
        )
        print("  Running Bootstrap SFS (Hybrid)...")
        E_bs, E_bs_whole, P_bs, P_bs_whole, kl_bs, kl_bs_whole, _, _ = bootstrap_sfs(
            M_norm, P, E_opt_whole_all, R=10, mutation_count=list(mutation_counts_all), decomposition_method=decomposeQP, max_iter=20000, check=500, eps=1e-10
        )
            
        E_sfs_all = np.concatenate([np.expand_dims(E_reg, axis=-1), E_bs], axis=-1)
        E_sfs_whole_all = np.concatenate([E_reg_whole, E_bs_whole], axis=-1)
        
        E_reg_whole_all = E_reg_whole
        E_bs_whole_all = E_bs_whole

        P_sfs_all = np.concatenate([np.expand_dims(P_reg, axis=-1), P_bs], axis=-1)
        P_sfs_whole_all = np.concatenate([P_reg_whole, P_bs_whole], axis=-1)
        kl_errors_sfs_all = np.concatenate([[kl_reg], kl_bs])
        kl_errors_sfs_whole_all = np.concatenate([kl_reg_whole, kl_bs_whole])
        n_reg_sfs_whole_all = E_reg_whole.shape[-1]
        t1_sfs = time.time()
        print(f"  [SFS completed in {t1_sfs - t0_sfs:.2f} seconds]")
        
        # Save global matrices
        global_data_path = os.path.join(out_path, "global_computed_data.npz")
        np.savez(global_data_path,
                 E_reg_whole_all=E_reg_whole_all,
                 P_sfs_whole_all=P_sfs_whole_all,
                 M_norm=M_norm,
                 n_reg_sfs_whole=n_reg_sfs_whole_all)
        print(f"  Saved global data to {global_data_path}")
    
    for pt in patients:
        print(f"Processing patient {pt}")
        if pt not in patient_names:
            print(f"Patient {pt} not found!")
            continue
            
        pt_dir = os.path.join(out_path, pt)
        os.makedirs(pt_dir, exist_ok=True)
            
        pt_idx = np.where(patient_names == pt)[0][0]
        m = samples[:, pt_idx]
        
        m_sum = m.sum()
        if m_sum < 2:
            # Assumed to be probabilities since sum <= 1.0 (with a bit of margin)
            mutation_count = 4000
            m_norm = m / m_sum
        else:
            mutation_count = int(np.round(m_sum))
            m_norm = m / mutation_count

        if mutation_count == 0:
            print(f"No mutations for {pt}")
            continue
                
        if whole:
            # Select the particular patient's results
            E_opt = E_opt_whole_all[:, pt_idx]
            frob_opt = frob_opt_whole_all[pt_idx]
            
            E_boot_reg = E_boot_reg_whole[:, pt_idx, :]
            E_boot_pois = E_boot_pois_whole[:, pt_idx, :]
            kl_errs_boot = kl_errs_boot_reg_whole[pt_idx, :]
            kl_errs_boot_pois = kl_errs_boot_pois_whole[pt_idx, :]
            
            E_sfs = E_sfs_all[:, pt_idx, :]
            E_sfs_whole = E_sfs_whole_all[:, pt_idx, :]
            E_sfs_reg_whole = E_reg_whole_all[:, pt_idx, :]
            E_sfs_bs_whole = E_bs_whole_all[:, pt_idx, :]
            
            P_sfs = P_sfs_all
            P_sfs_whole = P_sfs_whole_all
            
            # Calculate patient-specific errors from the SFS whole-matrix trajectory
            eps = 1e-10
            m_expanded = m_norm[:, np.newaxis]
            
            m_approx_sfs = np.einsum('kni,ni->ki', P_sfs, E_sfs)
            kl_matrix = m_expanded * np.log((m_expanded + eps) / (m_approx_sfs + eps)) - m_expanded + m_approx_sfs
            kl_errors_sfs = np.sum(kl_matrix, axis=0)
            
            m_approx_sfs_whole = np.einsum('kni,ni->ki', P_sfs_whole, E_sfs_whole)
            kl_matrix_whole = m_expanded * np.log((m_expanded + eps) / (m_approx_sfs_whole + eps)) - m_expanded + m_approx_sfs_whole
            kl_errors_sfs_whole = np.sum(kl_matrix_whole, axis=0)
            
            n_reg_sfs_whole = n_reg_sfs_whole_all
        else:
            # Optimal
            E_opt_all_pt, frob_opt_all_pt = findSigExposures(m.reshape(-1, 1), P)
            E_opt = E_opt_all_pt[:, 0]
            frob_opt = frob_opt_all_pt[0]
            
            print(f"  Running Bootstrap (Regular)...")
            R_boot = 1000
            E_boot_reg, frob_errs_boot_reg, kl_errs_boot = bootstrapSigExposures(
                m, P, R_boot, mutation_count=mutation_count, decomposition_method=decomposeQP
            )
            print(f"  Running Bootstrap (Poisson)...")
            E_boot_pois, frob_errs_boot_pois, kl_errs_boot_pois = bootstrapPoissonSigExposures(
                m, P, R_boot, mutation_count=mutation_count, decomposition_method=decomposeQP
            )
            
            print(f"  Running SFS...")
            E_reg, E_reg_whole, P_reg, P_reg_whole, kl_reg, kl_reg_whole, _, _ = sample_sfs(
                m_norm, P, E_opt, max_iter=20000, check=500, eps=1e-10
            )
            print(f"  Running Bootstrap SFS (Hybrid)...")
            E_bs, E_bs_whole, P_bs, P_bs_whole, kl_bs, kl_bs_whole, _, _ = bootstrap_sfs(
                m_norm, P, E_opt, R=10, mutation_count=mutation_count, decomposition_method=decomposeQP, max_iter=20000, check=500, eps=1e-10
            )
                
            E_sfs = np.concatenate([np.expand_dims(E_reg, axis=-1), E_bs], axis=-1)
            E_sfs_whole = np.concatenate([E_reg_whole, E_bs_whole], axis=-1)
            E_sfs_reg_whole = E_reg_whole
            E_sfs_bs_whole = E_bs_whole
            
            P_sfs = np.concatenate([np.expand_dims(P_reg, axis=-1), P_bs], axis=-1)
            P_sfs_whole = np.concatenate([P_reg_whole, P_bs_whole], axis=-1)
            kl_errors_sfs = np.concatenate([[kl_reg], kl_bs])
            kl_errors_sfs_whole = np.concatenate([kl_reg_whole, kl_bs_whole])
            n_reg_sfs_whole = E_reg_whole.shape[-1]
        
        if len(kl_errors_sfs) == 0:
            print(f"  SFS generated no samples for {pt}.")
            continue
            
        print(f"  Generated {E_sfs.shape[-1]} SFS samples.")
        
        # Opt errors for baseline relative calculations
        kl_opt = kl_divergence(m_norm, P @ E_opt)
        
        # Save computed data to npz for later visualisation
        data_path = os.path.join(pt_dir, "computed_data.npz")
        np.savez(data_path,
                 E_sfs=E_sfs,
                 E_sfs_whole=E_sfs_whole,
                 P_sfs=P_sfs,
                 P_sfs_whole=P_sfs_whole,
                 E_boot_reg=E_boot_reg,
                 E_boot_pois=E_boot_pois,
                 E_sfs_reg_whole=E_sfs_reg_whole,
                 E_sfs_bs_whole=E_sfs_bs_whole,
                 E_opt=E_opt,
                 kl_errors_sfs=kl_errors_sfs,
                 kl_errors_sfs_whole=kl_errors_sfs_whole,
                 kl_errs_boot=kl_errs_boot,
                 kl_errs_boot_pois=kl_errs_boot_pois,
                 sig_names_filtered=sig_names_filtered,
                 n_reg_sfs_whole=n_reg_sfs_whole)
        print(f"  Saved computation results to {data_path}")

    end_time = time.time()
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SFS and Bootstrap signature exposures.")
    parser.add_argument("--sample_file", default="tests/data/tumorBRCA.txt", help="Path to sample file")
    parser.add_argument("--sig_file", default="sigconfide/utils/data/COSMIC_v2_SBS_GRCh37.txt", help="Path to signatures file")
    parser.add_argument("--patients", nargs="+", default=["PD24196", "PD8609", "PD13608"], help="List of patients")
    parser.add_argument("--output_dir", default="comparison_output", help="Output directory")
    parser.add_argument("--bootstrap_type", choices=["regular", "poisson", "all"], default="all", help="Bootstrap type (legacy param, defaults to all)")
    parser.add_argument("--whole", action="store_true", default=True, help="Compute on whole matrix (default: True)")
    parser.add_argument("--no-whole", dest="whole", action="store_false", help="Compute per patient")
    
    args = parser.parse_args()
    compute_comparison(args.sample_file, args.sig_file, args.patients, args.whole, args.output_dir, args.bootstrap_type)
