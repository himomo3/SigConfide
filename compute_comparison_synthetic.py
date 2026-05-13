import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from sigconfide.utils.utils import load_samples_file, load_signatures_file, kl_divergence, FrobeniusNorm
from sigconfide.estimates.bootstrap import bootstrapSigExposures, bootstrapPoissonSigExposures
from sigconfide.estimates.sfs import sample_sfs, bootstrap_sfs, bootstrap_poisson_sfs
from sigconfide.estimates.standard import findSigExposures
from sigconfide.decompose.qp import decomposeQP

def compute_comparison_synthetic(sample_file, sig_file, truth_file, output_dir="comparison_output"):
    out_path = os.path.join(output_dir, "synthetic2700_all")
    os.makedirs(out_path, exist_ok=True)
    
    start_time = time.time()    
    
    samples, patient_names = load_samples_file(sample_file)
    signatures, sig_names = load_signatures_file(sig_file)
    if sig_names[0] == 'Samples' or sig_names[0] == 'Type':
        sig_names = sig_names[1:]

    # Load ground truth exposures
    truth_df = pd.read_csv(truth_file, index_col=0)
    truth_sigs = truth_df.index.tolist()
    
    # Filter signatures to those present in ground truth
    target_indices = []
    short_names = []
    
    for i, name in enumerate(sig_names):
        # Name in COSMIC matches truth exactly?
        if name in truth_sigs:
            target_indices.append(i)
            short_names.append(name)
            
    if not target_indices:
        print("Could not match any signatures between COSMIC and ground truth.")
        sys.exit(1)
        
    print(f"Matched {len(target_indices)} signatures from ground truth.", flush=True)
    
    P = signatures[:, target_indices]
    sig_names_filtered = np.array(short_names)

    missing_samples = [p for p in patient_names if p not in truth_df.columns]
    if missing_samples:
        print(f"Warning: {len(missing_samples)} samples from input missing in ground truth.", flush=True)
        
    E_truth_whole_all = np.zeros((len(target_indices), len(patient_names)))
    for i, sig in enumerate(sig_names_filtered):
        if sig in truth_df.index:
            for j, pat in enumerate(patient_names):
                if pat in truth_df.columns:
                    E_truth_whole_all[i, j] = truth_df.loc[sig, pat]
    
    print("Running computations on the whole matrix of all synthetic samples...", flush=True)
    M = samples
    M_sums = M.sum(axis=0)
    
    mutation_counts_all = np.zeros(M.shape[1], dtype=int)
    M_norm = np.zeros_like(M, dtype=float)
    
    for i in range(M.shape[1]):
        m_sum = M_sums[i]
        if m_sum < 2:
            mutation_counts_all[i] = 4000 
            M_norm[:, i] = M[:, i] / m_sum if m_sum > 0 else 0
        else:
            mutation_counts_all[i] = int(np.round(m_sum))
            M_norm[:, i] = M[:, i] / mutation_counts_all[i] if mutation_counts_all[i] > 0 else 0
            
    print("  Running Optimal (QP)...", flush=True)
    E_opt_whole_all, frob_opt_whole_all = findSigExposures(M, P)
    
    t0_boot = time.time()
    R_boot = 1000
    print("  Running Bootstrap (Poisson)...", flush=True)
    E_boot_pois_whole, frob_errs_boot_pois_whole, kl_errs_boot_pois_whole = bootstrapPoissonSigExposures(
        M, P, R_boot, mutation_count=list(mutation_counts_all), decomposition_method=decomposeQP
    )
    t1_boot = time.time()
    print(f"  [Bootstraps completed in {t1_boot - t0_boot:.2f} seconds]", flush=True)
    
    print("  Running SFS...", flush=True)
    t0_sfs = time.time()
    E_reg, E_reg_whole, P_reg, P_reg_whole, kl_reg, kl_reg_whole, _, _ = sample_sfs(
        M_norm, P, E_opt_whole_all, max_iter=20000, check=500, eps=1e-10
    )
    print("  Running Bootstrap SFS (Hybrid)...", flush=True)
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
    print(f"  [SFS completed in {t1_sfs - t0_sfs:.2f} seconds]", flush=True)
    
    # Save global matrices
    global_data_path = os.path.join(out_path, "global_computed_data.npz")
    np.savez(global_data_path,
             E_opt_whole_all=E_opt_whole_all,
             E_truth_whole_all=E_truth_whole_all,
             E_reg_whole_all=E_reg_whole_all,
             E_bs_whole_all=E_bs_whole_all,
             E_boot_pois_whole=E_boot_pois_whole,
             P_original=P,
             P_sfs_whole_all=P_sfs_whole_all,
             M_norm=M_norm,
             kl_errors_sfs_whole_all=kl_errors_sfs_whole_all,
             n_reg_sfs_whole=n_reg_sfs_whole_all,
             sig_names_filtered=sig_names_filtered)
    print(f"  Saved global data to {global_data_path}", flush=True)

    end_time = time.time()
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SFS and Bootstrap signature exposures on Synthetic Benchmark.")
    parser.add_argument("--sample_file", default="tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/Samples.txt", help="Path to sample file")
    parser.add_argument("--sig_file", default="tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/COSMIC_v3.3_SBS_GRCh37.txt", help="Path to signatures file")
    parser.add_argument("--truth_file", default="tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/ground.truth.syn.exposures.csv", help="Path to ground truth exposures")
    parser.add_argument("--output_dir", default="comparison_output", help="Output directory")
    
    args = parser.parse_args()
    compute_comparison_synthetic(args.sample_file, args.sig_file, args.truth_file, args.output_dir)
