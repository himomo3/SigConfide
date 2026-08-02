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
from SigProfilerAssignment import Analyzer as Analyze

def compute_comparison_synthetic(sample_file, sig_file, truth_file, output_dir="synthetic2700_all", run_hybrid=0):
    out_path = os.path.join("comparison_output", output_dir)
    os.makedirs(out_path, exist_ok=True)
    
    start_time = time.time()    
    
    samples, patient_names = load_samples_file(sample_file)
    signatures, sig_names = load_signatures_file(sig_file)
    if sig_names[0] == 'Samples' or sig_names[0] == 'Type' or sig_names[0] == 'Sampl' or sig_names[0].startswith('Samp'):
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
    t1_sfs = time.time()
    print(f"  [SFS completed in {t1_sfs - t0_sfs:.2f} seconds]", flush=True)

    E_reg_whole_all = E_reg_whole
    
    if run_hybrid == 1:
        print("  Running Bootstrap SFS (Hybrid)...", flush=True)
        t0_hybrid = time.time()
        E_bs, E_bs_whole, P_bs, P_bs_whole, kl_bs, kl_bs_whole, _, _ = bootstrap_sfs(
            M_norm, P, E_opt_whole_all, R=10, mutation_count=list(mutation_counts_all), decomposition_method=decomposeQP, max_iter=20000, check=500, eps=1e-10
        )
        t1_hybrid = time.time()
        print(f"  [Bootstrap SFS (Hybrid) completed in {t1_hybrid - t0_hybrid:.2f} seconds]", flush=True)
            
        E_sfs_all = np.concatenate([np.expand_dims(E_reg, axis=-1), E_bs], axis=-1)
        E_sfs_whole_all = np.concatenate([E_reg_whole, E_bs_whole], axis=-1)
        
        E_bs_whole_all = E_bs_whole

        P_sfs_all = np.concatenate([np.expand_dims(P_reg, axis=-1), P_bs], axis=-1)
        P_sfs_whole_all = np.concatenate([P_reg_whole, P_bs_whole], axis=-1)
        kl_errors_sfs_all = np.concatenate([[kl_reg], kl_bs])
        kl_errors_sfs_whole_all = np.concatenate([kl_reg_whole, kl_bs_whole])
    else:
        E_sfs_all = np.expand_dims(E_reg, axis=-1)
        E_sfs_whole_all = E_reg_whole
        P_sfs_all = np.expand_dims(P_reg, axis=-1)
        P_sfs_whole_all = P_reg_whole
        kl_errors_sfs_all = np.array([kl_reg])
        kl_errors_sfs_whole_all = kl_reg_whole
    n_reg_sfs_whole_all = E_reg_whole.shape[-1]
    
    print("  Running SigProfilerAssignment...", flush=True)
    spa_output_dir = os.path.join(out_path, "spa_output")
    try:
        Analyze.cosmic_fit(
            samples=sample_file,
            output=spa_output_dir,
            input_type="matrix",
            signature_database=sig_file,
            genome_build="GRCh37",
            make_plots=False,
            verbose=False
        )
        
        spa_activities_path = os.path.join(spa_output_dir, "Assignment_Solution", "Activities", "Assignment_Solution_Activities.txt")
        spa_df = pd.read_csv(spa_activities_path, sep="\t", index_col=0)
        
        E_spa_whole_all = np.zeros((len(target_indices), len(patient_names)))
        for i, sig in enumerate(sig_names_filtered):
            if sig in spa_df.columns:
                for j, pat in enumerate(patient_names):
                    if pat in spa_df.index:
                        E_spa_whole_all[i, j] = spa_df.loc[pat, sig]
        
        # Normalize SPA exposures to 0-1 proportions
        spa_sums = E_spa_whole_all.sum(axis=0)
        E_spa_whole_all = np.divide(E_spa_whole_all, spa_sums, out=np.zeros_like(E_spa_whole_all), where=spa_sums!=0)
    except Exception as e:
        print(f"  [SPA failed: {e}]", flush=True)
        E_spa_whole_all = np.zeros((len(target_indices), len(patient_names)))
    
    # ---- METRICS CALCULATION ----
    print("  Computing evaluation metrics...", flush=True)
    
    # Normalize truth to fractional exposures to match predictions
    truth_sums = np.sum(E_truth_whole_all, axis=0)
    truth_sums[truth_sums == 0] = 1.0 # avoid div zero
    E_truth_frac = E_truth_whole_all / truth_sums

    def compute_f1(truth, pred, thresh=1e-5):
        t_bool = truth > thresh
        p_bool = pred > thresh
        tp = np.sum(t_bool & p_bool)
        fp = np.sum(~t_bool & p_bool)
        fn = np.sum(t_bool & ~p_bool)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def compute_l1(truth, pred):
        return np.mean(np.abs(truth - pred))

    def compute_cosine(truth, pred):
        num_samples = truth.shape[1]
        cos_sims = []
        for i in range(num_samples):
            t = truth[:, i]
            p = pred[:, i]
            norm_t = np.linalg.norm(t)
            norm_p = np.linalg.norm(p)
            if norm_t > 0 and norm_p > 0:
                cos_sims.append(np.dot(t, p) / (norm_t * norm_p))
            elif norm_t == 0 and norm_p == 0:
                cos_sims.append(1.0)
            else:
                cos_sims.append(0.0)
        return np.mean(cos_sims)

    def compute_coverage(truth, lower, upper):
        eps = 1e-7
        covered = (truth >= lower - eps) & (truth <= upper + eps)
        return np.mean(covered) * 100.0

    def compute_interval_width(lower, upper):
        return np.mean(upper - lower)

    E_sfs_mean = np.mean(E_reg_whole_all, axis=2)
    E_boot_mean = np.mean(E_boot_pois_whole, axis=2)

    E_sfs_lower = np.min(E_reg_whole_all, axis=2)
    E_sfs_upper = np.max(E_reg_whole_all, axis=2)
    
    E_boot_lower = np.percentile(E_boot_pois_whole, 2.5, axis=2)
    E_boot_upper = np.percentile(E_boot_pois_whole, 97.5, axis=2)

    num_samples = E_truth_frac.shape[1]

    metrics = {
        "Method": ["QP", "SFS", "Bootstrap", "SigProfilerAssignment"],
        "Sample Size": [num_samples, num_samples, num_samples, num_samples],
        "F1-Score": [
            compute_f1(E_truth_frac, E_opt_whole_all),
            compute_f1(E_truth_frac, E_sfs_mean),
            compute_f1(E_truth_frac, E_boot_mean),
            compute_f1(E_truth_frac, E_spa_whole_all)
        ],
        "L1 Error": [
            compute_l1(E_truth_frac, E_opt_whole_all),
            compute_l1(E_truth_frac, E_sfs_mean),
            compute_l1(E_truth_frac, E_boot_mean),
            compute_l1(E_truth_frac, E_spa_whole_all)
        ],
        "Cosine Similarity": [
            compute_cosine(E_truth_frac, E_opt_whole_all),
            compute_cosine(E_truth_frac, E_sfs_mean),
            compute_cosine(E_truth_frac, E_boot_mean),
            compute_cosine(E_truth_frac, E_spa_whole_all)
        ],
        "Coverage (%)": [
            None,
            compute_coverage(E_truth_frac, E_sfs_lower, E_sfs_upper),
            compute_coverage(E_truth_frac, E_boot_lower, E_boot_upper),
            None
        ],
        "Interval Width": [
            None,
            compute_interval_width(E_sfs_lower, E_sfs_upper),
            compute_interval_width(E_boot_lower, E_boot_upper),
            None
        ]
    }
    
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(out_path, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved metrics to {metrics_csv_path}", flush=True)

    # Save global matrices
    global_data_path = os.path.join(out_path, "global_computed_data.npz")
    save_dict = dict(
        E_opt_whole_all=E_opt_whole_all,
        E_truth_whole_all=E_truth_whole_all,
        E_reg_whole_all=E_reg_whole_all,
        E_boot_pois_whole=E_boot_pois_whole,
        E_spa_whole_all=E_spa_whole_all,
        P_original=P,
        P_sfs_whole_all=P_sfs_whole_all,
        M_norm=M_norm,
        kl_errors_sfs_whole_all=kl_errors_sfs_whole_all,
        n_reg_sfs_whole=n_reg_sfs_whole_all,
        sig_names_filtered=sig_names_filtered,
        patient_names=patient_names
    )
    if run_hybrid == 1:
        save_dict['E_bs_whole_all'] = E_bs_whole_all
        
    np.savez(global_data_path, **save_dict)
    print(f"  Saved global data to {global_data_path}", flush=True)

    end_time = time.time()
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SFS and Bootstrap signature exposures on Synthetic Benchmark.")
    parser.add_argument("--sample_file", default="tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/Samples.txt", help="Path to sample file")
    parser.add_argument("--sig_file", default="tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/COSMIC_v3.3_SBS_GRCh37.txt", help="Path to signatures file")
    parser.add_argument("--truth_file", default="tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/ground.truth.syn.exposures.csv", help="Path to ground truth exposures")
    parser.add_argument("--output_dir", default="synthetic2700_all", help="Output directory")
    parser.add_argument("--hybrid", type=int, default=0, help="Run hybrid bootstrap SFS (1) or skip it (0)")
    
    args = parser.parse_args()
    compute_comparison_synthetic(args.sample_file, args.sig_file, args.truth_file, args.output_dir, args.hybrid)
