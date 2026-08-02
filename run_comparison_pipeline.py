import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run SFS and Bootstrap comparison pipeline (Computation + Visualisation)")
    parser.add_argument("--sample_file", default="tests/data/counts_tumorBRCA.txt", help="Path to sample file")
    parser.add_argument("--sig_file", default="sigconfide/utils/data/COSMIC_v2_SBS_GRCh37.txt", help="Path to signatures file")
    parser.add_argument("--patients", nargs="+", default=["PD24196", "PD8609", "PD13608"], help="List of patients")
    parser.add_argument("--output_dir", default="brca560", help="Output directory inside comparison_output")
    parser.add_argument("--bootstrap_type", choices=["regular", "poisson", "all", "none"], default="poisson", help="Bootstrap type")
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid bootstrap SFS")
    parser.add_argument("--no-spa", dest="run_spa", action="store_false", help="Disable SigProfilerAssignment")
    parser.add_argument("--whole", action="store_true", default=True, help="Compute on whole matrix (default: True)")
    parser.add_argument("--no-whole", dest="whole", action="store_false", help="Compute per patient")
    parser.add_argument("--run_all", action="store_true", help="Run all combinations of hard-coded samples and bootstrap types")
    args = parser.parse_args()
    
    if args.run_all:
        samples_to_run = ["tests/data/tumorBRCA.txt", "tests/data/counts_tumorBRCA.txt"]
        bootstrap_types_to_run = ["regular", "poisson"]
    else:
        samples_to_run = [args.sample_file]
        bootstrap_types_to_run = [args.bootstrap_type]
        
    for current_sample in samples_to_run:
        for current_bootstrap in bootstrap_types_to_run:
            print(f"\n=======================================================")
            print(f"Running pipeline for sample: {current_sample}, bootstrap: {current_bootstrap}")
            print(f"=======================================================\n")
            
            # 1. Computation step
            print(f"=== Starting Computation ===")
            compute_script = os.path.join(os.path.dirname(__file__), "compute_comparison.py")
            
            compute_cmd = [
                sys.executable, compute_script,
                "--sample_file", current_sample,
                "--sig_file", args.sig_file,
                "--output_dir", args.output_dir,
                "--bootstrap_type", current_bootstrap,
            ]
            if args.hybrid:
                compute_cmd.append("--hybrid")
            if not args.run_spa:
                compute_cmd.append("--no-spa")
            if not args.whole:
                compute_cmd.append("--no-whole")
                
            compute_cmd.extend(["--patients"] + args.patients)
            
            result_compute = subprocess.run(compute_cmd)
            if result_compute.returncode != 0:
                print(f"Error: Computation script failed for {current_sample} ({current_bootstrap}).")
                sys.exit(result_compute.returncode)
                
            # 2. Visualisation step
            print(f"\n=== Starting Visualisation ===")
            visualise_script = os.path.join(os.path.dirname(__file__), "visualise_comparison.py")
            
            vis_cmd = [
                sys.executable, visualise_script,
                "--sample_file", current_sample,
                "--sig_file", args.sig_file,
                "--output_dir", args.output_dir,
                "--bootstrap_type", current_bootstrap,
            ]
            vis_cmd.extend(["--patients"] + args.patients)
            
            result_vis = subprocess.run(vis_cmd)
            if result_vis.returncode != 0:
                print(f"Error: Visualisation script failed for {current_sample} ({current_bootstrap}).")
                sys.exit(result_vis.returncode)

    print(f"\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
