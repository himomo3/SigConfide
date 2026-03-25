import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run SFS and Bootstrap comparison pipeline (Computation + Visualisation)")
    parser.add_argument("--sample_file", default="tests/data/tumorBRCA.txt", help="Path to sample file")
    parser.add_argument("--sig_file", default="sigconfide/utils/data/COSMIC_v2_SBS_GRCh37.txt", help="Path to signatures file")
    parser.add_argument("--patients", nargs="+", default=["PD24196", "PD8609", "PD13608"], help="List of patients")
    parser.add_argument("--output_dir", default="comparison_output", help="Output directory")
    parser.add_argument("--whole", action="store_true", default=True, help="Compute on whole matrix (default: True)")
    parser.add_argument("--no-whole", dest="whole", action="store_false", help="Compute per patient")
    args = parser.parse_args()
    
    # 1. Computation step
    print(f"=== Starting Computation ===")
    compute_script = os.path.join(os.path.dirname(__file__), "compute_comparison.py")
    
    compute_cmd = [
        sys.executable, compute_script,
        "--sample_file", args.sample_file,
        "--sig_file", args.sig_file,
        "--output_dir", args.output_dir,
    ]
    if not args.whole:
        compute_cmd.append("--no-whole")
        
    compute_cmd.extend(["--patients"] + args.patients)
    
    result_compute = subprocess.run(compute_cmd)
    if result_compute.returncode != 0:
        print("Error: Computation script failed.")
        sys.exit(result_compute.returncode)
        
    # 2. Visualisation step
    print(f"\n=== Starting Visualisation ===")
    visualise_script = os.path.join(os.path.dirname(__file__), "visualise_comparison.py")
    
    vis_cmd = [
        sys.executable, visualise_script,
        "--output_dir", args.output_dir,
    ]
    vis_cmd.extend(["--patients"] + args.patients)
    
    result_vis = subprocess.run(vis_cmd)
    if result_vis.returncode != 0:
        print("Error: Visualisation script failed.")
        sys.exit(result_vis.returncode)

    print(f"\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
