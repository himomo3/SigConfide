import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run SFS and Bootstrap comparison pipeline for each organ separately")
    parser.add_argument("--input_base", default="tests/data/Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS_separate", help="Path to SBS_separate directory")
    parser.add_argument("--output_base", default="comparison_output", help="Output base directory")
    args = parser.parse_args()
    
    input_base_path = Path(args.input_base)
    if not input_base_path.exists():
        print(f"Error: {input_base_path} does not exist.")
        sys.exit(1)
        
    for organ_dir in input_base_path.iterdir():
        if not organ_dir.is_dir():
            continue
            
        organ_name = organ_dir.name
        print(f"\n=======================================================")
        print(f"Running pipeline for organ: {organ_name}")
        print(f"=======================================================\n")
        
        # Files for this organ
        sample_file = str(organ_dir / "Samples.txt")
        sig_file = str(organ_dir / "COSMIC_v3.3_SBS_GRCh37.txt")
        truth_file = str(organ_dir / "ground.truth.syn.exposures.csv")
        
        if not os.path.exists(sample_file) or not os.path.exists(sig_file) or not os.path.exists(truth_file):
            print(f"Skipping {organ_name} due to missing required files.")
            continue
            
        output_dir = os.path.join(args.output_base, f"synthetic2700_{organ_name}")
        
        # 1. Computation step
        print(f"=== Starting Computation ===")
        compute_script = os.path.join(os.path.dirname(__file__), "compute_comparison_synthetic.py")
        
        compute_cmd = [
            sys.executable, compute_script,
            "--sample_file", sample_file,
            "--sig_file", sig_file,
            "--truth_file", truth_file,
            "--output_dir", output_dir
        ]
        
        result_compute = subprocess.run(compute_cmd)
        if result_compute.returncode != 0:
            print(f"Error: Computation script failed for {organ_name}.")
            continue
            
        # 2. Visualisation step
        print(f"\n=== Starting Visualisation ===")
        visualise_script = os.path.join(os.path.dirname(__file__), "visualise_comparison_synthetic.py")
        
        vis_cmd = [
            sys.executable, visualise_script,
            "--output_dir", output_dir
        ]
        
        result_vis = subprocess.run(vis_cmd)
        if result_vis.returncode != 0:
            print(f"Error: Visualisation script failed for {organ_name}.")
            continue
            
        # 3. Patient Plots step
        print(f"\n=== Starting Patient Plots ===")
        plots_script = os.path.join(os.path.dirname(__file__), "generate_synthetic_patient_plots.py")
        
        plots_cmd = [
            sys.executable, plots_script,
            "--output_dir", os.path.join(output_dir, "synthetic2700_all"),
            "--sample_file", sample_file
        ]
        
        result_plots = subprocess.run(plots_cmd)
        if result_plots.returncode != 0:
            print(f"Error: Patient plots script failed for {organ_name}.")
            continue

    print(f"\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
