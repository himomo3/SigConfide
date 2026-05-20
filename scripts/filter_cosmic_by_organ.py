import argparse
import pandas as pd
from pathlib import Path

def filter_cosmic_for_organs(base_dir):
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"Error: {base_dir} is not a valid directory.")
        return

    # Iterate over all organ subdirectories
    for organ_dir in base_path.iterdir():
        if not organ_dir.is_dir():
            continue
            
        print(f"\nProcessing {organ_dir.name}...")
        
        gt_file = organ_dir / "ground.truth.syn.exposures.csv"
        cosmic_file = organ_dir / "COSMIC_v3.3_SBS_GRCh37.txt"
        
        if not gt_file.exists() or not cosmic_file.exists():
            print(f"  Missing required files in {organ_dir.name}. Skipping.")
            continue
            
        # 1. Load ground truth and find active signatures
        df_gt = pd.read_csv(gt_file, index_col=0)
        active_sigs_mask = df_gt.sum(axis=1) > 0
        active_sigs = df_gt[active_sigs_mask].index.tolist()
        print(f"  Found {len(active_sigs)} active signatures: {', '.join(active_sigs)}")
        
        # 2. Filter the ground truth file to remove all-zero rows (optional but keeps things clean)
        df_gt_filtered = df_gt[active_sigs_mask]
        df_gt_filtered.to_csv(gt_file)
        print(f"  Cleaned ground truth to only include active signatures.")
        
        # 3. Load and filter the COSMIC file
        df_cosmic = pd.read_csv(cosmic_file, sep='\t', index_col=0)
        
        # Keep only the active signatures that are also in the COSMIC file
        valid_cols = [sig for sig in active_sigs if sig in df_cosmic.columns]
        df_cosmic_filtered = df_cosmic[valid_cols]
        
        # 4. Save the filtered COSMIC file
        df_cosmic_filtered.to_csv(cosmic_file, sep='\t')
        print(f"  Saved organ-specific COSMIC matrix.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter COSMIC matrices based on active ground truth signatures.")
    parser.add_argument('base_dir', type=str, help="Path to the SBS_separate directory")
    args = parser.parse_args()
    
    filter_cosmic_for_organs(args.base_dir)
