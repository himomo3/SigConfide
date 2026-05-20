import argparse
import os
import shutil
import pandas as pd
from pathlib import Path

def get_delimiter(filename):
    if filename.endswith('.csv'):
        return ','
    elif filename.endswith('.txt') or filename.endswith('.tsv'):
        return '\t'
    return None

def process_directory(input_dir):
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        return

    output_path = input_path.parent / f"{input_path.name}_separate"
    
    # Identify all files
    files_to_process = [f for f in input_path.iterdir() if f.is_file() and get_delimiter(f.name)]
    
    # First pass: identify all unique organs
    organs = set()
    for file_path in files_to_process:
        sep = get_delimiter(file_path.name)
        try:
            # Only read the first row (header) to find organs
            df_header = pd.read_csv(file_path, sep=sep, index_col=0, nrows=0)
            for col in df_header.columns:
                if '::' in col:
                    organs.add(col.split('::')[0])
        except Exception as e:
            print(f"Error reading header of {file_path.name}: {e}")
            
    if not organs:
        print("No organs found in the dataset (no columns with '::' detected).")
        return

    print(f"Found organs: {', '.join(organs)}")
    
    # Create output directories
    output_path.mkdir(exist_ok=True, parents=True)
    for organ in organs:
        (output_path / organ).mkdir(exist_ok=True)
        
    # Second pass: split or copy files
    for file_path in files_to_process:
        sep = get_delimiter(file_path.name)
        print(f"Processing {file_path.name}...")
        
        try:
            df = pd.read_csv(file_path, sep=sep, index_col=0)
            
            # Check if this file has sample columns
            sample_cols = [col for col in df.columns if '::' in col]
            
            if sample_cols:
                # Split the file by organ
                for organ in organs:
                    # Select columns for this organ
                    organ_cols = [col for col in df.columns if col.startswith(f"{organ}::")]
                    
                    if organ_cols:
                        df_organ = df[organ_cols]
                        output_file = output_path / organ / file_path.name
                        df_organ.to_csv(output_file, sep=sep)
            else:
                # No sample columns, copy to all organ directories
                print(f"  No sample columns found. Copying to all organ folders.")
                for organ in organs:
                    output_file = output_path / organ / file_path.name
                    shutil.copy2(file_path, output_file)
                    
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print(f"Success: Separated data saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Separate genomic data by organ.")
    parser.add_argument('input_dir', type=str, help="Path to the input directory (e.g., SBS)")
    args = parser.parse_args()
    
    process_directory(args.input_dir)
