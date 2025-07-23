import pandas as pd
import os
from pathlib import Path

"""
This script splits a large CSV file into a specified number of smaller, interleaved CSV files under ./datasets. 
"""

INPUT_CSV = './transfuser_dataset_index.csv'
NUM_SPLITS = 50
OUTPUT_DIR = './datasets

def split_csv_interleaved(input_file, output_dir, num_splits):

    input_path = Path(input_file)
    if not input_path.is_file():
        print(f"Error: Input file not found at '{input_file}'")
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    total_rows = len(df)
    original_columns = df.columns.tolist()
    print(f"Successfully loaded {total_rows} rows.")
    if total_rows == 0:
        print("Input file empty.")
        return

    df['split_id'] = df.index % num_splits
    for split_id, group_df in df.groupby('split_id'):
        part_num = split_id + 1
        output_filename = output_path / f'part_{part_num}.csv'
        group_df.to_csv(output_filename, columns=original_columns, index=False)

    print(f"\nInterleaved splitting completed.")

if __name__ == '__main__':
    split_csv_interleaved(INPUT_CSV, OUTPUT_DIR, NUM_SPLITS)
    
