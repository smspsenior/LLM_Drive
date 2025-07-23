# -*- coding: utf-8 -*-
import os
import gzip
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

"""
This script preprocesses original TransFuser++ dataset, generating an index CSV for training. 
"""

def generate_dataset_index(root_dir, output_csv):
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"Root directory not exists: {root_dir}")
        return

    dataset_records = []
    route_folders = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'rgb_augmented' in dirnames and 'bev_semantics_augmented' in dirnames and 'measurements' in dirnames:
            route_folders.append(Path(dirpath))

    if not route_folders:
        print("Folders not found: 'rgb_augmented', 'bev_semantics_augmented', 'measurements'")
        return

    for route_path in tqdm(route_folders, desc="Routes Processing"):
        rgb_dir = route_path / 'rgb_augmented'
        bev_dir = route_path / 'bev_semantics_augmented'
        measurement_dir = route_path / 'measurements'
        frame_ids = sorted([p.stem for p in rgb_dir.glob('*.jpg')])
        
        for frame_id in frame_ids:
            try:
                rgb_file = rgb_dir / f"{frame_id}.jpg"
                bev_file = bev_dir / f"{frame_id}.png"
                measurement_file = measurement_dir / f"{frame_id}.json.gz"
                
                if not (rgb_file.exists() and bev_file.exists() and measurement_file.exists()):
                    continue
                with gzip.open(measurement_file, 'rt', encoding='utf-8') as f:
                    measurement_data = json.load(f)
                throttle = measurement_data.get('throttle', 0.0)
                brake = float(measurement_data.get('brake', False))
                dataset_records.append({
                    'rgb_path': str(rgb_file),
                    'bev_path': str(bev_file),
                    'throttle': throttle,
                    'brake': brake
                })
            except Exception as e:
                continue

    if not dataset_records:
        print("No valid data! Please check root directory. ")
        return

    df = pd.DataFrame(dataset_records)
    df.to_csv(output_csv, index=False)
    print(f"Dataset index saved to: {output_csv}")

if __name__ == '__main__':
    DATASET_ROOT = '/home/dwy1379/data_disk/CARLA/TFpp_dataset'
    OUTPUT_CSV_FILE = './transfuser_dataset_index.csv'
    
    generate_dataset_index(DATASET_ROOT, OUTPUT_CSV_FILE)
    
