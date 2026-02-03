
import numpy as np
import argparse

def inspect_npz(file_path):
    data = np.load(file_path)
    print(f"--- File: {file_path} ---")
    print(f"Keys: {list(data.keys())}")
    for key in data.keys():
        val = data[key]
        print(f"  {key}: shape={val.shape}, type={val.dtype}")
        if val.ndim == 1:
            print(f"    first 5 elements: {val[:5]}")
        elif val.ndim == 2:
            print(f"    top-left 2x2: \n{val[:2, :2]}")
    print("-" * 30)

def export_to_csv(dataset_dir):
    """
    Converts all .npz and .npy files in the directory to CSVs in a 'csv_dump' folder.
    """
    import pandas as pd
    import os
    
    dump_dir = os.path.join(dataset_dir, "csv_dump")
    os.makedirs(dump_dir, exist_ok=True)
    print(f"Exporting data to {dump_dir}...")
    
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    files.sort()
    
    for f in files:
        path = os.path.join(dataset_dir, f)
        step_name = f.replace('.npz', '')
        step_dir = os.path.join(dump_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)
        
        data = np.load(path)
        for k in data.keys():
            val = data[k]
            if val.ndim <= 2:
                csv_path = os.path.join(step_dir, f"{k}.csv")
                pd.DataFrame(val).to_csv(csv_path, index=False, header=False)
    
    print("Done export.")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .npz file or directory")
    parser.add_argument("--export", action="store_true", help="Export directory to CSV")
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        if args.export:
            export_to_csv(args.path)
        else:
            print("Directory provided. Use --export to convert all NPZ to CSV.")
    else:
        inspect_npz(args.path)
