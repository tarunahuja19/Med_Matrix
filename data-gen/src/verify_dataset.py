import os
import glob
import json
import numpy as np

def verify_generated_dataset(data_dir: str = "data/processed_slices"):
    print("=== Dataset Verification ===")
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Total .npz files found in '{data_dir}': {len(npz_files)}")
    
    if len(npz_files) == 0:
        print("No files found yet.")
        return
        
    # Inspect first file
    sample_file = npz_files[0]
    print(f"\nInspecting sample file: {sample_file}")
    
    with np.load(sample_file) as data:
        print("Keys present in .npz:")
        for key in data.files:
            val = data[key]
            if key == "meta":
                print(f"  - {key:20s}: JSON String ({len(str(val))} chars)")
                meta_json = json.loads(str(val))
                print(f"    Meta content: {meta_json}")
            else:
                print(f"  - {key:20s}: shape {str(val.shape):15s}, dtype {val.dtype}")
                
    # Class distribution check across all files
    class_counts = {}
    for f in npz_files:
        with np.load(f) as data:
            lbl = int(data["class_label"])
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
            
    print("\nClass distribution across generated slices:")
    for lbl, count in sorted(class_counts.items()):
        print(f"  Class {lbl}: {count} slices")
        
    print("\n=== Dataset Verification Complete ===")

if __name__ == "__main__":
    verify_generated_dataset()
