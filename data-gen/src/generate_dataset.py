import os
import json
import random
import argparse
import numpy as np
from typing import List, Dict, Any

from src.loader import load_brainweb_phantom, extract_axial_slices, BRAINWEB_SUBJECT_SET
from src.physics import generate_parameter_maps, normalize_label_slice
from src.anomaly import insert_anomalies_into_slice, ANOMALY_CLASSES
from src.signal_kspace import generate_spin_echo_image, compute_susceptibility_phase, generate_kspace_from_image
from src.cest import compute_z_spectrum_slice

def process_subject(
    sub_id: int,
    output_dir: str,
    snr_db: float = 25.0
) -> List[Dict[str, Any]]:
    """
    Processes all selected axial slices for a single BrainWeb subject phantom.
    Generates parameter maps, injects anomalies, simulates MR spin-echo image,
    k-space with phase distortions, and CEST Z-spectra, then saves to .npz.
    """
    print(f"\nProcessing Subject ID: {sub_id}...")
    try:
        crisp_vol = load_brainweb_phantom(sub_id=sub_id, contrast="crisp")
    except Exception as e:
        print(f"Warning: Failed to load subject {sub_id} ({e}). Skipping.")
        return []
        
    slices, slice_indices, _ = extract_axial_slices(crisp_vol, start_pct=0.20, end_pct=0.80)
    print(f"  Extracted {len(slice_indices)} axial slices for Subject {sub_id}.")
    
    os.makedirs(output_dir, exist_ok=True)
    slice_records = []
    
    for i, orig_slice_idx in enumerate(slice_indices):
        seg_raw = slices[i]
        norm_seg = normalize_label_slice(seg_raw)
        
        # 1. Parameter maps
        T1_base, T2_base, PD_base, T2star_base, chi_base = generate_parameter_maps(norm_seg)
        
        # 2. Anomaly insertion
        T1, T2, PD, T2star, chi, seg_mask, cls_label, meta_inst, k_sw, f_s = insert_anomalies_into_slice(
            norm_seg, T1_base, T2_base, PD_base, T2star_base, chi_base
        )
        
        # 3. Spin-Echo magnitude image (U-Net input)
        se_img = generate_spin_echo_image(T1, T2, PD, TR=2500.0, TE=85.0)
        image_input = np.expand_dims(se_img, axis=0).astype(np.float32)  # (1, H, W)
        
        # 4. Complex phase & k-space (Mamba input)
        phase_img = compute_susceptibility_phase(chi, TE_sec=0.020, B0_Tesla=3.0)
        kspace, _ = generate_kspace_from_image(se_img, phase_img, target_snr_db=snr_db)  # (2, H, W)
        
        # 5. CEST Z-spectrum & exchange maps (PINN input)
        z_spectrum, exchange_rate_map, concentration_map = compute_z_spectrum_slice(T1, T2, k_sw, f_s)
        
        # 6. Save per-slice .npz file
        file_name = f"sub_{sub_id:02d}_slice_{orig_slice_idx:03d}.npz"
        file_path = os.path.join(output_dir, file_name)
        
        meta_dict = {
            "subject_id": sub_id,
            "slice_index": orig_slice_idx,
            "class_label": cls_label,
            "class_name": ANOMALY_CLASSES[cls_label],
            "instances": meta_inst,
            "h_w": [int(norm_seg.shape[0]), int(norm_seg.shape[1])]
        }
        
        np.savez_compressed(
            file_path,
            image=image_input,                     # (1, H, W) float32
            segmentation_mask=seg_mask,            # (H, W) int32
            kspace=kspace,                         # (2, H, W) float32
            class_label=np.int32(cls_label),       # scalar int
            z_spectrum=z_spectrum,                 # (20, H, W) float32
            exchange_rate_map=exchange_rate_map,   # (H, W) float32
            concentration_map=concentration_map,   # (H, W) float32
            meta=json.dumps(meta_dict)             # JSON metadata string
        )
        
        slice_records.append({
            "file_name": file_name,
            "file_path": file_path,
            "subject_id": sub_id,
            "slice_index": orig_slice_idx,
            "class_label": cls_label,
            "class_name": ANOMALY_CLASSES[cls_label]
        })
        
    return slice_records

def create_dataset_manifest(
    all_records: List[Dict[str, Any]],
    output_manifest_path: str,
    subject_ids: List[int],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Creates train / val / test split manifest grouped by subject ID.
    """
    random.seed(seed)
    shuffled_subjects = subject_ids.copy()
    random.shuffle(shuffled_subjects)
    
    n_sub = len(shuffled_subjects)
    n_train = max(1, int(n_sub * 0.70))
    n_val = max(1, int(n_sub * 0.15))
    
    train_subs = set(shuffled_subjects[:n_train])
    val_subs = set(shuffled_subjects[n_train:n_train + n_val])
    test_subs = set(shuffled_subjects[n_train + n_val:])
    
    splits = {"train": [], "val": [], "test": []}
    
    for rec in all_records:
        sub = rec["subject_id"]
        if sub in train_subs:
            splits["train"].append(rec)
        elif sub in val_subs:
            splits["val"].append(rec)
        else:
            splits["test"].append(rec)
            
    manifest = {
        "summary": {
            "total_subjects": n_sub,
            "total_slices": len(all_records),
            "train_subjects": list(train_subs),
            "val_subjects": list(val_subs),
            "test_subjects": list(test_subs),
            "counts": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"])
            }
        },
        "splits": splits
    }
    
    with open(output_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nSaved dataset manifest to: {output_manifest_path}")
    print(f"Dataset split counts -> Train: {len(splits['train'])} slices ({len(train_subs)} subs), Val: {len(splits['val'])} slices ({len(val_subs)} subs), Test: {len(splits['test'])} slices ({len(test_subs)} subs)")
    
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Synthetic Brain Anomaly Dataset Generator")
    parser.add_argument("--output_dir", type=str, default="data/processed_slices", help="Output directory for .npz files")
    parser.add_argument("--manifest_path", type=str, default="data/dataset_manifest.json", help="Output JSON manifest path")
    parser.add_argument("--subjects", type=int, nargs="+", default=[0], help="List of subject IDs to process (default: 0 for testing)")
    args = parser.parse_args()
    
    all_records = []
    for sub in args.subjects:
        recs = process_subject(sub, args.output_dir)
        all_records.extend(recs)
        
    create_dataset_manifest(all_records, args.manifest_path, args.subjects)
    print("\nDataset generation process complete!")

if __name__ == "__main__":
    main()
