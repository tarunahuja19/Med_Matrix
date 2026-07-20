import numpy as np
from typing import Dict, Tuple

# Standard literature tissue parameters at 3T
# T1 (ms), T2 (ms), PD (0..1 relative proton density), T2* (ms), chi (susceptibility in ppm)
DEFAULT_3T_TISSUE_PARAMS = {
    0: {"name": "Background",   "T1": 0.0,    "T2": 0.0,   "PD": 0.00, "T2star": 0.0,   "chi": 0.00},
    1: {"name": "CSF",          "T1": 4000.0, "T2": 2000.0,"PD": 1.00, "T2star": 1000.0,"chi": 0.00},
    2: {"name": "Gray Matter",  "T1": 1350.0, "T2": 95.0,  "PD": 0.85, "T2star": 50.0,  "chi": 0.00},
    3: {"name": "White Matter", "T1": 840.0,  "T2": 75.0,  "PD": 0.70, "T2star": 45.0,  "chi": 0.00},
    4: {"name": "Fat",          "T1": 300.0,  "T2": 70.0,  "PD": 0.90, "T2star": 30.0,  "chi": -0.10},
    5: {"name": "Muscle",       "T1": 1400.0, "T2": 50.0,  "PD": 0.75, "T2star": 30.0,  "chi": 0.00},
    6: {"name": "Skin",         "T1": 400.0,  "T2": 40.0,  "PD": 0.60, "T2star": 25.0,  "chi": 0.00},
    7: {"name": "Skull",        "T1": 200.0,  "T2": 2.0,   "PD": 0.05, "T2star": 1.5,   "chi": 0.00},
    8: {"name": "Vessels",      "T1": 1600.0, "T2": 150.0, "PD": 0.90, "T2star": 60.0,  "chi": -0.05},
    9: {"name": "Around Fat",   "T1": 350.0,  "T2": 65.0,  "PD": 0.80, "T2star": 35.0,  "chi": -0.10},
    10:{"name": "Dura",         "T1": 1000.0, "T2": 50.0,  "PD": 0.70, "T2star": 25.0,  "chi": 0.00},
    11:{"name": "Bone Marrow",  "T1": 500.0,  "T2": 80.0,  "PD": 0.85, "T2star": 40.0,  "chi": -0.05},
}

def normalize_label_slice(seg_slice: np.ndarray) -> np.ndarray:
    """
    Normalizes tissue segmentation labels if they are scaled (e.g. BrainWeb 455 scaling).
    """
    unique_vals = np.unique(seg_slice.astype(int))
    unique_vals = unique_vals[unique_vals > 0]
    if len(unique_vals) > 0:
        min_pos = np.min(unique_vals)
        if min_pos >= 100 and min_pos % 5 == 0:
            # Scaled labels detected (e.g., multiplier 455)
            scale = min_pos
            return np.round(seg_slice / scale).astype(int)
    return seg_slice.astype(int)

def generate_parameter_maps(
    seg_slice: np.ndarray,
    param_dict: Dict[int, Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a 2D integer crisp tissue segmentation slice into physical parameter maps at 3T.
    """
    if param_dict is None:
        param_dict = DEFAULT_3T_TISSUE_PARAMS
        
    norm_seg = normalize_label_slice(seg_slice)
    shape = norm_seg.shape
    
    T1_map = np.zeros(shape, dtype=np.float32)
    T2_map = np.zeros(shape, dtype=np.float32)
    PD_map = np.zeros(shape, dtype=np.float32)
    T2star_map = np.zeros(shape, dtype=np.float32)
    chi_map = np.zeros(shape, dtype=np.float32)
    
    unique_labels = np.unique(norm_seg)
    
    for lbl in unique_labels:
        mask = (norm_seg == lbl)
        if lbl in param_dict:
            p = param_dict[lbl]
        else:
            p = DEFAULT_3T_TISSUE_PARAMS.get(0)
            
        T1_map[mask] = p["T1"]
        T2_map[mask] = p["T2"]
        PD_map[mask] = p["PD"]
        T2star_map[mask] = p["T2star"]
        chi_map[mask] = p["chi"]
        
    return T1_map, T2_map, PD_map, T2star_map, chi_map
