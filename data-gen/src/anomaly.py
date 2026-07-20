import numpy as np
import random
from typing import Tuple, Dict, Any, List

ANOMALY_CLASSES = {
    0: "Normal",
    1: "Low-grade glioma",
    2: "High-grade glioma",
    3: "Necrosis",
    4: "Edema",
    5: "Meningioma",
    6: "Microbleed"
}

def generate_blob_mask(shape: Tuple[int, int], center: Tuple[int, int], radius_x: float, radius_y: float, angle_deg: float = 0, roughness: float = 0.2) -> np.ndarray:
    """
    Generates a 2D boolean mask representing an irregular blob/ellipse shape.
    """
    H, W = shape
    cy, cx = center
    y, x = np.ogrid[:H, :W]
    
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    xr = cos_a * (x - cx) + sin_a * (y - cy)
    yr = -sin_a * (x - cx) + cos_a * (y - cy)
    
    dist_eq = (xr / radius_x)**2 + (yr / radius_y)**2
    
    if roughness > 0:
        # Add spatial perturbation / noise to the boundary
        np.random.seed(int(abs(cx * 1000 + cy * 100)) % 10000)
        noise = np.random.normal(0, roughness, size=(H, W))
        mask = (dist_eq + noise) <= 1.0
    else:
        mask = dist_eq <= 1.0
        
    return mask

def generate_microbleed_mask(shape: Tuple[int, int], center: Tuple[int, int], radius: float = 3.0) -> np.ndarray:
    """
    Generates a small circular disk mask for microbleeds (2-10 voxels diameter).
    """
    H, W = shape
    cy, cx = center
    y, x = np.ogrid[:H, :W]
    dist = (x - cx)**2 + (y - cy)**2
    return dist <= (radius**2)

def insert_anomalies_into_slice(
    norm_seg: np.ndarray,
    T1_map: np.ndarray,
    T2_map: np.ndarray,
    PD_map: np.ndarray,
    T2star_map: np.ndarray,
    chi_map: np.ndarray,
    force_class: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Inserts synthetic lesions or microbleeds into a 2D slice based on tissue constraints.
    
    Returns:
        Updated (T1_map, T2_map, PD_map, T2star_map, chi_map),
        segmentation_mask (0=bg, 1..11 tissue, 12..17 anomaly classes),
        class_label (int: 0..6),
        meta_instances (list of dicts),
        exchange_rate_map (Hz),
        concentration_map (solute fraction f_s)
    """
    H, W = norm_seg.shape
    
    # Initialize output maps
    T1_out = T1_map.copy()
    T2_out = T2_map.copy()
    PD_out = PD_map.copy()
    T2star_out = T2star_map.copy()
    chi_out = chi_map.copy()
    
    # Base CEST exchange maps (baseline healthy tissue has near-zero solute exchange)
    k_sw_out = np.zeros((H, W), dtype=np.float32)
    f_s_out = np.zeros((H, W), dtype=np.float32)
    
    # Base background exchange in GM/WM (small baseline APT/NOE proxy)
    brain_mask = np.isin(norm_seg, [2, 3])  # GM & WM
    k_sw_out[brain_mask] = 30.0    # 30 Hz baseline exchange
    f_s_out[brain_mask] = 0.001     # 0.1% baseline pool fraction
    
    seg_mask = norm_seg.copy()
    
    # Decide anomaly class
    if force_class is not None:
        chosen_class = force_class
    else:
        prob = random.random()
        if prob < 0.55:
            chosen_class = 0  # Normal
        else:
            chosen_class = random.randint(1, 6)
            
    meta_instances = []
    
    if chosen_class == 0:
        return T1_out, T2_out, PD_out, T2star_out, chi_out, seg_mask, 0, meta_instances, k_sw_out, f_s_out
        
    # Find valid target candidate coordinates
    if chosen_class == 5:
        # Meningioma: placed near peripheral brain tissue / Dura / Skull boundary
        valid_coords = np.argwhere(np.isin(norm_seg, [2, 6, 9, 10]))
    else:
        # Gliomas, Necrosis, Edema, Microbleeds: placed in GM (2) or WM (3)
        valid_coords = np.argwhere(np.isin(norm_seg, [2, 3]))
        
    if len(valid_coords) == 0:
        # No valid tissue found in slice -> return normal
        return T1_out, T2_out, PD_out, T2star_out, chi_out, seg_mask, 0, meta_instances, k_sw_out, f_s_out
        
    # Pick random center
    center_idx = random.randint(0, len(valid_coords) - 1)
    cy, cx = valid_coords[center_idx]
    
    # Generate anomaly mask based on class
    if chosen_class == 6:
        # Microbleed (small disk, 2-5 mm radius)
        r = random.uniform(2.0, 5.0)
        anomaly_mask = generate_microbleed_mask((H, W), (cy, cx), radius=r)
    else:
        # Tumor/lesion blob
        rx = random.uniform(8.0, 22.0)
        ry = random.uniform(8.0, 22.0)
        angle = random.uniform(0, 180)
        rough = random.uniform(0.1, 0.25) if chosen_class in [1, 2, 3, 4] else 0.05
        anomaly_mask = generate_blob_mask((H, W), (cy, cx), rx, ry, angle, rough)
        
    # Restrict anomaly to brain interior (exclude skull/background overlap)
    valid_inside = (norm_seg > 0) & (norm_seg != 7)
    anomaly_mask = anomaly_mask & valid_inside
    
    if not np.any(anomaly_mask):
        return T1_out, T2_out, PD_out, T2star_out, chi_out, seg_mask, 0, meta_instances, k_sw_out, f_s_out
        
    # Modify physical parameters according to class profile
    if chosen_class == 1:
        # Low-grade glioma: mildly ↑ T1, mildly ↑ T2, low-mod APT
        T1_out[anomaly_mask] = random.uniform(1600.0, 2100.0)
        T2_out[anomaly_mask] = random.uniform(110.0, 150.0)
        T2star_out[anomaly_mask] = random.uniform(60.0, 90.0)
        k_sw_out[anomaly_mask] = random.uniform(50.0, 80.0)
        f_s_out[anomaly_mask] = random.uniform(0.003, 0.005)
        
    elif chosen_class == 2:
        # High-grade glioma: ↑↑ T1, ↑↑ T2, high APT
        T1_out[anomaly_mask] = random.uniform(2200.0, 3000.0)
        T2_out[anomaly_mask] = random.uniform(160.0, 250.0)
        T2star_out[anomaly_mask] = random.uniform(80.0, 130.0)
        k_sw_out[anomaly_mask] = random.uniform(120.0, 180.0)
        f_s_out[anomaly_mask] = random.uniform(0.008, 0.015)
        
    elif chosen_class == 3:
        # Necrosis: fluid-like ↑↑ T1, ↑↑ T2, low APT
        T1_out[anomaly_mask] = random.uniform(3200.0, 3800.0)
        T2_out[anomaly_mask] = random.uniform(250.0, 450.0)
        T2star_out[anomaly_mask] = random.uniform(150.0, 300.0)
        k_sw_out[anomaly_mask] = random.uniform(10.0, 30.0)
        f_s_out[anomaly_mask] = random.uniform(0.0005, 0.0015)
        
    elif chosen_class == 4:
        # Edema: mild ↑ T1, ↑ T2, low APT
        T1_out[anomaly_mask] = random.uniform(1500.0, 1900.0)
        T2_out[anomaly_mask] = random.uniform(120.0, 170.0)
        T2star_out[anomaly_mask] = random.uniform(70.0, 100.0)
        k_sw_out[anomaly_mask] = random.uniform(20.0, 40.0)
        f_s_out[anomaly_mask] = random.uniform(0.001, 0.002)
        
    elif chosen_class == 5:
        # Meningioma: iso/mild ↑ T1, mild ↑ T2, mod exchange
        T1_out[anomaly_mask] = random.uniform(1300.0, 1700.0)
        T2_out[anomaly_mask] = random.uniform(95.0, 135.0)
        T2star_out[anomaly_mask] = random.uniform(50.0, 75.0)
        k_sw_out[anomaly_mask] = random.uniform(70.0, 110.0)
        f_s_out[anomaly_mask] = random.uniform(0.004, 0.007)
        
    elif chosen_class == 6:
        # Microbleed: strong paramagnetic susceptibility shift (hemosiderin)
        chi_out[anomaly_mask] += random.uniform(0.8, 2.5)  # +0.8 to +2.5 ppm shift
        T2star_out[anomaly_mask] = random.uniform(5.0, 15.0)  # Shorter T2*
        T2_out[anomaly_mask] = np.minimum(T2_out[anomaly_mask], random.uniform(30.0, 50.0))
        
    # Update segmentation mask (offset label by 11 so tissue=1..11, anomaly=12..17)
    seg_mask[anomaly_mask] = 11 + chosen_class
    
    # Record metadata
    ys, xs = np.where(anomaly_mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    meta_instances.append({
        "class_id": chosen_class,
        "class_name": ANOMALY_CLASSES[chosen_class],
        "center": [int(cx), int(cy)],
        "bbox": bbox,
        "area_voxels": int(np.sum(anomaly_mask))
    })
    
    return T1_out, T2_out, PD_out, T2star_out, chi_out, seg_mask, chosen_class, meta_instances, k_sw_out, f_s_out
