import numpy as np
from brainweb_dl import get_mri
from typing import Tuple, List, Dict

# Standard 20-subject set for anatomical variety
BRAINWEB_SUBJECT_SET = [0, 4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

def load_brainweb_phantom(sub_id: int = 0, contrast: str = "crisp") -> np.ndarray:
    """
    Downloads and loads a BrainWeb phantom volume for a given subject ID and contrast mode.
    
    Parameters:
        sub_id (int): Subject ID (0 for baseline detailed phantom, or 4..54 from subject set).
        contrast (str): Contrast type ('crisp', 'fuzzy', 'T1', 'T2', 'PD').
        
    Returns:
        np.ndarray: 3D or 4D numpy array containing the volumetric data.
    """
    data = get_mri(sub_id=sub_id, contrast=contrast)
    if hasattr(data, "get_fdata"):
        data = data.get_fdata()
    return np.asarray(data)

def inspect_tissue_labels(crisp_vol: np.ndarray) -> Dict[int, int]:
    """
    Inspects unique tissue label IDs and their voxel counts in a crisp segmentation volume.
    """
    labels, counts = np.unique(crisp_vol.astype(int), return_counts=True)
    return dict(zip(labels, counts))

def extract_axial_slices(volume: np.ndarray, 
                         start_pct: float = 0.20, 
                         end_pct: float = 0.80) -> Tuple[np.ndarray, List[int], int]:
    """
    Extracts central 2D axial slices from a 3D BrainWeb volume.
    
    Parameters:
        volume (np.ndarray): 3D volume of shape (X, Y, Z) or similar.
        start_pct (float): Starting slice percentile (default 0.20 for central 60%).
        end_pct (float): Ending slice percentile (default 0.80).
        
    Returns:
        Tuple[np.ndarray, List[int], int]:
            - 3D stack of 2D axial slices, shape (N_slices, H, W)
            - List of slice indices selected from axis 0 (or central volume axis)
            - The chosen axial axis index
    """
    # Identify axial axis: BrainWeb phantoms usually have dimensions (181, 217, 181) or (217, 181, 181)
    # Axial slices are typically taken along axis 0 or axis 2.
    # We choose axis 0 as default axial orientation or check shape.
    shape = volume.shape
    axial_axis = 0
    
    n_slices = shape[axial_axis]
    start_idx = int(n_slices * start_pct)
    end_idx = int(n_slices * end_pct)
    
    slice_indices = list(range(start_idx, end_idx))
    
    if axial_axis == 0:
        slices = volume[start_idx:end_idx, :, :]
    elif axial_axis == 1:
        slices = volume[:, start_idx:end_idx, :]
    else:
        slices = volume[:, :, start_idx:end_idx].transpose(2, 0, 1)
        
    return slices, slice_indices, axial_axis
