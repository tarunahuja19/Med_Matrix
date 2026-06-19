import os
import re
import struct
import warnings
import numpy as np
import h5py

def generate_coil_sensitivities(height: int, width: int, num_coils: int) -> np.ndarray:
    """
    Generates coil sensitivity maps of shape [num_coils, height, width] using a birdcage model.
    Normalized such that the root-sum-of-squares (RSS) profile is uniform.
    """
    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    coils = []
    r_coil = 1.2  # radius of coil circle (slightly outside the brain phantom)
    sigma = 1.5   # spatial roll-off factor
    
    for c in range(num_coils):
        angle = 2 * np.pi * c / num_coils
        cx = r_coil * np.cos(angle)
        cy = r_coil * np.sin(angle)
        
        # Distance squared to coil element
        dist_sq = (x - cx)**2 + (y - cy)**2
        magnitude = np.exp(-dist_sq / (2 * sigma**2))
        
        # Spatial phase profile
        phase = np.arctan2(y - cy, x - cx)
        
        sens = magnitude * np.exp(1j * phase)
        coils.append(sens)
        
    sens_maps = np.stack(coils, axis=0)
    
    # RSS normalization
    rss_sens = np.sqrt(np.sum(np.abs(sens_maps)**2, axis=0))
    rss_sens[rss_sens == 0] = 1e-8
    sens_maps = sens_maps / rss_sens[np.newaxis, ...]
    
    return sens_maps

def generate_brain_phantom(height: int, width: int, slice_idx: int, num_slices: int) -> np.ndarray:
    """
    Generates a 2D brain-like phantom image (values in range [0, 1]) for a given slice index
    by combining nested ellipses of varying shapes and intensities.
    """
    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    
    # Scale phantom according to slice index to simulate 3D brain volume progression
    z_frac = slice_idx / max(1, num_slices - 1) if num_slices > 1 else 0.5
    scale = np.sin(np.pi * (0.15 + 0.7 * z_frac))
    
    image = np.zeros((height, width), dtype=np.float64)
    
    # Define ellipses: {'cx', 'cy', 'rx', 'ry', 'angle', 'intensity'}
    ellipses = [
        # Skull outer
        {'cx': 0.0, 'cy': -0.05, 'rx': 0.75 * scale, 'ry': 0.85 * scale, 'angle': 0, 'intensity': 0.8},
        # Skull inner (hollow out)
        {'cx': 0.0, 'cy': -0.05, 'rx': 0.71 * scale, 'ry': 0.81 * scale, 'angle': 0, 'intensity': -0.6},
        # Brain tissue (gray matter background)
        {'cx': 0.0, 'cy': -0.05, 'rx': 0.70 * scale, 'ry': 0.80 * scale, 'angle': 0, 'intensity': 0.5},
        # Ventricles (left and right CSF spaces)
        {'cx': -0.15 * scale, 'cy': 0.05 * scale, 'rx': 0.12 * scale * (1.2 - 0.5 * abs(z_frac - 0.5)), 'ry': 0.25 * scale, 'angle': -10, 'intensity': -0.4},
        {'cx': 0.15 * scale, 'cy': 0.05 * scale, 'rx': 0.12 * scale * (1.2 - 0.5 * abs(z_frac - 0.5)), 'ry': 0.25 * scale, 'angle': 10, 'intensity': -0.4},
        # Internal white matter structures (deep gray/white matter tracts)
        {'cx': -0.25 * scale, 'cy': -0.2 * scale, 'rx': 0.18 * scale, 'ry': 0.25 * scale, 'angle': 15, 'intensity': 0.3},
        {'cx': 0.25 * scale, 'cy': -0.2 * scale, 'rx': 0.18 * scale, 'ry': 0.25 * scale, 'angle': -15, 'intensity': 0.3},
        # Posterior lobe structure
        {'cx': 0.0, 'cy': 0.35 * scale, 'rx': 0.25 * scale, 'ry': 0.15 * scale, 'angle': 0, 'intensity': 0.25},
    ]
    
    for ell in ellipses:
        cx, cy = ell['cx'], ell['cy']
        rx, ry = ell['rx'], ell['ry']
        angle = np.radians(ell['angle'])
        intensity = ell['intensity']
        
        # Coordinate rotation
        xr = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        yr = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
        
        mask = (xr / rx)**2 + (yr / ry)**2 <= 1.0
        image[mask] += intensity
        
    return np.clip(image, 0.0, 1.0)

def generate_synthetic_kspace(
    slices: int = 10,
    coils: int = 8,
    height: int = 256,
    width: int = 256,
    noise_level: float = 0.0,
    motion_shift: float = 0.0,
    artifact: str = None
) -> np.ndarray:
    """
    Generates synthetic multi-coil K-space tensors with shape [slices, coils, height, width]
    values are complex numbers simulating a brain-like phantom.
    
    Parameters:
        slices (int): Number of slices.
        coils (int): Number of coils/channels.
        height (int): K-space matrix height (phase encoding steps).
        width (int): K-space matrix width (readout samples).
        noise_level (float): Scale of standard complex Gaussian noise to inject.
        motion_shift (float): Pixel shift representing motion during the scan.
        artifact (str): Type of artifact to simulate ('ghosting', 'aliasing', 'zipper').
    """
    kspace = np.zeros((slices, coils, height, width), dtype=np.complex128)
    
    # 1. Generate multi-coil sensitivity maps (assumed static across slices for simplicity)
    sens_maps = generate_coil_sensitivities(height, width, coils)
    
    # 2. Generate phantom for each slice, apply sensitivity maps, and transform to K-space
    for s in range(slices):
        img_slice = generate_brain_phantom(height, width, s, slices)
        for c in range(coils):
            # Apply coil sensitivity profile
            coil_img = img_slice * sens_maps[c]
            
            # 2D FFT to generate centered K-space data
            # Forward transform: image -> kspace
            coil_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
            kspace[s, c, :, :] = coil_kspace
            
    # 3. Inject motion shifts
    # A translational shift in image space corresponds to a linear phase ramp in K-space.
    if motion_shift > 0.0:
        # Simulate motion during a subset of phase encoding lines (e.g. middle 20% of acquisition)
        start_row = int(height * 0.4)
        end_row = int(height * 0.6)
        
        kx = np.arange(-width // 2, width // 2)
        # Shift formula in 1D along columns: exp(-2j * pi * kx * shift / width)
        phase_ramp = np.exp(-2j * np.pi * kx * motion_shift / width)
        
        # Apply motion shift along readout for selected phase-encoding lines
        kspace[:, :, start_row:end_row, :] *= phase_ramp[np.newaxis, np.newaxis, np.newaxis, :]

    # 4. Inject specific artifacts
    if artifact == 'ghosting':
        # Periodic modulation of phase encoding lines (e.g. simulating breathing/pulsation ghosting)
        modulation = 1.0 + 0.3 * np.cos(2 * np.pi * np.arange(height) * 6 / height)
        kspace *= modulation[np.newaxis, np.newaxis, :, np.newaxis]
        
    elif artifact == 'aliasing':
        # Zero out every 2nd phase encoding line (undersampling factor of 2)
        # This will wrap the image in the phase-encoding direction during IFFT
        kspace[:, :, 1::2, :] = 0.0
        
    elif artifact == 'zipper':
        # Zipper lines are caused by RF noise at a specific frequency (vertical line in image space)
        # This is represented by a spike/delta function along a column in K-space
        zipper_col = width // 2 + 15
        kspace[..., :, zipper_col] += 8.0 * np.max(np.abs(kspace))

    # 5. Inject complex Gaussian noise
    if noise_level > 0.0:
        kspace_std = np.std(np.abs(kspace))
        noise_std = noise_level * kspace_std
        noise = (np.random.normal(0, noise_std, kspace.shape) + 
                 1j * np.random.normal(0, noise_std, kspace.shape)) / np.sqrt(2)
        kspace += noise

    return kspace

class TwixReader:
    """
    Parser for Siemens Twix (.dat) raw data files.
    Supports basic VB and VD/VE measurement headers and MDH parsing.
    If parsing fails or file is corrupt, falls back gracefully to synthetic K-space.
    """
    def __init__(self, file_path: str, fallback_on_error: bool = True):
        self.file_path = file_path
        self.fallback_on_error = fallback_on_error
        self.is_vd = False
        self.header_size = 0
        self.num_measurements = 1
        self.measurements = []

    def read(self) -> np.ndarray:
        metadata = {}
        try:
            with open(self.file_path, 'rb') as f:
                sig = f.read(8)
                if len(sig) < 8:
                    raise ValueError("File too small to be Siemens Twix format.")
                
                first_val = struct.unpack('<I', sig[:4])[0]
                second_val = struct.unpack('<I', sig[4:8])[0]
                
                if first_val == 0:
                    self.is_vd = True
                    self.num_measurements = second_val
                    if self.num_measurements < 1 or self.num_measurements > 100:
                        raise ValueError(f"Suspect measurement count: {self.num_measurements}")
                    
                    # Read measurement table
                    for _ in range(self.num_measurements):
                        meas_data = f.read(152)
                        if len(meas_data) < 152:
                            raise ValueError("Truncated measurement table in VD/VE file.")
                        meas_id, file_id = struct.unpack('<II', meas_data[:8])
                        offset, length = struct.unpack('<QQ', meas_data[8:24])
                        self.measurements.append({
                            'id': meas_id,
                            'offset': offset,
                            'length': length
                        })
                else:
                    self.is_vd = False
                    self.header_size = first_val
                    self.measurements.append({
                        'id': 1,
                        'offset': 0,
                        'length': 0
                    })
                
                # Retrieve the first measurement
                meas = self.measurements[0]
                f.seek(meas['offset'])
                
                if self.is_vd:
                    header_size = struct.unpack('<I', f.read(4))[0]
                else:
                    header_size = self.header_size
                
                header_bytes = f.read(header_size - (4 if self.is_vd else 0))
                header_text = header_bytes.decode('latin1', errors='ignore')
                
                # Parse metadata from ASCII protocol header
                metadata = self._parse_ascii_header(header_text)
                
                # Seek to raw data starting position
                f.seek(meas['offset'] + header_size)
                
                # Parse MDH blocks and construct complex K-space array
                kspace_data = self._parse_data_blocks(f, metadata)
                return kspace_data
                
        except Exception as e:
            if self.fallback_on_error:
                warnings.warn(f"Twix Reader failed to parse {self.file_path}: {e}. Falling back to synthetic mock data.")
                slices = metadata.get('slices', 10)
                coils = metadata.get('coils', 8)
                height = metadata.get('height', 256)
                width = metadata.get('width', 256)
                return generate_synthetic_kspace(slices=slices, coils=coils, height=height, width=width)
            else:
                raise e

    def _parse_ascii_header(self, header_text: str) -> dict:
        metadata = {}
        
        # Width (Base resolution / Column samples)
        match = re.search(r'sKSpace\.lBaseResolution\s*=\s*(\d+)', header_text)
        metadata['width'] = int(match.group(1)) if match else 256
        
        # Height (Phase encoding lines)
        match = re.search(r'sKSpace\.lPhaseEncodingLines\s*=\s*(\d+)', header_text)
        metadata['height'] = int(match.group(1)) if match else 256
        
        # Slices
        match = re.search(r'sSliceArray\.lSize\s*=\s*(\d+)', header_text)
        metadata['slices'] = int(match.group(1)) if match else 1
        
        # Coil count
        match = re.search(r'iMaxNoOfRxChannels\s*=\s*(\d+)', header_text)
        metadata['coils'] = int(match.group(1)) if match else 8
        
        return metadata

    def _parse_data_blocks(self, f, metadata: dict) -> np.ndarray:
        est_width = metadata.get('width', 256)
        
        data_dict = {}
        max_slice = 0
        max_coil = 0
        max_line = 0
        
        # Store current position, get file size, then seek back
        current_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(current_pos)
        
        current_offset = current_pos
        
        while current_offset + 128 <= file_size:
            f.seek(current_offset)
            if self.is_vd:
                dma_bytes = f.read(16)
                if len(dma_bytes) < 16:
                    break
                dma_len = struct.unpack('<I', dma_bytes[:4])[0]
                if dma_len == 0 or dma_len > 10000000:
                    current_offset += 16
                    continue
                
                f.seek(current_offset + 16)
                mdh_bytes = f.read(192)
                if len(mdh_bytes) < 192:
                    break
                
                samples_in_scan = struct.unpack('<H', mdh_bytes[28:30])[0]
                used_channels = struct.unpack('<H', mdh_bytes[30:32])[0]
                loop_counters = struct.unpack('<14H', mdh_bytes[40:68])
            else:
                mdh_bytes = f.read(128)
                if len(mdh_bytes) < 128:
                    break
                dma_len, _, _, _, _, _, _, samples_in_scan, used_channels = struct.unpack('<I i I I I I I H H', mdh_bytes[:32])
                loop_counters = struct.unpack('<14H', mdh_bytes[32:60])
                
            line = loop_counters[0]
            slice_idx = loop_counters[2]
            
            max_slice = max(max_slice, slice_idx)
            max_line = max(max_line, line)
            
            if self.is_vd:
                channel_offset = current_offset + 16 + 192
                for c in range(used_channels):
                    f.seek(channel_offset)
                    chan_hdr = f.read(32)
                    if len(chan_hdr) < 32:
                        break
                    chan_id = struct.unpack('<H', chan_hdr[:2])[0]
                    max_coil = max(max_coil, chan_id)
                    
                    data_bytes = f.read(samples_in_scan * 8)
                    if len(data_bytes) < samples_in_scan * 8:
                        break
                    
                    raw_data = np.frombuffer(data_bytes, dtype=np.float32)
                    complex_data = raw_data[0::2] + 1j * raw_data[1::2]
                    data_dict[(slice_idx, chan_id, line)] = complex_data
                    
                    channel_offset += 32 + samples_in_scan * 8
                
                next_offset = current_offset + max(dma_len, 192 + used_channels * (32 + samples_in_scan * 8))
                if next_offset <= current_offset:
                    next_offset = current_offset + 16
                current_offset = next_offset
            else:
                if used_channels == 1:
                    data_bytes = f.read(samples_in_scan * 8)
                    if len(data_bytes) < samples_in_scan * 8:
                        break
                    raw_data = np.frombuffer(data_bytes, dtype=np.float32)
                    complex_data = raw_data[0::2] + 1j * raw_data[1::2]
                    data_dict[(slice_idx, 0, line)] = complex_data
                    current_offset += 128 + samples_in_scan * 8
                else:
                    channel_offset = current_offset + 128
                    for c in range(used_channels):
                        f.seek(channel_offset)
                        chan_hdr = f.read(32)
                        if len(chan_hdr) < 32:
                            break
                        chan_id = struct.unpack('<H', chan_hdr[:2])[0]
                        max_coil = max(max_coil, chan_id)
                        
                        data_bytes = f.read(samples_in_scan * 8)
                        if len(data_bytes) < samples_in_scan * 8:
                            break
                        raw_data = np.frombuffer(data_bytes, dtype=np.float32)
                        complex_data = raw_data[0::2] + 1j * raw_data[1::2]
                        data_dict[(slice_idx, chan_id, line)] = complex_data
                        
                        channel_offset += 32 + samples_in_scan * 8
                    
                    current_offset = channel_offset
                    
        if not data_dict:
            raise ValueError("No valid K-space data blocks parsed from file.")
            
        num_slices = max_slice + 1
        num_coils = max_coil + 1
        num_lines = max_line + 1
        
        first_val = next(iter(data_dict.values()))
        num_cols = len(first_val)
        
        kspace = np.zeros((num_slices, num_coils, num_lines, num_cols), dtype=np.complex64)
        for (s, c, l), val in data_dict.items():
            if len(val) == num_cols:
                kspace[s, c, l, :] = val
            elif len(val) < num_cols:
                kspace[s, c, l, :len(val)] = val
            else:
                kspace[s, c, l, :] = val[:num_cols]
                
        return kspace

def load_kspace(file_path: str, **kwargs) -> np.ndarray:
    """
    Loads K-space data from different formats (.h5, .dat, .npy, .npz).
    
    Parameters:
        file_path (str): Path to the k-space file.
        **kwargs: Additional parameters passed to loaders (e.g. fallback_on_error for TwixReader).
        
    Returns:
        np.ndarray: Complex K-space tensor with shape [slices, coils, height, width].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.h5', '.hdf5']:
        with h5py.File(file_path, 'r') as f:
            if 'kspace' not in f:
                keys = list(f.keys())
                if len(keys) == 1:
                    kspace = f[keys[0]][()]
                else:
                    raise KeyError(f"Could not find 'kspace' dataset in HDF5 file. Available keys: {keys}")
            else:
                kspace = f['kspace'][()]
                
            # fastMRI complex data representation helper (shape is [..., 2])
            if kspace.ndim >= 3 and kspace.shape[-1] == 2:
                kspace = kspace[..., 0] + 1j * kspace[..., 1]
                
            # Ensure shape is 4D [slices, coils, height, width]
            if kspace.ndim == 3:
                # [coils, height, width] -> [1, coils, height, width]
                kspace = kspace[np.newaxis, ...]
            elif kspace.ndim == 2:
                # [height, width] -> [1, 1, height, width]
                kspace = kspace[np.newaxis, np.newaxis, ...]
                
            return kspace
            
    elif ext == '.dat':
        fallback = kwargs.get('fallback_on_error', True)
        reader = TwixReader(file_path, fallback_on_error=fallback)
        return reader.read()
        
    elif ext == '.npy':
        kspace = np.load(file_path)
        if kspace.ndim == 3:
            kspace = kspace[np.newaxis, ...]
        elif kspace.ndim == 2:
            kspace = kspace[np.newaxis, np.newaxis, ...]
        return kspace
        
    elif ext == '.npz':
        data = np.load(file_path)
        if 'kspace' in data:
            kspace = data['kspace']
        elif 'data' in data:
            kspace = data['data']
        else:
            keys = list(data.keys())
            if len(keys) > 0:
                kspace = data[keys[0]]
            else:
                raise ValueError("Empty .npz file.")
                
        if kspace.ndim == 3:
            kspace = kspace[np.newaxis, ...]
        elif kspace.ndim == 2:
            kspace = kspace[np.newaxis, np.newaxis, ...]
        return kspace
        
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .h5, .dat, .npy, .npz")
