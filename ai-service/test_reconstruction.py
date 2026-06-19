import os
import tempfile
import numpy as np
import h5py
import pytest
from kspace_reader import generate_synthetic_kspace, load_kspace, TwixReader
from reconstruction import reconstruct_kspace

def test_generate_synthetic_kspace():
    # Test shape and type
    slices, coils, height, width = 2, 4, 64, 64
    kspace = generate_synthetic_kspace(slices=slices, coils=coils, height=height, width=width)
    assert kspace.shape == (slices, coils, height, width)
    assert np.iscomplexobj(kspace)

def test_synthetic_artifacts():
    # Test generation with noise and artifacts
    slices, coils, height, width = 1, 2, 32, 32
    
    # 1. Noise
    k_noisy = generate_synthetic_kspace(slices, coils, height, width, noise_level=0.1)
    assert k_noisy.shape == (slices, coils, height, width)
    
    # 2. Motion shift
    k_motion = generate_synthetic_kspace(slices, coils, height, width, motion_shift=2.0)
    assert k_motion.shape == (slices, coils, height, width)
    
    # 3. Artifacts
    for art in ['ghosting', 'aliasing', 'zipper']:
        k_art = generate_synthetic_kspace(slices, coils, height, width, artifact=art)
        assert k_art.shape == (slices, coils, height, width)

def test_reconstruction():
    slices, coils, height, width = 2, 4, 64, 64
    kspace = generate_synthetic_kspace(slices=slices, coils=coils, height=height, width=width)
    
    # 1. Reconstruct without phase correction
    img_no_pc = reconstruct_kspace(kspace, phase_correction=False)
    assert img_no_pc.shape == (slices, height, width)
    assert np.isrealobj(img_no_pc)
    
    # 2. Reconstruct with phase correction
    img_pc = reconstruct_kspace(kspace, phase_correction=True)
    assert img_pc.shape == (slices, height, width)
    assert np.isrealobj(img_pc)
    
    # 3. Test 3D and 2D shapes
    img_3d = reconstruct_kspace(kspace[0], phase_correction=True)
    assert img_3d.shape == (height, width)

def test_file_loaders():
    slices, coils, height, width = 2, 4, 32, 32
    kspace = generate_synthetic_kspace(slices=slices, coils=coils, height=height, width=width)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Test .npy loader
        npy_path = os.path.join(tmpdir, 'kspace.npy')
        np.save(npy_path, kspace)
        loaded_npy = load_kspace(npy_path)
        assert loaded_npy.shape == kspace.shape
        assert np.allclose(loaded_npy, kspace)
        
        # 2. Test .npz loader
        npz_path = os.path.join(tmpdir, 'kspace.npz')
        np.savez(npz_path, kspace=kspace)
        loaded_npz = load_kspace(npz_path)
        assert loaded_npz.shape == kspace.shape
        assert np.allclose(loaded_npz, kspace)
        
        # 3. Test .h5 loader
        h5_path = os.path.join(tmpdir, 'kspace.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('kspace', data=kspace)
        loaded_h5 = load_kspace(h5_path)
        assert loaded_h5.shape == kspace.shape
        assert np.allclose(loaded_h5, kspace)
        
        # 4. Test .h5 loader with fastMRI float array representation
        h5_fmri_path = os.path.join(tmpdir, 'kspace_fmri.h5')
        kspace_fmri_data = np.stack([kspace.real, kspace.imag], axis=-1)
        with h5py.File(h5_fmri_path, 'w') as f:
            f.create_dataset('kspace', data=kspace_fmri_data)
        loaded_fmri = load_kspace(h5_fmri_path)
        assert loaded_fmri.shape == kspace.shape
        assert np.allclose(loaded_fmri, kspace)
        
        # 5. Test Siemens .dat loader (corrupt / fallback)
        dat_path = os.path.join(tmpdir, 'invalid_twix.dat')
        with open(dat_path, 'wb') as f:
            f.write(b'corrupt_header_data_bytes_here')
            
        # Should raise warning and return synthetic data
        with pytest.warns(UserWarning, match="Twix Reader failed to parse"):
            loaded_dat = load_kspace(dat_path, fallback_on_error=True)
            # Should fallback to standard dimensions (1, 8, 256, 256)
            assert loaded_dat.shape == (1, 8, 256, 256)
            
        # Without fallback, should raise error
        with pytest.raises(Exception):
            load_kspace(dat_path, fallback_on_error=False)
