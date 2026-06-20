import os
import numpy as np
import urllib.request
import json
import uuid

# Import the phantom and K-space simulation code from our modules
from generate_synthetic_dataset import generate_quantitative_maps, simulate_mri_signal, generate_coil_sensitivities

def upload_study_urllib(backend_url, patient_id, npy_path):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    
    parts = []
    
    # Add form fields
    fields = {
        "patientId": patient_id,
        "modality": "MRI",
        "studyDate": "2026-06-21T00:00:00.000Z",
        "phaseCorrection": "true",
        "denoiseMethod": "nlm"
    }
    for name, value in fields.items():
        parts.append(f"--{boundary}")
        parts.append(f'Content-Disposition: form-data; name="{name}"')
        parts.append('')
        parts.append(value)
        
    # Add file field
    parts.append(f"--{boundary}")
    parts.append(f'Content-Disposition: form-data; name="kspace"; filename="demo_kspace_228.npy"')
    parts.append('Content-Type: application/octet-stream')
    parts.append('')
    
    # Read file content
    with open(npy_path, 'rb') as f:
        file_content = f.read()
        
    # Build binary body
    binary_parts = []
    for p in parts:
        binary_parts.append(p.encode('utf-8'))
        binary_parts.append(b'\r\n')
    binary_parts.append(file_content)
    binary_parts.append(b'\r\n')
    binary_parts.append(f"--{boundary}--".encode('utf-8'))
    binary_parts.append(b'\r\n')
    
    body = b''.join(binary_parts)
    
    req = urllib.request.Request(
        f"{backend_url}/studies/upload",
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body))
        }
    )
    
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode('utf-8'))

def main():
    slices = 8
    coils = 16
    resolution = 228  # High resolution as requested
    category_id = 10  # Cerebral Microbleeds (to verify detail reconstruction)
    difficulty = 0.3  # Moderate difficulty
    
    print(f"Generating synthetic K-space volume for Demo Patient...")
    print(f"Dimensions: {slices} slices, {coils} coils, {resolution}x{resolution} resolution")
    
    patient_volume = np.zeros((slices, coils, resolution, resolution), dtype=np.complex64)
    sens_maps = generate_coil_sensitivities(resolution, resolution, coils)
    
    pathology_start = slices // 4
    pathology_end = 3 * slices // 4
    
    for s in range(slices):
        if s == 0:
            # T1 localize
            TR, TE = 600.0, 15.0
        else:
            # T2 weighted
            TR, TE = 3000.0, 90.0
            
        curr_cat_id = category_id if (s >= pathology_start and s < pathology_end) else 0
        curr_diff = difficulty if (s >= pathology_start and s < pathology_end) else 0.0
        
        t1, t2, pd, t2dash, b0 = generate_quantitative_maps(resolution, s, slices, curr_cat_id, curr_diff)
        img_slice = simulate_mri_signal(t1, t2, pd, t2dash, b0, TR, TE)
        
        for c in range(coils):
            coil_img = img_slice * sens_maps[c]
            coil_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
            patient_volume[s, c, :, :] = coil_kspace.astype(np.complex64)
            
    # Add complex Gaussian noise (~30 dB SNR)
    vol_std = np.std(np.abs(patient_volume))
    noise_std = 0.03 * vol_std
    noise = (np.random.normal(0, noise_std, patient_volume.shape) + 
             1j * np.random.normal(0, noise_std, patient_volume.shape)) / np.sqrt(2)
    patient_volume += noise.astype(np.complex64)
    
    npy_path = "demo_kspace_228.npy"
    np.save(npy_path, patient_volume)
    print(f"Saved raw K-space file to {npy_path}")
    
    # 2. Create Patient on Express backend
    backend_url = "http://backend:3000"
    patient_payload = {
        "name": "Demo Patient (228x228 MB)",
        "dateOfBirth": "1988-08-08T00:00:00.000Z",
        "gender": "M"
    }
    
    print(f"Creating patient on Express backend: {patient_payload['name']}...")
    
    req = urllib.request.Request(
        f"{backend_url}/patients",
        data=json.dumps(patient_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as res:
            patient = json.loads(res.read().decode('utf-8'))
            patient_id = patient["id"]
            print(f"Created patient successfully. ID: {patient_id}")
    except Exception as e:
        print(f"Failed to create patient: {e}")
        return
    
    # 3. Upload K-space file to backend
    print(f"Uploading raw K-space to trigger AI pipeline...")
    try:
        study = upload_study_urllib(backend_url, patient_id, npy_path)
        print(f"Study uploaded and queued successfully. Study ID: {study['studyId']}, Job ID: {study['jobId']}")
        print("Success! You can now view the Demo Patient's study reconstruction on the Study Archive and AI Reports tabs.")
    except Exception as e:
        print(f"Failed to upload study: {e}")
        return

if __name__ == "__main__":
    main()
