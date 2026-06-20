import os
import numpy as np
import urllib.request
import json
import uuid

# Import the phantom and K-space simulation code from our modules
from generate_synthetic_dataset import generate_quantitative_maps, simulate_mri_signal, generate_coil_sensitivities

def upload_study_urllib(backend_url, patient_id, npy_path, filename):
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
    parts.append(f'Content-Disposition: form-data; name="kspace"; filename="{filename}"')
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

def generate_patient_kspace(slices, coils, resolution, category_id, difficulty, output_path):
    patient_volume = np.zeros((slices, coils, resolution, resolution), dtype=np.complex64)
    sens_maps = generate_coil_sensitivities(resolution, resolution, coils)
    
    pathology_start = slices // 4
    pathology_end = 3 * slices // 4
    
    for s in range(slices):
        if s == 0:
            TR, TE = 600.0, 15.0
        else:
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
    
    np.save(output_path, patient_volume)

def main():
    slices = 8
    coils = 16
    resolution = 228
    backend_url = "http://backend:3000"
    
    # 5 pathologies: Category 1 (Glioma), 2 (Stroke), 3 (MS), 4 (Hydrocephalus), 10 (Microbleeds)
    cases = [
        {"name": "John Doe (Glioma)", "category_id": 1, "dob": "1975-03-12T00:00:00.000Z", "gender": "M", "filename": "glioma_kspace_228.npy"},
        {"name": "Sarah Connor (Stroke)", "category_id": 2, "dob": "1965-11-20T00:00:00.000Z", "gender": "F", "filename": "stroke_kspace_228.npy"},
        {"name": "Bruce Wayne (MS Lesions)", "category_id": 3, "dob": "1980-05-15T00:00:00.000Z", "gender": "M", "filename": "ms_kspace_228.npy"},
        {"name": "Ellen Ripley (Hydrocephalus)", "category_id": 4, "dob": "1992-07-04T00:00:00.000Z", "gender": "F", "filename": "hydrocephalus_kspace_228.npy"},
        {"name": "Tony Stark (Microbleeds)", "category_id": 10, "dob": "1970-04-29T00:00:00.000Z", "gender": "M", "filename": "microbleeds_kspace_228.npy"},
    ]
    
    for case in cases:
        print(f"\n--- Processing: {case['name']} ---")
        temp_file = case["filename"]
        
        # 1. Generate K-space data
        print(f"Generating K-space data for {case['name']}...")
        generate_patient_kspace(slices, coils, resolution, case["category_id"], 0.25, temp_file)
        print(f"K-space data saved to {temp_file}")
        
        # 2. Create Patient
        patient_payload = {
            "name": case["name"],
            "dateOfBirth": case["dob"],
            "gender": case["gender"]
        }
        print(f"Registering patient {case['name']} on backend...")
        req = urllib.request.Request(
            f"{backend_url}/patients",
            data=json.dumps(patient_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as res:
                patient = json.loads(res.read().decode('utf-8'))
                patient_id = patient["id"]
                print(f"Patient registered with ID: {patient_id}")
        except Exception as e:
            print(f"Failed to create patient: {e}")
            continue
            
        # 3. Upload Study
        print(f"Uploading study K-space file {temp_file}...")
        try:
            study = upload_study_urllib(backend_url, patient_id, temp_file, temp_file)
            print(f"Successfully uploaded and enqueued. Study ID: {study['studyId']}, Job ID: {study['jobId']}")
        except Exception as e:
            print(f"Failed to upload study: {e}")
            continue
            
        # Clean up temp file on disk
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    print("\n✓ All 5 demo patients generated, registered, and queued for AI reconstruction!")

if __name__ == "__main__":
    main()
