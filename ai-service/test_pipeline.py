import os
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Mock the Minio client connection during import to prevent connection errors
with patch("minio.Minio") as mock_minio:
    from main import app, minio_client
    # Set the mocked client on the app module
    import main
    main.minio_client = MagicMock()

from kspace_reader import generate_synthetic_kspace

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_reconstruct_endpoint_integration():
    # Generate a realistic small synthetic kspace data shape (e.g. 2 slices, 4 coils, 32x32 resolution)
    slices, coils, height, width = 2, 4, 32, 32
    kspace_data = generate_synthetic_kspace(
        slices=slices,
        coils=coils,
        height=height,
        width=width,
        noise_level=0.05,
        artifact="ghosting"
    )

    # Temporary directory to simulate MinIO object file downloads
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_kspace_file = os.path.join(tmpdir, "mock_kspace.npy")
        np.save(temp_kspace_file, kspace_data)

        # Mock fget_object to copy our mock file to the local download path
        def mock_fget_object(bucket_name, object_name, file_path):
            assert bucket_name == "kspace-raw"
            assert object_name == "input_kspace_key.npy"
            # Copy temp_kspace_file contents to file_path
            import shutil
            shutil.copy(temp_kspace_file, file_path)

        # Mock fput_object to verify the reconstructed image is uploaded correctly
        uploaded_files = {}
        def mock_fput_object(bucket_name, object_name, file_path):
            assert bucket_name == "reconstructed"
            assert object_name == "output_image_key.npy"
            # Verify file exists and is readable as a numpy array
            data = np.load(file_path)
            assert data.shape == (slices, 256, 256)
            uploaded_files[object_name] = data

        # Apply mocks to the main.minio_client
        main.minio_client.fget_object = mock_fget_object
        main.minio_client.fput_object = mock_fput_object

        # Prepare request payload
        payload = {
            "study_id": "test-study-uuid-12345",
            "kspace_key": "input_kspace_key.npy",
            "reconstructed_key": "output_image_key.npy",
            "phase_correction": True,
            "denoise_method": "nlm"  # Use NLM to avoid training a deep model during tests
        }

        # Make the request
        response = client.post("/reconstruct", json=payload)

        # Assertions
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["status"] == "success"
        assert json_data["study_id"] == "test-study-uuid-12345"
        assert json_data["reconstructed_key"] == "output_image_key.npy"
        
        # Verify artifact report contents
        report = json_data["artifact_report"]
        assert "ghosting" in report
        assert "wrap_around" in report
        assert "zipper_noise" in report
        for k, v in report.items():
            assert 0.0 <= v <= 1.0

        # Verify file upload occurred
        assert "output_image_key.npy" in uploaded_files


def test_reconstruct_endpoint_invalid_file():
    # Mock fget_object to write corrupt data
    def mock_fget_object(bucket_name, object_name, file_path):
        with open(file_path, "w") as f:
            f.write("completely_invalid_garbage_data")

    main.minio_client.fget_object = mock_fget_object

    payload = {
        "study_id": "test-study-uuid-12345",
        "kspace_key": "invalid_kspace.npy",
        "reconstructed_key": "output.npy"
    }

    # Make request
    response = client.post("/reconstruct", json=payload)
    # The endpoint should return 422 Unprocessable Entity due to parsing failure
    assert response.status_code == 422
    assert "Failed to parse K-space file format" in response.json()["detail"]


def test_reconstruct_endpoint_minio_error():
    # Mock fget_object to raise an error (simulating MinIO down or key not found)
    def mock_fget_object(bucket_name, object_name, file_path):
        raise Exception("NoSuchKey: The specified key does not exist.")

    main.minio_client.fget_object = mock_fget_object

    payload = {
        "study_id": "test-study-uuid-12345",
        "kspace_key": "missing_kspace.npy",
        "reconstructed_key": "output.npy"
    }

    # Make request
    response = client.post("/reconstruct", json=payload)
    # The endpoint should return 404 Not Found
    assert response.status_code == 404
    assert "Failed to retrieve K-space file" in response.json()["detail"]


def test_predict_endpoint_with_kspace_explainability():
    # 1. Generate realistic raw kspace data
    slices, coils, height, width = 4, 12, 64, 64
    kspace_data = generate_synthetic_kspace(
        slices=slices,
        coils=coils,
        height=height,
        width=width,
        noise_level=0.05,
        artifact="ghosting"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_kspace_file = os.path.join(tmpdir, "mock_kspace.npy")
        np.save(temp_kspace_file, kspace_data)

        # Mock fget_object
        def mock_fget_object(bucket_name, object_name, file_path):
            import shutil
            shutil.copy(temp_kspace_file, file_path)

        # Mock fput_object to capture uploaded files
        uploaded_files = {}
        def mock_fput_object(bucket_name, object_name, file_path):
            uploaded_files[object_name] = np.load(file_path)

        main.minio_client.fget_object = mock_fget_object
        main.minio_client.fput_object = mock_fput_object

        # Prepare request payload for /predict
        payload = {
            "study_id": "test-predict-study-123",
            "kspace_key": "input_kspace_key.npy",
            "anomaly_threshold": 0.5,
            "phase_correction": True,
            "denoise_method": "nlm"
        }

        # Make the /predict request
        response = client.post("/predict", json=payload)

        # Assertions
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["status"] == "success"
        
        # Check pathology predictions
        assert "predicted_pathology" in json_data
        assert "pathology_confidence" in json_data
        assert "pathology_probabilities" in json_data
        
        # Check explainability keys
        assert json_data["kspace_gradcam_key"] == "test-predict-study-123/kspace_gradcam.npy"
        assert json_data["kspace_log_mag_key"] == "test-predict-study-123/kspace_log_mag.npy"
        assert json_data["reconstructed_gradcam_key"] == "test-predict-study-123/reconstructed_gradcam.npy"
        
        # Verify the files were uploaded to MinIO correctly
        assert "test-predict-study-123/kspace_gradcam.npy" in uploaded_files
        assert "test-predict-study-123/kspace_log_mag.npy" in uploaded_files
        assert "test-predict-study-123/reconstructed_gradcam.npy" in uploaded_files
        
        # Verify arrays dimensions
        assert uploaded_files["test-predict-study-123/kspace_gradcam.npy"].shape == (8, 128, 128)
        assert uploaded_files["test-predict-study-123/kspace_log_mag.npy"].shape == (8, 128, 128)
        assert uploaded_files["test-predict-study-123/reconstructed_gradcam.npy"].shape == (8, 256, 256)


def test_progression_projection_endpoint():
    # Test Tumor_Glioma projection
    payload = {
        "pathology": "Tumor_Glioma",
        "initial_pathology_volume_cm3": 12.5
    }
    response = client.post("/predict/progression", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert json_data["pathology"] == "Tumor_Glioma"
    assert json_data["initial_volume_cm3"] == 12.5
    assert len(json_data["timeline"]) == 6
    
    # Test a point in the timeline
    point = json_data["timeline"][0]
    assert point["month"] == 0
    assert point["pathology_volume_cm3"] == 12.5
    assert "edema_volume_cm3" in point
    assert "healthy_brain_volume_cm3" in point
    assert "cognitive_impact_pct" in point
    assert "severity_level" in point
    assert "clinical_note" in point

    # Test default volume fallback
    payload_default = {
        "pathology": "Hydrocephalus"
    }
    response_default = client.post("/predict/progression", json=payload_default)
    assert response_default.status_code == 200
    json_default = response_default.json()
    assert json_default["initial_volume_cm3"] == 60.0