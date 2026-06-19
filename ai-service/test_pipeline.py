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
            assert data.shape == (slices, height, width)
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
