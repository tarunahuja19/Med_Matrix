"""
KVision 4.0 — Rust PDF Compilation Service
Wraps the Rust report_pdf binary for generating styled clinical PDF reports.
"""

import os
import subprocess
import tempfile
import logging
import numpy as np
from PIL import Image as PILImage

from config import RUST_PDF_BINARY_SEARCH_PATHS
from services.minio_service import minio_client, download_object
from kspace_reader import load_kspace

logger = logging.getLogger("ai-service")


def _find_rust_pdf_binary() -> str | None:
    """Locates the Rust PDF binary from configured search paths."""
    for p in RUST_PDF_BINARY_SEARCH_PATHS:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


def _numpy_slice_to_png(data: np.ndarray, output_path: str) -> None:
    """Converts a 2D numpy array to a grayscale PNG (normalized to 0–255)."""
    s_min, s_max = data.min(), data.max()
    denom = s_max - s_min
    normalized = (data - s_min) / denom * 255.0 if denom > 1e-8 else np.zeros_like(data)
    img = PILImage.fromarray(normalized.astype(np.uint8))
    img.save(output_path, "PNG")


def generate_kspace_log_mag(study_id: str) -> np.ndarray | None:
    """Dynamically generates a k-space log-magnitude array for a study."""
    try:
        from services.minio_service import list_objects_with_prefix
        keys = list_objects_with_prefix("kspace-raw", prefix=f"{study_id}/")
        kspace_key = next((k for k in keys if "kspace_input" in k), None)

        if not kspace_key:
            logger.warning(f"No raw K-space found for study {study_id}")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "kspace.npy")
            download_object("kspace-raw", kspace_key, local_path)

            kspace = load_kspace(local_path)

            # Combine coils via RSS
            if kspace.ndim == 4:
                kspace_mag = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=1))
            elif kspace.ndim == 3:
                kspace_mag = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))[np.newaxis, ...]
            else:
                kspace_mag = np.abs(kspace)[np.newaxis, ...]

            kspace_log_mag = np.log(1.0 + kspace_mag)

            # Per-slice normalization
            for s in range(kspace_log_mag.shape[0]):
                s_min, s_max = kspace_log_mag[s].min(), kspace_log_mag[s].max()
                denom = s_max - s_min
                kspace_log_mag[s] = (kspace_log_mag[s] - s_min) / denom if denom > 1e-8 else 0.0

            # Cache it in MinIO
            from services.minio_service import upload_numpy
            upload_numpy("reconstructed", f"{study_id}/kspace_log_mag.npy", kspace_log_mag, tmpdir)
            logger.info(f"Generated and cached kspace_log_mag.npy for study {study_id}")
            return kspace_log_mag

    except Exception as e:
        logger.error(f"Failed to generate kspace log mag for study {study_id}: {e}")
        return None


def compile_pdf(
    report_text: str,
    patient_metadata: dict,
    study_id: str | None = None,
    for_patient: bool = False,
) -> bytes:
    """
    Compiles a clinical PDF report using the Rust binary.

    Args:
        report_text: The report text content.
        patient_metadata: Patient info dict (name, age, sex, physician, etc.).
        study_id: Optional study ID for embedding MRI/K-space images.
        for_patient: Whether this is a patient-friendly report.

    Returns:
        PDF file as bytes.

    Raises:
        FileNotFoundError: If Rust binary is not found.
        subprocess.CalledProcessError: If Rust binary fails.
    """
    binary_path = _find_rust_pdf_binary()
    if not binary_path:
        raise FileNotFoundError("report_pdf Rust binary not found in search paths.")

    f_in_name = None
    f_out_name = None
    mri_png_path = None
    kspace_png_path = None

    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f_in:
            f_in.write(report_text)
            f_in_name = f_in.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f_out:
            f_out_name = f_out.name

        cmd = [binary_path, "--input", f_in_name, "--output", f_out_name]

        if for_patient:
            cmd.append("--for-patient")

        # Add metadata flags
        metadata_flags = {
            "name": "--name", "age": "--age", "sex": "--sex",
            "physician": "--physician", "report_id": "--id",
            "patient_id": "--patient-id", "study_date": "--study-date",
            "modality": "--modality", "date": "--date",
        }
        for key, flag in metadata_flags.items():
            if key in patient_metadata:
                cmd.extend([flag, str(patient_metadata[key])])

        # Embed MRI and K-space images if study_id provided
        if study_id and minio_client:
            mri_png_path = _try_embed_mri_image(study_id, cmd)
            if not for_patient:
                kspace_png_path = _try_embed_kspace_image(study_id, cmd)

        logger.info(f"Invoking Rust PDF generator: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Rust compiler stdout: {res.stdout}")

        with open(f_out_name, "rb") as f_pdf:
            return f_pdf.read()

    finally:
        for path in [f_in_name, f_out_name, mri_png_path, kspace_png_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


def _try_embed_mri_image(study_id: str, cmd: list) -> str | None:
    """Attempts to download and convert reconstructed MRI for PDF embedding."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_npy = tmp.name

        download_object("reconstructed", f"{study_id}/reconstructed.npy", tmp_npy)
        data = np.load(tmp_npy)
        slice_data = data[data.shape[0] // 2] if data.ndim >= 3 else data

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            png_path = tmp_png.name

        _numpy_slice_to_png(slice_data, png_path)
        cmd.extend(["--mri", png_path])
        os.remove(tmp_npy)
        return png_path

    except Exception as e:
        logger.error(f"Failed to embed MRI image for study {study_id}: {e}")
        return None


def _try_embed_kspace_image(study_id: str, cmd: list) -> str | None:
    """Attempts to download or generate and convert K-space image for PDF embedding."""
    kspace_data = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_npy = tmp.name
        download_object("reconstructed", f"{study_id}/kspace_log_mag.npy", tmp_npy)
        kspace_data = np.load(tmp_npy)
        os.remove(tmp_npy)
    except Exception:
        logger.warning(f"kspace_log_mag.npy not found for study {study_id}, generating dynamically...")
        kspace_data = generate_kspace_log_mag(study_id)

    if kspace_data is None:
        return None

    try:
        slice_data = kspace_data[kspace_data.shape[0] // 2] if kspace_data.ndim >= 3 else kspace_data
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            png_path = tmp_png.name
        _numpy_slice_to_png(slice_data, png_path)
        cmd.extend(["--kspace", png_path])
        return png_path
    except Exception as e:
        logger.error(f"Failed to convert K-space to PNG: {e}")
        return None
