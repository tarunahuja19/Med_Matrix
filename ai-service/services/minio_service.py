"""
KVision 4.0 — MinIO Object Storage Service
Handles all MinIO client initialization and common storage operations.
"""

import os
import tempfile
import logging
import numpy as np
from minio import Minio
from config import (
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE,
    BUCKET_KSPACE_RAW, BUCKET_RECONSTRUCTED, BUCKET_REPORTS,
)

logger = logging.getLogger("ai-service")


def create_minio_client() -> Minio | None:
    """Creates and returns a MinIO client, or None if connection fails."""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        logger.info(f"Connected to MinIO at {MINIO_ENDPOINT}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        return None


# Module-level singleton
minio_client = create_minio_client()


def download_object(bucket: str, key: str, local_path: str) -> None:
    """Downloads an object from MinIO to a local file path."""
    if minio_client is None:
        raise ConnectionError("MinIO client not configured or connected")
    minio_client.fget_object(bucket_name=bucket, object_name=key, file_path=local_path)


def upload_object(bucket: str, key: str, local_path: str) -> None:
    """Uploads a local file to MinIO."""
    if minio_client is None:
        raise ConnectionError("MinIO client not configured or connected")
    minio_client.fput_object(bucket_name=bucket, object_name=key, file_path=local_path)


def upload_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Uploads raw bytes to MinIO."""
    import io
    if minio_client is None:
        raise ConnectionError("MinIO client not configured or connected")
    minio_client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def upload_numpy(bucket: str, key: str, array: np.ndarray, tmpdir: str | None = None) -> None:
    """Saves a numpy array and uploads it to MinIO."""
    if tmpdir is None:
        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, "array.npy")
            np.save(local_path, array)
            upload_object(bucket, key, local_path)
    else:
        local_path = os.path.join(tmpdir, os.path.basename(key))
        np.save(local_path, array)
        upload_object(bucket, key, local_path)


def list_objects_with_prefix(bucket: str, prefix: str) -> list[str]:
    """Lists all object keys in a bucket matching a prefix."""
    if minio_client is None:
        return []
    objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]
