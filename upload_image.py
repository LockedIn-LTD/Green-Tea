import mimetypes
import os
import time
from datetime import timedelta
from typing import Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, storage


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-()" else "_" for c in name)


def upload_image_to_firebase(
    *,
    key_path: str,
    bucket_name: str,
    file_path: str,
    prefix: str = "uploads",
    expires_min: int = 60,
    max_mb: float = 1.0,
    app_name: str = "uploader",
) -> Dict[str, Any]:
    """
    Upload an image to Firebase Storage using Admin SDK and return metadata.

    Enforces: if file size > max_mb, raises ValueError and DOES NOT upload.

    Returns dict:
      {
        "gs_uri": "gs://bucket/path",
        "object_path": "uploads/....jpg",
        "signed_url": "https://....",
        "size_bytes": 12345,
        "content_type": "image/jpeg"
      }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = os.path.getsize(file_path)
    max_bytes = int(max_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        raise ValueError(
            f"Not uploading: file is {size_bytes / (1024*1024):.2f} MB "
            f"(limit is {max_mb:.2f} MB)."
        )

    # Initialize (reuse existing app if already initialized)
    try:
        app = firebase_admin.get_app(app_name)
    except ValueError:
        cred = credentials.Certificate(key_path)
        app = firebase_admin.initialize_app(cred, {"storageBucket": bucket_name}, name=app_name)

    bucket = storage.bucket(app=app)

    filename = _sanitize_filename(os.path.basename(file_path))
    object_path = f"{prefix}/{int(time.time())}_{filename}"

    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    blob = bucket.blob(object_path)
    blob.upload_from_filename(file_path, content_type=content_type)

    signed_url = blob.generate_signed_url(
        expiration=timedelta(minutes=expires_min),
        method="GET",
    )

    return {
        "gs_uri": f"gs://{bucket_name}/{object_path}",
        "object_path": object_path,
        "signed_url": signed_url,
        "size_bytes": size_bytes,
        "content_type": content_type,
    }

