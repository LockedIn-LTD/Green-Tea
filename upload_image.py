import mimetypes
import os
import time
from datetime import timedelta
from typing import Optional, Dict, Any
import tempfile

import firebase_admin
from firebase_admin import credentials, storage
from PIL import Image 

def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-()" else "_" for c in name)


# <-- NEW PREPROCESSING FUNCTION (see above)

def preprocess_image(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
    target_size_mb: float,
    initial_quality: int = 95,
    min_quality: int = 40,
) -> str:
    # ... (Insert the full code for preprocess_image here)
    target_bytes = int(target_size_mb * 1024 * 1024)
    img = Image.open(input_path)
    img.thumbnail((target_width, target_height))
    quality = initial_quality
    processed_size = float('inf')
    
    # Check if the format is compatible with quality-based compression (e.g., JPEG)
    # If it's a PNG, compression is lossless, so quality will be ignored.
    img_format = img.format if img.format in ['JPEG', 'WEBP'] else 'JPEG'

    # Save initial version
    img.save(output_path, format=img_format, optimize=True, quality=quality)
    processed_size = os.path.getsize(output_path)

    while processed_size > target_bytes and quality > min_quality:
        quality = max(min_quality, quality - 5)
        
        # Re-save with lower quality
        img.save(output_path, format=img_format, optimize=True, quality=quality)
        processed_size = os.path.getsize(output_path)

        if quality == min_quality and processed_size > target_bytes:
            print(f"Warning: Image is still {processed_size / (1024*1024):.2f} MB "
                  f"after max compression (Quality {min_quality}). Upload might fail.")
            break

    return output_path


def upload_image_to_firebase(
    key_path: str,
    bucket_name: str,
    file_path: str,
    prefix: str = "uploads",
    expires_min: int = 60,
    max_mb: float = 1.0,
    app_name: str = "uploader",
    target_resolution: str = "1280x720", # <-- NEW ARGUMENT
) -> Dict[str, Any]:
    """
    Uploads a pre-processed image to Firebase Storage, enforcing size/resolution.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Parse target resolution
    try:
        width_str, height_str = target_resolution.split('x')
        target_width = int(width_str)
        target_height = int(height_str)
    except ValueError:
        raise ValueError("target_resolution must be in 'WxH' format, e.g., '1280x720'.")

    # --- 1. Preprocess: Resize and Compress ---
    temp_file_name = f"processed_{os.path.basename(file_path)}"
    # Use tempfile.NamedTemporaryFile for automatic cleanup in the 'finally' block
    temp_file_descriptor, processed_file_path = tempfile.mkstemp(suffix=os.path.splitext(temp_file_name)[1])
    os.close(temp_file_descriptor) # Close the descriptor immediately

    try:
        # Preprocess the original file to the temporary path
        print(f"Processing image: Resizing to max {target_resolution} and compressing to under {max_mb} MB...")
        preprocess_image(
            input_path=file_path,
            output_path=processed_file_path,
            target_width=target_width,
            target_height=target_height,
            target_size_mb=max_mb,
        )
        
        # We now proceed with the *processed* file
        file_to_upload = processed_file_path
        size_bytes = os.path.getsize(file_to_upload)
        max_bytes = int(max_mb * 1024 * 1024)

        # --- 2. Check Size (Final check) ---
        if size_bytes > max_bytes:
            # This should ideally not happen if preprocess_image worked, but serves as a final guard
            raise ValueError(
                f"Not uploading: Processed file is still {size_bytes / (1024*1024):.2f} MB "
                f"(limit is {max_mb:.2f} MB)."
            )

        # --- 3. Upload to Firebase ---
        
        # Initialize (reuse existing app if already initialized)
        try:
            app = firebase_admin.get_app(app_name)
        except ValueError:
            cred = credentials.Certificate(key_path)
            app = firebase_admin.initialize_app(cred, {"storageBucket": bucket_name}, name=app_name)

        bucket = storage.bucket(app=app)

        # Use the name of the *original* file, but the content of the *processed* file
        filename = _sanitize_filename(os.path.basename(file_path))
        object_path = f"{prefix}/{int(time.time())}_{filename}"

        content_type = mimetypes.guess_type(file_to_upload)[0] or "application/octet-stream"
        blob = bucket.blob(object_path)
        blob.upload_from_filename(file_to_upload, content_type=content_type)

        signed_url = blob.generate_signed_url(
            expiration=timedelta(minutes=expires_min),
            method="GET",
        )
        
        print(f"Upload successful. Final size: {size_bytes / (1024*1024):.2f} MB")

        return {
            "gs_uri": f"gs://{bucket_name}/{object_path}",
            "object_path": object_path,
            "signed_url": signed_url,
            "size_bytes": size_bytes,
            "content_type": content_type,
        }
    
    finally:
        # --- 4. Cleanup Temporary File ---
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)
            print(f"Cleaned up temporary file: {processed_file_path}")