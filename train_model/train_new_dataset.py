"""
Improved YOLOv8n Training Script for Drowsiness Detection
Focus: Includes eyes + mouth (not just mouth)
Author: [Your Name]
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

# ================================
# 1. CONFIGURATION
# ================================
DATASET_PATH = "FaceImages"  
MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"
CROPPED_FACE_DIR = "cropped_faces_focused"
YOLO_DATASET_DIR = "yolo_drowsiness_focused"
IMG_SIZE = 224  # a bit larger to capture fine facial details
YOLO_MODEL = "yolov8n-cls.pt"
VAL_SPLIT = 0.2  

CLASS_MAPPING = {
    "active": "no_yawn",
    "fatigue": "yawn"
}

# ================================
# 2. MEDIAPIPE SETUP
# ================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

# ================================
# 3. FACE REGION EXTRACTION (Eyes + Mouth)
# ================================
def extract_focused_face(frame, landmarker):
    """
    Extracts a focused face region that includes both eyes and mouth (eyebrows → chin),
    cropped and resized for YOLO classification.
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]

    # Landmarks to anchor bounding box
    left_eye = lm[33]
    right_eye = lm[263]
    mouth_top = lm[13]
    chin = lm[152]
    left_cheek = lm[234]
    right_cheek = lm[454]
    forehead = lm[10]

    # Convert to pixel coordinates
    keypoints = np.array([
        [left_eye.x * w, left_eye.y * h],
        [right_eye.x * w, right_eye.y * h],
        [mouth_top.x * w, mouth_top.y * h],
        [chin.x * w, chin.y * h],
        [left_cheek.x * w, left_cheek.y * h],
        [right_cheek.x * w, right_cheek.y * h],
        [forehead.x * w, forehead.y * h]
    ])

    x_min, y_min = keypoints.min(axis=0).astype(int)
    x_max, y_max = keypoints.max(axis=0).astype(int)

    # Add padding for context (forehead + chin margin)
    pad_x = int((x_max - x_min) * 0.25)
    pad_y = int((y_max - y_min) * 0.35)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    if x_max <= x_min or y_max <= y_min:
        return None

    focused_face = frame[y_min:y_max, x_min:x_max]
    if focused_face.size == 0:
        return None

    return cv2.resize(focused_face, (IMG_SIZE, IMG_SIZE))


# ================================
# 4. DATASET PREPROCESSING
# ================================
def preprocess_and_structure_yolo():
    """Extract focused faces (eyes + mouth) and structure data for YOLO classification."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Face Landmarker model not found at {MODEL_PATH}")
        return False
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset folder '{DATASET_PATH}' not found.")
        return False

    for d in [CROPPED_FACE_DIR, YOLO_DATASET_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    print(f"\n🚀 Extracting focused face regions (eyes + mouth)...")

    data = []
    with FaceLandmarker.create_from_options(options) as landmarker:
        for original_class, mapped_class in CLASS_MAPPING.items():
            class_path = os.path.join(DATASET_PATH, original_class)
            if not os.path.exists(class_path):
                continue

            for filename in tqdm(os.listdir(class_path), desc=f"Processing {original_class}"):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                input_path = os.path.join(class_path, filename)
                frame = cv2.imread(input_path)
                if frame is None:
                    continue

                focused_face = extract_focused_face(frame, landmarker)
                if focused_face is not None:
                    output_path = os.path.join(CROPPED_FACE_DIR, f"{mapped_class}_{filename}")
                    cv2.imwrite(output_path, focused_face)
                    data.append((output_path, mapped_class))

    print(f"\n✅ Extraction complete. Total valid samples: {len(data)}")

    if not data:
        print("❌ No valid samples found. Check landmarks or dataset paths.")
        return False

    # Split train/val
    train_data, val_data = train_test_split(data, test_size=VAL_SPLIT, random_state=42)

    # Structure YOLO folders
    for subset, subset_data in [('train', train_data), ('val', val_data)]:
        for cls_name in CLASS_MAPPING.values():
            os.makedirs(os.path.join(YOLO_DATASET_DIR, subset, cls_name), exist_ok=True)
        for src, cls_name in tqdm(subset_data, desc=f"Structuring {subset}"):
            dest_dir = os.path.join(YOLO_DATASET_DIR, subset, cls_name)
            shutil.copy(src, dest_dir)

    print(f"\n✅ YOLOv8 dataset prepared at {YOLO_DATASET_DIR}")
    return True


# ================================
# 5. YOLOv8 TRAINING
# ================================
def fine_tune_yolo():
    """Fine-tune YOLOv8n classification on the eyes+mouth focused dataset."""
    if not os.path.exists(YOLO_DATASET_DIR):
        print("❌ Data not found. Run preprocess_and_structure_yolo() first.")
        return

    print("\n🔥 Starting YOLOv8n Fine-tuning (Eyes + Mouth Focus)...")
    model = YOLO(YOLO_MODEL)

    results = model.train(
        data=YOLO_DATASET_DIR,
        epochs=40,
        imgsz=IMG_SIZE,
        batch=32,
        device='mps',          # or 'cuda'
        name='yolo_drowsiness_focused_finetuned2',
        augment=True,
        patience=10
    )

    print("\n✅ Training complete. Results saved in 'runs/classify/yolo_drowsiness_focused_finetuned'")
    return results


# ================================
# 6. EXECUTION
# ================================
if __name__ == "__main__":
    if preprocess_and_structure_yolo():
        fine_tune_yolo()
