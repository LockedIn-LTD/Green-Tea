import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import YOLO
import time

# ================================
# 1. CONFIGURATION
# ================================
MODEL_PATH = "best.pt"
FACEMESH_MODEL = "face_landmarker_v2_with_blendshapes.task"
IMG_SIZE = 224
DEVICE = 'mps'  # or 'cuda', 'cpu' CHANGE TO CUGA FOR JETSON

# Fatigue logic parameters
CLOSED_FRAME_THRESHOLD = 15   # mark fatigue if eyes closed > 15 frames
MAX_HISTORY = 20              # size of sliding window for history
CONFIDENCE_THRESHOLD = 0.7    # model confidence cutoff

# ================================
# 2. LOAD MODELS
# ================================
yawn_model = YOLO(MODEL_PATH)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
RunningMode = mp.tasks.vision.RunningMode

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACEMESH_MODEL),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

# ================================
# 3. FACE CROP FUNCTION (Eyes + Mouth Focus)
# ================================
def extract_focused_face(frame, landmarker):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]

    # key landmarks
    left_eye = lm[33]
    right_eye = lm[263]
    mouth_top = lm[13]
    chin = lm[152]
    left_cheek = lm[234]
    right_cheek = lm[454]
    forehead = lm[10]

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

    resized = cv2.resize(focused_face, (IMG_SIZE, IMG_SIZE))
    rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb_resized

# ================================
# 4. REAL-TIME DROWSINESS DETECTION
# ================================
def run_drowsiness_detection():
    cap = cv2.VideoCapture(0)
    pred_history = deque(maxlen=MAX_HISTORY)
    fatigue_counter = 0

    with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_crop = extract_focused_face(frame, landmarker)
            if face_crop is None:
                cv2.putText(frame, "No face detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            results = yawn_model.predict(face_crop, verbose=False, device=DEVICE)
            pred_class = results[0].names[int(results[0].probs.top1)]
            conf = float(results[0].probs.top1conf)

            # confidence filter
            if conf < CONFIDENCE_THRESHOLD:
                pred_class = "no_yawn"

            pred_history.append(pred_class)

            # Count how many "yawn" or "fatigue" frames in last N
            recent_fatigues = sum(1 for p in pred_history if p == "yawn")

            # If many consecutive fatigue frames → flag fatigue
            if recent_fatigues >= CLOSED_FRAME_THRESHOLD:
                status = "FATIGUE DETECTED"
                color = (0, 0, 255)
            else:
                status = "Alert"
                color = (0, 255, 0)

            cv2.putText(frame, f"{status}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(frame, f"Conf: {conf:.2f} | Pred: {pred_class}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

# ================================
# 5. RUN
# ================================
if __name__ == "__main__":
    run_drowsiness_detection()
