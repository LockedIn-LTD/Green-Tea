import cv2
import numpy as np
import time
from collections import deque
import torch
import sys
import face_alignment
from ultralytics import YOLO
from zmq_video_client import ZMQVideoReceiver
# ================================
# CONFIG & MODEL SETUP
# ================================
ZMQ_ADDRESS = "tcp://127.0.0.1:5555"
MODEL_PATH = "fatigueModel.pt"
IMG_SIZE = 224 # Image size for the model input

CLOSED_FRAME_THRESHOLD = 15
MAX_HISTORY = 20
CONFIDENCE_THRESHOLD = 0.7

# --- PYTORCH SETUP ---
# Automatically select CUDA if available, otherwise fall back to CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLO classification model and move it to the defined device
try:
    # Use ultralytics.YOLO to load the model
    yolo_model = YOLO(MODEL_PATH)
    # Move model parameters to GPU
    yolo_model.to(DEVICE)
    yolo_model.eval() # Set model to evaluation mode
    print(f"Loaded classification model {MODEL_PATH} onto {DEVICE}")
except Exception as e:
    print(f"Error loading PyTorch classification model: {e}")
    sys.exit(1) # Use sys.exit(1) for a clean script exit

# Global variable to hold the PyTorch Face Alignment model
FA_MODEL = None 

# ================================
# FACE ALIGNMENT MODEL SETUP (PYTORCH)
# ================================

def setup_face_alignment():
    """Loads the face alignment model onto the defined DEVICE (cuda/cpu)."""
    try:
        # Pass the string name "2D" instead of the class attribute
        fa_model = face_alignment.FaceAlignment(
            # Pass the string name "2D"
            '2D', 
            face_detector='dlib', 
            device=DEVICE 
        )
        print(f"Loaded PyTorch Face Alignment model onto {DEVICE}")
        return fa_model
    except Exception as e:
        print(f"Error loading PyTorch Face Alignment model: {e}")
        return None

# ================================
# OPENCV CUDA INITIALIZATION
# ================================
import cv2.cuda as cuda
try:
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise RuntimeError("No CUDA devices found by OpenCV.")

    # Create reusable GpuMat objects outside the loop for efficiency
    GPU_FRAME = cuda.GpuMat()
    # GPU_CROP and GPU_RESIZED are no longer strictly needed globally due to GpuMat return values
    
    print(f"OpenCV CUDA GpuMat objects initialized on device: {cv2.cuda.getDevice()}")

except Exception as e:
    print(f"FATAL WARNING: OpenCV CUDA Initialization failed. Using CPU fallback. Error: {e}")
    # Set cuda to None *if* we are truly falling back to prevent GpuMat calls
    cuda = None

# ================================
# CAMERA SETUP FOR JETSON NANO
# ================================
"""
def gst_pipeline(sensor_id, width, height, fps=15, flip=2): 
    return (
        f"nvarguscamerasrc sensor-id={int(sensor_id)} "
        f"bufapi-version=1 ! "
        f"video/x-raw(memory:NVMM), width=(int){int(width)}, height=(int){int(height)}, "
        f"framerate=(fraction){int(fps)}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={int(flip)} ! "
        f"video/x-raw, format=(string)BGRx, width=(int){int(width)}, height=(int){int(height)} ! "
        f"videoconvert ! "
        f"appsink caps=video/x-raw,format=(string)BGR,width=(int){int(width)},height=(int){int(height)} "
        f"drop=true max-buffers=1 sync=false"
    )
"""
def open_camera():
    #pipeline = gst_pipeline(0, 1280, 720, 15, 2) 
    cap = cap = ZMQVideoReceiver(address=ZMQ_ADDRESS)
    return cap
# ================================
# FACE CROP (PYTORCH FACE ALIGNMENT & CUDA ACCELERATED)
# ================================
def extract_focused_face(frame, fa_model):
    h, w, _ = frame.shape
    
    # --- Part 1: PyTorch Landmark Detection ---
    # get_landmarks_from_image handles color conversion internally.
    landmarks = fa_model.get_landmarks_from_image(frame)
    
    if not landmarks:
        return None # No face found
    
    # We only process the first detected face
    lm = landmarks[0] # lm is a NumPy array of shape (68, 2) in pixel coordinates

    # --- Part 2: Bounding Box Calculation ---
    # Find the tightest box around all 68 landmarks
    x_min, y_min = lm.min(axis=0)
    x_max, y_max = lm.max(axis=0)

    # Apply padding (adjusting these values can improve detection stability)
    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_x = int(face_w * 0.25) 
    pad_y = int(face_h * 0.35)
    
    # Ensure coordinates are integers and stay within frame boundaries
    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = min(w, int(x_max + pad_x))
    y_max = min(h, int(y_max + pad_y))

    if x_min >= x_max or y_min >= y_max:
        return None

    # --- Part 3: Cropping and Resizing (CUDA or CPU) ---
    
    if cuda is not None:
        # GPU Cropping and Resizing
        
        # 1. Upload frame to GPU once (if not already uploaded)
        GPU_FRAME.upload(frame)
        
        # 2. Use GpuMat slicing to extract the crop (no need for global crop mat)
        gpu_crop_temp = GPU_FRAME.colRange(x_min, x_max).rowRange(y_min, y_max)
        
        # 3. Resize the cropped GpuMat
        resized_gpu = cuda.resize(
            gpu_crop_temp, 
            dsize=(IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR
        )
        
        # 4. Download the result (still BGR at this point)
        resized_bgr_cpu = resized_gpu.download()
        
        # 5. Final Color Conversion (BGR -> RGB) on CPU for PyTorch
        rgb_resized = cv2.cvtColor(resized_bgr_cpu, cv2.COLOR_BGR2RGB)
    else:
        # CPU Cropping and Resizing
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) 
        
    return rgb_resized

# ================================
# PYTORCH INFERENCE
# ================================
@torch.no_grad() # Disable gradient calculation for faster inference
def pt_predict(img):
    # 1. Preprocess: Normalize (0-255 -> 0-1.0) and convert to float32
    img_tensor = img.astype(np.float32) / 255.0
    
    # 2. Reshape: HWC -> CHW (RGB image is now expected)
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    
    # 3. Add Batch Dimension: CHW -> BCHW (B=1)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # 4. Convert to PyTorch Tensor and move to CUDA
    input_tensor = torch.from_numpy(img_tensor).to(DEVICE)

    # 5. Run Inference
    output = yolo_model(input_tensor)
    
    # Extract classification probabilities tensor
    probs = output[0].probs.data

    # 6. Move results back to CPU (Numpy)
    return probs.cpu().numpy()

# ================================
# MAIN LOOP
# ================================
def run():
    cap = open_camera()
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
        
    pred_history = deque(maxlen=MAX_HISTORY)
    
    # Define class names based on your trained model
    CLASS_NAMES = ["no_yawn", "yawn"] 

    # Use the global FA_MODEL loaded in __main__
    global FA_MODEL

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # --- FIX: Ensure the frame is 3-channel BGR before GPU upload ---
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # -----------------------------------------------------------
        
        # Start timer for FPS calculation
        t0 = time.time()

        # Pass the global FA_MODEL object for PyTorch face detection
        face_crop = extract_focused_face(frame, FA_MODEL) # <-- PyTorch Face Alignment

        if face_crop is None:
            cv2.putText(frame, "No face", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Fatigue", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # Run PyTorch inference
        probs = pt_predict(face_crop)
        
        probs = probs.flatten()

        top_idx = int(np.argmax(probs))
        conf = float(probs[top_idx])

        pred_class = CLASS_NAMES[top_idx]
        
        # Apply confidence threshold
        if conf < CONFIDENCE_THRESHOLD:
            # If confidence is low, default to the safe state
            pred_class = CLASS_NAMES[0] # assuming index 0 is 'no_yawn'

        pred_history.append(pred_class)

        # Check history for fatigue condition
        recent_fatigues = pred_history.count("yawn")

        if recent_fatigues >= CLOSED_FRAME_THRESHOLD:
            status = "🚨 FATIGUE DETECTED 🚨"
            color = (0, 0, 255) # Red
        else:
            status = "Alert"
            color = (0, 255, 0) # Green
        
        # Calculate FPS
        fps = 1.0 / (time.time() - t0)

        # Display Status
        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        # Display Prediction and Confidence
        cv2.putText(frame, f"{pred_class} ({conf:.2f})", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


        cv2.imshow("Fatigue", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize the PyTorch Face Alignment model globally
    FA_MODEL = setup_face_alignment()
    if FA_MODEL is None:
        print("Fatal error: Face Alignment model failed to load. Exiting.")
        sys.exit(1)
        
    # Check for OpenCV CUDA availability at runtime
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA-enabled OpenCV running on device: {cv2.cuda.getDevice()}")
    else:
        print("Warning: OpenCV CUDA not available. Image processing will run on CPU.")
        
    run()