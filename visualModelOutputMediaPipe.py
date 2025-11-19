import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
import torch
from ultralytics import YOLO
from zmq_video_client import ZMQVideoReceiver

# ================================
# CONFIG & MODEL SETUP
# ================================
ZMQ_ADDRESS = "tcp://127.0.0.1:5555"
MODEL_PATH = "fatigueModel.pt"
FACEMESH_MODEL = "face_landmarker_v2_with_blendshapes.task"
IMG_SIZE = 224 # Image size for the model input

CLOSED_FRAME_THRESHOLD = 15
MAX_HISTORY = 20
CONFIDENCE_THRESHOLD = 0.7

# --- PYTORCH SETUP ---
# Automatically select CUDA if available, otherwise fall back to CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLO model and move it to the defined device
try:
    # Use ultralytics.YOLO to load the model
    yolo_model = YOLO(MODEL_PATH)
    # Move model parameters to GPU
    yolo_model.to(DEVICE)
    yolo_model.eval() # Set model to evaluation mode
    print(f"Loaded model {MODEL_PATH} onto {DEVICE}")
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    exit()

# ================================
# MEDIA PIPE & CUDA INITIALIZATION
# ================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
RunningMode = mp.tasks.vision.RunningMode

# *** NOTE: THIS LINE SHOULD BE UPDATED IF YOU INSTALL A GPU-ENABLED MEDIA PIPE WHEEL ***
# delegate=BaseOptions.Delegate.GPU should be added here if available
landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACEMESH_MODEL
                             #,delegate=BaseOptions.Delegate.GPU
                             ),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

# --- OPENCV CUDA SETUP (NEW) ---
import cv2.cuda as cuda
try:
    # 1. Check for basic CUDA support
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise RuntimeError("No CUDA devices found by OpenCV.")

    # Create reusable GpuMat objects outside the loop for efficiency
    # These should be fine even if createCvtColor fails.
    GPU_FRAME = cuda.GpuMat()
    GPU_CROP = cuda.GpuMat()
    GPU_RESIZED = cuda.GpuMat()
    
    # Flag to indicate if the advanced functions failed
    use_simple_cuda = True 
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
# FACE CROP (CUDA ACCELERATED - STREAMLINED FINAL)
# ================================
def extract_focused_face(frame, landmarker):
    h, w, _ = frame.shape
    rgb_cpu = None

    # --- PART 1: Color Conversion for MediaPipe Input ---
    if cuda is not None:
        # GPU Path: Upload, convert (BGR->RGB) on GPU, then download for MediaPipe
        GPU_FRAME.upload(frame)
        
        RGB_GPU = cuda.GpuMat(h, w, cv2.CV_8UC3)
        cuda.cvtColor(GPU_FRAME, cv2.COLOR_BGR2RGB, RGB_GPU) 
        
        rgb_cpu = RGB_GPU.download()
    else:
        # CPU Path: Convert on CPU
        rgb_cpu = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- PART 2: MediaPipe Landmark Detection (CPU, unchanged) ---
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cpu)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    # Landmark extraction and Bounding Box calculation (CPU, unchanged)
    lm = result.face_landmarks[0]
    # ... (Bounding box and padding calculation logic) ...
    pts = np.array([
        [lm[33].x*w, lm[33].y*h], 
        [lm[263].x*w, lm[263].y*h],
        [lm[13].x*w, lm[13].y*h],
        [lm[152].x*w, lm[152].y*h],
        [lm[234].x*w, lm[234].y*h],
        [lm[454].x*w, lm[454].y*h],
        [lm[10].x*w, lm[10].y*h]
    ], dtype=np.int32)
    x_min, y_min = pts.min(axis=0); x_max, y_max = pts.max(axis=0)
    pad_x = int((x_max - x_min) * 0.25); pad_y = int((y_max - y_min) * 0.35)
    x_min = max(0, x_min - pad_x); y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x); y_max = min(h, y_max + pad_y)

    if x_min >= x_max or y_min >= y_max:
        return None

    # --- PART 3: Cropping and Resizing ---
    
    if cuda is not None:
        # GPU Cropping and Resizing
        
        # 1. Use direct GpuMat slicing to extract the crop (Source)
        gpu_crop_temp = GPU_FRAME.colRange(x_min, x_max).rowRange(y_min, y_max)
        
        # 2. FIX: Omit the 'dst' argument. Capture the returned GpuMat.
        # This forces the function to allocate and return a valid GpuMat object.
        resized_gpu = cuda.resize(
            gpu_crop_temp, 
            dsize=(IMG_SIZE, IMG_SIZE), # Pass dsize explicitly
            interpolation=cv2.INTER_LINEAR
        )
        
        # Download the result
        resized_bgr_cpu = resized_gpu.download()
        
        # Final Color Conversion (BGR -> RGB) on CPU
        rgb_resized = cv2.cvtColor(resized_bgr_cpu, cv2.COLOR_BGR2RGB)
    else:
        # CPU Cropping and Resizing (unchanged)
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
    
    # 2. Reshape: HWC -> CHW
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    
    # 3. Add Batch Dimension: CHW -> BCHW (B=1)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # 4. Convert to PyTorch Tensor and move to CUDA (CORRECT PLACEMENT)
    input_tensor = torch.from_numpy(img_tensor).to(DEVICE)

    # 5. Run Inference
    output = yolo_model(input_tensor)
    
    # The output format depends on the specific YOLOv8 classification model.
    # Assuming it returns a list of results with .probs attribute for classification
    probs = output[0].probs.data # Extract classification probabilities tensor

    # 6. Move results back to CPU (Numpy) (CORRECT PLACEMENT)
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

    with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # --- FIX: Ensure the frame is 3-channel BGR before GPU upload ---
            # Use cv2.COLOR_BGRA2BGR, which is more universal than COLOR_BGRX2BGR 
            # for dropping the 4th channel (Alpha/Ignored).
            if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
                # The input is 4 channels (BGRA or BGRX), convert to 3 channels (BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # -----------------------------------------------------------
            
            # Start timer for FPS calculation
            t0 = time.time()

            face_crop = extract_focused_face(frame, landmarker)

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
    # Check for OpenCV CUDA availability at runtime
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA-enabled OpenCV running on device: {cv2.cuda.getDevice()}")
    else:
        print("Warning: OpenCV CUDA not available. Image processing will run on CPU.")
        
    run()