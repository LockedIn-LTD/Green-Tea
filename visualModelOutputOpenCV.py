import cv2
import numpy as np
import time
from collections import deque
import torch
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

# --- OPENCV DNN FACE DETECTOR CONFIG ---
PROTOTXT_PATH = "deploy.prototxt"
CAFFE_MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for a face detection

# --- PYTORCH SETUP ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLO model
try:
    yolo_model = YOLO(MODEL_PATH)
    yolo_model.to(DEVICE)
    yolo_model.eval() # Set model to evaluation mode
    print(f"Loaded YOLO model {MODEL_PATH} onto {DEVICE}")
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    exit()

# --- OPENCV DNN SETUP ---
try:
    FACE_DETECTOR = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_MODEL_PATH)
    FACE_DETECTOR.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) 
    FACE_DETECTOR.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Loaded lightweight OpenCV DNN Face Detector (running on CUDA GPU).")
    #FACE_DETECTOR.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) 
    #FACE_DETECTOR.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    #print("OpenCV DNN Face Detector running on CPU.")
except Exception as e:
    print(f"ERROR: Could not set DNN to CUDA. Falling back to CPU. Error: {e}")
    # Fallback to CPU if CUDA setting fails

# ================================
# CAMERA & CUDA INITIALIZATION
# ================================
import cv2.cuda as cuda
try:
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise RuntimeError("No CUDA devices found by OpenCV.")

    # GpuMat objects for reusable GPU processing (Upload, Resize)
    GPU_FRAME = cuda.GpuMat()
    # The resize operation will return a new GpuMat, we only need the upload object.
    print(f"OpenCV CUDA GpuMat objects initialized on device: {cv2.cuda.getDevice()}")
except Exception as e:
    print(f"WARNING: OpenCV CUDA Initialization failed. Using CPU fallback for image ops. Error: {e}")
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
# FACE CROP (DNN CPU DETECTOR & CUDA ACCELERATED IMAGE OPS)
# ================================
def extract_focused_face(frame, net, width=300, height=300, confidence_threshold=DNN_CONFIDENCE_THRESHOLD):
    """
    Finds the largest face using the OpenCV DNN SSD detector and crops the face.
    """
    (h, w) = frame.shape[:2]
    
    # 1. DNN Detection (CPU): Convert to blob for network input
    # Mean subtraction values are standard for this model (104, 177, 123)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (width, height)), 
        1.0, 
        (width, height), 
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    # 2. Find the face with the largest area
    max_area = 0
    best_box = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # Scale the bounding box coordinates back to the original frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            area = (endX - startX) * (endY - startY)
            
            if area > max_area:
                max_area = area
                
                # Apply padding (approx 10% on each side)
                pad_x = int(0.1 * (endX - startX))
                pad_y = int(0.1 * (endY - startY))

                # Clip coordinates with padding
                startX = max(0, startX - pad_x); startY = max(0, startY - pad_y)
                endX = min(w, endX + pad_x); endY = min(h, endY + pad_y)
                
                best_box = (startY, endY, startX, endX)
                
    if best_box is None:
        return None, None
        
    (startY, endY, startX, endX) = best_box
    
    # 3. Cropping and Resizing (Use CUDA if available)
    if cuda is not None and max_area > 0:
        # GPU Path: Upload, Crop, Resize on GPU
        GPU_FRAME.upload(frame)
        
        # GpuMat slicing for cropping
        gpu_crop_temp = GPU_FRAME.colRange(startX, endX).rowRange(startY, endY)
        
        # CUDA resize (returns a new GpuMat)
        resized_gpu = cuda.resize(
            gpu_crop_temp, 
            dsize=(IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR
        )
        
        resized_bgr_cpu = resized_gpu.download()
    else:
        # CPU Path: Crop and Resize
        face_crop = frame[startY:endY, startX:endX]
        if face_crop.size == 0:
            return None, None
            
        resized_bgr_cpu = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

    # 4. Final Color Conversion (BGR -> RGB) for PyTorch input
    rgb_resized = cv2.cvtColor(resized_bgr_cpu, cv2.COLOR_BGR2RGB)
    
    return rgb_resized, best_box


# ================================
# PYTORCH INFERENCE
# ================================
@torch.no_grad()
def pt_predict(img):
    # Preprocess: Normalize (0-255 -> 0-1.0) and convert to float32
    img_tensor = img.astype(np.float32) / 255.0
    
    # Reshape and Add Batch Dimension: HWC -> CHW -> BCHW (B=1)
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # Convert to PyTorch Tensor and move to CUDA
    input_tensor = torch.from_numpy(img_tensor).to(DEVICE)

    # Run Inference
    output = yolo_model(input_tensor)
    
    probs = output[0].probs.data # Extract classification probabilities tensor

    # Move results back to CPU
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
    CLASS_NAMES = ["no_yawn", "yawn"] 

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Clean up 4-channel input if necessary
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        t0 = time.time()

        # Call the new DNN-based detector
        face_crop, face_box = extract_focused_face(frame, FACE_DETECTOR)

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
            pred_class = CLASS_NAMES[0]

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

        # Draw Bounding Box
        if face_box is not None:
            (startY, endY, startX, endX) = face_box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Display Info
        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"{pred_class} ({conf:.2f})", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


        cv2.imshow("Fatigue", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA-enabled OpenCV running on device: {cv2.cuda.getDevice()}")
    else:
        print("Warning: OpenCV CUDA not available. Image processing will run on CPU.")
        
    run()