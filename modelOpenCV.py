import zmq
import time
import json
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import sys
import os
import torch
import gc # Import garbage collector
from collections import deque

# --- ZMQ Helper Class Imports (External files are now required) ---
# Assuming these classes are available in the execution environment
from zmq_video_client import ZMQVideoReceiver
from data_publisher import DataPublisher
        
# ================================
# ZMQ PUBLISHER SETUP
# ================================

data_publisher = None

def setup_data_publisher():
    global data_publisher
    # Use the topic/port defined in the class default or config
    data_publisher = DataPublisher(port=5557, topic='model_out')

def publish_status_standardized(perclos_time_s=0.0):
    """
    Standardized 1Hz publishing function using the DataPublisher class.
    """
    if data_publisher:
        data_publisher.publish(perclos_time_s=perclos_time_s)
    else:
        # print("Data publisher not initialized.") # Suppress internal log noise
        pass

# ================================
# CONFIG & MODEL SETUP
# ================================

ZMQ_ADDRESS = "tcp://127.0.0.1:5555"
MODEL_PATH = "fatigueModel.pt"
IMG_SIZE = 224 

CONFIDENCE_THRESHOLD = 0.7 # YOLO inference confidence

# --- PERFORMANCE FIX: FRAME DECIMATION ---
# Only run the expensive PyTorch inference every N frames.
# Set to 10 to run ~3 times/second (much faster overall frame rate when face is present).
INFERENCE_DECIMATION_FACTOR = 10 

# --- OPENCV DNN FACE DETECTOR CONFIG ---
PROTOTXT_PATH = "deploy.prototxt"
CAFFE_MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
# Increased confidence for stability
DNN_CONFIDENCE_THRESHOLD = 0.7 

# --- PYTORCH SETUP ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define thresholds needed for the main loop logic (Perclos)
THRESHOLDS = {
    "PERCLOS_DROWSY_S": 1.5, 
    "PERCLOS_CRITICAL_S": 4.0,
}

# Load the YOLO model
try:
    yolo_model = YOLO(MODEL_PATH)
    yolo_model.to(DEVICE)
    yolo_model.eval() 
    # Disable the verbose output from YOLO 
    yolo_model.verbose = True
    print(f"Loaded YOLO model {MODEL_PATH} onto {DEVICE}")
    setup_data_publisher()
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    sys.exit(1)

# --- OPENCV DNN SETUP ---
try:
    FACE_DETECTOR = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_MODEL_PATH)
    # Configure for high performance
    FACE_DETECTOR.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) 
    FACE_DETECTOR.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Loaded lightweight OpenCV DNN Face Detector (running on CUDA GPU).")
except Exception as e:
    print(f"WARNING: Could not set DNN to CUDA. Falling back to CPU. Error: {e}")
    # If setting to CUDA fails, try to load it again, implicitly falling back to CPU
    try:
        FACE_DETECTOR = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_MODEL_PATH)
        FACE_DETECTOR.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Loaded lightweight OpenCV DNN Face Detector (running on CPU fallback).")
    except Exception as e_cpu:
        print(f"FATAL ERROR: Failed to load OpenCV DNN model even on CPU. Error: {e_cpu}")
        sys.exit(1)


# ================================
# CAMERA & CUDA INITIALIZATION
# ================================
import cv2.cuda as cuda
GPU_FRAME = None # Define globally for scope access

try:
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise RuntimeError("No CUDA devices found by OpenCV.")

    GPU_FRAME = cuda.GpuMat()
    print(f"OpenCV CUDA GpuMat objects initialized on device: {cv2.cuda.getDevice()}")
except Exception as e:
    print(f"WARNING: OpenCV CUDA Initialization failed. Using CPU fallback for image ops. Error: {e}")
    cuda = None

# ================================
# CAMERA SETUP 
# ================================
def open_camera():
    # Uses the imported ZMQVideoReceiver class
    cap = ZMQVideoReceiver(address=ZMQ_ADDRESS)
    return cap

# ================================
# FACE CROP (DNN CPU DETECTOR & CUDA ACCELERATED IMAGE OPS)
# ================================
def extract_focused_face(frame, net, width=300, height=300, confidence_threshold=DNN_CONFIDENCE_THRESHOLD):
    """
    Finds the largest face using the OpenCV DNN SSD detector and crops the face.
    """
    global GPU_FRAME
    
    (h, w) = frame.shape[:2]
    
    # 1. DNN Face Detection 
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (width, height)), 
        1.0, 
        (width, height), 
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    max_area = 0
    best_box = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Using the stricter DNN_CONFIDENCE_THRESHOLD (0.7)
        if confidence > confidence_threshold: 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            area = (endX - startX) * (endY - startY)
            
            if area > max_area:
                max_area = area
                
                # Apply padding 
                pad_x = int(0.1 * (endX - startX))
                pad_y = int(0.1 * (endY - startY))

                # Clip coordinates with padding
                startX = max(0, startX - pad_x); startY = max(0, startY - pad_y)
                endX = min(w, endX + pad_x); endY = min(h, endY + pad_y)
                
                best_box = (startY, endY, startX, endX)
                
    if best_box is None:
        # If no face is found above the stricter threshold, return None
        return None, None
        
    (startY, endY, startX, endX) = best_box
    
    # 3. Cropping and Resizing (Use CUDA if available)
    if cuda is not None and GPU_FRAME is not None and max_area > 0:
        # CUDA Path
        GPU_FRAME.upload(frame)
        # Note: Slicing GpuMat creates a view, no copy, which is efficient
        gpu_crop_temp = GPU_FRAME.colRange(startX, endX).rowRange(startY, endY)
        
        resized_gpu = cuda.resize(
            gpu_crop_temp, 
            dsize=(IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR
        )
        resized_bgr_cpu = resized_gpu.download()
        
        # Explicit memory cleanup is good practice
        del gpu_crop_temp
        del resized_gpu
    else:
        # CPU Fallback Path
        face_crop = frame[startY:endY, startX:endX]
        if face_crop.size == 0:
            return None, None
            
        resized_bgr_cpu = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

    # 4. Final Color Conversion (BGR -> RGB) for PyTorch input
    rgb_resized = cv2.cvtColor(resized_bgr_cpu, cv2.COLOR_BGR2RGB)
    
    # Final cleanup of CPU resource copy
    del resized_bgr_cpu
    
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
    input_tensor = torch.from_numpy(img_tensor).to(DEVICE).unsqueeze(0)
    
    # CRITICAL: Delete input tensor immediately after moving to GPU
    del img_tensor 

    # Explicitly set verbose=False to suppress logging and stream=False
    output = yolo_model(input_tensor, verbose=False, stream=False)
    
    # CRITICAL MEMORY MANAGEMENT: Clear GPU resources immediately
    del input_tensor
    torch.cuda.empty_cache()
    gc.collect()

    probs = output[0].probs.data 
    return probs.cpu().numpy()

# ================================
# MAIN LOOP
# ================================
def run():
    cap = open_camera()
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
        
    # --- Continuous Fatigue Timer Variables ---
    PUBLISH_INTERVAL_S = 1.0 # Standardize model output to 1 Hz
    last_publish_time = time.time()
    
    # drowsy_start_time resets to None when a face is lost or no_yawn is detected
    drowsy_start_time = None 
    continuous_fatigue_s = 0.0 
    # -----------------------------------------------
    
    # --- Performance and Status Variables ---
    
    # FIX: Initialize frame counter and last known status
    inference_frame_counter = 0 
    # Start with "no_yawn" so the timer doesn't start immediately on detection
    last_known_yolo_status = "no_yawn" 
    
    # --- FPS Measurement Variables ---
    fps_start_time = time.time()
    total_frames = 0
    FPS_REPORT_FRAMES = 30 # Report FPS every 30 frames processed
    # ---------------------------------

    CLASS_NAMES = ["no_yawn", "yawn"] 
    print(f"Model loop started. YOLO inference runs every {INFERENCE_DECIMATION_FACTOR} frames (~3 FPS). Publishing status every 1.0s.")

    while True:
        
        current_time = time.time()
        
        # 1. READ FRAME
        ret, frame = cap.read() 
        
        if not ret:
            # Frame not received or stream lost (handles ZMQ failure/no source)
            
            # CRITICAL RESET: If no frame, reset timer
            instant_status = "No Frame" 
            drowsy_start_time = None 
            
            time.sleep(0.01) 
            pass # Skip processing, run publishing check below
        else:
            # Frame received, proceed with detection and inference
            
            total_frames += 1
            
            # This is the performance gate! Only proceed to YOLO if counter hits 0.
            inference_frame_counter = (inference_frame_counter + 1) % INFERENCE_DECIMATION_FACTOR

            # Clean up 4-channel input if necessary
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 2. Extract Face (Fast/Cheap)
            face_crop, face_box = extract_focused_face(frame, FACE_DETECTOR)

            instant_status = "No Face"

            if face_crop is not None:
                
                if inference_frame_counter == 0:
                    # Time to run the EXPENSIVE YOLO Inference
                    probs = pt_predict(face_crop).flatten()
                    top_idx = int(np.argmax(probs))
                    conf = float(probs[top_idx])
                    pred_class = CLASS_NAMES[top_idx]
                    
                    if conf < CONFIDENCE_THRESHOLD:
                        last_known_yolo_status = CLASS_NAMES[0] # Default to awake if confidence is low
                    else:
                        last_known_yolo_status = pred_class
                    
                    # Cleanup immediately
                    del probs
                    gc.collect()

                # For this frame, use the last known result from YOLO
                instant_status = last_known_yolo_status
                
                # 3. Continuous Timer Logic 
                if instant_status == "yawn":
                    # If yawn detected, start the timer if it wasn't running
                    if drowsy_start_time is None:
                        drowsy_start_time = current_time
                else:
                    # Face present but awake (no_yawn): Reset the continuous timer
                    drowsy_start_time = None
            else:
                # No face detected (handles user walking out of frame)
                # CRITICAL RESET: If face is lost, reset the continuous timer
                drowsy_start_time = None
                # Also reset the last known status
                last_known_yolo_status = "no_yawn"
        
        # 4. Standardized Publishing Check (Once per second)
        if current_time - last_publish_time >= PUBLISH_INTERVAL_S:
            
            # A. Calculate Total Continuous Fatigue Time
            if drowsy_start_time is not None:
                continuous_fatigue_s = current_time - drowsy_start_time
            else:
                # If timer is None, PERCLOS is 0.0s 
                continuous_fatigue_s = 0.0
            
            # B. Determine Model Status for the interval
            if continuous_fatigue_s >= THRESHOLDS["PERCLOS_CRITICAL_S"]:
                status_to_publish = "FATIGUE_CRITICAL"
            elif continuous_fatigue_s >= THRESHOLDS["PERCLOS_DROWSY_S"]:
                status_to_publish = "FATIGUE_DROWSY"
            else:
                # If continuous_fatigue_s is 0.0, this is "Alert"
                status_to_publish = "Alert"

            # C. Publish the aggregated status
            publish_status_standardized(
                perclos_time_s=continuous_fatigue_s
            )

            # D. Reset publish time
            last_publish_time = current_time
            
        # 5. FPS Reporting
        if total_frames >= FPS_REPORT_FRAMES:
            elapsed_time = current_time - fps_start_time
            if elapsed_time > 0:
                fps = total_frames / elapsed_time
                print(f"[FPS] Total Frame Processing Speed: {fps:.2f} FPS (Processed {total_frames} frames)")
            
            # Reset counter and time for the next batch
            fps_start_time = current_time
            total_frames = 0


if __name__ == "__main__":
    # Check for required external files
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFE_MODEL_PATH):
        print("Error: Required DNN model files (deploy.prototxt and .caffemodel) not found.")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: YOLO model '{MODEL_PATH}' not found.")
        sys.exit(1)
        
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA-enabled OpenCV running on device: {cv2.cuda.getDevice()}")
    else:
        print("Warning: OpenCV CUDA not available. Image processing will run on CPU.")
    run()