import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import dlib
import os
import cProfile
import pstats
import time  # Import the time module

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the CNN model for yawn detection
class YawnCNN(nn.Module):
    def __init__(self):
        super(YawnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Method to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the trained yawn detection model
model = YawnCNN()
model_path = os.path.expanduser('~/Desktop/Green-Tea/Yawning/yawn_detection.pth')
model.load_state_dict(torch.load(model_path, map_location=device))  # Load onto the detected device
model.to(device)  # Move the model to the detected device
model.eval()

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
landmarks_path = os.path.expanduser('~/Desktop/Green-Tea/Yawning/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(landmarks_path)

# Define the indices for eyes and mouth
left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))
mouth_indices = list(range(48, 68))

# Eye state threshold
EAR_THRESHOLD = 0.26

# For PiCamera v2 via CSI - Lowest Resolution:
pipeline_picam_csi_lowres = (
    "nvarguscamerasrc ! "
    "'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1' ! "
    "nvvidconv ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! " #added BGR format to allow for cv2
    "appsink"
)

cap = cv2.VideoCapture(pipeline_picam_csi_lowres, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
# Define colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

def main():
    while True:
        start_time = time.time()  # Start timing the frame processing

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            # ===== Eye Detection =====
            # Get landmarks for eyes
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_indices])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_indices])

            # Calculate eye aspect ratios
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Determine eye states
            left_eye_state = "Closed" if left_ear < EAR_THRESHOLD else "Open"
            right_eye_state = "Closed" if right_ear < EAR_THRESHOLD else "Open"

            # Draw eye landmarks (blue for better visibility)
            for point in left_eye:
                cv2.circle(frame, tuple(point), 2, BLUE, -1)
            for point in right_eye:
                cv2.circle(frame, tuple(point), 2, BLUE, -1)

            # ===== Mouth/Yawn Detection =====
            # Extract mouth landmarks
            mouth_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in mouth_indices])

            # Get bounding rectangle of mouth with padding
            x, y, w, h = cv2.boundingRect(mouth_points)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)

            # Extract mouth region
            mouth_roi = frame[y:y+h, x:x+w]

            yawn_state = "Unknown"  # Initialize yawn_state with a default value

            if mouth_roi.size > 0:
                try:
                    # Use OpenCV for resizing and directly create tensor
                    resized_roi = cv2.resize(mouth_roi, (224, 224))
                    img_tensor = torch.from_numpy(resized_roi.transpose((2, 0, 1))).float() / 255.0
                    input_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img_tensor).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        prediction_value = output.item()
                        prediction = (output > 0.5).float()

                    yawn_state = "Yawning" if prediction_value == 1 else "Not Yawning"
                    yawn_color = RED if yawn_state == "Yawning" else GREEN
                    cv2.putText(frame, yawn_state, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)

                except Exception as e:
                    print(f"Error processing mouth region: {e}")
                    yawn_state = "Error"  # Set yawn_state to indicate an error

            # Draw mouth landmarks (blue) and rectangle (green)
            for point in mouth_points:
                cv2.circle(frame, tuple(point), 2, BLUE, -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)

            # Display eye information (green text)
            cv2.putText(frame, f"Left: {left_eye_state} ({left_ear:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            cv2.putText(frame, f"Right: {right_eye_state} ({right_ear:.2f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

            # Combined state information (now in green with improved formatting)
            state_text = f"State: Eyes {left_eye_state[0]}/{right_eye_state[0]} | Mouth: {yawn_state}"
            cv2.putText(frame, state_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

            # Add a subtle background for the state text for better visibility
            text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (5, 70), (15 + text_size[0], 95), (0, 0, 0), -1)
            cv2.putText(frame, state_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        cv2.imshow('Fatigue Detection System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Frame processing time: {processing_time:.4f} seconds")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)