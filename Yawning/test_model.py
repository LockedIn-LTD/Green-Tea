import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import dlib

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

# Load the trained yawn detection model (on CPU)
model = YawnCNN()
model.load_state_dict(torch.load('yawn_detection.pth', map_location=torch.device('cpu')))
model.eval()

# Transformation to match input format for yawn detection
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the indices for eyes and mouth
left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))
mouth_indices = list(range(48, 68))

# Eye state threshold
EAR_THRESHOLD = 0.26

# Open video stream
cap = cv2.VideoCapture(0)

# Define colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

while True:
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
        
        # Draw mouth landmarks (blue) and rectangle (green)
        for point in mouth_points:
            cv2.circle(frame, tuple(point), 2, BLUE, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)
        
        # Yawn detection
        if mouth_roi.size > 0:
            try:
                input_image = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB)
                input_image = Image.fromarray(input_image)
                input_tensor = transform(input_image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = (output > 0.5).float()
                
                yawn_state = "Yawning" if prediction.item() == 1 else "Not Yawning"
                yawn_color = RED if yawn_state == "Yawning" else GREEN
                cv2.putText(frame, yawn_state, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)
                
            except Exception as e:
                print(f"Error processing mouth region: {e}")
        
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

cap.release()
cv2.destroyAllWindows()